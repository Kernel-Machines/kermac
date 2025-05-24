import threading
from typing import Dict, Any, Tuple, Optional, List
import sys
import torch

from cuda.core.experimental._module import Kernel
from cuda.core.experimental import Device, Program, ProgramOptions, ObjectCode

from .paths import *
from .common import hash_cuda_include_files, get_compute_capability
from .cubin_database import *

def compile_functions(
    arch,
    function_names,
    debug = False
):
    module_cubin = Program(
        '#include <kermac.cuh>',
        code_type="c++", 
        options= \
            ProgramOptions(
                std="c++17",
                arch=f"sm_{arch}",
                device_as_default_execution_space=True,
                # diag_suppress cutlass: 64-D: declaration does not declare anything
                # diag_suppress cutlass: 1055-D: declaration does not declare anything
                diag_suppress=[64,1055],
                ptxas_options=['-v'] if debug else None,
                # some good ones
                # device_code_optimize=True,
                # extensible_whole_program=True,
                # ftz=True,
                # extra_device_vectorization=True,
                # restrict=True,
                # use_fast_math=True,
                # prec_sqrt=False,
                # prec_div=False,
                # split_compile=8,
                include_path=[
                    get_include_local_cuda_dir(),   # include/*.cuh
                    get_include_dir_cutlass(),      # thirdparty/cutlass/include
                    get_include_dir_cuda()          # cuda toolkit for <cuda/src/assert>, etc.. (dependency of cutlass)
                ],
            )
    ).compile(
        "cubin", 
        logs=sys.stdout,
        name_expressions=function_names
    )
    return module_cubin

def compile_and_cache_functions(
    database: CubinDatabase,
    cuda_version: str,
    arch: str,
    function_names: List[str],
    debug = False
):
    function_db_keys_to_compile = []
    function_names_to_compile = []

    if debug:
        print(f'(Kermac Debug) Checking which functions need to be compiled (sm_{arch})')
    for function_name in function_names:
        function_db_key = \
            FunctionDBKey(
                package_name=get_package_name(),
                package_version=get_package_version(),
                cuda_version=cuda_version,
                arch = arch,
                function_name=function_name
            )
        function_db_value = database.get_function_mapping(function_db_key)
        if not function_db_value:
            function_db_keys_to_compile.append(function_db_key)
            function_names_to_compile.append(function_name)

    if function_names_to_compile == []:
        if debug:
            print(f'(Kermac Debug) Nothing needs to compile (sm_{arch})')
        return True
    if debug:
        for function_name_to_compile in function_names_to_compile:
            print(f'(Kermac Debug) Compiling (sm_{arch}): {function_name_to_compile}') 
    module_cubin = compile_functions(
        arch, 
        function_names_to_compile,
        debug
    )

    cubin_data_hash = hashlib.sha256(module_cubin.code).digest()
    if debug:
        print(f'(Kermac Debug) Storing function mappings to database')
    for function_db_key in function_db_keys_to_compile:
        lowered_name = module_cubin._sym_map[function_db_key.function_name]
        function_db_value = \
            FunctionDBValue(
                lowered_name=lowered_name,
                cubin_data_hash=cubin_data_hash
            )
        database.put_function_mapping(key=function_db_key, value=function_db_value)
    if debug:
        print(f'(Kermac Debug) Storing cubin to database')
    database.put_cubin(data_hash=cubin_data_hash, cubin_data=module_cubin.code)
    if debug:
        print(f'(Kermac Debug) Stored all mappings to database')
    return True

class Singleton(type):
    """Metaclass for creating singleton classes."""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
class ModuleCache(metaclass=Singleton):
    """Singleton class mapping device IDs to lazily loaded modules/functions."""
    
    def __init__(self, debug = False):
        # A loaded cubin module is stored in device memory
        # Should have a dictionary to keep it live to pull kernel functions out of
        # (device_id, cubin_data_hash) -> cubin module
        self._loaded_modules : Dict[Tuple[int, bytes], ObjectCode] = {}

        # A loaded kernel function is stored in device memory also
        # (device_id, function_name) -> Kernel
        self._loaded_kernel_functions: Dict[Tuple[int, str], Kernel] = {}  
        self._lock = threading.Lock()
        if debug:
            print(f'(Kermac Debug) Using database at: {cache_root().resolve()}')
        directory = get_include_local_cuda_dir()
        hash_result = hash_cuda_include_files(directory)
        if debug:
            print(f"(Kermac Debug) Combined hash of cuda source files: {hash_result}")
        self._db = \
            CubinDatabase(
                cache_dir=str(cache_root().resolve()),
                max_size_mb=1024,
                current_file_src_hash=hash_result.encode(),
                debug=debug
            )
        self._cuda_version = str(torch.version.cuda)

    def compile_and_cache_functions(
        self,
        device,
        function_names: List[str],
        debug = False
    ):
        arch = get_compute_capability(device)
        compile_and_cache_functions(
            database=self._db,
            cuda_version=self._cuda_version,
            arch=arch,
            function_names=function_names,
            debug=debug
        )

    def get_function(self, device: Device, function_name : str, debug = False) -> Any:
        device_id = device.device_id
        if device.compute_capability.major < 8:
            raise ValueError(f"Invalid device compute capability, (device:{device.compute_capability}, requrires at least:8.0")

        function_dict_key = (device_id, function_name)
        with self._lock:
            # Check if this function is already loaded on this device
            if function_dict_key in self._loaded_kernel_functions:
                if debug: 
                    print(f'(Kermac Debug) Loaded function found for (device:{device_id}, function:{function_name})')
                kernel = self._loaded_kernel_functions[function_dict_key]
                return kernel

            if debug: 
                print(f'(Kermac Debug) Loaded function found not found for (device:{device_id}, function:{function_name})')
            arch = get_compute_capability(device)
            function_db_key = \
                FunctionDBKey(
                    package_name=get_package_name(),
                    package_version=get_package_version(),
                    cuda_version=self._cuda_version,
                    arch=arch,
                    function_name=function_name
                )

            # Check database if this function is already built for this arch
            # The cubin module may or may not be loaded on this device
            function_db_value = self._db.get_function_mapping(function_db_key)
            if not function_db_value:
                if debug: 
                    print(f'(Kermac Debug) Mapping does not exist for {function_db_key}')
                # The cubin for this function doesn't exist
                # Need to compile it
                success = compile_and_cache_functions(
                    database=self._db,
                    cuda_version=self._cuda_version,
                    arch=arch, 
                    function_names=[function_name], 
                    debug=debug
                )

                assert success
            else: 
                if debug: 
                    print(f'(Kermac Debug) Mapping does exist for {function_db_key}')
            # The entry should exist now
            function_db_value = self._db.get_function_mapping(function_db_key)
            if not function_db_value:
                assert False
            # There is a mapping of the function to a cubin in the database
            cubin_data_hash = function_db_value.cubin_data_hash
            lowered_name = function_db_value.lowered_name

            # Need to check if the cubin is in a loaded module for this device
            module_dict_key = (device_id, cubin_data_hash)
            if not module_dict_key in self._loaded_modules:
                if debug: 
                    print(f'(Kermac Debug) Module not already loaded belonging to {function_name}')
                # The module is not already loaded on this device
                cubin_code = self._db.get_cubin(function_db_value.cubin_data_hash)
                loaded_module = ObjectCode.from_cubin(cubin_code)
                # Store this loaded module in the dict for later
                self._loaded_modules[module_dict_key] = loaded_module
            else:
                if debug: 
                    print(f'(Kermac Debug) Module already loaded belonging to {function_name}')

            loaded_module = self._loaded_modules[module_dict_key]
            # Need to construct a mapping for the function to the lowered name
            symbol_map = {function_name: lowered_name}
            loaded_module._sym_map = symbol_map
            kernel = loaded_module.get_kernel(function_name)
            # Update the dict so it knows the function for this device is loaded
            self._loaded_kernel_functions[function_dict_key] = kernel
            return kernel
