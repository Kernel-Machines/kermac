import threading
from typing import Dict, Any, Tuple, Optional, List
import sys
import torch

from cuda.core.experimental._module import Kernel
from cuda.core.experimental import Device, Program, ProgramOptions, ObjectCode

from .paths import *
from .common import hash_cuda_include_files, get_compute_capability

import lmdb
import os

import struct

class FunctionDBKey:
    """Represents the key structure for the function database."""
    def __init__(
        self,
        package_name: str,
        package_version: str,
        cuda_version: str,
        arch: str,
        function_name: str
    ):
        self.package_name = package_name
        self.package_version = package_version
        self.cuda_version = cuda_version
        self.arch = arch
        self.function_name = function_name

    def to_bytes(self) -> bytes:
        """Serialize the key to a bytes object for LMDB storage."""
        key_dict = {
            'package_name': self.package_name,
            'package_version': self.package_version,
            'cuda_version': self.cuda_version,
            'arch': self.arch,
            'function_name': self.function_name
        }
        return json.dumps(key_dict, sort_keys=True).encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FunctionDBKey':
        """Deserialize bytes to a FunctionDBKey object."""
        key_dict = json.loads(data.decode('utf-8'))
        return cls(
            package_name=key_dict['package_name'],
            package_version=key_dict['package_version'],
            cuda_version=key_dict['cuda_version'],
            arch=key_dict['arch'],
            function_name=key_dict['function_name']
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionDBKey):
            return False
        return (
            self.package_name == other.package_name and
            self.package_version == other.package_version and
            self.cuda_version == other.cuda_version and
            self.arch == other.arch and
            self.function_name == other.function_name
        )

    def __repr__(self) -> str:
        return (f"FunctionDBKey(package_name={self.package_name}, "
                f"package_version={self.package_version}, "
                f"cuda_version={self.cuda_version}, "
                f"arch={self.arch}, "
                f"function_name={self.function_name})")

class FunctionDBValue:
    """Represents the value structure (lowered_name, cubin_data_hash) for the function database."""
    def __init__(self, lowered_name: bytes, cubin_data_hash: bytes):
        self.lowered_name = lowered_name
        self.cubin_data_hash = cubin_data_hash

    def to_bytes(self) -> bytes:
        """Serialize the value to a bytes object for LMDB storage."""
        lowered_len = len(self.lowered_name)
        return struct.pack('>I', lowered_len) + self.lowered_name + self.cubin_data_hash

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FunctionDBValue':
        """Deserialize bytes to a FunctionDBValue object."""
        lowered_len = struct.unpack('>I', data[:4])[0]
        lowered_name = data[4:4 + lowered_len]
        cubin_data_hash = data[4 + lowered_len:]
        return cls(lowered_name=lowered_name, cubin_data_hash=cubin_data_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionDBValue):
            return False
        return (
            self.lowered_name == other.lowered_name and
            self.cubin_data_hash == other.cubin_data_hash
        )

    def __repr__(self) -> str:
        return f"FunctionDBValue(lowered_name={self.lowered_name!r}, cubin_data_hash={self.cubin_data_hash!r})"

class CubinDatabase:
    """Manages the LMDB database for cubin data and source files hash."""
    def __init__(
            self, 
            cache_dir: str, 
            max_size_mb,
            current_file_src_hash,
            debug = False
        ):  # 10MB default
        os.makedirs(cache_dir, exist_ok=True)
        db_max_size_bytes = max_size_mb * 1024 * 1024
        self.env = lmdb.open(cache_dir, map_size=db_max_size_bytes, max_dbs=2)
        self.function_map_db = self.env.open_db(b'function_map_db')  # First mapping: key -> (lowered_name, cubin_data_hash)
        self.data_db = self.env.open_db(b'data_db')    # Second mapping: data_hash -> cubin_data, plus source files hash
        self._put_source_files_hash(current_file_src_hash, debug = debug)

    def put_function_mapping(self, key: FunctionDBKey, value: FunctionDBValue) -> None:
        """Store a key-value pair in the function_map_db."""
        with self.env.begin(write=True, db=self.function_map_db) as txn:
            txn.put(key.to_bytes(), value.to_bytes())

    def get_function_mapping(self, key: FunctionDBKey) -> Optional[FunctionDBValue]:
        """Retrieve a value from the function_map_db by key."""
        with self.env.begin(db=self.function_map_db) as txn:
            data = txn.get(key.to_bytes())
            return FunctionDBValue.from_bytes(data) if data is not None else None

    def put_cubin(self, data_hash: bytes, cubin_data: bytes) -> None:
        """Store a data_hash to cubin_data mapping in the data_db."""
        with self.env.begin(write=True, db=self.data_db) as txn:
            txn.put(data_hash, cubin_data)

    def get_cubin(self, data_hash: bytes) -> Optional[bytes]:
        """Retrieve cubin_data by data_hash from the data_db."""
        with self.env.begin(db=self.data_db) as txn:
            return txn.get(data_hash)

    def _put_source_files_hash(self, source_hash: bytes, debug = False) -> bool:
        """Store the source files hash in the data_db, clearing databases if hash doesn't match."""
        with self.env.begin(write=True) as txn:
            stored_hash = txn.get(b'source_files_hash', db=self.data_db)
            if stored_hash is not None and stored_hash != source_hash:
                if debug:
                    print(f"(Kermac Debug) File source hash mismatch (stored: {stored_hash}, provided: {source_hash}).")
                    print(f"(Kermac Debug) Clearing database of pre-built cubin entries")
                # Hash mismatch: drop both databases
                txn.drop(self.function_map_db, delete=False)  # Clear function_map_db
                txn.drop(self.data_db, delete=False)   # Clear data_db (including old hash)
                # Store new hash
                txn.put(b'source_files_hash', source_hash, db=self.data_db)
                if debug:
                    print(f"(Kermac Debug) Updated stored src hash to: {source_hash}")
                return True  # Indicate databases were cleared
            else:
                # No mismatch or no stored hash: just store the new hash
                txn.put(b'source_files_hash', source_hash, db=self.data_db)
                if debug:
                    print("(Kermac Debug) Hashes match. Keeping database pre-built cubin entries.")
                return False  # Indicate no clearing was needed

    def _get_source_files_hash(self) -> Optional[bytes]:
        """Retrieve the source files hash from the data_db."""
        with self.env.begin(db=self.data_db) as txn:
            return txn.get(b'source_files_hash')

    def close(self):
        """Close the LMDB environment."""
        self.env.close()

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
        print(f'(Kermac Debug) Checking which functions need to be compiled for arch: sm_{arch}')
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
            print(f'(Kermac Debug) Nothing needs to compile for arch sm_{arch}')
        return True
    if debug:
        for function_name_to_compile in function_names_to_compile:
            print(f'(Kermac Debug) Need to compile for arch sm_{arch}: {function_name_to_compile}') 
    module_cubin = compile_functions(
        arch, 
        function_names_to_compile,
        debug
    )

    cubin_data_hash = hashlib.sha256(module_cubin.code).digest()
    print(f'(Kermac Debug) Storing function mappings to database')
    for function_db_key in function_db_keys_to_compile:
        lowered_name = module_cubin._sym_map[function_db_key.function_name]
        function_db_value = \
            FunctionDBValue(
                lowered_name=lowered_name,
                cubin_data_hash=cubin_data_hash
            )
        database.put_function_mapping(key=function_db_key, value=function_db_value)
    print(f'(Kermac Debug) Storing cubin to database')
    database.put_cubin(data_hash=cubin_data_hash, cubin_data=module_cubin.code)
    print(f'(Kermac Debug) Stored')
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

            # if debug: 
            #     print(f'(Kermac Debug) Loaded module not found for (device:{device_id}, function:{function_name})')
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
                # The module is not already loaded on this device
                cubin_code = self._db.get_cubin(function_db_value.cubin_data_hash)
                loaded_module = ObjectCode.from_cubin(cubin_code)
                # Store this loaded module in the dict for later
                self._loaded_modules[module_dict_key] = loaded_module

            loaded_module = self._loaded_modules[module_dict_key]
            # Need to construct a mapping for the function to the lowered name
            symbol_map = {function_name: lowered_name}
            loaded_module._sym_map = symbol_map
            kernel = loaded_module.get_kernel(function_name)
            # Update the dict so it knows the function for this device is loaded
            self._loaded_kernel_functions[function_dict_key] = kernel
            return kernel
