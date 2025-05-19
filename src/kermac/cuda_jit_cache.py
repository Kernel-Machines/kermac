import threading
from typing import Dict, Any, Tuple
import sys
import gzip
import io

from cuda.core.experimental import Device, Program, ProgramOptions, ObjectCode

from .paths import *

class Singleton(type):
    """Metaclass for creating singleton classes."""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
class DeviceModuleMap(metaclass=Singleton):
    """Singleton class mapping device IDs to lazily loaded modules."""
    
    def __init__(self):
        self._modules: Dict[Tuple[int, str], Any] = {}  # device_id -> module
        self._lock = threading.Lock()

    def get_module(self, device: Device, function_name : str, storage_name : str, debug = False) -> Any:
        device_id = device.device_id
        if device_id < 0:
            raise ValueError(f"Invalid device ID: {device_id}")

        key = (device_id, function_name)
        with self._lock:
            if key not in self._modules:
                arch = "".join(f"{i}" for i in device.compute_capability)
                package_name = get_package_name()
                package_version = get_package_version()
                cubin_path = get_cache_cubin_dir() / f'{package_name}.{package_version}.{arch}.{storage_name}.cubin'
                if debug:
                    print(f'(Kermac Debug) Loaded module not found for (device:{device_id}, function:{function_name})')
                    print(f'(Kermac Debug) For {cubin_path}')
                if cubin_path.is_file():
                    if debug:
                        print(f'(Kermac Debug)\t\tFound pre-built cubin')
                    module_cubin = ObjectCode.from_cubin(str(cubin_path))
                else:
                    if debug:
                        print(f'(Kermac Debug)\t\tNot found, building..')
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

                                include_path=[
                                    get_include_local_cuda_dir(),   # *.cuh
                                    get_include_dir_cutlass(),      # main cutlass include
                                    get_include_dir_cuda()          # cuda toolkit for <cuda/src/assert>, etc.. (dependency of cutlass)
                                ],
                            )
                    ).compile(
                        "cubin", 
                        logs=sys.stdout,
                        name_expressions=[function_name]
                    )
                    with open(cubin_path, 'wb') as file:
                        file.write(module_cubin.code)
                    if debug:
                        print(f'(Kermac Debug)\t\tBuilt and saved')
                self._modules[key] = module_cubin
                return module_cubin
            else:
                if debug:
                    print(f'(Kermac Debug) Loaded module found for (device:{device_id}, function:{function_name})')
            return self._modules[key]
