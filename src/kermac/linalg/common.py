import threading
import nvmath

from ..common import FillMode

class Singleton(type):
    """Metaclass for creating singleton classes."""
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
class cusolverDnHandle(metaclass=Singleton):
    def __init__(self):
        self._cusolver_handle = nvmath.bindings.cusolverDn.create()
        self._cusolver_params = nvmath.bindings.cusolverDn.create_params()
    
    def __del__(self):
        nvmath.bindings.cusolverDn.destroy_params(self._cusolver_params)
        nvmath.bindings.cusolverDn.destroy(self._cusolver_handle)


def map_fill_mode(fill_mode):
    if fill_mode == FillMode.UPPER:
        return nvmath.bindings.cublas.FillMode.UPPER
    else:
        return nvmath.bindings.cublas.FillMode.LOWER
