from cuda.core.experimental import Device
from enum import Enum, auto
from .module_cache import *

class PowerType(Enum):
    FRACTIONAL = auto()
    FIXED_L1 = auto()
    FIXED_L2 = auto()

class InnerOperator(Enum):
    DIFF = auto()
    DOT = auto()

class KernelType(Enum):
    NONE = auto()
    LAPLACE = auto()
    GAUSSIAN = auto()

class Symmetry(Enum):
    NonSymmetric = auto()
    Symmetric = auto()

class BuildAKernelDescriptor():
    def __init__(
        self,
        inner_power : PowerType = PowerType.FIXED_L2,
        inner_operator : InnerOperator = InnerOperator.DIFF,
        outer_power : PowerType = PowerType.FIXED_L2,
        kernel_type : KernelType = KernelType.LAPLACE,
        symmetry : Symmetry = Symmetry.NonSymmetric,
    ):
        self._inner_power = inner_power
        self._inner_operator = inner_operator
        self._outer_power = outer_power
        self._kernel_type = kernel_type
        self._symmetrc = symmetry

def build_a_kernel(
    device,
    inner_power : PowerType = PowerType.FIXED_L2,
    inner_operator: InnerOperator = InnerOperator.DIFF,
    outer_power : PowerType = PowerType.FIXED_L2,
    kernel_type : KernelType = KernelType.LAPLACE,
    debug = False
):
    pt_device_id = device.index
    device = Device(pt_device_id)
    print(device.compute_capability)

    device_module_map = DeviceModuleMap(debug)

    function_string = f'cute_build_a_kernel_m128n128k8p3<true>'
    module_cubin = device_module_map.get_module(device, function_string, debug=debug)
    kernel = module_cubin.get_kernel(function_string)

    build_a_kernel_descriptor = BuildAKernelDescriptor(
        inner_power=inner_power,
        inner_operator=inner_operator,
        outer_power=outer_power,
        kernel_type=kernel_type,
        symmetry=Symmetry.NonSymmetric
    )

    return build_a_kernel_descriptor