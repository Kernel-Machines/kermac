from enum import Enum, auto

class PowerType(Enum):
    FRACTIONAL = auto()
    FIXED_L1 = auto()
    FIXED_L2 = auto()

class KernelType(Enum):
    NONE = auto()
    LAPLACE = auto()
    GAUSSIAN = auto()

def build_a_kernel(
    inner_power : PowerType = PowerType.FIXED_L2,
    outer_power : PowerType = PowerType.FIXED_L2,
    kernel_type : KernelType = KernelType.LAPLACE
):
    pass