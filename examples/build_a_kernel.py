import torch
import kermac

kernel_descriptor = kermac.KernelDescriptor(
    inner_operator=kermac.InnerOperator.DIFF,
    inner_power=kermac.PowerType.POW,
    outer_power=kermac.PowerType.POW,
    kernel_type=kermac.KernelType.LAPLACE,
)

print(kernel_descriptor._function_name)