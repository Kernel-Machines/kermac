import torch
import kermac

device = torch.device('cuda')

kermac.build_a_kernel(
    device,
    inner_power=kermac.PowerType.FIXED_L2,
    inner_operator=kermac.InnerOperator.DIFF,
    outer_power=kermac.PowerType.FIXED_L2,
    kernel_type=kermac.KernelType.LAPLACE,
    debug = True
)