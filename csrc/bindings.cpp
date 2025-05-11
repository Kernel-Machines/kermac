#include <torch/extension.h>

#include "kermac_pytorch.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_p_norm_pytorch", &_p_norm_pytorch, "_p_norm_pytorch");
}