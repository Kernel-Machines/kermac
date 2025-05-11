#include <torch/extension.h>

#include "kermac.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_p_norm", &tensor_p_norm, "tensor_p_norm");
}