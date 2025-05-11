#include <torch/extension.h>

#include "kermac.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_stats", &tensor_stats, "tensor_stats");
}