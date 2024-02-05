#pragma once

#include "operators.hpp"
#include "tensor_autodiff.hpp"

namespace tensor {
    struct Tensor;
}

namespace tensor_functions {

    using tensor::Tensor;
    using tensor_autodiff::Context;

    struct Add {
        static Tensor forward(Context&, const Tensor&, const Tensor&);
    };
}  // namespace tensor_functions