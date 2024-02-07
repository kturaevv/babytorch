#pragma once

#include "operators.hpp"
#include "tensor_autodiff.hpp"

namespace tensor {
    struct Tensor;
}

namespace tensor_functions {

    using tensor::Tensor;
    using tensor_autodiff::Context;

    struct Id {
        static Tensor forward(Context&, const Tensor&);
    };

    struct Neg {
        static Tensor forward(Context&, const Tensor&);
    };

    struct Inv {
        static Tensor forward(Context&, const Tensor&);
    };

    struct Relu {
        static Tensor forward(Context&, const Tensor&);
    };

    struct Sigmoid {
        static Tensor forward(Context&, const Tensor&);
    };

    struct Log {
        static Tensor forward(Context&, const Tensor&);
    };

    struct Exp {
        static Tensor forward(Context&, const Tensor&);
    };

    struct Add {
        static Tensor forward(Context&, const Tensor&, const Tensor&);
    };

    struct Mul {
        static Tensor forward(Context&, const Tensor&, const Tensor&);
    };

    struct Lt {
        static Tensor forward(Context&, const Tensor&, const Tensor&);
    };

    struct Eq {
        static Tensor forward(Context&, const Tensor&, const Tensor&);
    };

    struct Max {
        static Tensor forward(Context&, const Tensor&, const Tensor&);
    };

    struct Is_close {
        static Tensor forward(Context&, const Tensor&, const Tensor&);
    };

}