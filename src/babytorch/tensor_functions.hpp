#pragma once

#include "operators.hpp"
#include "tensor_autodiff.hpp"
#include "ptr.hpp"

namespace tensor {
    struct Tensor;
}

namespace tensor_functions {

    using tensor::Tensor;
    using tensor_autodiff::Context;

    struct Neg {
        static Tensor forward(Context&, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Inv {
        static Tensor forward(Context&, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Relu {
        static Tensor forward(Context&, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Sigmoid {
        static Tensor forward(Context&, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Log {
        static Tensor forward(Context&, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Exp {
        static Tensor forward(Context&, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Add {
        static Tensor forward(Context&, const sptr<Tensor>, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Mul {
        static Tensor forward(Context&, const sptr<Tensor>, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Lt {
        static Tensor forward(Context&, const sptr<Tensor>, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Eq {
        static Tensor forward(Context&, const sptr<Tensor>, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Max {
        static Tensor forward(Context&, const sptr<Tensor>, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Is_close {
        static Tensor forward(Context&, const sptr<Tensor>, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

    struct Copy {
        static Tensor forward(Context&, const sptr<Tensor>);
        static std::array<Tensor, 2> backward(Context&, const sptr<Tensor>);
    };

}