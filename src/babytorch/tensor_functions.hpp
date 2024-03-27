#pragma once

#include "ptr.hpp"
#include "tensor_autodiff.hpp"

namespace tensor {
    class Tensor;
}

namespace tensor_functions {

    using tensor::Tensor;
    using tensor_autodiff::Context;

    struct Neg {
        static sptr<Tensor> forward(Context&, const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Inv {
        static sptr<Tensor> forward(Context&, const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Relu {
        static sptr<Tensor> forward(Context&, const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Sigmoid {
        static sptr<Tensor> forward(Context&, const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Log {
        static sptr<Tensor> forward(Context&, const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Exp {
        static sptr<Tensor> forward(Context&, const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Add {
        static sptr<Tensor> forward(Context&,
                                    const sptr<Tensor>&,
                                    const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Mul {
        static sptr<Tensor> forward(Context&,
                                    const sptr<Tensor>&,
                                    const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Lt {
        static sptr<Tensor> forward(Context&,
                                    const sptr<Tensor>&,
                                    const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Eq {
        static sptr<Tensor> forward(Context&,
                                    const sptr<Tensor>&,
                                    const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Max {
        static sptr<Tensor> forward(Context&,
                                    const sptr<Tensor>&,
                                    const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Is_close {
        static sptr<Tensor> forward(Context&,
                                    const sptr<Tensor>&,
                                    const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

    struct Copy {
        static sptr<Tensor> forward(Context&, const sptr<Tensor>&);
        static std::array<sptr<Tensor>, 2> backward(Context&,
                                                    const sptr<Tensor>&);
    };

}
