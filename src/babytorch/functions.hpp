#pragma once

#include <memory>

namespace scalar {
    struct Scalar;
}

namespace functions {

    using namespace scalar;

    struct Func {
        static double forward(double x, double y) noexcept;
        static double backward(double x, double y) noexcept;
    };

    struct ScalarFunction {
        template <typename F, typename... Args>
        static std::shared_ptr<Scalar> apply(Args&&... args);
    };

    struct Id;
    struct Neg;
    struct Inv;
    struct Relu;
    struct Sigmoid;
    struct Log;
    struct Exp;
    struct Add;
    struct Mul;
    struct Lt;
    struct Eq;
    struct Max;
    struct Is_close;
}