#pragma once

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "operators.hpp"
#include "scalarlike.hpp"

namespace functions {

    struct Func {
        static double forward();
        static double backward();
    };

    struct ScalarFunction {
        template <typename F, typename... Args>
        static auto apply(Args&&... args) noexcept {
            double result = F::forward(args.data...);
            return ScalarLike(result);
        }
    };

    struct Id : Func {
        static double forward(const double self) {
            return operators::id(self);
        }
    };

    struct Neg : Func {
        static double forward(const double self) {
            return operators::neg(self);
        }
    };

    struct Inv : Func {
        static double forward(const double self) {
            return operators::inv(self);
        }
    };

    struct Relu : Func {
        static double forward(const double self) {
            return operators::relu(self);
        }
    };

    struct Sigmoid : Func {
        static double forward(const double self) {
            return operators::sigmoid(self);
        }
    };

    struct Log : Func {
        static double forward(const double self) {
            return operators::log_func(self);
        }
    };

    struct Exp : Func {
        static double forward(const double self) {
            return operators::exp_func(self);
        }
    };

    struct Add : Func {
        static double forward(const double self, const double other) {
            return operators::add(self, other);
        }
    };

    struct Mul : Func {
        static double forward(const double x, const double y) {
            return operators::mul(x, y);
        }
    };

    struct Lt : Func {
        static double forward(const double x, const double y) {
            return operators::lt(x, y);
        }
    };

    struct Eq : Func {
        static double forward(const double x, const double y) {
            return operators::eq(x, y);
        }
    };

    struct Max : Func {
        static double forward(const double x, const double y) {
            return operators::max(x, y);
        }
    };

    struct Is_close : Func {
        static double forward(const double x, const double y) {
            return operators::is_close(x, y);
        }
    };

    struct Log_back : Func {
        static double forward(const double x, const double d) {
            return operators::log_back(x, d);
        }
    };

    struct Inv_back : Func {
        static double forward(const double x, const double d) {
            return operators::inv_back(x, d);
        }
    };

    struct Relu_back : Func {
        static double forward(const double x, const double d) {
            return operators::relu_back(x, d);
        }
    };
}  // namespace minitorch
