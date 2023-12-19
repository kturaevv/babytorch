#pragma once

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "autodiff.hpp"
#include "operators.hpp"
#include "scalarlike.hpp"

namespace functions {

    using namespace autodiff;

    struct Func {
        static double forward();
        static double backward();
    };

    struct ScalarFunction {
        template <typename F, typename... Args>
        static auto apply(Args&&... args) noexcept {
            std::vector<ScalarLike> scalars{ args... };
            std::vector<double> raw_vals{ (args.data)... };

            auto ctx = autodiff::Context();

            double result = F::forward(ctx, args.data...);
            return ScalarLike(result);
        }
    };

    struct Id : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::id(self);
        }

        static double backward(const Context ctx, const double deriv) {
            return 1.0;
        }
    };

    struct Neg : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::neg(self);
        }

        static double backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return 1.0;
        }
    };

    struct Inv : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::inv(self);
        }

        static double backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return operators::inv_back(self, deriv);
        }
    };

    struct Relu : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::relu(self);
        }

        static double backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return operators::relu_back(self, deriv);
        }
    };

    struct Sigmoid : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::sigmoid(self);
        }

        static double backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return operators::sigmoid_back(self, deriv);
        }
    };

    struct Log : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::log_func(self);
        }

        static double backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return operators::log_back(self, deriv);
        }
    };

    struct Exp : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::exp_func(self);
        }

        static double backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return self * deriv;
        }
    };

    struct Add : Func {
        static double forward(Context ctx, const double self, const double other) {
            ctx.save_for_backwards(self);
            return operators::add(self, other);
        }

        static std::tuple<double, double> backward(const Context ctx,
                                                   const double deriv) {
            double self = ctx.saved_values[0];
            return { deriv, deriv };
        }
    };

    struct Mul : Func {
        static double forward(Context ctx, const double self, const double other) {
            ctx.save_for_backwards(self, other);
            return operators::mul(self, other);
        }

        static std::tuple<double, double> backward(const Context ctx,
                                                   const double deriv) {
            double self = ctx.saved_values[0];
            double other = ctx.saved_values[1];
            return { other * deriv, self * deriv };
        }
    };

    struct Lt : Func {
        static double forward(Context ctx, const double self, const double other) {
            return operators::lt(self, other);
        }

        static std::tuple<double, double> backward(const Context ctx,
                                                   const double deriv) {
            return { 0.0, 0.0 };
        }
    };

    struct Eq : Func {
        static double forward(Context ctx, const double self, const double other) {
            return operators::eq(self, other);
        }

        static std::tuple<double, double> backward(const Context ctx,
                                                   const double deriv) {
            return { 0.0, 0.0 };
        }
    };

    struct Max : Func {
        static double forward(Context ctx, const double self, const double other) {
            ctx.save_for_backwards(self, other);
            return operators::max(self, other);
        }

        static std::tuple<double, double> backward(const Context ctx,
                                                   const double deriv) {
            return { 0.0, 0.0 };
        }
    };

    struct Is_close : Func {
        static double forward(Context ctx, const double self, const double other) {
            return operators::is_close(self, other);
        }

        static std::tuple<double, double> backward(const Context ctx,
                                                   const double deriv) {
            return { 0.0, 0.0 };
        }
    };
}
