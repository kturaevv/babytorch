#pragma once

#include <memory>

#include "autodiff.hpp"
#include "operators.hpp"

namespace functions {
    using namespace autodiff;

    struct Func {
        static double forward(double x, double y) noexcept;
        static std::tuple<double> backward(double x, double y) noexcept;
    };

    struct Id : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::id(self);
        }

        static std::tuple<double> backward(const Context ctx, const double deriv) {
            return 1.0;
        }
    };

    struct Neg : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::neg(self);
        }

        static std::tuple<double> backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return { 1.0 };
        }
    };

    struct Inv : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::inv(self);
        }

        static std::tuple<double> backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return { operators::inv_back(self, deriv) };
        }
    };

    struct Relu : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::relu(self);
        }

        static std::tuple<double> backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return { operators::relu_back(self, deriv) };
        }
    };

    struct Sigmoid : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::sigmoid(self);
        }

        static std::tuple<double> backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return { operators::sigmoid_back(self, deriv) };
        }
    };

    struct Log : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::log_func(self);
        }

        static std::tuple<double> backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return { operators::log_back(self, deriv) };
        }
    };

    struct Exp : Func {
        static double forward(Context ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::exp_func(self);
        }

        static std::tuple<double> backward(const Context ctx, const double deriv) {
            double self = ctx.saved_values[0];
            return { self * deriv };
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