#pragma once

#include <array>
#include <memory>

#include "autodiff.hpp"
#include "operators.hpp"

namespace functions {
    using namespace autodiff;

    struct Id {
        static double forward(Context& ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::id(self);
        }

        static std::array<double, 2> backward(const Context&, const double deriv) {
            return { 1.0 * deriv };
        }
    };

    struct Neg {
        static double forward(Context& ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::neg(self);
        }

        static std::array<double, 2> backward(const Context&, const double deriv) {
            return { -1.0 * deriv };
        }
    };

    struct Inv {
        static double forward(Context& ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::inv(self);
        }

        static std::array<double, 2> backward(const Context& ctx,
                                              const double deriv) {
            double self = ctx.saved_values[0];
            return { operators::inv_back(self, deriv) };
        }
    };

    struct Relu {
        static double forward(Context& ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::relu(self);
        }

        static std::array<double, 2> backward(const Context& ctx,
                                              const double deriv) {
            double self = ctx.saved_values[0];
            return { operators::relu_back(self, deriv) };
        }
    };

    struct Sigmoid {
        static double forward(Context& ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::sigmoid(self);
        }

        static std::array<double, 2> backward(const Context& ctx,
                                              const double deriv) {
            double self = ctx.saved_values[0];
            return { operators::sigmoid_back(self, deriv) };
        }
    };

    struct Log {
        static double forward(Context& ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::log_func(self);
        }

        static std::array<double, 2> backward(const Context& ctx,
                                              const double deriv) {
            double self = ctx.saved_values[0];
            return { operators::log_back(self, deriv) };
        }
    };

    struct Exp {
        static double forward(Context& ctx, const double self) {
            ctx.save_for_backwards(self);
            return operators::exp_func(self);
        }

        static std::array<double, 2> backward(const Context& ctx,
                                              const double deriv) {
            double self = ctx.saved_values[0];
            return { self * deriv };
        }
    };

    struct Add {
        static double forward(Context& ctx, const double self,
                              const double other) {
            ctx.save_for_backwards(self);
            return operators::add(self, other);
        }

        static std::array<double, 2> backward(const Context&, const double deriv) {
            return { deriv, deriv };
        }
    };

    struct Mul {
        static double forward(Context& ctx, const double self,
                              const double other) {
            ctx.save_for_backwards(self, other);
            return operators::mul(self, other);
        }

        static std::array<double, 2> backward(const Context& ctx,
                                              const double deriv) {
            double self = ctx.saved_values[0];
            double other = ctx.saved_values[1];
            return { other * deriv, self * deriv };
        }
    };

    struct Lt {
        static double forward(Context&, const double self, const double other) {
            return operators::lt(self, other);
        }

        static std::array<double, 2> backward(const Context&, const double) {
            return { 0.0, 0.0 };
        }
    };

    struct Eq {
        static double forward(Context&, const double self, const double other) {
            return operators::eq(self, other);
        }

        static std::array<double, 2> backward(const Context&, const double) {
            return { 0.0, 0.0 };
        }
    };

    struct Max {
        static double forward(Context& ctx, const double self,
                              const double other) {
            ctx.save_for_backwards(self, other);
            return operators::max(self, other);
        }

        static std::array<double, 2> backward(const Context&, const double) {
            return { 0.0, 0.0 };
        }
    };

    struct Is_close {
        static double forward(Context&, const double self, const double other) {
            return operators::is_close(self, other);
        }

        static std::array<double, 2> backward(const Context&, const double) {
            return { 0.0, 0.0 };
        }
    };
}