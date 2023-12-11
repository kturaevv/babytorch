#pragma once

#include <functional>  // Include for std::function
#include <optional>
#include <vector>

#include "functions.hpp"
#include "scalarlike.hpp"

namespace scalar {
    using namespace functions;

    struct History {
        std::optional<Func> fn;
        std::vector<ScalarLike> inputs;

        History(std::optional<Func> fn_ = std::nullopt,
                const std::vector<ScalarLike>& inputs_ = {})
            : fn(fn_)
            , inputs(inputs_) {
        }
    };

    struct Scalar : ScalarLike {
        //
        double data;
        double grad;
        History history;

        template <typename Other>
        Scalar operator+(const Other& other) const {
            return ScalarFunction::apply<Add>(this, other);
        }

        // Scalar operator*(const Scalar& other) const {
        //     return ScalarFunction::apply<Mul>(this, other);
        // }

        // Scalar operator^(const Scalar& power) const {
        //     return Functions::apply<Pow>(*this, power);
        // }

        // Scalar operator-() const {
        //     return *this * -1;
        // }

        // Scalar operator-() const {
        //     return *this + (-other);
        // }

        // Scalar operator+(const Scalar& other) const {
        //     return *this + other;
        // }

        // Scalar operator-(const Scalar& other) const {
        //     return *this + (-other);
        // }

        // Scalar operator*(const Scalar& other) const {
        //     return *this * other;
        // }

        // Scalar operator/(const Scalar& other) const {
        //     return *this * pow(other, -1);
        // }

        Scalar log() const;
        Scalar exp() const;
        Scalar sigmoid() const;
        Scalar relu() const;
        Scalar tanh() const;

        void accumulate_grad(double d_x);
        bool is_leaf() const;
        void backward();
        int id() const;
        void chain_rule(Scalar deriv);
        std::vector<Scalar> parents() const;
    };

}