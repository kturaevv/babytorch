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
        // using parent constructors
        using ScalarLike::ScalarLike;

        Scalar(const ScalarLike& other)
            : ScalarLike(other) {
        }

        // basic operations

        Scalar operator+(const Scalar& other) {
            return ScalarFunction::apply<Add>(*this, other);
        }

        friend Scalar operator+(const Scalar& self, const Scalar& other) {
            return ScalarFunction::apply<Add>(self, other);
        }

        Scalar operator*(const Scalar& other) {
            return ScalarFunction::apply<Mul>(*this, other);
        }

        friend Scalar operator*(const Scalar& self, const Scalar& other) {
            return ScalarFunction::apply<Mul>(self, other);
        }

        // other operations, derived from basic ones

        Scalar operator-(const Scalar& other) {
            return *this + ScalarFunction::apply<Neg>(other);
        }

        friend Scalar operator-(const Scalar& self, const Scalar& other) {
            return self + ScalarFunction::apply<Neg>(other);
        }

        Scalar operator/(const Scalar& other) {
            return *this * ScalarFunction::apply<Inv>(other);
        }

        friend Scalar operator/(const Scalar& self, const Scalar& other) {
            return self * ScalarFunction::apply<Inv>(other);
        }

        // comparisons

        Scalar operator<(const Scalar& other) {
            return ScalarFunction::apply<Lt>(*this, other);
        }

        friend Scalar operator<(const Scalar& self, const Scalar& other) {
            return ScalarFunction::apply<Lt>(self, other);
        }

        Scalar operator>(const Scalar& other) {
            return ScalarFunction::apply<Lt>(other, *this);
        }

        friend Scalar operator>(const Scalar& self, const Scalar& other) {
            return ScalarFunction::apply<Lt>(other, self);
        }

        friend Scalar operator==(const Scalar& self, const Scalar& other) {
            return ScalarFunction::apply<Eq>(self, other);
        }

        template <typename T>
        friend Scalar operator==(const Scalar& self, const T& other) {
            return ScalarFunction::apply<Eq>(self, Scalar(other));
        }

        template <typename T>
        friend Scalar operator==(const T& self, const Scalar& other) {
            return ScalarFunction::apply<Eq>(Scalar(self), other);
        }

        friend std::ostream& operator<<(std::ostream& os, Scalar& v) {
            os << "Scalar(data=" << v.data << ", grad=" << v.grad << ")\n";
            return os;
        }

        // functions
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