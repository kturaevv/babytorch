#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "autodiff.hpp"
#include "functions.hpp"

namespace scalar {

    // using autodiff::Context;
    using namespace autodiff;
    using namespace functions;

    struct Scalar;

    struct ScalarFunction {
        template <typename F, typename... Args>
        static std::shared_ptr<Scalar> apply(Args&&... args);
    };

    struct History {
        Context ctx;
        std::vector<std::shared_ptr<Scalar>> inputs;
        std::function<std::array<double, 2>(Context&, double)> backward;
    };

    struct Scalar {
        // members

        double id;
        double data;
        double grad;
        History history;

        static inline double next_id = 0.0;

        // constructors

        Scalar(double data)
            : data(data)
            , id(next_id++)
            , grad(0) {
        }

        Scalar(History history, double data)
            : history(history)
            , data(data)
            , id(next_id++)
            , grad(0) {
        }

        Scalar(Scalar* v)
            : id(next_id++) {
            this->data = v->data;
            this->grad = v->grad;
            this->history = v->history;
        }

        // overloads

        // +
        friend auto operator+(std::shared_ptr<Scalar> self,
                              std::shared_ptr<Scalar> other) {
            return ScalarFunction::apply<Add>(self, other);
        }

        template <typename T>
        friend auto operator+(std::shared_ptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return ScalarFunction::apply<Add>(self, other);
        }

        template <typename T>
        friend auto operator+(const T& lhs, std::shared_ptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return ScalarFunction::apply<Add>(self, other);
        }

        // *

        friend auto operator*(std::shared_ptr<Scalar> self,
                              std::shared_ptr<Scalar> other) {
            return ScalarFunction::apply<Mul>(self, other);
        }

        template <typename T>
        friend auto operator*(std::shared_ptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return ScalarFunction::apply<Mul>(self, other);
        }

        template <typename T>
        friend auto operator*(const T& lhs, std::shared_ptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return ScalarFunction::apply<Mul>(self, other);
        }

        // -

        friend auto operator-(std::shared_ptr<Scalar> self,
                              std::shared_ptr<Scalar> other) {
            return self + ScalarFunction::apply<Neg>(other);
        }

        template <typename T>
        friend auto operator-(std::shared_ptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return self + ScalarFunction::apply<Neg>(other);
        }

        template <typename T>
        friend auto operator-(const T& lhs, std::shared_ptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return self + ScalarFunction::apply<Neg>(other);
        }

        // /

        friend auto operator/(std::shared_ptr<Scalar> self,
                              std::shared_ptr<Scalar> other) {
            return self * ScalarFunction::apply<Inv>(other);
        }

        template <typename T>
        friend auto operator/(std::shared_ptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return self * ScalarFunction::apply<Inv>(other);
        }

        template <typename T>
        friend auto operator/(const T& lhs, std::shared_ptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return self * ScalarFunction::apply<Inv>(other);
        }

        // <

        friend auto operator<(std::shared_ptr<Scalar> self,
                              std::shared_ptr<Scalar> other) {
            return ScalarFunction::apply<Lt>(self, other);
        }

        template <typename T>
        friend auto operator<(std::shared_ptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return ScalarFunction::apply<Lt>(self, other);
        }

        template <typename T>
        friend auto operator<(const T& lhs, std::shared_ptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return ScalarFunction::apply<Lt>(self, other);
        }

        // >

        friend auto operator>(std::shared_ptr<Scalar> self,
                              std::shared_ptr<Scalar> other) {
            return ScalarFunction::apply<Lt>(other, self);
        }

        template <typename T>
        friend auto operator>(std::shared_ptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return ScalarFunction::apply<Lt>(other, self);
        }

        template <typename T>
        friend auto operator>(const T& lhs, std::shared_ptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return ScalarFunction::apply<Lt>(other, self);
        }

        // ==

        friend auto operator==(std::shared_ptr<Scalar> self,
                               std::shared_ptr<Scalar> other) {
            return ScalarFunction::apply<Eq>(self, other);
        }

        template <typename T>
        friend auto operator==(std::shared_ptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return ScalarFunction::apply<Eq>(self, other);
        }

        template <typename T>
        friend auto operator==(const T& lhs, std::shared_ptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return ScalarFunction::apply<Eq>(other, self);
        }

        friend std::ostream& operator<<(std::ostream& os, Scalar& v) {
            os << "Scalar(data=" << v.data << ", grad=" << v.grad << ")\n";
            return os;
        }

        // functions
        std::shared_ptr<Scalar> log();
        std::shared_ptr<Scalar> exp();
        std::shared_ptr<Scalar> sigmoid();
        std::shared_ptr<Scalar> relu();

        bool is_leaf();
        void backward();
        void accumulate_grad(double d_x);
        std::vector<std::shared_ptr<Scalar>> parents();
        std::vector<std::tuple<std::shared_ptr<Scalar>, double>> chain_rule(
            double deriv);

        static std::shared_ptr<Scalar> create(double data);
        static std::shared_ptr<Scalar> create(History history, double data);
    };

    template <typename F, typename... Args>
    std::shared_ptr<scalar::Scalar> ScalarFunction::apply(Args&&... args) {
        Context ctx;

        double result = F::forward(ctx, args->data...);

        History history;
        history.ctx = std::move(ctx);
        history.backward = F::backward;
        (history.inputs.emplace_back(args), ...);

        return Scalar::create(history, result);
    }
}
