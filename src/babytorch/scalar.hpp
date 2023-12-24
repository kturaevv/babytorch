#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <tuple>
#include <variant>
#include <vector>

#include "autodiff.hpp"
#include "functions.hpp"

namespace scalar {

    using autodiff::Context;
    using namespace functions;

    struct Scalar;

    using backward0 = double;
    using backward1 = std::tuple<double>;
    using backward2 = std::tuple<double, double>;
    using backward_return_type = std::variant<backward1, backward2>;

    struct History {
        Context ctx;
        std::vector<std::shared_ptr<Scalar>> inputs;
        std::function<backward_return_type(Context&, double)> backward;
    };

    struct Scalar {
        // members

        double id;
        double data;
        double grad;
        History history;

        static inline double next_id = 0.0;

        // constructors

        template <typename T>
        explicit Scalar(T data)
            : data(static_cast<double>(data))
            , id(next_id++)
            , grad(0) {
        }

        template <typename T>
        Scalar(History history, T data)
            : history(history)
            , data(static_cast<double>(data))
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
        void chain_rule(double deriv);
        void accumulate_grad(double d_x);

        std::vector<std::shared_ptr<Scalar>> parents();

        static std::shared_ptr<Scalar> create(double data);
        static std::shared_ptr<Scalar> create(History history, double data);
    };
