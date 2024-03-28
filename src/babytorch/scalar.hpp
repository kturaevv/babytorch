#pragma once

#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

#include <fmt/core.h>

#include "autodiff.hpp"
#include "functions.hpp"
#include "ptr.hpp"

namespace scalar {

    // using autodiff::Context;
    using namespace autodiff;
    using namespace functions;

    struct Scalar;

    struct ScalarFunction {
        template <typename F, typename... Args>
        static sptr<Scalar> apply(Args&&... args);
    };

    struct History {
        Context ctx;
        std::vector<sptr<Scalar>> inputs;
        std::function<std::array<double, 2>(Context&, double)> backward;
    };

    struct Scalar {
        // members

        History history;

        double id;
        double data;
        double grad;

        static inline double next_id = 0.0;

        // constructors

        Scalar()
            : id(next_id++)
            , data(0)
            , grad(0) {
        }

        Scalar(double data)
            : id(next_id++)
            , data(data)
            , grad(0) {
        }

        Scalar(History history, double data)
            : history(history)
            , id(next_id++)
            , data(data)
            , grad(0) {
        }

        Scalar(Scalar* v)
            : id(next_id++) {
            this->data    = v->data;
            this->grad    = v->grad;
            this->history = v->history;
        }

        // overloads

        // +
        friend auto operator+(const sptr<Scalar> self, const sptr<Scalar> other) {
            return ScalarFunction::apply<Add>(self, other);
        }

        template <typename T>
        friend auto operator+(const sptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return ScalarFunction::apply<Add>(self, other);
        }

        template <typename T>
        friend auto operator+(const T& lhs, const sptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return ScalarFunction::apply<Add>(self, other);
        }

        // *

        friend auto operator*(const sptr<Scalar> self, const sptr<Scalar> other) {
            return ScalarFunction::apply<Mul>(self, other);
        }

        template <typename T>
        friend auto operator*(const sptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return ScalarFunction::apply<Mul>(self, other);
        }

        template <typename T>
        friend auto operator*(const T& lhs, const sptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return ScalarFunction::apply<Mul>(self, other);
        }

        // -

        friend auto operator-(const sptr<Scalar> self, const sptr<Scalar> other) {
            return self + ScalarFunction::apply<Neg>(other);
        }

        template <typename T>
        friend auto operator-(const sptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return self + ScalarFunction::apply<Neg>(other);
        }

        template <typename T>
        friend auto operator-(const T& lhs, const sptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return self + ScalarFunction::apply<Neg>(other);
        }

        // /

        friend auto operator/(const sptr<Scalar> self, const sptr<Scalar> other) {
            return self * ScalarFunction::apply<Inv>(other);
        }

        template <typename T>
        friend auto operator/(const sptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return self * ScalarFunction::apply<Inv>(other);
        }

        template <typename T>
        friend auto operator/(const T& lhs, const sptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return self * ScalarFunction::apply<Inv>(other);
        }

        // <

        friend auto operator<(const sptr<Scalar> self, const sptr<Scalar> other) {
            return ScalarFunction::apply<Lt>(self, other);
        }

        template <typename T>
        friend auto operator<(const sptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return ScalarFunction::apply<Lt>(self, other);
        }

        template <typename T>
        friend auto operator<(const T& lhs, const sptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return ScalarFunction::apply<Lt>(self, other);
        }

        // >

        friend auto operator>(const sptr<Scalar> self, const sptr<Scalar> other) {
            return ScalarFunction::apply<Lt>(other, self);
        }

        template <typename T>
        friend auto operator>(const sptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return ScalarFunction::apply<Lt>(other, self);
        }

        template <typename T>
        friend auto operator>(const T& lhs, const sptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return ScalarFunction::apply<Lt>(other, self);
        }

        // ==

        friend auto operator==(const sptr<Scalar> self, const sptr<Scalar> other) {
            return ScalarFunction::apply<Eq>(self, other);
        }

        template <typename T>
        friend auto operator==(const sptr<Scalar> self, const T& rhs) {
            auto other = Scalar::create(rhs);
            return ScalarFunction::apply<Eq>(self, other);
        }

        template <typename T>
        friend auto operator==(const T& lhs, const sptr<Scalar> other) {
            auto self = Scalar::create(lhs);
            return ScalarFunction::apply<Eq>(other, self);
        }

        friend std::ostream& operator<<(std::ostream& os, Scalar& v) {
            os << "Scalar(data=" << v.data << ", grad=" << v.grad << ")\n";
            return os;
        }

        // functions
        sptr<Scalar> log();
        sptr<Scalar> exp();
        sptr<Scalar> sigmoid();
        sptr<Scalar> relu();

        bool is_leaf();
        void backward();
        void accumulate_grad(double d_x);
        std::vector<sptr<Scalar>> parents();
        std::vector<std::tuple<sptr<Scalar>, double>> chain_rule(double deriv);

        static sptr<Scalar> create();
        static sptr<Scalar> create(double data);
        static sptr<Scalar> create(History history, double data);
    };

    template <typename F, typename... Args>
    sptr<scalar::Scalar> ScalarFunction::apply(Args&&... args) {
        Context ctx;

        double result = F::forward(ctx, args->data...);

        History history;
        history.ctx      = std::move(ctx);
        history.backward = F::backward;
        (history.inputs.emplace_back(args), ...);

        return Scalar::create(history, result);
    }
}  // namespace scalar

template <>
struct fmt::formatter<scalar::Scalar> : formatter<string_view> {
    auto format(const scalar::Scalar& s, format_context& ctx) {
        return fmt::format_to(ctx.out(),
                              "Scalar(data={}, grad={})\n",
                              s.data,
                              s.grad);
    }
};
