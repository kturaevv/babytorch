#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <ranges>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "generic_operators.hpp"
#include "operators.hpp"
#include "tensor_autodiff.hpp"
#include "tensor_data.hpp"
#include "tensor_functions.hpp"
#include "tensor_ops.hpp"
#include "utils.hpp"

namespace tensor_ops {
    struct TensorBackend;
}

namespace tensor {

    using namespace tensor_autodiff;
    using namespace tensor_functions;
    using namespace tensor_data;

    using generic_operators::Arithmetic;
    using tensor_ops::TensorBackend;

    struct Tensor;

    struct TensorFunction {
        template <typename Fn, typename... Args>
        static Tensor apply(Args&&... args);
    };

    struct History {
        Context ctx;
        std::vector<Tensor*> inputs;
        std::function<std::array<Tensor, 2>(Context&, Tensor)> backward;
    };

    struct Tensor {
        // members

        size_t id;

        TensorData data;
        Tensor* grad;
        History history;
        TensorBackend backend;

        static inline size_t next_id = 0;
        Index passed_idx;

        // constructors

        Tensor()
            : id(next_id++) {
        }

        Tensor(TensorData data)
            : id(next_id++)
            , data(std::move(data)) {
        }

        Tensor(TensorData&& data, History&& hist, TensorBackend&& back)
            : id(next_id++)
            , data(std::move(data))
            , history(std::move(hist))
            , backend(std::move(back)) {
        }

        template <typename T>
        Tensor(std::vector<T> data)
            : id(next_id++) {
            this->data = TensorData(std::move(data));
        }

        template <typename... Sizes>
        Tensor(Sizes... dims)
            : id(next_id++) {
            utils::check_dimensions(dims...);

            Shape input_shapes{};
            (input_shapes.push_back(dims), ...);

            this->data = TensorData::rand(input_shapes);
        }

        Tensor(const Tensor& other)
            : id(next_id++)
            , data(other.data)
            , grad(other.grad)
            , history(other.history)
            , backend(other.backend) {
        }

        Tensor(Tensor&& other) noexcept
            : id(next_id++)
            , data(std::move(other.data))
            , grad(std::move(other.grad))
            , history(std::move(other.history))
            , backend(std::move(other.backend)) {
        }

        template <typename... Sizes>
        static Tensor create(Sizes... dims);
        static Tensor create(TensorData data);
        static Tensor create(History hist, TensorData data);
        static Tensor create(Tensor* tensor);
        static Tensor create(std::vector<double> data);

        // overloads

        template <typename... size_t>
        Tensor operator[](const size_t... dims) {
            Index passed_idx;
            (passed_idx.push_back(dims), ...);

            TensorStorageView storage_view = this->data.view(passed_idx);

            // Copy a view to a Tensor
            Storage new_storage = { storage_view.begin(), storage_view.end() };

            auto shape_view = this->data.shape
                              | std::views::drop(passed_idx.size());
            Shape new_shape = { shape_view.begin(), shape_view.end() };

            return Tensor(TensorData(std::move(new_storage), new_shape));
        }

        friend auto operator+(const Tensor& self, const Tensor& other) {
            return TensorFunction::apply<Add>(self, other);
        }

        template <typename T>
        friend auto operator+(const Tensor self, const T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor(val);
            return TensorFunction::apply<Add>(self, other);
        }

        template <typename T>
        friend auto operator+(const T& lhs, const Tensor other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor(val);
            return TensorFunction::apply<Add>(self, other);
        }

        // *

        friend auto operator*(const Tensor& self, const Tensor& other) {
            return TensorFunction::apply<Mul>(self, other);
        }

        template <typename T>
        friend auto operator*(const Tensor& self, const T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor(val);
            return TensorFunction::apply<Mul>(self, other);
        }

        template <typename T>
        friend auto operator*(const T& lhs, const Tensor& other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor(val);
            return TensorFunction::apply<Mul>(self, other);
        }

        // -

        friend auto operator-(const Tensor& self, const Tensor& other) {
            return self + TensorFunction::apply<Neg>(other);
        }

        template <typename T>
        friend auto operator-(const Tensor& self, const T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor(val);
            return self + TensorFunction::apply<Neg>(other);
        }

        template <typename T>
        friend auto operator-(const T& lhs, const Tensor& other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor(val);
            return self + TensorFunction::apply<Neg>(other);
        }

        // /

        friend auto operator/(const Tensor& self, const Tensor& other) {
            return self * TensorFunction::apply<Inv>(other);
        }

        template <typename T>
        friend auto operator/(const Tensor& self, const T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor(val);
            return self * TensorFunction::apply<Inv>(other);
        }

        template <typename T>
        friend auto operator/(const T& lhs, const Tensor& other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor(val);
            return self * TensorFunction::apply<Inv>(other);
        }

        // <

        friend auto operator<(const Tensor& self, const Tensor& other) {
            return TensorFunction::apply<Lt>(self, other);
        }

        template <typename T>
        friend auto operator<(const Tensor& self, const T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor(val);
            return TensorFunction::apply<Lt>(self, other);
        }

        template <typename T>
        friend auto operator<(const T& lhs, const Tensor& other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor(val);
            return TensorFunction::apply<Lt>(self, other);
        }

        // >

        friend auto operator>(const Tensor& self, const Tensor& other) {
            return TensorFunction::apply<Lt>(other, self);
        }

        template <typename T>
        friend auto operator>(const Tensor& self, const T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor(val);
            return TensorFunction::apply<Lt>(other, self);
        }

        template <typename T>
        friend auto operator>(const T& lhs, const Tensor& other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor(val);
            return TensorFunction::apply<Lt>(other, self);
        }

        // ==

        friend auto operator==(const Tensor& self, const Tensor& other) {
            return TensorFunction::apply<Eq>(self, other);
        }

        template <typename T>
        friend auto operator==(const Tensor& self, const T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor(val);
            return TensorFunction::apply<Eq>(self, other);
        }

        template <typename T>
        friend auto operator==(const T& lhs, const Tensor& other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor(val);
            return TensorFunction::apply<Eq>(other, self);
        }

        // +=

        Tensor& operator+=(const Tensor& other) {
            *this = std::move(TensorFunction::apply<Add>(*this, other));
            return *this;
        }

        template <typename T>
        Tensor& operator+=(const T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor(val);
            *this      = std::move(TensorFunction::apply<Add>(*this, other));
            return *this;
        }

        Tensor& operator=(Tensor&& other) noexcept = default;

        // functions
        size_t size();
        size_t dims();
        Shape shape() const;
        Tensor is_close();
        Tensor sigmoid();
        Tensor relu();
        Tensor log();
        Tensor exp();
        Tensor item();
        Tensor sum(size_t dim);
        Tensor mean(size_t dim);
        Tensor contiguous();
        Tensor view(Shape shape);
        Tensor permute(ReOrderIndex order);
        TensorDataInfo info() const;
        Tensor zeros() const;
        static Tensor zeros(Shape shape);

        bool is_leaf();
        void backward();
        void accumulate_grad(Tensor& d_x);
        std::vector<Tensor*> parents() const;
        std::vector<std::tuple<Tensor*, Tensor>> chain_rule(Tensor* deriv);
    };

    template <typename Fn, typename... Args>
    Tensor TensorFunction::apply(Args&&... args) {
        Context ctx;

        Tensor result = Fn::forward(ctx, args.data...);

        History history;
        history.ctx      = std::move(ctx);
        history.backward = Fn::backward;

        (history.inputs.emplace_back(&args), ...);

        return Tensor(std::move(result.data),
                      std::move(history),
                      std::move(result.backend));
    }

    // helper functions
}  // namespace tensor

template <>
struct fmt::formatter<tensor::Tensor> : formatter<string_view> {
    auto format(const tensor::Tensor& s, format_context& ctx) {
        std::string tensor_view = s.data.string_view();
        return fmt::format_to(ctx.out(), "Tensor({})\n", tensor_view);
    }
};
