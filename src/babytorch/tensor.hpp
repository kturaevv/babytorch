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
        static std::shared_ptr<Tensor> apply(Args&&... args);
    };

    struct History {
        Context ctx;
        std::vector<std::shared_ptr<Tensor>> inputs;
        std::function<std::array<std::shared_ptr<Tensor>, 2>(Context&, std::shared_ptr<Tensor>)> backward;
    };

    struct Tensor {
        // members

        size_t id;

        TensorData data;
        std::shared_ptr<Tensor> grad;
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

        Tensor(Tensor& other)
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
        static std::shared_ptr<Tensor> create(Sizes... dims);
        static std::shared_ptr<Tensor> create(TensorData data);
        static std::shared_ptr<Tensor> create(History hist, TensorData data);
        static std::shared_ptr<Tensor> create(std::shared_ptr<Tensor> tensor);
        static std::shared_ptr<Tensor> create(std::vector<double> data);

        // overloads

        template <typename... size_t>
        std::shared_ptr<Tensor> operator[](const size_t... dims) {
            Index passed_idx;
            (passed_idx.push_back(dims), ...);

            TensorStorageView storage_view = this->data.view(passed_idx);

            // Copy a view to a Tensor
            Storage new_storage = { storage_view.begin(), storage_view.end() };

            auto shape_view = this->data.shape
                              | std::views::drop(passed_idx.size());
            Shape new_shape = { shape_view.begin(), shape_view.end() };

            return std::make_shared<Tensor>(TensorData(std::move(new_storage), new_shape));
        }

        friend auto operator+(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other) {
            return TensorFunction::apply<Add>(self, other);
        }

        template <typename T>
        friend auto operator+(std::shared_ptr<Tensor> self, T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = std::make_shared<Tensor>(val);
            return TensorFunction::apply<Add>(self, other);
        }

        template <typename T>
        friend auto operator+(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other) {
            return TensorFunction::apply<Add>(self, other);
        }

        template <typename T>
        friend auto operator+(T& lhs, std::shared_ptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = std::make_shared<Tensor>(val);
            return TensorFunction::apply<Add>(self, other);
        }

        // *

        friend auto operator*(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other) {
            return TensorFunction::apply<Mul>(self, other);
        }

        template <typename T>
        friend auto operator*(std::shared_ptr<Tensor> self, T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = std::make_shared<Tensor>(val);
            return TensorFunction::apply<Mul>(self, other);
        }

        template <typename T>
        friend auto operator*(T& lhs, std::shared_ptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = std::make_shared<Tensor>(val);
            return TensorFunction::apply<Mul>(self, other);
        }

        // -

        friend auto operator-(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other) {
            return *self + TensorFunction::apply<Neg>(other);
        }

        template <typename T>
        friend auto operator-(std::shared_ptr<Tensor> self, T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = std::make_shared<Tensor>(val);
            return *self + TensorFunction::apply<Neg>(other);
        }

        template <typename T>
        friend auto operator-(T& lhs, std::shared_ptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = std::make_shared<Tensor>(val);
            return *self + TensorFunction::apply<Neg>(other);
        }

        // /

        friend auto operator/(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other) {
            return *self * TensorFunction::apply<Inv>(other);
        }

        template <typename T>
        friend auto operator/(std::shared_ptr<Tensor> self, T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = std::make_shared<Tensor>(val);
            return *self * TensorFunction::apply<Inv>(other);
        }

        template <typename T>
        friend auto operator/(T& lhs, std::shared_ptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = std::make_shared<Tensor>(val);
            return *self * TensorFunction::apply<Inv>(other);
        }

        // <

        friend auto operator<(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other) {
            return TensorFunction::apply<Lt>(self, other);
        }

        template <typename T>
        friend auto operator<(std::shared_ptr<Tensor> self, T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = std::make_shared<Tensor>(val);
            return TensorFunction::apply<Lt>(self, other);
        }

        template <typename T>
        friend auto operator<(T& lhs, std::shared_ptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = std::make_shared<Tensor>(val);
            return TensorFunction::apply<Lt>(self, other);
        }

        // >

        friend auto operator>(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other) {
            return TensorFunction::apply<Lt>(other, self);
        }

        template <typename T>
        friend auto operator>(std::shared_ptr<Tensor> self, T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = std::make_shared<Tensor>(val);
            return TensorFunction::apply<Lt>(other, self);
        }

        template <typename T>
        friend auto operator>(T& lhs, std::shared_ptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = std::make_shared<Tensor>(val);
            return TensorFunction::apply<Lt>(other, self);
        }

        // ==

        friend auto operator==(std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> other) {
            return TensorFunction::apply<Eq>(self, other);
        }

        template <typename T>
        friend auto operator==(std::shared_ptr<Tensor> self, T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = std::make_shared<Tensor>(val);
            return TensorFunction::apply<Eq>(self, other);
        }

        template <typename T>
        friend auto operator==(T& lhs, std::shared_ptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = std::make_shared<Tensor>(val);
            return TensorFunction::apply<Eq>(other, self);
        }

        // +=

        std::shared_ptr<Tensor> operator+=(std::shared_ptr<Tensor> other) {
            *this = std::move(*TensorFunction::apply<Add>(std::make_shared<Tensor>(*this), other));
            return std::make_shared<Tensor>(*this);
        }

        template <typename T>
        std::shared_ptr<Tensor> operator+=(T& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = std::make_shared<Tensor>(val);
            *this      = std::move(*TensorFunction::apply<Add>(std::make_shared<Tensor>(*this), other));
            return std::make_shared<Tensor>(*this);
        }

        Tensor& operator=(Tensor&& other) noexcept = default;

        // functions
        size_t size();
        size_t dims();
        Shape shape() const;
        std::shared_ptr<Tensor> is_close();
        std::shared_ptr<Tensor> sigmoid();
        std::shared_ptr<Tensor> relu();
        std::shared_ptr<Tensor> log();
        std::shared_ptr<Tensor> exp();
        std::shared_ptr<Tensor> item();
        std::shared_ptr<Tensor> sum(size_t dim);
        std::shared_ptr<Tensor> mean(size_t dim);
        std::shared_ptr<Tensor> contiguous();
        std::shared_ptr<Tensor> view(Shape shape);
        std::shared_ptr<Tensor> permute(ReOrderIndex order);
        TensorDataInfo info() const;
        std::shared_ptr<Tensor> zeros() const;
        static std::shared_ptr<Tensor> zeros(Shape shape);

        bool is_leaf();
        void backward();
        void accumulate_grad(std::shared_ptr<Tensor> d_x);
        std::vector<std::shared_ptr<Tensor>> parents() const;
        std::vector<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> chain_rule(std::shared_ptr<Tensor> deriv);
    };

    template <typename Fn, typename... Args>
    std::shared_ptr<Tensor> TensorFunction::apply(Args&&... args) {
        Context ctx;

        auto result = Fn::forward(ctx, args->data...);

        History history;
        history.ctx      = std::move(ctx);
        history.backward = Fn::backward;

        (history.inputs.emplace_back(args), ...);

        return std::make_shared<Tensor>(std::move(result.data),
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
