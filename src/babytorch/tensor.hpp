#pragma once

#include <functional>
#include <memory>
#include <ranges>
#include <type_traits>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "generic_operators.hpp"
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

    class Tensor;

    struct TensorFunction {
        template <typename Fn, typename... Args>
        static sptr<Tensor> apply(Args&&... args);
    };

    struct History {
        Context ctx;
        std::vector<sptr<Tensor>> inputs;
        std::function<std::array<sptr<Tensor>, 2>(Context&, sptr<Tensor>)> backward;
    };

    class Tensor : public std::enable_shared_from_this<Tensor> {
    public:
        // members

        size_t id;

        uptr<TensorData> data;
        sptr<Tensor> grad;
        History history;
        static inline sptr<TensorBackend> backend;
        static inline size_t next_id = 0;

        // static functions

        static void set_backend() {
            backend = std::make_shared<TensorBackend>();
        }

        template <typename... Args>
            requires(std::is_same_v<int, Args> && ...)
        static sptr<Tensor> create(Args&&... args) {
            return std::make_shared<Tensor>(std::forward<Args>(args)...);
        }

        static sptr<Tensor> create(uptr<TensorData> data) {
            return std::make_shared<Tensor>(std::move(data));
        }

        static sptr<Tensor> create(std::vector<double> data) {
            return std::make_shared<Tensor>(std::move(data));
        }

        static sptr<Tensor> create(History hist, uptr<TensorData> data) {
            return std::make_shared<Tensor>(std::move(data), std::move(hist));
        }

        // constructors
        Tensor()
            : id(next_id++) {
        }

        Tensor(uptr<TensorData> data)
            : id(next_id++)
            , data(std::move(data)) {
        }

        Tensor(uptr<TensorData>&& data, History&& hist)
            : id(next_id++)
            , data(std::move(data))
            , history(std::move(hist)) {
        }

        template <typename T>
        Tensor(std::vector<T> input_arr)
            : id(next_id++) {
            this->data = std::make_unique<TensorData>(std::move(input_arr));
        }

        template <typename... Sizes>
            requires(std::is_same_v<int, Sizes> && ...)
        Tensor(Sizes... dims)
            : id(next_id++) {
            utils::check_dimensions(dims...);

            Shape input_shapes{};
            (input_shapes.push_back(dims), ...);

            this->data = TensorData::rand(input_shapes);
        }

        Tensor(const Tensor& other)
            : std::enable_shared_from_this<Tensor>()
            , id(next_id++)
            , data(other.data ? std::make_unique<TensorData>(*other.data)
                              : nullptr)
            , grad(other.grad)
            , history(other.history) {
        }

        Tensor(Tensor&& other) noexcept
            : id(next_id++)
            , data(std::move(other.data))
            , grad(std::move(other.grad))
            , history(std::move(other.history)) {
        }

        // functions
        size_t size();
        size_t dims();
        Shape shape() const;
        sptr<Tensor> adjust_for_broadcast(sptr<Tensor> other);
        sptr<Tensor> is_close();
        sptr<Tensor> sigmoid();
        sptr<Tensor> relu();
        sptr<Tensor> log();
        sptr<Tensor> exp();
        sptr<Tensor> item();
        sptr<Tensor> sum(size_t dim);
        sptr<Tensor> mean(size_t dim);
        sptr<Tensor> contiguous();
        sptr<Tensor> view(Shape shape);
        sptr<Tensor> permute(ReOrderIndex order);
        TensorDataInfo info() const;
        sptr<Tensor> zeros() const;
        static sptr<Tensor> zeros(Shape shape);

        bool is_leaf();
        void backward();
        void accumulate_grad(sptr<Tensor>&& d_x);
        std::vector<sptr<Tensor>> parents() const;
        std::vector<std::tuple<sptr<Tensor>, sptr<Tensor>>> chain_rule(
            sptr<Tensor> deriv);

        template <typename... size_t>
        sptr<Tensor> at(const size_t... dims) {
            Index ix;
            (ix.push_back(dims), ...);

            TensorStorageView storage_view = this->data->view(ix);

            // Copy a view to a Tensor
            Storage new_storage = { storage_view.begin(), storage_view.end() };

            auto shape_view = this->data->shape | std::views::drop(ix.size());
            Shape new_shape = { shape_view.begin(), shape_view.end() };

            return Tensor::create(
                std::make_unique<TensorData>(std::move(new_storage), new_shape));
        }

        // overloads

        template <typename... size_t>
        Tensor operator[](const size_t... dims) {
            Index ix;
            (ix.push_back(dims), ...);

            TensorStorageView storage_view = this->data->view(ix);

            // Copy a view to a Tensor
            Storage new_storage = { storage_view.begin(), storage_view.end() };

            auto shape_view = this->data->shape | std::views::drop(ix.size());
            Shape new_shape = { shape_view.begin(), shape_view.end() };

            return Tensor(
                std::make_unique<TensorData>(std::move(new_storage), new_shape));
        }

        friend auto operator+(sptr<Tensor> self, sptr<Tensor> other) {
            return TensorFunction::apply<Add>(self, other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator+(sptr<Tensor> self, T&& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor::create(val);
            return TensorFunction::apply<Add>(self, other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator+(T&& lhs, sptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor::create(val);
            return TensorFunction::apply<Add>(self, other);
        }

        // *

        friend auto operator*(sptr<Tensor> self, sptr<Tensor> other) {
            return TensorFunction::apply<Mul>(self, other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator*(sptr<Tensor> self, T&& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor::create(val);
            return TensorFunction::apply<Mul>(self, other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator*(T&& lhs, sptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor::create(val);
            return TensorFunction::apply<Mul>(self, other);
        }

        // -

        friend auto operator-(sptr<Tensor> self, sptr<Tensor> other) {
            return self + TensorFunction::apply<Neg>(other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator-(sptr<Tensor> self, T&& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor::create(val);
            return self + TensorFunction::apply<Neg>(other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator-(T&& lhs, sptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor::create(val);
            return self + TensorFunction::apply<Neg>(other);
        }

        // /

        friend auto operator/(sptr<Tensor> self, sptr<Tensor> other) {
            return self * TensorFunction::apply<Inv>(other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator/(sptr<Tensor> self, T&& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor::create(val);
            return self * TensorFunction::apply<Inv>(other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator/(T&& lhs, sptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor::create(val);
            return self * TensorFunction::apply<Inv>(other);
        }

        // <

        friend auto operator<(sptr<Tensor> self, sptr<Tensor> other) {
            return TensorFunction::apply<Lt>(self, other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator<(sptr<Tensor> self, T&& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor::create(val);
            return TensorFunction::apply<Lt>(self, other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator<(T&& lhs, sptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor::create(val);
            return TensorFunction::apply<Lt>(self, other);
        }

        // >

        friend auto operator>(sptr<Tensor> self, sptr<Tensor> other) {
            return TensorFunction::apply<Lt>(other, self);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator>(sptr<Tensor> self, T&& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor::create(val);
            return TensorFunction::apply<Lt>(other, self);
        }

        template <typename T>
        friend auto operator>(T&& lhs, sptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor::create(val);
            return TensorFunction::apply<Lt>(other, self);
        }

        // ==

        friend auto operator==(sptr<Tensor> self, sptr<Tensor> other) {
            return TensorFunction::apply<Eq>(self, other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator==(sptr<Tensor> self, T&& rhs) {
            auto val   = Storage{ static_cast<double>(rhs) };
            auto other = Tensor::create(val);
            return TensorFunction::apply<Eq>(self, other);
        }

        template <typename T>
            requires std::is_arithmetic_v<T>
        friend auto operator==(T&& lhs, sptr<Tensor> other) {
            auto val  = Storage{ static_cast<double>(lhs) };
            auto self = Tensor::create(val);
            return TensorFunction::apply<Eq>(other, self);
        }

        Tensor& operator=(Tensor&& other) noexcept = default;
    };

    template <typename Fn, typename... Args>
    sptr<Tensor> TensorFunction::apply(Args&&... args) {
        Context ctx;

        auto result = Fn::forward(ctx, args...);

        History history;
        history.ctx      = std::move(ctx);
        history.backward = Fn::backward;

        (history.inputs.emplace_back(args), ...);

        return Tensor::create(std::move(history), std::move(result->data));
    }

    // helper functions
}  // namespace tensor

template <>
struct fmt::formatter<tensor::Tensor> : formatter<string_view> {
    auto format(const tensor::Tensor& s, format_context& ctx) const {
        std::string tensor_view = s.data->string_view();
        return fmt::format_to(ctx.out(), "Tensor({})\n", tensor_view);
    }
};
