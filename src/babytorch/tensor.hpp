#pragma once

#include <functional>
#include <iostream>
#include <memory>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "autodiff.hpp"
#include "operators.hpp"
#include "tensor_data.hpp"
#include "utils.hpp"

namespace tensor {

    using namespace autodiff;
    using namespace tensor_data;

    struct Tensor;

    struct History {
        Context ctx;
        std::vector<Tensor*> inputs;
        std::function<std::array<double, 2>(Context&, double)> backward;
    };

    struct Tensor {
        // members

        size_t id;

        TensorData data;
        History history;

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

        Tensor(std::vector<double> data)
            : id(next_id++)
            , data(std::move(data)) {
        }

        template <typename... Sizes>
        Tensor(Sizes... dims)
            : id(next_id++) {
            utils::check_dimensions(dims...);

            Shape input_shapes{};
            (input_shapes.push_back(dims), ...);

            this->data = TensorData::rand(input_shapes);
        }

        Tensor(const Tensor& other) = delete;

        template <typename... Sizes>
        static Tensor create(Sizes... dims);
        static Tensor create(TensorData data);
        static Tensor create(History hist, TensorData data);
        static Tensor create(Tensor* tensor);
        static Tensor create(std::vector<double> data);

        // overloads

        // friend std::ostream& operator<<(std::ostream& os, Tensor& v) {
        //     os << "Tensor(data=" << v.data << ", grad=" << v.grad << ")\n";
        //     return os;
        // }

        template <typename... size_t>
        Tensor operator[](const size_t... dims) {
            (passed_idx.push_back(dims), ...);

            fmt::print("Passed index: {}\n", passed_idx);
            return Tensor(TensorData({ 1 }));
        }

        // Tensor operator[](const int) {
        //     return Tensor(TensorData({ 1 }));
        // }

        // functions
        size_t size();
        size_t dims();
        Shape shape();
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
        Tensor zeros(Shape shape);

        bool is_leaf();
        void backward();
        void accumulate_grad(double d_x);
        std::vector<Tensor> parents();
        std::vector<std::tuple<Tensor, double>> chain_rule(double deriv);
    };

    // helper functions
}  // namespace tensor

template <>
struct fmt::formatter<tensor::Tensor> : formatter<string_view> {
    auto format(const tensor::Tensor& s, format_context& ctx) {
        std::string tensor_view = "";
        return fmt::format_to(ctx.out(), "Tensor(storage={})\n", s.data._storage);
    }
};
