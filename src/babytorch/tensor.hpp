#pragma once

#include <functional>
#include <iostream>
#include <memory>

#include "autodiff.hpp"
#include "generic_operators.hpp"
#include "operators.hpp"
#include "tensor_data.hpp"
#include "utils.hpp"

namespace tensor {

    using namespace autodiff;
    using namespace tensor_data;

    struct Tensor;

    struct History {
        Context ctx;
        std::vector<std::shared_ptr<Tensor>> inputs;
        std::function<std::array<double, 2>(Context&, double)> backward;
    };

    struct Tensor {
        // members

        size_t id;

        TensorData data;
        History history;

        static inline size_t next_id = 0;

        // constructors

        Tensor()
            : id(next_id++) {
        }

        Tensor(TensorData data)
            : id(next_id++)
            , data(std::move(data)) {
        }

        Tensor(History history, TensorData data)
            : id(next_id++)
            , data(std::move(data))
            , history(history) {
        }

        template <typename... Ints>
        Tensor(Ints... dims)
            : id(next_id++) {
            UserShape input_shapes{};
            (input_shapes.push_back(dims), ...);

            double storage_size = generic_operators::prod(input_shapes);
            Storage storage = utils::rand(storage_size);

            data = TensorData(std::move(storage), input_shapes);
        }

        // Tensor(Tensor* v)
        //     : id(next_id++) {
        //     this->data = v->data;
        //     this->history = v->history;
        // }

        template <typename... Ints>
        static Tensor create(Ints... dims);
        static Tensor create(TensorData data);
        static Tensor create(History hist, TensorData data);
        static Tensor create(Tensor* tensor);
        static Tensor create(std::vector<double> data);

        // functions

        size_t size();
        size_t dims();
        UserShape shape();
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
        Tensor zeros(UserShape shape);

        bool is_leaf();
        void backward();
        void accumulate_grad(double d_x);
        std::vector<Tensor> parents();
        std::vector<std::tuple<Tensor, double>> chain_rule(double deriv);
    };
}  // namespace tensor