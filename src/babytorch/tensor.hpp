#pragma once

#include <functional>

#include "autodiff.hpp"
#include "tensor_data.hpp"

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
            , data(data) {
        }

        Tensor(History history, TensorData data)
            : id(next_id++)
            , data(data)
            , history(history) {
        }

        Tensor(Tensor* v)
            : id(next_id++) {
            this->data = v->data;
            this->history = v->history;
        }

        template <typename... Args>
        static Tensor create(Args... dims);
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