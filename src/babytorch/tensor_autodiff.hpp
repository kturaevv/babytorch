#pragma once

#include <any>
#include <memory>
#include <unordered_set>
#include <vector>

namespace tensor {
    struct Tensor;
}

namespace tensor_autodiff {

    using namespace tensor;

    std::vector<Tensor> topological_sort(Tensor& v);

    void backpropagate(Tensor* variable);
    void backpropagate(Tensor* variable, Tensor* deriv);

    struct Context {
        std::vector<Tensor> saved_values;

        template <typename... Args>
        void save_for_backwards(Args&&... args) {
            (saved_values.push_back(args), ...);
            return;
        }
    };
}