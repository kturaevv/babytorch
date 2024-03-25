#pragma once

#include <any>
#include <memory>
#include <unordered_set>
#include <vector>
#include "ptr.hpp"

namespace tensor {
    struct Tensor;
}

namespace tensor_autodiff {

    using namespace tensor;

    std::vector<sptr<Tensor>> topological_sort(sptr<Tensor> v);

    void backpropagate(sptr<Tensor> variable);
    void backpropagate(sptr<Tensor> variable, sptr<Tensor> deriv);

    struct Context {
        std::vector<sptr<Tensor>> saved_values;

        template <typename... Args>
        void save_for_backwards(Args&&... args) {
            (saved_values.push_back(args), ...);
            return;
        }
    };
}