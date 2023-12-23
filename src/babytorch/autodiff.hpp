#pragma once

#include <any>
#include <memory>
#include <unordered_set>
#include <vector>

namespace scalar {
    struct Scalar;
}

namespace autodiff {

    using namespace scalar;

    std::vector<Scalar> topological_sort(std::shared_ptr<Scalar> v);
    std::vector<Scalar> topological_sort(std::shared_ptr<Scalar> v,
                                         std::unordered_set<int>& visited);

    void backpropagate(std::shared_ptr<Scalar> variable);
    void backpropagate(std::shared_ptr<Scalar> variable, double deriv);

    struct Context {
        std::vector<double> saved_values;

        template <typename... Args>
        void save_for_backwards(Args... args) {
            (saved_values.push_back(args), ...);
            return;
        }
    };
}