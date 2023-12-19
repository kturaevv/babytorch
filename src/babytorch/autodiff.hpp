#pragma once

#include <any>
#include <unordered_set>
#include <vector>

#include "scalarlike.hpp"

namespace autodiff {
    std::vector<ScalarLike> topological_sort(const ScalarLike& v);
    std::vector<ScalarLike> topological_sort(const ScalarLike& v,
                                             std::unordered_set<int>& visited);

    void backpropagate(const ScalarLike& variable);
    void backpropagate(const ScalarLike& variable, double deriv);

    struct Context {
        std::vector<double> saved_values;

        template <typename... Args>
        void save_for_backwards(Args... args) {
            (saved_values.push_back(args), ...);
            return;
        }
    };
}