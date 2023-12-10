#pragma once

#include <any>
#include <unordered_set>
#include <vector>

#include "scalarlike.hpp"

namespace autodiff {
    std::vector<ScalarLike> topological_sort(const ScalarLike& v);
    std::vector<ScalarLike> topological_sort(const ScalarLike& v,
                                             std::unordered_set<int>& visited);

    void backpropagate(const ScalarLike& variable, int deriv);

}