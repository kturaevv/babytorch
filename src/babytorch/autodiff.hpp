#pragma once

#include <vector>

#include "ptr.hpp"

namespace scalar {
    struct Scalar;
}

namespace autodiff {

    using namespace scalar;

    std::vector<sptr<Scalar>> topological_sort(sptr<Scalar> v);

    void backpropagate(sptr<Scalar> variable);
    void backpropagate(sptr<Scalar> variable, double deriv);

    struct Context {
        std::vector<double> saved_values;

        template <typename... Args>
        void save_for_backwards(Args... args) {
            (saved_values.push_back(args), ...);
            return;
        }
    };
}
