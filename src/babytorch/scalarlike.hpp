#pragma once

#include <vector>

struct ScalarLike {
    double id;
    double data;
    double grad;
    std::vector<ScalarLike> parents;

    ScalarLike() = default;

    ScalarLike(double data)
        : data(data) {
    }

    ScalarLike(double data, double grad)
        : data(data)
        , grad(grad) {
    }

    bool is_leaf() const;
    void accumulate_grad(double grad);
    std::vector<std::tuple<ScalarLike, double>> chain_rule(int i) const;
};
