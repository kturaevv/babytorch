#pragma once

#include <vector>

struct ScalarLike {
    double id;
    double data;
    double grad;
    std::vector<ScalarLike> parents;

    bool is_leaf() const;
    void accumulate_grad(double grad);
    std::vector<std::tuple<ScalarLike, double>> chain_rule(int i) const;

    ScalarLike(double data, double grad)
        : data(data)
        , grad(grad) {
    }
};
