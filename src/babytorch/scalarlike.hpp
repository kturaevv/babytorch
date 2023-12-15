#pragma once

#include <iostream>
#include <type_traits>
#include <vector>

struct ScalarLike {
    double id;
    double data;
    double grad;
    std::vector<ScalarLike> parents;

    ScalarLike() = default;

    ScalarLike(double data) noexcept
        : data(data)
        , grad(0) {
    }

    ScalarLike(double data, double grad) noexcept
        : data(data)
        , grad(grad) {
    }

    bool is_leaf() const;
    void accumulate_grad(double grad);
    std::vector<std::tuple<ScalarLike, double>> chain_rule(int i) const;

    friend std::ostream& operator<<(std::ostream& os, const ScalarLike& self) {
        os << "Scalar(data=" << self.data << ", grad=" << self.grad << ")\n";
        return os;
    }
};