#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "./babytorch/scalar.hpp"
#include "./babytorch/tensor.hpp"
#include "./babytorch/tensor_data.hpp"

int main() {
    fmt::print("Autograd project!\n\n");

    using scalar::Scalar;
    auto x      = Scalar::create(1.0);
    auto y      = Scalar::create(2.0);
    auto z      = Scalar::create(3.0);
    auto k      = Scalar::create(4.0);
    auto j      = Scalar::create(5.0);
    auto result = (x * y + z - k / j);
    result->backward();
    fmt::print("{}\n", *result);

    using tensor::Tensor;
    auto tensor_sample = Tensor(3, 3, 5);
    auto a             = Tensor(3, 3, 5);
    auto b             = Tensor(3, 1, 5);
    auto c             = Tensor(5);
    auto d             = Tensor(3, 3, 1);
    auto e             = Tensor(3, 5);
    auto tensor_result = a / 1.2 + b * c / d - 3 - e;
    fmt::print("{}", tensor_result);

    return 0;
}
