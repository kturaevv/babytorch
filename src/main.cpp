#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "./babytorch/scalar.hpp"
#include "./babytorch/tensor.hpp"

int main() {
    fmt::print("Autograd project!\n\n");

    using scalar::Scalar;
    auto x = Scalar::create(1.0);
    auto y = Scalar::create(2.0);
    auto z = Scalar::create(3.0);
    auto k = Scalar::create(4.0);
    auto j = Scalar::create(5.0);

    fmt::print("Example -> Scalars:\n\n");
    fmt::print("x = {}", *x);
    fmt::print("y = {}", *y);
    fmt::print("z = {}", *z);
    fmt::print("k = {}", *k);
    fmt::print("j = {}", *j);
    fmt::print("\n");

    auto result = (x * y + z - k / j);
    fmt::print("result = (x * y + z - k / j) => {}\n", *result);

    result->backward();
    fmt::print("After .backward() pass, Scalars are upgraded: \n");
    fmt::print("x = {}", *x);
    fmt::print("y = {}", *y);
    fmt::print("z = {}", *z);
    fmt::print("k = {}", *k);
    fmt::print("\n\n");

    using tensor::Tensor;

    fmt::print("Example -> Tensors:\n\n");

    auto a = Tensor(3, 3, 5);
    fmt::print("Creating an n-dimensional tensor of shape (3,3,5) ");
    fmt::print("i.e. Tensor(3,3,5) ->:\n{} \n", a);
    fmt::print("Indexing to a certain dimensions would return a sub-Tensor:\n");

    fmt::print("Tensor[1] = {} \n", a[1]);
    fmt::print("Tensor[1, 2] = {} \n", a[1, 2]);
    fmt::print("Tensor[1, 2, 3] = {} \n", a[1, 2, 3]);
    return 0;
}
