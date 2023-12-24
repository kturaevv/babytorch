#include <iostream>

#include <fmt/core.h>

#include "./babytorch/scalar.hpp"

int main() {
    fmt::print("Auto-diff project!\n");

    using namespace scalar;

    auto x = Scalar::create(1.0);
    auto y = Scalar::create(2.0);
    auto z = Scalar::create(3.0);
    auto k = Scalar::create(4.0);
    auto j = Scalar::create(5.0);

    auto result = (x * y + z - k / j);

    std::cout << result;
    return 0;
}
