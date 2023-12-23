#include <iostream>

#include <fmt/core.h>

#include "./babytorch/scalar.hpp"

int main() {
    fmt::print("Auto-diff project!\n");

    using namespace scalar;
    auto x = Scalar::create(1.0);
    auto y = Scalar::create(2.0);
    auto z = Scalar::create(2.0);

    Scalar result = (x * y + z);

    std::cout << result;
    return 0;
}
