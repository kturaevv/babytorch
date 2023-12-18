#include <iostream>

#include <fmt/core.h>

#include "./babytorch/scalar.hpp"

int main() {
    fmt::print("Auto-diff project!\n");

    using namespace scalar;
    Scalar x{ 1.0 };
    Scalar y{ 2.0 };
    Scalar z{ 3.0 };

    Scalar result = (x * y + z);

    std::cout << result;
    result.backward();
    return 0;
}
