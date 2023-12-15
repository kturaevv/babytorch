#include <iostream>

#include <fmt/core.h>

#include "./babytorch/scalar.hpp"

int main() {
    fmt::print("Auto-diff project!\n");

    using namespace scalar;
    Scalar x{ 1.0 };
    Scalar y{ 2.0 };
    std::cout << x << y;

    std::cout << " + \n";
    std::cout << x + y;
    std::cout << x + int(2);
    std::cout << x + float(3);
    std::cout << x + double(4);
    std::cout << float(5) + x;
    std::cout << float(6) + x;

    std::cout << " - \n";
    std::cout << x - y;
    std::cout << x - int(2);
    std::cout << x - float(3);
    std::cout << x - double(4);
    std::cout << float(5) - x;
    std::cout << float(6) - x;

    std::cout << " / \n";
    std::cout << x / y;
    std::cout << x / int(2);
    std::cout << x / float(3);
    std::cout << x / double(4);
    std::cout << float(5) / x;
    std::cout << float(6) / x;

    return 0;
}
