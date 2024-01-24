#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace utils {
    std::vector<double> rand(const size_t size);
    std::vector<double> rand(const size_t size, const int min, const int max);

    template <typename... Dims>
    void check_dimensions(const Dims... dimensions) {
        (([&]() {
             if (dimensions < 0) {
                 std::ostringstream oss;
                 oss << "Dimension size cannot be negative: " << dimensions;
                 throw std::invalid_argument(oss.str());
             }
         }()),
         ...);
    }

}  // namespace utils
