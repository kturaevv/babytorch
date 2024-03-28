#pragma once

#include <sstream>
#include <vector>

#include "generic_operators.hpp"

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

    template <typename T = double>
    std::vector<T> zeros(const T size) {
        return std::vector<T>(size, 0);
    };

    template <typename T = double>
    std::vector<T> zeros(const std::vector<size_t> shape) {
        std::vector<T> _shape{ shape.begin(), shape.end() };
        T size = generic_operators::prod<T>(_shape);
        return zeros<T>(size);
    };

}  // namespace utils
