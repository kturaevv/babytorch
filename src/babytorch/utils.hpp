#pragma once

#include <iostream>
#include <memory>
#include <vector>

namespace utils {
    std::unique_ptr<std::vector<double>> rand(size_t size);
    std::unique_ptr<std::vector<double>> rand(size_t size, int min, int max);

    template <typename T>
    void print_vec(std::vector<T>& vec) {
        std::cout << "[ ";
        for (auto element : vec)
            std::cout << element << " ";
        std::cout << "]\n";
    }
}  // namespace utils
