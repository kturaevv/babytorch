#include <random>
#include <vector>

#include "utils.hpp"

namespace utils {
    std::vector<double> rand(size_t size, int min, int max) {
        std::random_device rd;   // Obtain a random number from hardware
        std::mt19937 gen(rd());  // Seed the generator

        auto random_values = std::vector<double>();
        random_values.reserve(size);

        std::uniform_real_distribution<double> distr(min, max);

        for (size_t i = 0; i < size; ++i)
            random_values.push_back(distr(gen));

        return random_values;
    }

    std::vector<double> rand(size_t size) {
        return rand(size, -1, 1);
    }

}