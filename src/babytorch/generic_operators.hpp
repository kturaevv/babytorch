#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

namespace generic_operators {

    const double EPS = 1e-10;

    // A concept to check if a type supports basic arithmetic operations
    template <typename T>
    concept Arithmetic = requires(T a, T b) {
                             { a + b } -> std::convertible_to<T>;
                             { a - b } -> std::convertible_to<T>;
                             { a* b } -> std::convertible_to<T>;
                             { a / b } -> std::convertible_to<T>;
                             { -a } -> std::convertible_to<T>;
                         };

    // Basic operations
    template <Arithmetic T>
    auto mul(const T& x, const T& y) {
        return x * y;
    }

    template <Arithmetic T>
    auto id(const T& x) {
        return x;
    }

    template <Arithmetic T>
    auto add(const T& x, const T& y) {
        return x + y;
    }

    template <Arithmetic T>
    auto neg(const T& x) {
        return -x;
    }

    template <Arithmetic T>
    auto lt(const T& x, const T& y) {
        return x < y ? 1 : 0;
    }

    template <Arithmetic T>
    auto eq(const T& x, const T& y) {
        return x == y ? 1 : 0;
    }

    template <Arithmetic T>
    auto max(const T& x, const T& y) {
        return x > y ? x : y;
    }

    template <Arithmetic T>
    auto is_close(const T& x, const T& y) {
        const T EPS = 1e-2;
        return fabs(x - y) < EPS ? 1 : 0;
    }

    template <Arithmetic T>
    auto sigmoid(const T& x) {
        if (x >= 0)
            return 1 / (1 + exp(-x));
        else
            return exp(x) / (1 + exp(x));
    }

    template <Arithmetic T>
    auto sigmoid_back(const T& x, const T& d) {
        return x * (1 - x) * d;
    }

    template <Arithmetic T>
    auto relu(const T& x) {
        return x > 0 ? x : 0;
    }

    template <Arithmetic T>
    auto log_func(const T& x) {
        const T EPS = 1e-2;
        return std::log(x + EPS);
    }

    template <Arithmetic T>
    auto exp_func(const T& x) {
        return std::exp(x);
    }

    template <Arithmetic T>
    auto log_back(const T& x, const T& d) {
        const T EPS = 1e-2;
        return 1.0 / (x * std::log(d + EPS) + EPS);
    }

    template <Arithmetic T>
    auto inv(const T& x) {
        const T EPS = 1e-2;
        return 1.0 / (x + EPS);
    }

    template <Arithmetic T>
    auto inv_back(const T& x, const T& d) {
        const T EPS = 1e-2;
        return -1.0 / (x * x + EPS) * d;
    }

    template <Arithmetic T>
    auto relu_back(const T& x, const T& d) {
        return x > 0.0 ? d : 0.0;
    }

    // High Order functions Definitions

    template <typename F, Arithmetic T>
    auto map(const F& fn, const std::vector<T>& args) {
        std::vector<decltype(fn(T{}))> result;
        result.reserve(args.size());

        for (const auto& i : args)
            result.push_back(fn(i));

        return result;
    }

    template <typename F, Arithmetic T>
    auto zipWith(const F& fn, const std::vector<T>& x, const std::vector<T>& y) {
        std::vector<decltype(fn(T{}, T{}))> result;
        auto minSize = std::min(x.size(), y.size());
        result.reserve(minSize);

        for (size_t i = 0; i < minSize; ++i)
            result.push_back(fn(x[i], y[i]));
        return result;
    }

    template <typename F, Arithmetic T>
    auto reduce(F fn, T start, const std::vector<T>& ls) {
        return std::accumulate(ls.begin(), ls.end(), start, fn);
    }

    // Utility functions using high order function defined above

    template <Arithmetic T>
    auto sum(const std::vector<T>& ls) {
        return std::accumulate(ls.begin(), ls.end(), T(0));
    }

    template <Arithmetic T>
    auto prod(const std::vector<T>& ls) {
        return reduce(std::multiplies<T>(), T(1), ls);
    }

    template <Arithmetic T>
    auto addLists(const std::vector<T>& ls1, const std::vector<T>& ls2) {
        return zipWith(add<T>, ls1, ls2);
    }

    template <Arithmetic T>
    auto negList(const std::vector<T>& ls) {
        return map(neg<T>, ls);
    }

}  // namespace operators