#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

namespace operators {

    const double EPS = 1e-8;

    double mul(const double x, const double y) {
        return x * y;
    }

    double id(const double x) {
        return x;
    }

    double add(const double x, const double y) {
        return x + y;
    }

    double neg(const double x) {
        return -x;
    }

    double lt(const double x, const double y) {
        return x < y ? 1.0 : 0.0;
    }

    double eq(const double x, const double y) {
        return x == y ? 1.0 : 0.0;
    }

    double max(const double x, const double y) {
        return x > y ? x : y;
    }

    double is_close(const double x, const double y) {
        const double EPS = 1e-2;
        return fabs(x - y) < EPS ? 1.0 : 0.0;
    }

    double sigmoid(const double x) {
        if (x >= 0)
            return 1.0 / (1.0 + exp(-x));
        else
            return exp(x) / (1.0 + exp(x));
    }

    double sigmoid_back(const double x, const double d) {
        return x * (1 - x) * d;
    }

    double relu(const double x) {
        return x > 0 ? x : 0;
    }

    double log_func(const double x) {
        return std::log(x + EPS);
    }

    double exp_func(const double x) {
        return std::exp(x);
    }

    double log_back(const double x, const double d) {
        return 1.0 / (x * std::log(d + EPS) + EPS);
    }

    double inv(const double x) {
        return 1.0 / (x + EPS);
    }

    double inv_back(const double x, const double d) {
        return -1.0 / (x * x + EPS);
    }

    double relu_back(const double x, const double d) {
        return x > 0.0 ? d : 0.0;
    }

    // High Order functions Definitions

    std::vector<double> map(const std::function<double(double)>& fn,
                            const std::vector<double>& args) {
        std::vector<double> result;
        result.reserve(args.size());

        for (const auto& i : args)
            result.push_back(fn(i));

        return result;
    }

    std::vector<double> zipWith(const std::function<double(double, double)>& fn,
                                const std::vector<double>& x,
                                const std::vector<double>& y) {
        std::vector<double> result;

        double minSize = std::min(x.size(), y.size());
        result.reserve(minSize);

        for (const auto i : std::views::iota(0, minSize))
            result.push_back(fn(x[i], y[i]));
        return result;
    }

    double reduce(std::function<double(double, double)> fn, double start,
                  const std::vector<double>& ls) {
        return std::accumulate(ls.begin(), ls.end(), start, fn);
    }

    // Utility functions using high order function defined above

    double sum(const std::vector<double>& ls) {
        return std::accumulate(ls.begin(), ls.end(), 0);
    }

    double prod(const std::vector<double>& ls) {
        return reduce(std::multiplies<double>(), 1.0, ls);
    }

    std::vector<double> addLists(const std::vector<double>& ls1,
                                 const std::vector<double>& ls2) {
        return zipWith(add, ls1, ls2);
    }

    std::vector<double> negList(const std::vector<double>& ls) {
        return map(neg, ls);
    }
}