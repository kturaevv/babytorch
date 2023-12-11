#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

namespace operators {

    double id(const double x);
    double neg(const double x);
    double inv(const double x);
    double relu(const double x);
    double sigmoid(const double x);
    double log_func(const double x);
    double exp_func(const double x);
    double mul(const double x, const double y);
    double add(const double x, const double y);
    double lt(const double x, const double y);
    double eq(const double x, const double y);
    double max(const double x, const double y);
    double is_close(const double x, const double y);
    double log_back(const double x, const double d);
    double inv_back(const double x, const double d);
    double relu_back(const double x, const double d);

    // High Order functions Definitions
    std::vector<double> map(const std::function<double(double)>& fn,
                            const std::vector<double>& args);
    std::vector<double> zipWith(const std::function<double(double, double)>& fn,
                                const std::vector<double>& x,
                                const std::vector<double>& y);
    double reduce(std::function<double(double, double)> fn, double start,
                  const std::vector<double>& ls);

    // Utility functions using high order function defined above
    double sum(const std::vector<double>& ls);
    double prod(const std::vector<double>& ls);
    std::vector<double> addLists(const std::vector<double>& ls1,
                                 const std::vector<double>& ls2);
    std::vector<double> negList(const std::vector<double>& ls);
}