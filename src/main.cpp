#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>

#include <boost/lambda/lambda.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "./babytorch/operators.hpp"
#include "./babytorch/scalar.hpp"
#include "./babytorch/tensor.hpp"
#include "./babytorch/utils.hpp"

int main() {
    fmt::print("Auto-diff project!\n");

    using namespace scalar;
    using namespace tensor;
    using namespace boost::lambda;

    // typedef std::istream_iterator<int> in;
    // std::for_each(in(std::cin), in(), std::cout << (_1 * 3) << " ");

    auto x = Scalar::create(1.0);
    auto y = Scalar::create(2.0);
    auto z = Scalar::create(3.0);
    auto k = Scalar::create(4.0);
    auto j = Scalar::create(5.0);

    std::shared_ptr<Scalar> result = (x * y + z - k / j);
    fmt::print("Results: {}", *result);

    result->backward();

    auto a = Tensor(3, 3, 5);
    fmt::print("Tensor storage: {}\n", *a.data._storage);
    fmt::print("Tensor strides: {}\n", a.data.strides);
    return 0;
}
