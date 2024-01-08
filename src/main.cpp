#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>

#include <boost/lambda/lambda.hpp>
#include <fmt/core.h>

#include "./babytorch/scalar.hpp"

int main() {
    fmt::print("Auto-diff project!\n");

    using namespace scalar;
    using namespace boost::lambda;

    typedef std::istream_iterator<int> in;

    std::for_each(in(std::cin), in(), std::cout << (_1 * 3) << " ");

    auto x = Scalar::create(1.0);
    auto y = Scalar::create(2.0);
    auto z = Scalar::create(3.0);
    auto k = Scalar::create(4.0);
    auto j = Scalar::create(5.0);

    std::shared_ptr<Scalar> result = (x * y + z - k / j);
    std::cout << "Results: " << *result;

    result->backward();

    return 0;
}
