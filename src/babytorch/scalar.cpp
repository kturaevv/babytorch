#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "functions.hpp"
#include "scalar.hpp"

namespace scalar {
    using namespace functions;

    Scalar Scalar::log() const {
        return ScalarFunction::apply<Log>(*this);
    }

    Scalar Scalar::exp() const {
        return ScalarFunction::apply<Exp>(*this);
    };

    Scalar Scalar::sigmoid() const {
        return ScalarFunction::apply<Sigmoid>(*this);
    };

    Scalar Scalar::relu() const {
        return ScalarFunction::apply<Relu>(*this);
    };

}