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

    std::shared_ptr<Scalar> Scalar::create(double data) {
        return std::make_shared<Scalar>(data);
    }

    std::shared_ptr<Scalar> Scalar::create(History hist, double data) {
        return std::make_shared<Scalar>(hist, data);
    }

    std::shared_ptr<Scalar> Scalar::log() {
        return ScalarFunction::apply<Log>(std::make_shared<Scalar>(this));
    }

    std::shared_ptr<Scalar> Scalar::exp() {
        return ScalarFunction::apply<Exp>(std::make_shared<Scalar>(this));
    };

    std::shared_ptr<Scalar> Scalar::sigmoid() {
        return ScalarFunction::apply<Sigmoid>(std::make_shared<Scalar>(this));
    };

    std::shared_ptr<Scalar> Scalar::relu() {
        return ScalarFunction::apply<Relu>(std::make_shared<Scalar>(this));
    };

    std::vector<std::shared_ptr<Scalar>> Scalar::parents() {
        return history.inputs;
    };

    bool Scalar::is_leaf() {
        return parents().empty();
    }

    void Scalar::chain_rule(double deriv) {
        this->history.backward(this->history.ctx, deriv);
        return;
    }
}