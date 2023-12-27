#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "autodiff.hpp"
#include "functions.hpp"
#include "operators.hpp"
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
    }

    std::shared_ptr<Scalar> Scalar::sigmoid() {
        return ScalarFunction::apply<Sigmoid>(std::make_shared<Scalar>(this));
    }

    std::shared_ptr<Scalar> Scalar::relu() {
        return ScalarFunction::apply<Relu>(std::make_shared<Scalar>(this));
    }

    std::vector<std::shared_ptr<Scalar>> Scalar::parents() {
        return history.inputs;
    }

    bool Scalar::is_leaf() {
        return parents().empty();
    }

    std::variant<backward_return_type> Scalar::chain_rule(double deriv) {
        auto history = this->history;
        auto grads = history.backward(history.ctx, deriv);
        return grads;
    }

    void Scalar::backward() {
        autodiff::backpropagate(std::make_shared<Scalar>(this), 1.0);
        return;
    }

    void accumulate_grad(double d_x);
}