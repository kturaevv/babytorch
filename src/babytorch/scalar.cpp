#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <vector>

#include "autodiff.hpp"
#include "functions.hpp"
#include "ptr.hpp"
#include "scalar.hpp"

namespace scalar {
    using namespace functions;

    sptr<Scalar> Scalar::create() {
        return std::make_shared<Scalar>();
    }

    sptr<Scalar> Scalar::create(double data) {
        return std::make_shared<Scalar>(data);
    }

    sptr<Scalar> Scalar::create(History hist, double data) {
        return std::make_shared<Scalar>(hist, data);
    }

    sptr<Scalar> Scalar::log() {
        return ScalarFunction::apply<Log>(std::make_shared<Scalar>(this));
    }

    sptr<Scalar> Scalar::exp() {
        return ScalarFunction::apply<Exp>(std::make_shared<Scalar>(this));
    }

    sptr<Scalar> Scalar::sigmoid() {
        return ScalarFunction::apply<Sigmoid>(std::make_shared<Scalar>(this));
    }

    sptr<Scalar> Scalar::relu() {
        return ScalarFunction::apply<Relu>(std::make_shared<Scalar>(this));
    }

    std::vector<sptr<Scalar>> Scalar::parents() {
        return history.inputs;
    }

    bool Scalar::is_leaf() {
        return parents().empty();
    }

    void Scalar::accumulate_grad(double deriv) {
        this->grad += deriv;
        return;
    }

    std::vector<std::tuple<sptr<Scalar>, double>> Scalar::chain_rule(
        double deriv) {
        auto hist                   = this->history;
        std::array<double, 2> grads = hist.backward(hist.ctx, deriv);

        std::vector<std::tuple<sptr<Scalar>, double>> vals;
        for (size_t i = 0; i < hist.inputs.size() && i < 2; i++)
            vals.emplace_back(hist.inputs[i], grads[i]);

        return vals;
    }

    void Scalar::backward() {
        autodiff::backpropagate(std::make_shared<Scalar>(this), 1.0);
        return;
    }
}
