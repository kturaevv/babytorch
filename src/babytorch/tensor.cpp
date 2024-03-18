#include <algorithm>
#include <cctype>
#include <ranges>
#include <string>
#include <memory>

#include "tensor.hpp"
#include "tensor_autodiff.hpp"
#include "tensor_data.hpp"
#include "tensor_functions.hpp"
#include "utils.hpp"

namespace tensor {
    using namespace tensor_autodiff;

    Shape Tensor::shape() const {
        return this->data.shape;
    }

    std::shared_ptr<Tensor> Tensor::zeros(Shape shape) {
        return std::make_shared<Tensor>(TensorData(utils::zeros(shape), shape));
    }

    std::shared_ptr<Tensor> Tensor::zeros() const {
        return Tensor::zeros(this->shape());
    }

    TensorDataInfo Tensor::info() const {
        return this->data.info();
    }

    std::vector<std::shared_ptr<Tensor>> Tensor::parents() const {
        return this->history.inputs;
    }

    bool Tensor::is_leaf() {
        return parents().empty();
    }

    void Tensor::accumulate_grad(std::shared_ptr<Tensor> deriv) {
        (*this->grad) += deriv;
        return;
    }

    // std::vector<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> Tensor::chain_rule(std::shared_ptr<Tensor> deriv) {
    //     History history             = this->history;
    //     std::array<std::shared_ptr<Tensor>, 2> grads = history.backward(history.ctx, deriv);

    //     std::vector<std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> vals;
    //     // for (size_t i = 0; i < history.inputs.size() && i < 2; i++)
    //     //     vals.emplace_back(history.inputs[i], std::move(grads[i]));

    //     return vals;
    // }

    void Tensor::backward() {
        auto deriv = std::make_shared<Tensor>(TensorData({ 1.0 }, Shape{}));
        tensor_autodiff::backpropagate(this, deriv);
        return;
    }

}  // namespace tensor
