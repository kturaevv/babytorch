#include <cassert>
#include <cctype>
#include <memory>

#include "tensor.hpp"
#include "tensor_autodiff.hpp"
#include "tensor_data.hpp"
#include "tensor_functions.hpp"
#include "utils.hpp"

namespace tensor {
    using namespace tensor_autodiff;

    Shape Tensor::shape() const {
        return this->data->shape;
    }

    sptr<Tensor> Tensor::zeros(Shape shape) {
        auto tensor_data = std::make_unique<TensorData>(utils::zeros(shape),
                                                        shape);
        return Tensor::create(std::move(tensor_data));
    }

    sptr<Tensor> Tensor::zeros() const {
        return Tensor::zeros(this->shape());
    }

    TensorDataInfo Tensor::info() const {
        return this->data->info();
    }

    std::vector<sptr<Tensor>> Tensor::parents() const {
        return this->history.inputs;
    }

    bool Tensor::is_leaf() {
        return parents().empty();
    }

    void Tensor::accumulate_grad(sptr<Tensor>&& deriv) {
        (*this->grad) += deriv;
        return;
    }

    std::vector<std::tuple<sptr<Tensor>, sptr<Tensor>>> Tensor::chain_rule(
        sptr<Tensor> deriv) {
        auto backward_fn = this->history.backward;
        auto grads       = backward_fn(this->history.ctx, deriv);

        std::vector<std::tuple<sptr<Tensor>, sptr<Tensor>>> zip_inputs_grads;
        for (size_t i = 0; i < history.inputs.size() && i < 2; i++)
            zip_inputs_grads.emplace_back(history.inputs[i], std::move(grads[i]));

        return zip_inputs_grads;
    }

    void Tensor::backward() {
        auto deriv = Tensor::create({ 1.0 });
        auto self  = shared_from_this();
        tensor_autodiff::backpropagate(self, deriv);
        return;
    }
}  // namespace tensor
