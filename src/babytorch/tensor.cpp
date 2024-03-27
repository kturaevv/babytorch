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
        return this->data.shape;
    }

    sptr<Tensor> Tensor::zeros(Shape shape) {
        return std::make_shared<Tensor>(TensorData(utils::zeros(shape), shape));
    }

    sptr<Tensor> Tensor::zeros() const {
        return Tensor::zeros(this->shape());
    }

    TensorDataInfo Tensor::info() const {
        return this->data.info();
    }

    std::vector<sptr<Tensor>> Tensor::parents() const {
        return this->history.inputs;
    }

    bool Tensor::is_leaf() {
        return parents().empty();
    }

    void Tensor::accumulate_grad(sptr<Tensor> deriv) {
        (*this->grad) += deriv;
        return;
    }

    // std::vector<std::tuple<sptr<Tensor>, sptr<Tensor>>>
    // Tensor::chain_rule(sptr<Tensor> deriv) {
    //     History history             = this->history;
    //     std::array<sptr<Tensor>, 2> grads = history.backward(history.ctx, deriv);

    //     std::vector<std::tuple<sptr<Tensor>, sptr<Tensor>>> vals;
    //     // for (size_t i = 0; i < history.inputs.size() && i < 2; i++)
    //     //     vals.emplace_back(history.inputs[i], std::move(grads[i]));

    //     return vals;
    // }

    void Tensor::backward() {
        Storage storage    = Storage{ 1.0 };
        TensorData _data   = TensorData{ storage };
        sptr<Tensor> _this = shared_from_this();
        auto deriv         = std::make_shared<Tensor>(Tensor(_data));

        tensor_autodiff::backpropagate(_this, deriv);
        return;
    }

}  // namespace tensor
