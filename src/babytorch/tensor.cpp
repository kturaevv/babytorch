#include <algorithm>
#include <cctype>
#include <ranges>
#include <string>

#include "tensor.hpp"
#include "tensor_data.hpp"
#include "tensor_functions.hpp"
#include "utils.hpp"

namespace tensor {
    using tensor_data::TensorData;

    Shape Tensor::shape() {
        return this->data.shape;
    }

    Tensor Tensor::zeros(Shape shape) {
        return Tensor(TensorData(utils::zeros(shape), shape));
    }

    TensorDataInfo Tensor::info() const {
        return this->data.info();
    }
}  // namespace tensor