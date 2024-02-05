#include "operators.hpp"
#include "tensor.hpp"
#include "tensor_functions.hpp"

namespace tensor_functions {

    using tensor::Tensor;
    using tensor_autodiff::Context;

    Tensor Add::forward(Context& ctx, const Tensor& self, const Tensor& other) {
        ctx.save_for_backwards(self, other);
        return self.backend->add_zip(self, other);
    }
}  // namespace tensor_functions