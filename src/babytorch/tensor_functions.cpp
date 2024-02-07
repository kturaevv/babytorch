#include "operators.hpp"
#include "tensor.hpp"
#include "tensor_functions.hpp"

namespace tensor_functions {

    using tensor::Tensor;
    using tensor_autodiff::Context;

    Tensor Add::forward(Context& ctx, const Tensor& self, const Tensor& other) {
        ctx.save_for_backwards(self, other);
        return self.backend.add_zip(self, other);
    }

    Tensor Id::forward(Context& ctx, const Tensor& self) {
        ctx.save_for_backwards(self);
        return self.backend.id_map(self);
    }

    Tensor Neg::forward(Context& ctx, const Tensor& self) {
        ctx.save_for_backwards(self);
        return self.backend.neg_map(self);
    }

    Tensor Inv::forward(Context& ctx, const Tensor& self) {
        ctx.save_for_backwards(self);
        return self.backend.inv_map(self);
    }

    Tensor Relu::forward(Context& ctx, const Tensor& self) {
        ctx.save_for_backwards(self);
        return self.backend.relu_map(self);
    }

    Tensor Sigmoid::forward(Context& ctx, const Tensor& self) {
        ctx.save_for_backwards(self);
        return self.backend.sigmoid_map(self);
    }

    Tensor Log::forward(Context& ctx, const Tensor& self) {
        ctx.save_for_backwards(self);
        return self.backend.log_map(self);
    }

    Tensor Exp::forward(Context& ctx, const Tensor& self) {
        ctx.save_for_backwards(self);
        return self.backend.exp_map(self);
    }

    Tensor Mul::forward(Context& ctx, const Tensor& self, const Tensor& other) {
        ctx.save_for_backwards(self, other);
        return self.backend.mul_zip(self, other);
    }

    Tensor Lt::forward(Context& ctx, const Tensor& self, const Tensor& other) {
        ctx.save_for_backwards(self, other);
        return self.backend.lt_zip(self, other);
    }

    Tensor Eq::forward(Context& ctx, const Tensor& self, const Tensor& other) {
        ctx.save_for_backwards(self, other);
        return self.backend.eq_zip(self, other);
    }

    Tensor Is_close::forward(Context& ctx,
                             const Tensor& self,
                             const Tensor& other) {
        ctx.save_for_backwards(self, other);
        return self.backend.is_close_zip(self, other);
    }

}  // namespace tensor_functions