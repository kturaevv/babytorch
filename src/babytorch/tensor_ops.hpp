#pragma once

#include <functional>
#include <vector>

#include "operators.hpp"

namespace tensor {
    struct Tensor;
}

namespace tensor_ops {
    // Tensor base operations that leverage high order functions
    // to apply base operators on a Tensor

    // Forward declarations
    using tensor::Tensor;
    using Shape   = std::vector<size_t>;
    using Storage = std::vector<double>;
    using Strides = std::vector<size_t>;

    // Aliases
    using TensorFunction       = std::function<Tensor(Tensor&, Tensor&)>;
    using TensorReduceFunction = std::function<Tensor(Tensor&, double)>;

    using SingleVariableFunction = std::function<double(double)>;
    using TwoVariableFunction    = std::function<double(double, double)>;

    namespace TensorFuncs {
        TensorFunction map(SingleVariableFunction);
        TensorFunction zip(TwoVariableFunction);
        TensorReduceFunction reduce(TwoVariableFunction, double);
        Tensor matrix_multiply(Tensor&, Tensor&);
    };

    namespace TensorOps {
        // Map operations
        TensorFunction id_map      = TensorFuncs::map(operators::id);
        TensorFunction neg_map     = TensorFuncs::map(operators::neg);
        TensorFunction inv_map     = TensorFuncs::map(operators::inv);
        TensorFunction relu_map    = TensorFuncs::map(operators::relu);
        TensorFunction log_map     = TensorFuncs::map(operators::log_func);
        TensorFunction exp_map     = TensorFuncs::map(operators::exp_func);
        TensorFunction sigmoid_map = TensorFuncs::map(operators::sigmoid);

        // Zip operations
        TensorFunction add_zip       = TensorFuncs::zip(operators::add);
        TensorFunction mul_zip       = TensorFuncs::zip(operators::mul);
        TensorFunction lt_zip        = TensorFuncs::zip(operators::lt);
        TensorFunction eq_zip        = TensorFuncs::zip(operators::eq);
        TensorFunction is_close_zip  = TensorFuncs::zip(operators::is_close);
        TensorFunction relu_back_zip = TensorFuncs::zip(operators::relu_back);
        TensorFunction log_back_zip  = TensorFuncs::zip(operators::log_back);
        TensorFunction inv_back_zip  = TensorFuncs::zip(operators::inv_back);

        // Reduce operations
        TensorReduceFunction add_reduce = TensorFuncs::reduce(operators::add, 0);
        TensorReduceFunction mul_reduce = TensorFuncs::reduce(operators::mul, 1);

        // Additional methods
        Tensor (*matrix_multiply)(Tensor&, Tensor&);  // Pointer to
                                                      // matrix_multiply
                                                      // function
    };

    // Helper functions
    void tensor_map(SingleVariableFunction);
    void tensor_zip(TwoVariableFunction);
    void tensor_reduce(TensorReduceFunction);

}  // namespace tensor_ops
