#pragma once

#include <functional>
#include <vector>

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
        static TensorFunction map(SingleVariableFunction);
        static TensorFunction zip(TwoVariableFunction);
        static TensorReduceFunction reduce(TwoVariableFunction, double);
        static Tensor matrix_multiply(Tensor&, Tensor&);
    };

    namespace TensorOps {
        // Map operations
        static TensorFunction neg_map;
        static TensorFunction sigmoid_map;
        static TensorFunction relu_map;
        static TensorFunction log_map;
        static TensorFunction exp_map;
        static TensorFunction id_map;
        static TensorFunction id_cmap;
        static TensorFunction inv_map;

        // Zip operations
        static TensorFunction add_zip;
        static TensorFunction mul_zip;
        static TensorFunction lt_zip;
        static TensorFunction eq_zip;
        static TensorFunction is_close_zip;
        static TensorFunction relu_back_zip;
        static TensorFunction log_back_zip;
        static TensorFunction inv_back_zip;

        // Reduce operations
        static TensorReduceFunction add_reduce;
        static TensorReduceFunction mul_reduce;

        // Additional methods
        Tensor (*matrix_multiply)(Tensor&, Tensor&);  // Pointer to
                                                      // matrix_multiply
                                                      // function
    };

    // Helper functions
    void tensor_map();
    void tensor_zip();
    void tensor_reduce();

}  // namespace tensor_ops
