#pragma once

#include <functional>
#include <vector>

#include "operators.hpp"
#include "tensor_data.hpp"

// Forward declaration of Tensor instead of including tensor.hpp
namespace tensor {
    struct Tensor;
}

namespace tensor_ops {
    // Tensor base operations that leverage high order functions
    // to apply base operators on a Tensor

    // Forward declarations
    using tensor::Tensor;
    using tensor_data::TensorDataInfo;

    // Aliases
    using UnivariateFn = std::function<double(double)>;
    using BivariateFn  = std::function<double(double, double)>;

    using UnivariateTensorFn = std::function<Tensor(const TensorDataInfo&)>;
    using BivariateTensorFn
        = std::function<Tensor(const TensorDataInfo&, const TensorDataInfo&)>;
    using ReduceTensorFn
        = std::function<Tensor(const TensorDataInfo&, const size_t)>;

    // Function factories
    using MapFuncFactory = std::function<UnivariateTensorFn(UnivariateFn)>;
    using ZipFuncFactory = std::function<BivariateTensorFn(BivariateFn)>;
    using ReduceFuncFactory = std::function<ReduceTensorFn(BivariateFn, double)>;

    struct TensorOps {
        MapFuncFactory map;
        ZipFuncFactory zip;
        ReduceFuncFactory reduce;
        UnivariateTensorFn matrix_multiply;
    };

    struct TensorBackend {
        UnivariateTensorFn id_map;
        UnivariateTensorFn neg_map;
        UnivariateTensorFn inv_map;
        UnivariateTensorFn relu_map;
        UnivariateTensorFn log_map;
        UnivariateTensorFn exp_map;
        UnivariateTensorFn sigmoid_map;

        // Zip operations
        BivariateTensorFn add_zip;
        BivariateTensorFn mul_zip;
        BivariateTensorFn lt_zip;
        BivariateTensorFn eq_zip;
        BivariateTensorFn is_close_zip;
        BivariateTensorFn relu_back_zip;
        BivariateTensorFn log_back_zip;
        BivariateTensorFn inv_back_zip;

        // Reduce operations
        ReduceTensorFn add_reduce;
        ReduceTensorFn mul_reduce;

        TensorBackend(const TensorOps* ops) {
            if (ops == nullptr)
                throw std::invalid_argument("TensorOps pointer cannot be null");

            this->id_map      = ops->map(operators::id);
            this->neg_map     = ops->map(operators::neg);
            this->inv_map     = ops->map(operators::inv);
            this->relu_map    = ops->map(operators::relu);
            this->log_map     = ops->map(operators::log_func);
            this->exp_map     = ops->map(operators::exp_func);
            this->sigmoid_map = ops->map(operators::sigmoid);

            this->add_zip       = ops->zip(operators::add);
            this->mul_zip       = ops->zip(operators::mul);
            this->lt_zip        = ops->zip(operators::lt);
            this->eq_zip        = ops->zip(operators::eq);
            this->is_close_zip  = ops->zip(operators::is_close);
            this->relu_back_zip = ops->zip(operators::relu_back);
            this->log_back_zip  = ops->zip(operators::log_back);
            this->inv_back_zip  = ops->zip(operators::inv_back);

            this->add_reduce = ops->reduce(operators::add, 0);
            this->mul_reduce = ops->reduce(operators::mul, 1);
        }

        // Additional methods
        Tensor (*matrix_multiply)(Tensor&, Tensor&);  // Pointer to
                                                      // matrix_multiply
                                                      // function
    };

    UnivariateTensorFn tensor_map(UnivariateFn);
    BivariateTensorFn tensor_zip(BivariateFn);
    ReduceTensorFn tensor_reduce(BivariateFn, double);

}  // namespace tensor_ops
