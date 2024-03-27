#pragma once

#include <concepts>
#include <functional>
#include <vector>

#include "operators.hpp"
#include "tensor_data.hpp"
#include "ptr.hpp"

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

    using UnivariateTensorFn = std::function<sptr<Tensor>(const sptr<Tensor>&)>;
    using BivariateTensorFn = std::function<sptr<Tensor>(const sptr<Tensor>&, const sptr<Tensor>&)>;
    using ReduceTensorFn = std::function<sptr<Tensor>(const sptr<Tensor>&, const size_t)>;

    using UnivariateTensorDataFn  //
        = std::function<sptr<Tensor>(const TensorDataInfo&)>;
    using BivariateTensorDataFn
        = std::function<sptr<Tensor>(const TensorDataInfo&, const TensorDataInfo&)>;
    using ReduceTensorDataFn
        = std::function<sptr<Tensor>(const TensorDataInfo&, const size_t)>;

    // 1layer =
    // Function factories
    using MapFuncFactory = std::function<UnivariateTensorFn(UnivariateFn)>;
    using ZipFuncFactory = std::function<BivariateTensorFn(BivariateFn)>;
    using ReduceFuncFactory = std::function<ReduceTensorFn(BivariateFn, double)>;

    struct TensorOps {
        static MapFuncFactory map;
        static ZipFuncFactory zip;
        static ReduceFuncFactory reduce;
        static UnivariateTensorFn matrix_multiply;
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

        TensorBackend() {
            this->id_map      = TensorOps::map(operators::id);
            this->neg_map     = TensorOps::map(operators::neg);
            this->inv_map     = TensorOps::map(operators::inv);
            this->relu_map    = TensorOps::map(operators::relu);
            this->log_map     = TensorOps::map(operators::log_func);
            this->exp_map     = TensorOps::map(operators::exp_func);
            this->sigmoid_map = TensorOps::map(operators::sigmoid);

            this->add_zip       = TensorOps::zip(operators::add);
            this->mul_zip       = TensorOps::zip(operators::mul);
            this->lt_zip        = TensorOps::zip(operators::lt);
            this->eq_zip        = TensorOps::zip(operators::eq);
            this->is_close_zip  = TensorOps::zip(operators::is_close);
            this->relu_back_zip = TensorOps::zip(operators::relu_back);
            this->log_back_zip  = TensorOps::zip(operators::log_back);
            this->inv_back_zip  = TensorOps::zip(operators::inv_back);

            this->add_reduce = TensorOps::reduce(operators::add, 0);
            this->mul_reduce = TensorOps::reduce(operators::mul, 1);
        }

        // Additional methods
        Tensor (*matrix_multiply)(sptr<Tensor>, sptr<Tensor>);  // Pointer to
                                                      // matrix_multiply
                                                      // function
    };

    UnivariateTensorDataFn tensor_map(UnivariateFn);
    BivariateTensorDataFn tensor_zip(BivariateFn);
    ReduceTensorDataFn tensor_reduce(BivariateFn, double);

}  // namespace tensor_ops
