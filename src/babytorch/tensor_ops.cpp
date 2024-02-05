#include <ranges>

#include "tensor.hpp"
#include "tensor_data.hpp"
#include "tensor_ops.hpp"
#include "utils.hpp"

namespace tensor_ops {

    using tensor_data::Index;
    using tensor_data::Shape;
    using tensor_data::Storage;
    using tensor_data::Strides;

    using tensor_data::broadcast_index;
    using tensor_data::index_to_position;
    using tensor_data::shape_broadcast;
    using tensor_data::to_tensor_index;

    UnivariateTensorDataFn tensor_map(UnivariateFn fn) {
        return [fn](const TensorDataInfo& a) {
            auto& [in_storage, in_shape, in_strides] = a;

            auto out_tensor = Tensor::zeros(in_shape);
            auto data_tuple = out_tensor.data.tuple();

            auto& [out_storage, out_shape, out_strides] = data_tuple;

            Index out_index = utils::zeros<size_t>(out_shape.size());
            Index in_index  = utils::zeros<size_t>(out_shape.size());

            for (size_t idx : std::views::iota(in_storage.size())) {
                to_tensor_index(idx, out_index, out_shape);
                out_index     = broadcast_index(out_index, out_shape, in_shape);
                size_t in_pos = index_to_position(in_index, in_strides);
                size_t out_pos = index_to_position(out_index, out_strides);

                out_storage[out_pos] = fn(in_storage[in_pos]);
            }
            return out_tensor;
        };
    }

    BivariateTensorDataFn tensor_zip(BivariateFn fn) {
        return [fn](const TensorDataInfo& a, const TensorDataInfo& b) {
            auto& [a_storage, a_shape, a_strides] = a;
            auto& [b_storage, b_shape, b_strides] = b;

            Shape out_shape = a_shape != b_shape
                                  ? shape_broadcast(a_shape, b_shape)
                                  : a_shape;

            auto out_tensor = Tensor::zeros(out_shape);
            auto data_tuple = out_tensor.data.tuple();

            auto& [out_storage, _, out_strides] = data_tuple;

            Index out_index = utils::zeros<size_t>(out_shape.size());
            Index a_index   = utils::zeros<size_t>(a_shape.size());
            Index b_index   = utils::zeros<size_t>(b_shape.size());

            size_t idx    = 0;
            size_t a_size = a_storage.size();
            while (idx < a_size) {
                to_tensor_index(idx, out_index, out_shape);
                a_index   = broadcast_index(a_index, a_shape, out_shape);
                b_index   = broadcast_index(b_index, b_shape, out_shape);
                size_t ai = index_to_position(a_index, a_strides);
                size_t bi = index_to_position(b_index, b_strides);
                size_t oi = index_to_position(out_index, out_tensor.data.strides);

                out_storage[oi] = fn(a_storage[ai], b_storage[bi]);
            }
            return out_tensor;
        };
    }

    ReduceTensorDataFn tensor_reduce(BivariateFn fn, double start) {
        return [fn, start](const TensorDataInfo& a, size_t dim) {
            auto& [in_storage, in_shape, in_strides] = a;

            Shape out_shape = in_shape;
            out_shape[dim]  = 1;

            auto out_tensor = Tensor::zeros(out_shape);
            auto data_tuple = out_tensor.data.tuple();

            auto& [out_storage, _, out_strides] = data_tuple;

            Index out_index = utils::zeros<size_t>(out_shape.size());

            for (size_t idx = 0; idx < out_storage.size(); idx++) {
                to_tensor_index(idx, out_index, out_shape);
                auto pos = index_to_position(out_index, out_strides);

                for (auto j : std::views::iota(0ull, in_shape[dim])) {
                    Index in_index = out_index;
                    in_index[dim]  = j;
                    size_t pos_a   = index_to_position(in_index, in_strides);

                    out_storage[pos] = fn(in_storage[pos_a], out_storage[pos]);
                }
            }
            return out_tensor;
        };
    }

    MapFuncFactory TensorOps::map = [](UnivariateFn fn) -> UnivariateTensorFn {
        UnivariateTensorDataFn f = tensor_map(fn);
        UnivariateTensorFn ret   = [&](const Tensor& a) {
            return f(a.info());
        };
        return ret;
    };

    ZipFuncFactory TensorOps::zip;
    ReduceFuncFactory TensorOps::reduce;
    UnivariateTensorFn matrix_multiply;

}  // tensor_ops