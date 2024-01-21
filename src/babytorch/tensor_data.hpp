#pragma once

#include <memory>
#include <vector>

#include "generic_operators.hpp"
#include "utils.hpp"

namespace tensor_data {

    // Type - aliases
    using Storage = std::unique_ptr<std::vector<double>>;
    using OutIndex = std::vector<size_t>;

    using Index = std::vector<size_t>;
    using Shape = std::vector<size_t>;
    using Strides = std::vector<size_t>;

    using UserIndex = std::vector<size_t>;
    using UserShape = std::vector<size_t>;
    using UserStrides = std::vector<size_t>;
    using ReOrderIndex = std::vector<size_t>;

    // Map n-dim pos. to 1-dim storage
    void to_index(size_t& ordinal, const Shape& shape, const OutIndex& out_index);
    void broadcast_index(Index& index, const Shape in_shape,
                         const Shape out_shape, const OutIndex out_index);
    size_t index_to_position(const Index& index, const Strides& strides);
    UserShape shape_broadcast();
    UserStrides strides_from_shape(UserShape shape);

    struct TensorData {
        Storage _storage;
        Shape _shape;
        Strides _strides;

        UserShape shape;
        UserStrides strides;

        size_t size = 0;
        int dims = 0;

        TensorData(){};

        TensorData(UserShape shape) {
            size = generic_operators::prod(shape);
            _storage = utils::rand(size);
            strides = strides_from_shape(shape);
            dims = strides.size();
        }

        TensorData(Storage storage, UserShape shape)
            : _storage(std::move(storage))
            , shape(shape) {
            strides = strides_from_shape(shape);
            size = generic_operators::prod(shape);
            dims = strides.size();
        }

        TensorData(Storage storage, UserShape shape, UserStrides strides)
            : _storage(std::move(storage))
            , shape(shape)
            , strides(strides) {
            size = generic_operators::prod(shape);
            dims = strides.size();
        }

        bool is_contiguous();
        UserIndex sample();
        size_t index(UserIndex index);
        void set(UserIndex index);
        double get(UserIndex key);
        TensorData permute(ReOrderIndex order);

        static UserShape shape_broadcast(UserShape shape_a, UserShape shape_b);
    };
}  // namespace tensor_data
