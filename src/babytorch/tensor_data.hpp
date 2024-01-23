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

        TensorData(UserShape user_shape) {
            this->size = generic_operators::prod(user_shape);
            this->_storage = utils::rand(size);
            this->strides = strides_from_shape(user_shape);
            this->shape = user_shape;
            this->dims = strides.size();
        }

        TensorData(Storage storage, UserShape shape)
            : _storage(std::move(storage))
            , shape(shape) {
            this->strides = strides_from_shape(shape);
            this->size = generic_operators::prod(shape);
            this->dims = strides.size();
        }

        TensorData(Storage storage, UserShape shape, UserStrides strides)
            : _storage(std::move(storage))
            , shape(shape)
            , strides(strides) {
            this->size = generic_operators::prod(shape);
            this->dims = strides.size();
        }

        void info();
        bool is_contiguous();
        UserIndex sample();
        size_t index(const UserIndex index);
        void set(const UserIndex index);
        double get(const UserIndex key);
        TensorData permute(const ReOrderIndex order);

        static UserShape shape_broadcast(const UserShape shape_a,
                                         const UserShape shape_b);
    };
}  // namespace tensor_data
