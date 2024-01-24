#pragma once

#include <memory>
#include <span>
#include <vector>

#include "generic_operators.hpp"
#include "utils.hpp"

namespace tensor_data {

    // Type - aliases
    using Storage = std::vector<double>;
    using OutIndex = std::vector<size_t>;

    using Index = std::vector<size_t>;
    using Shape = std::vector<size_t>;
    using Strides = std::vector<size_t>;

    using TensorStorageView = std::span<double>;
    using ReOrderIndex = std::vector<size_t>;

    // Map n-dim pos. to 1-dim storage
    void to_index(size_t& ordinal, const Shape& shape, const OutIndex& out_index);
    void broadcast_index(Index& index, const Shape in_shape,
                         const Shape out_shape, const OutIndex out_index);
    size_t index_to_position(const Index& index, const Strides& strides);
    Shape shape_broadcast();
    Strides strides_from_shape(Shape shape);

    struct TensorData {
        Storage _storage;
        Shape shape;
        Strides strides;

        size_t size = 0;
        int dims = 0;

        TensorData() {
            this->_storage = { 0 };
            this->shape = { 0 };
            this->strides = { 0 };
        };

        TensorData(Storage storage)
            : _storage(std::move(storage)) {
            this->shape = { this->_storage.size() };
            this->strides = strides_from_shape(shape);
            this->size = generic_operators::prod(shape);
            this->dims = strides.size();
        }

        TensorData(Storage storage, Shape shape)
            : _storage(std::move(storage))
            , shape(shape) {
            this->strides = strides_from_shape(shape);
            this->size = generic_operators::prod(shape);
            this->dims = strides.size();
        }

        TensorData(Storage storage, Shape shape, Strides strides)
            : _storage(std::move(storage))
            , shape(shape)
            , strides(strides) {
            this->size = generic_operators::prod(shape);
            this->dims = strides.size();
        }

        void info();
        bool is_contiguous();
        Index sample();
        size_t index(const Index index);
        void set(const Index index);
        double get(const Index key);
        TensorData permute(const ReOrderIndex order);
        TensorStorageView view(const Index index);

        static TensorData rand(Shape user_shape) {
            size_t new_size = generic_operators::prod(user_shape);
            Storage new_storage = utils::rand(new_size);
            return TensorData(new_storage, user_shape);
        }

        static Shape shape_broadcast(const Shape shape_a, const Shape shape_b);
    };
}  // namespace tensor_data
