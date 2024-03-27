#pragma once

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "generic_operators.hpp"
#include "utils.hpp"

namespace tensor_data {

    struct IndexingError : std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // Type - aliases
    using Storage = std::vector<double>;
    using Index   = std::vector<size_t>;
    using Shape   = std::vector<size_t>;
    using Strides = std::vector<size_t>;

    using ReOrderIndex      = std::vector<size_t>;
    using TensorStorageView = std::span<const double>;
    using TensorDataTuple   = std::tuple<Storage&, Shape&, Strides&>;
    using TensorDataInfo
        = std::tuple<const Storage&, const Shape&, const Strides&>;

    Index to_tensor_index(const size_t storage_idx,
                          const Index& tensor_idx,
                          const Shape& shape);

    Index broadcast_index(const Index& original_index,
                          const Shape& original_shape,
                          const Shape& broadcasted_shape);

    Shape shape_broadcast(const Shape& shape1, const Shape& shape2);
    size_t index_to_position(const Index& index, const Strides& strides);
    Strides strides_from_shape(const Shape& shape);

    struct TensorData {
        Storage _storage;
        Shape shape;
        Strides strides;

        size_t size = 0;
        int dims    = 0;

        TensorData() {
            this->_storage = { 0 };
            this->shape    = { 0 };
            this->strides  = { 0 };
        };

        TensorData(Storage storage)
            : _storage(std::move(storage)) {
            // If vector passed construct 1-d Tensor
            this->shape   = { this->_storage.size() };
            this->strides = strides_from_shape(shape);
            this->size    = generic_operators::prod(shape);
            this->dims    = strides.size();
        }

        TensorData(Storage storage, Shape shape)
            : _storage(std::move(storage))
            , shape(shape) {
            if (shape.size() == 0)
                this->shape = { 1 };
            this->strides = strides_from_shape(shape);
            this->size    = generic_operators::prod(shape);
            this->dims    = strides.size();
        }

        TensorData(Storage storage, Shape shape, Strides strides)
            : _storage(std::move(storage))
            , shape(shape)
            , strides(strides) {
            if (shape.size() == 0)
                this->shape = { 1 };
            this->size = generic_operators::prod(shape);
            this->dims = strides.size();
        }

        void print_info() const;
        TensorDataInfo info() const;
        TensorDataTuple tuple();
        bool is_contiguous();
        Index sample();
        size_t index(const Index& index) const;
        void set(const Index& index);
        double get(const Index& key);
        TensorData permute(const ReOrderIndex order);

        TensorStorageView view() const;
        TensorStorageView view(const Index& index) const;
        std::string string_view() const;

        static TensorData rand(Shape user_shape) {
            size_t new_size     = generic_operators::prod(user_shape);
            Storage new_storage = utils::rand(new_size);
            return TensorData(new_storage, user_shape);
        }

        static Shape shape_broadcast(const Shape shape_a, const Shape shape_b);
    };
}  // namespace tensor_data
