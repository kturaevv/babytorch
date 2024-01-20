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
    using Shape = std::vector<int>;
    using Strides = std::vector<int>;
    using UserIndex = std::vector<int>;
    using UserShape = std::vector<int>;
    using UserStrides = std::vector<int>;
    using ReOrderIndex = std::vector<int>;

    // Map n-dim pos. to 1-dim storage
    void to_index();
    void broadcast_index();
    size_t index_to_position();
    UserShape shape_broadcast();
    UserStrides strides_from_shape();

    struct TensorData {
        int dims;
        Storage _storage;
        Shape _shape;
        Strides _strides;
        UserShape shape;
        UserStrides strides;
        size_t size;

        TensorData(){};

        TensorData(UserShape dims) {
            size = generic_operators::prod(dims);
            _storage = utils::rand(size);
        }

        TensorData(Storage storage, UserShape shape)
            : _storage(std::move(storage))
            , _shape(shape) {
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
