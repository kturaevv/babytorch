#pragma once

#include <vector>

namespace tensor_data {

    // Type - aliases
    using Storage = std::vector<double>;
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
        Storage _storage;
        Strides _strides;
        Shape _shape;
        UserStrides strides;
        UserShape shape;
        int dims;

        bool is_contiguous();
        UserIndex sample();
        size_t index(UserIndex index);
        void set(UserIndex index);
        double get(UserIndex key);
        static UserShape shape_broadcast(UserShape shape_a, UserShape shape_b);
        TensorData permute(ReOrderIndex order);
    };
}  // namespace tensor_data
