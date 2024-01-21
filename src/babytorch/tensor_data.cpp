#include <ranges>

#include "tensor_data.hpp"

namespace tensor_data {

    size_t index_to_position(const Index& index, const Strides& strides) {
        size_t pos = 0;
        for (auto i : std::ranges::views::iota(0, (int)index.size()))
            pos += index[i] * strides[i];
        return pos;
    }

    UserStrides strides_from_shape(UserShape shape) {
        UserStrides layout{ 1 };
        size_t offset = 1;
        for (auto s : shape | std::views::reverse) {
            layout.push_back(s * offset);
            offset *= s;
        }
        return layout;
    }

    size_t TensorData::index(UserIndex index) {
        std::cout << "LEN: " << index.size() << " " << strides.size();
        return index_to_position(index, strides);
    }

    double TensorData::get(UserIndex key) {
        return (*_storage)[index(key)];
    }
}  // namespace tensor_data