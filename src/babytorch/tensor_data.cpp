#include <ranges>
#include <sstream>

#include <fmt/ranges.h>

#include "tensor_data.hpp"

namespace tensor_data {

    size_t index_to_position(const Index& index, const Strides& strides) {
        size_t pos = 0;
        for (auto i : std::ranges::views::iota(0ull, index.size()))
            pos += index[i] * strides[i];
        return pos;
    }

    Strides strides_from_shape(const Shape shape) {
        Strides strides{ 1 };
        size_t offset = 1;

        for (auto s : shape | std::views::drop(1) | std::views::reverse) {
            strides.insert(strides.begin(), s * offset);
            offset *= s;
        }

        return strides;
    }

    size_t TensorData::index(const Index index) {
        if (index.size() != this->shape.size()) {
            fmt::print("Index {}\n", index);
            fmt::print("Shape {}\n", shape);
            throw std::runtime_error(
                "IndexingError: Index must be size of shape.");
        }

        for (auto i : std::views::iota(0ull, index.size())) {
            if (index[i] >= this->shape[i]) {
                std::ostringstream msg;
                msg << "IndexingError: Index " << index[i]
                    << " is out of range for dimension " << i << ".";
                throw std::runtime_error(msg.str());
            }
        }

        return index_to_position(index, this->strides);
    }

    double TensorData::get(const Index key) {
        return (this->_storage)[index(key)];
    }
}  // namespace tensor_data