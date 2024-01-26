#include <ranges>
#include <span>
#include <sstream>

#include <fmt/ranges.h>

#include "generic_operators.hpp"
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

        for (auto s : shape                      //
                          | std::views::drop(1)  //
                          | std::views::reverse) {
            strides.insert(strides.begin(), s * offset);
            offset *= s;
        }

        return strides;
    }

    size_t TensorData::index(const Index index) const {
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

    TensorStorageView TensorData::view(Index index) const {
        size_t start_idx = index_to_position(index, this->strides);

        auto slice_size = this->strides  //
                          | std::views::drop(index.size() - 1)
                          | std::views::take(1);

        size_t slice_width = std::accumulate(slice_size.begin(),
                                             slice_size.end(),
                                             1,
                                             std::multiplies<double>());

        return TensorStorageView(this->_storage.data() + start_idx, slice_width);
    }

    TensorStorageView TensorData::view() const {
        return TensorStorageView(this->_storage.data(), this->size);
    }

    double TensorData::get(const Index key) {
        return (this->_storage)[index(key)];
    }

    void TensorData::info() const {
        fmt::print("TensorData(shape={}, size={}, dims={}, strides={})\n",
                   this->shape,
                   this->size,
                   this->dims,
                   this->strides);
    }

    std::string TensorData::string_view() const {
        const TensorStorageView storage = this->view();
        Strides strides                 = this->strides;
        Shape shape                     = this->shape;

        std::string tensor_string = "[";
        tensor_string.reserve(storage.size() * 10);

        size_t offset = strides.size();  // offset braces
        offset += 7;                     // offset "Tensor("

        for (size_t idx = 0; idx < storage.size(); idx++) {
            // Closing brackets
            for (auto stride : strides | std::views::take(shape.size() - 1)) {
                if (idx != 0 && idx % stride == 0) {
                    if (tensor_string.back() == ',')
                        tensor_string.pop_back();
                    tensor_string += ']';
                }
            }

            // Newlines
            size_t n_newlines = 0;
            for (auto stride : strides | std::views::take(shape.size() - 1))
                if (idx != 0 && idx % stride == 0) {
                    tensor_string += '\n';
                    n_newlines++;
                }

            // Offset
            if (n_newlines)
                for (size_t i = 0; i < offset - n_newlines; i++)
                    tensor_string += ' ';

            // Opening brackets
            for (auto stride : strides | std::views::take(shape.size() - 1))
                if (idx % stride == 0)
                    tensor_string += '[';

            // Space between nums
            if (tensor_string.back() != '[')
                tensor_string += ' ';

            // Align for negative sign
            if (storage[idx] > 0)
                tensor_string += ' ';

            tensor_string += std::to_string(storage[idx]);

            // Comma
            if (idx % shape.back() != 0 && idx + 1 < storage.size())
                tensor_string += ',';
        }

        // Remove trailing space
        while (std::isspace(tensor_string.back()))
            tensor_string.pop_back();

        // Last closing bracket
        for (size_t i = 0; i <= strides.size() - 1; i++)
            tensor_string += ']';

        return tensor_string;
    }
}  // namespace tensor_data