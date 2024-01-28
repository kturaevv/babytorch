#include <ranges>
#include <span>
#include <sstream>
#include <tuple>

#include <fmt/ranges.h>

#include "generic_operators.hpp"
#include "tensor_data.hpp"

namespace tensor_data {

    Index broadcast_index(const Index& original_index,
                          const Shape& original_shape,
                          const Shape& broadcasted_shape) {
        Index broadcasted_index(broadcasted_shape.size(), 0);

        int oi = original_shape.size() - 1;
        int bi = broadcasted_shape.size() - 1;

        while (oi >= 0) {
            broadcasted_index[bi] = original_shape[oi] == 1 ? 0
                                                            : original_index[oi];
            oi--;
            bi--;
        }

        return broadcasted_index;
    }

    Shape shape_broadcast(const Shape& shape1, const Shape& shape2) {
        Shape max_shape = shape1.size() >= shape2.size() ? shape1 : shape2;
        Shape min_shape = shape1.size() < shape2.size() ? shape1 : shape2;

        int min_size = min_shape.size();
        int max_size = max_shape.size();
        int offset   = max_size - min_size;
        int idx      = max_size - 1;

        Shape new_shape(max_size);

        while (idx >= 0) {
            int min_idx = idx - offset;
            int max_val = max_shape[idx];
            int min_val = min_idx >= 0 ? min_shape[min_idx] : 1;

            // If neither of dimensions equal each other or 1
            if (min_val != 1 && max_val != 1 && min_val != max_val)
                throw std::runtime_error("IndexingError: Shape mismatch!");

            new_shape[idx] = max_val > min_val ? max_val : min_val;
            idx--;
        }

        return new_shape;
    }

    void to_tensor_index(const size_t storage_idx,
                         Index& tensor_idx,
                         const Shape& shape) {
        size_t _storage_idx = storage_idx;

        for (auto i : std::views::iota(0ull, shape.size())) {
            tensor_idx[i] = _storage_idx % shape[i];
            _storage_idx  = static_cast<int>(_storage_idx / shape[i]);
        }
    }

    size_t index_to_position(const Index& index, const Strides& strides) {
        size_t pos = 0;
        for (auto i : std::ranges::views::iota(0ull, index.size()))
            pos += index[i] * strides[i];
        return pos;
    }

    Strides strides_from_shape(const Shape& shape) {
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

    size_t TensorData::index(const Index& index) const {
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

    TensorStorageView TensorData::view(const Index& index) const {
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

    double TensorData::get(const Index& key) {
        return (this->_storage)[index(key)];
    }

    void TensorData::print_info() const {
        fmt::print("TensorData(shape={}, size={}, dims={}, strides={})\n",
                   this->shape,
                   this->size,
                   this->dims,
                   this->strides);
    }

    TensorDataInfo TensorData::tuple() {
        return TensorDataInfo(this->_storage, this->shape, this->strides);
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