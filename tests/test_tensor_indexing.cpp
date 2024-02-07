#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include "../src/babytorch/tensor_data.cpp"

using namespace tensor_data;

TEST_CASE("to_tensor_index") {
    SECTION("Simple case") {
        size_t storage_idx = 5;
        Index tensor_idx   = { 0, 1, 2 };
        Shape shape        = { 2, 2, 3 };

        auto result    = to_tensor_index(storage_idx, tensor_idx, shape);
        Index expected = { 1, 0, 1 };
        REQUIRE(result == expected);
    }

    SECTION("Empty shape") {
        size_t storage_idx = 0;
        Index tensor_idx   = {};
        Shape shape        = {};

        auto result    = to_tensor_index(storage_idx, tensor_idx, shape);
        Index expected = {};
        REQUIRE(result == expected);
    }

    SECTION("4D Tensor") {
        size_t storage_idx = 20;
        Index tensor_idx   = { 0, 1, 2, 3 };
        Shape shape        = { 4, 3, 2, 5 };

        auto result    = to_tensor_index(storage_idx, tensor_idx, shape);
        Index expected = { 0, 2, 1, 0 };

        REQUIRE(result == expected);
    }
}

TEST_CASE("Test shape_broadcast function", "[shape_broadcast]") {
    SECTION("Broadcast smaller shape to larger shape") {
        Shape a        = { 1 };
        Shape b        = { 5, 5 };
        Shape expected = { 5, 5 };
        REQUIRE(shape_broadcast(a, b) == expected);
    }

    SECTION("Broadcast larger shape to smaller shape") {
        Shape a        = { 5, 5 };
        Shape b        = { 1 };
        Shape expected = { 5, 5 };
        REQUIRE(shape_broadcast(a, b) == expected);
    }

    SECTION("Broadcast with leading 1s in shape") {
        Shape a        = { 1, 5, 5 };
        Shape b        = { 5, 5 };
        Shape expected = { 1, 5, 5 };
        REQUIRE(shape_broadcast(a, b) == expected);
    }

    SECTION("Broadcast with interleaved 1s in shapes") {
        Shape a        = { 5, 1, 5, 1 };
        Shape b        = { 1, 5, 1, 5 };
        Shape expected = { 5, 5, 5, 5 };
        REQUIRE(shape_broadcast(a, b) == expected);
    }

    SECTION("Broadcast incompatible shapes (should throw IndexingError)") {
        Shape a = { 5, 7, 5, 1 };
        Shape b = { 1, 5, 1, 5 };
        REQUIRE_THROWS_AS(shape_broadcast(a, b), IndexingError);
    }

    SECTION("Broadcast with mismatched dimensions (should throw IndexingError)") {
        Shape a = { 5, 2 };
        Shape b = { 5 };
        REQUIRE_THROWS_AS(shape_broadcast(a, b), IndexingError);
    }

    SECTION("Broadcast with second shape smaller and compatible") {
        Shape a        = { 2, 5 };
        Shape b        = { 5 };
        Shape expected = { 2, 5 };
        REQUIRE(shape_broadcast(a, b) == expected);
    }
}

TEST_CASE("Test TensorData initialization") {
    SECTION("Test layout") {
        Storage data(0, 3 * 5);
        TensorData tensor_data(data, { 3, 5 }, { 5, 1 });

        REQUIRE(tensor_data.index({ 1, 0 }) == 5);
        REQUIRE(tensor_data.index({ 1, 2 }) == 7);
    }
}

TEST_CASE("Test index broadcasting") {
    SECTION("Test broadcast inside") {
        Index to_index   = { 2, 1, 3 };
        Shape to_shape   = { 3, 3, 5 };
        Shape from_shape = { 3, 1, 5 };
        Index expected   = { 2, 0, 3 };
        REQUIRE(broadcast_index(to_index, to_shape, from_shape) == expected);
    }

    SECTION("Test broadcast outside") {
        Index to_index   = { 2, 1, 3 };
        Shape to_shape   = { 3, 3, 5 };
        Shape from_shape = { 3, 5 };
        Index expected   = { 1, 3 };
        REQUIRE(broadcast_index(to_index, to_shape, from_shape) == expected);
    }
}