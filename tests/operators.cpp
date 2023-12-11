// #define CATCH_CONFIG_MAIN
#include <cmath>
#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../src/babytorch/operators.cpp"

using namespace operators;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

TEST_CASE("Multiplication Function Tests") {
    SECTION("Basic Multiplication") {
        REQUIRE(mul(2.0, 3.0) == 6.0);
        REQUIRE(mul(-2.0, 3.0) == -6.0);
        REQUIRE(mul(0.0, 3.0) == 0.0);
    }

    SECTION("Multiplication with Large Values") {
        REQUIRE(mul(1.0e100, 1.0e50) == Approx(1.0e150));
    }

    SECTION("Multiplication with Small Values") {
        REQUIRE(mul(1.0e-100, 1.0e-50) == Approx(1.0e-150));
    }

    SECTION("Multiplication with Zero") {
        REQUIRE(mul(0.0, -3.0) == 0.0);
    }

    SECTION("Multiplication with Different Numeric Types") {
        REQUIRE(mul(static_cast<float>(2.0), static_cast<float>(3.0)) == 6.0f);
        REQUIRE(mul(static_cast<long>(2), static_cast<long>(3)) == 6);
    }
}

TEST_CASE("Identity Function Tests") {
    SECTION("Identity of Positive Numbers") {
        REQUIRE(id(5.0) == Approx(5.0));
    }

    SECTION("Identity of Negative Numbers") {
        REQUIRE(id(-2.0) == Approx(-2.0));
    }

    SECTION("Identity of Zero") {
        REQUIRE(id(0.0) == Approx(0.0));
    }
}

TEST_CASE("Addition Function Tests") {
    SECTION("Basic Addition") {
        REQUIRE(add(2.0, 3.0) == 5.0);
        REQUIRE(add(-2.0, 3.0) == 1.0);
        REQUIRE(add(0.0, 3.0) == 3.0);
    }

    SECTION("Addition with Large Values") {
        REQUIRE(add(1.0e100, 1.0e50) == Approx(1.0e100 + 1.0e50));
    }

    SECTION("Addition with Small Values") {
        REQUIRE(add(1.0e-100, 1.0e-50) == Approx(1.0e-100 + 1.0e-50));
    }

    SECTION("Addition with Zero") {
        REQUIRE(add(0.0, -3.0) == -3.0);
    }

    SECTION("Addition with Different Numeric Types") {
        REQUIRE(add(static_cast<float>(2.0), static_cast<float>(3.0)) == 5.0f);
        REQUIRE(add(static_cast<long>(2), static_cast<long>(3)) == 5);
    }
}

TEST_CASE("Negation Function Tests") {
    SECTION("Basic Negation") {
        REQUIRE(neg(5.0) == -5.0);
        REQUIRE(neg(-2.0) == 2.0);
        REQUIRE(neg(0.0) == -0.0);  // Handle floating point precision
    }

    SECTION("Negation with Large Values") {
        REQUIRE(neg(1.0e100) == -1.0e100);
    }

    SECTION("Negation with Small Values") {
        REQUIRE(neg(1.0e-100) == -1.0e-100);
    }

    SECTION("Negation with Zero") {
        REQUIRE(neg(0.0) == -0.0);
    }
}

TEST_CASE("Less Than Function Tests") {
    SECTION("Basic Less Than") {
        REQUIRE(lt(2.0, 3.0) == 1.0);
        REQUIRE(lt(3.0, 2.0) == 0.0);
        REQUIRE(lt(2.0, 2.0) == 0.0);
    }

    SECTION("Less Than with Large Values") {
        REQUIRE(lt(1.0e100, 1.0e101) == 1.0);
    }

    SECTION("Less Than with Small Values") {
        REQUIRE(lt(1.0e-100, 1.0e-99) == 1.0);
    }

    SECTION("Less Than with Zero") {
        REQUIRE(lt(0.0, -3.0) == 0.0);
    }
}

TEST_CASE("Equality Function Tests") {
    SECTION("Basic Equality") {
        REQUIRE(eq(2.0, 2.0) == 1.0);
        REQUIRE(eq(2.0, 3.0) == 0.0);
    }

    SECTION("Equality with Large Values") {
        REQUIRE(eq(1.0e100, 1.0e100) == 1.0);
    }

    SECTION("Equality with Small Values") {
        REQUIRE(eq(1.0e-100, 1.0e-100) == 1.0);
    }

    SECTION("Equality with Zero") {
        REQUIRE(eq(0.0, 0.0) == 1.0);
    }
}

TEST_CASE("Maximum Function Tests") {
    SECTION("Basic Maximum") {
        REQUIRE(max(2.0, 3.0) == 3.0);
        REQUIRE(max(3.0, 2.0) == 3.0);
        REQUIRE(max(2.0, 2.0) == 2.0);
    }

    SECTION("Maximum with Large Values") {
        REQUIRE(max(1.0e100, 1.0e101) == 1.0e101);
    }

    SECTION("Maximum with Small Values") {
        REQUIRE(max(1.0e-100, 1.0e-99) == 1.0e-99);
    }

    SECTION("Maximum with Zero") {
        REQUIRE(max(0.0, -3.0) == 0.0);
    }
}

// Tests for is_close function
TEST_CASE("Is Close Function Tests") {
    SECTION("Basic Is Close") {
        REQUIRE(is_close(5.0, 5.01) == 1.0);
        REQUIRE(is_close(5.0, 6.0) == 0.0);
    }

    SECTION("Is Close with Large Values") {
        REQUIRE(is_close(1.0e100, 1.0e101) == 0.0);
    }

    SECTION("Is Close with Small Values") {
        REQUIRE(is_close(1.0e-100, 1.0e-99) == 1.0);
    }

    SECTION("Is Close with Zero") {
        REQUIRE(is_close(0.0, -3.0) == 0.0);
    }
}

// Tests for sigmoid function
TEST_CASE("Sigmoid Function Tests") {
    SECTION("Sigmoid with Positive Values") {
        REQUIRE_THAT(sigmoid(1.0), WithinAbs(0.73105857863, EPS));
        REQUIRE_THAT(sigmoid(10.0), WithinAbs(0.99995460213, EPS));
    }

    SECTION("Sigmoid with Negative Values") {
        REQUIRE_THAT(sigmoid(-1.0), WithinAbs(0.2689414213, EPS));
        REQUIRE_THAT(sigmoid(-10.0), WithinAbs(4.53978687e-05, EPS));
    }

    SECTION("Sigmoid with Zero") {
        REQUIRE_THAT(sigmoid(0.0), WithinAbs(0.5, EPS));
    }
}

// Tests for relu function
TEST_CASE("ReLU Function Tests") {
    SECTION("ReLU with Positive Values") {
        REQUIRE_THAT(relu(5.0), WithinAbs(5.0, EPS));
        REQUIRE_THAT(relu(0.0), WithinAbs(0.0, EPS));
    }

    SECTION("ReLU with Negative Values") {
        REQUIRE_THAT(relu(-5.0), WithinAbs(0.0, EPS));
        REQUIRE_THAT(relu(-0.001), WithinAbs(0.0, EPS));
    }
}

// Tests for log_func function
TEST_CASE("Log Function Tests") {
    SECTION("Log with Positive Values") {
        REQUIRE_THAT(log_func(5.0), WithinAbs(1.6094379124, EPS));
        REQUIRE_THAT(log_func(1.0), WithinAbs(0.0, EPS));
    }

    SECTION("Log with Zero") {
        REQUIRE_THAT(log_func(0.0), WithinAbs(-18.420680743952367, EPS));
    }

    SECTION("Log with Negative Values") {
        REQUIRE(log_func(-1.0) != log_func(-1.0));  // NaN check
    }
}

// Tests for exp_func function
TEST_CASE("Exponential Function Tests") {
    SECTION("Exponential with Positive Values") {
        REQUIRE_THAT(exp_func(5.0), WithinAbs(148.4131591026, EPS));
        REQUIRE_THAT(exp_func(0.0), WithinAbs(1.0, EPS));
    }

    SECTION("Exponential with Negative Values") {
        REQUIRE_THAT(exp_func(-5.0), WithinAbs(0.006737946999, EPS));
    }
}

// Tests for log_back function
TEST_CASE("Log Back Function Tests") {
    SECTION("Log Back with Positive Values") {
        REQUIRE_THAT(log_back(2.0, 3.0), WithinAbs(0.45511961331341866, EPS));
    }

    SECTION("Log Back with Negative Values") {
        REQUIRE_THAT(log_back(-2.0, 3.0), WithinAbs(-0.45511961331341866, EPS));
    }

    SECTION("Log Back with Zero") {
        REQUIRE_THAT(log_back(0.0, 3.0), WithinAbs(1e+8, EPS));
    }
}

// Tests for inv function
TEST_CASE("Inverse Function Tests") {
    SECTION("Inverse with Positive Values") {
        REQUIRE_THAT(inv(2.0), WithinAbs(0.5, EPS));
    }

    SECTION("Inverse with Negative Values") {
        REQUIRE_THAT(inv(-2.0), WithinAbs(-0.5, EPS));
    }

    SECTION("Inverse with Zero") {
        REQUIRE_THAT(inv(0.0), WithinAbs(1e+8, EPS));
    }
}

// Tests for inv_back function
TEST_CASE("Inverse Back Function Tests") {
    SECTION("Inverse Back with Positive Values") {
        REQUIRE_THAT(inv_back(2.0, 3.0), WithinAbs(-0.25, EPS));
    }

    SECTION("Inverse Back with Negative Values") {
        REQUIRE_THAT(inv_back(-2.0, 3.0), WithinAbs(-0.25, EPS));
    }

    SECTION("Inverse Back with Zero") {
        REQUIRE(inv_back(0.0, 3.0) == -100000000.0);
    }
}

// Tests for relu_back function
TEST_CASE("ReLU Back Function Tests") {
    SECTION("ReLU Back with Positive Values") {
        REQUIRE_THAT(relu_back(5.0, 3.0), WithinAbs(3.0, EPS));
        REQUIRE_THAT(relu_back(0.5, 3.0), WithinAbs(3.0, EPS));
    }

    SECTION("ReLU Back with Negative Values") {
        REQUIRE_THAT(relu_back(-5.0, 3.0), WithinAbs(0.0, EPS));
        REQUIRE_THAT(relu_back(0.0, 3.0), WithinAbs(0.0, EPS));
    }
}