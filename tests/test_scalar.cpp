#include <cmath>
#include <limits>
#include <sstream>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../src/babytorch/autodiff.cpp"
#include "../src/babytorch/scalar.cpp"

using namespace scalar;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

#define EPS 1e-6

TEST_CASE("Scalar Constructors", "[Scalar]") {
    SECTION("Default Constructor") {
        auto a = Scalar::create();
        REQUIRE_THAT(a->data, WithinAbs(0.0, EPS));
        REQUIRE_THAT(a->grad, WithinAbs(0.0, EPS));
    }

    SECTION("Constructor with Data") {
        auto b = Scalar::create(5.0);
        REQUIRE_THAT(b->data, WithinAbs(5.0, EPS));
        REQUIRE_THAT(b->grad, WithinAbs(0.0, EPS));
    }

    SECTION("Constructor with Data and Grad") {
        auto c = Scalar::create(3.0);
        REQUIRE_THAT(c->data, WithinAbs(3.0, EPS));
    }

    SECTION("Copy Constructor") {
        auto d = Scalar::create(4.0);
        auto e = d;
        REQUIRE_THAT(e->data, WithinAbs(4.0, EPS));
    }
}

TEST_CASE("Scalar Addition", "[Scalar]") {
    auto a = Scalar::create(2.0);
    auto b = Scalar::create(3.0);

    double d = 4.0;
    int i    = 5;
    float f  = 6.0f;

    SECTION("Scalar + Scalar") {
        auto r      = Scalar::create();
        auto result = a + b;
        REQUIRE_THAT(result->data, WithinAbs(5.0, EPS));
    }

    SECTION("Scalar + double") {
        auto r      = Scalar::create();
        auto result = a + d;
        REQUIRE_THAT(result->data, WithinAbs(6.0, EPS));
    }

    SECTION("double + Scalar") {
        auto r      = Scalar::create();
        auto result = d + a;
        REQUIRE_THAT(result->data, WithinAbs(6.0, EPS));
    }

    SECTION("Scalar + int") {
        auto r      = Scalar::create();
        auto result = a + i;
        REQUIRE_THAT(result->data, WithinAbs(7.0, EPS));
    }

    SECTION("int + Scalar") {
        auto r      = Scalar::create();
        auto result = i + a;
        REQUIRE_THAT(result->data, WithinAbs(7.0, EPS));
    }

    SECTION("Scalar + float") {
        auto r      = Scalar::create();
        auto result = a + f;
        REQUIRE_THAT(result->data, WithinAbs(8.0, EPS));
    }

    SECTION("float + Scalar") {
        auto r      = Scalar::create();
        auto result = f + a;
        REQUIRE_THAT(result->data, WithinAbs(8.0, EPS));
    }
}

TEST_CASE("Scalar Multiplication", "[Scalar]") {
    auto a   = Scalar::create(2.0);
    auto b   = Scalar::create(3.0);
    double d = 4.0;
    int i    = 5;
    float f  = 6.0f;

    SECTION("Scalar * Scalar") {
        auto r      = Scalar::create();
        auto result = a * b;
        REQUIRE_THAT(result->data, WithinAbs(6.0, EPS));
    }

    SECTION("Scalar * double") {
        auto r      = Scalar::create();
        auto result = a * d;
        REQUIRE_THAT(result->data, WithinAbs(8.0, EPS));
    }

    SECTION("double * Scalar") {
        auto r      = Scalar::create();
        auto result = d * a;
        REQUIRE_THAT(result->data, WithinAbs(8.0, EPS));
    }

    SECTION("Scalar * int") {
        auto r      = Scalar::create();
        auto result = a * i;
        REQUIRE_THAT(result->data, WithinAbs(10.0, EPS));
    }

    SECTION("int * Scalar") {
        auto r      = Scalar::create();
        auto result = i * a;
        REQUIRE_THAT(result->data, WithinAbs(10.0, EPS));
    }

    SECTION("Scalar * float") {
        auto r      = Scalar::create();
        auto result = a * f;
        REQUIRE_THAT(result->data, WithinAbs(12.0, EPS));
    }

    SECTION("float * Scalar") {
        auto r      = Scalar::create();
        auto result = f * a;
        REQUIRE_THAT(result->data, WithinAbs(12.0, EPS));
    }
}

TEST_CASE("Scalar Subtraction", "[Scalar]") {
    auto a   = Scalar::create(10.0);
    auto b   = Scalar::create(4.0);
    double d = 3.0;
    int i    = 2;
    float f  = 1.0f;

    SECTION("Scalar - Scalar") {
        auto r      = Scalar::create();
        auto result = a - b;
        REQUIRE_THAT(result->data, WithinAbs(6.0, EPS));
    }

    SECTION("Scalar - double") {
        auto r      = Scalar::create();
        auto result = a - d;
        REQUIRE_THAT(result->data, WithinAbs(7.0, EPS));
    }

    SECTION("double - Scalar") {
        auto r      = Scalar::create();
        auto result = d - a;
        REQUIRE_THAT(result->data, WithinAbs(-7.0, EPS));
    }

    SECTION("Scalar - int") {
        auto r      = Scalar::create();
        auto result = a - i;
        REQUIRE_THAT(result->data, WithinAbs(8.0, EPS));
    }

    SECTION("int - Scalar") {
        auto r      = Scalar::create();
        auto result = i - a;
        REQUIRE_THAT(result->data, WithinAbs(-8.0, EPS));
    }

    SECTION("Scalar - float") {
        auto r      = Scalar::create();
        auto result = a - f;
        REQUIRE_THAT(result->data, WithinAbs(9.0, EPS));
    }

    SECTION("float - Scalar") {
        auto r      = Scalar::create();
        auto result = f - a;
        REQUIRE_THAT(result->data, WithinAbs(-9.0, EPS));
    }
}

TEST_CASE("Scalar Division", "[Scalar]") {
    auto a   = Scalar::create(20.0);
    auto b   = Scalar::create(4.0);
    double d = 5.0;
    int i    = 2;
    float f  = 10.0f;

    SECTION("Scalar / Scalar") {
        auto r      = Scalar::create();
        auto result = a / b;
        REQUIRE_THAT(result->data, WithinAbs(5.0, EPS));
    }

    SECTION("Scalar / double") {
        auto r      = Scalar::create();
        auto result = a / d;
        REQUIRE_THAT(result->data, WithinAbs(4.0, EPS));
    }

    SECTION("double / Scalar") {
        auto r      = Scalar::create();
        auto result = d / a;
        REQUIRE_THAT(result->data, WithinAbs(0.25, EPS));
    }

    SECTION("Scalar / int") {
        auto r      = Scalar::create();
        auto result = a / i;
        REQUIRE_THAT(result->data, WithinAbs(10.0, EPS));
    }

    SECTION("int / Scalar") {
        auto r      = Scalar::create();
        auto result = i / a;
        REQUIRE_THAT(result->data, WithinAbs(0.1, EPS));
    }

    SECTION("Scalar / float") {
        auto r      = Scalar::create();
        auto result = a / f;
        REQUIRE_THAT(result->data, WithinAbs(2.0, EPS));
    }

    SECTION("float / Scalar") {
        auto r      = Scalar::create();
        auto result = f / a;
        REQUIRE_THAT(result->data, WithinAbs(0.5, EPS));
    }
}

TEST_CASE("Scalar Less Than Comparison", "[Scalar]") {
    auto a   = Scalar::create(2.0);
    auto b   = Scalar::create(3.0);
    double d = 4.0;
    int i    = 5;
    float f  = 6.0f;

    SECTION("Scalar < Scalar") {
        auto r      = Scalar::create();
        auto result = a < b;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));  // Assuming true is
                                                        // represented as 1
    }

    SECTION("Scalar < double") {
        auto r      = Scalar::create();
        auto result = a < d;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("double < Scalar") {
        auto r      = Scalar::create();
        auto result = d < a;
        REQUIRE_THAT(result->data, WithinAbs(0, EPS));
    }

    SECTION("Scalar < int") {
        auto r      = Scalar::create();
        auto result = a < i;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("int < Scalar") {
        auto r      = Scalar::create();
        auto result = i < a;
        REQUIRE_THAT(result->data, WithinAbs(0, EPS));
    }

    SECTION("Scalar < float") {
        auto r      = Scalar::create();
        auto result = a < f;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("float < Scalar") {
        auto r      = Scalar::create();
        auto result = f < a;
        REQUIRE_THAT(result->data, WithinAbs(0, EPS));
    }
}

TEST_CASE("Scalar Greater Than Comparison", "[Scalar]") {
    auto a   = Scalar::create(4.0);
    auto b   = Scalar::create(3.0);
    double d = 2.0;
    int i    = 1;
    float f  = 0.0f;

    SECTION("Scalar > Scalar") {
        auto r      = Scalar::create();
        auto result = a > b;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("Scalar > double") {
        auto r      = Scalar::create();
        auto result = a > d;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("double > Scalar") {
        auto r      = Scalar::create();
        auto result = d > a;
        REQUIRE_THAT(result->data, WithinAbs(0, EPS));
    }

    SECTION("Scalar > int") {
        auto r      = Scalar::create();
        auto result = a > i;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("int > Scalar") {
        auto r      = Scalar::create();
        auto result = i > a;
        REQUIRE_THAT(result->data, WithinAbs(0, EPS));
    }

    SECTION("Scalar > float") {
        auto r      = Scalar::create();
        auto result = a > f;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("float > Scalar") {
        auto r      = Scalar::create();
        auto result = f > a;
        REQUIRE_THAT(result->data, WithinAbs(0, EPS));
    }
}

TEST_CASE("Scalar Equality Comparison", "[Scalar]") {
    auto a   = Scalar::create(3.0);
    auto b   = Scalar::create(3.0);
    double d = 3.0;
    int i    = 3;
    float f  = 3.0f;

    SECTION("Scalar == Scalar") {
        auto r      = Scalar::create();
        auto result = a == b;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("Scalar == double") {
        auto r      = Scalar::create();
        auto result = a == d;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("double == Scalar") {
        auto r      = Scalar::create();
        auto result = d == a;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("Scalar == int") {
        auto r      = Scalar::create();
        auto result = a == i;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("int == Scalar") {
        auto r      = Scalar::create();
        auto result = i == a;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("Scalar == float") {
        auto r      = Scalar::create();
        auto result = a == f;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }

    SECTION("float == Scalar") {
        auto r      = Scalar::create();
        auto result = f == a;
        REQUIRE_THAT(result->data, WithinAbs(1, EPS));
    }
}

TEST_CASE("Scalar Output Stream", "[Scalar]") {
    std::stringstream ss;
    auto a = Scalar::create(3.5);

    SECTION("Output Stream Format") {
        ss << *a;
        REQUIRE(ss.str() == "Scalar(data=3.5, grad=0)\n");
    }
}
