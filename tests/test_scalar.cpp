#include <cmath>
#include <limits>
#include <sstream>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../src/babytorch/scalar.hpp"

using namespace scalar;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

#define EPS 1e-8

TEST_CASE("Scalar Constructors", "[Scalar]") {
    SECTION("Default Constructor") {
        Scalar a;
        REQUIRE_THAT(a.data, WithinAbs(0.0, EPS));
        REQUIRE_THAT(a.grad, WithinAbs(0.0, EPS));
    }

    SECTION("Constructor with Data") {
        Scalar b(5.0);
        REQUIRE(b.data == 5.0);
        REQUIRE(b.grad == 0.);
    }

    SECTION("Constructor with Data and Grad") {
        Scalar c(3.0, 2.0);
        REQUIRE(c.data == 3.0);
        REQUIRE(c.grad == 2.0);
    }

    SECTION("Copy Constructor") {
        Scalar d(4.0, 1.0);
        Scalar e = d;
        REQUIRE(e.data == 4.0);
        REQUIRE(e.grad == 1.0);
    }
}

TEST_CASE("Scalar Addition", "[Scalar]") {
    Scalar a(2.0), b(3.0);
    double d = 4.0;
    int i = 5;
    float f = 6.0f;

    SECTION("Scalar + Scalar") {
        Scalar result = a + b;
        REQUIRE(result.data == Approx(5.0));
    }

    SECTION("Scalar + double") {
        Scalar result = a + d;
        REQUIRE(result.data == Approx(6.0));
    }

    SECTION("double + Scalar") {
        Scalar result = d + a;
        REQUIRE(result.data == Approx(6.0));
    }

    SECTION("Scalar + int") {
        Scalar result = a + i;
        REQUIRE(result.data == Approx(7.0));
    }

    SECTION("int + Scalar") {
        Scalar result = i + a;
        REQUIRE(result.data == Approx(7.0));
    }

    SECTION("Scalar + float") {
        Scalar result = a + f;
        REQUIRE(result.data == Approx(8.0));
    }

    SECTION("float + Scalar") {
        Scalar result = f + a;
        REQUIRE(result.data == Approx(8.0));
    }
}

TEST_CASE("Scalar Multiplication", "[Scalar]") {
    Scalar a(2.0), b(3.0);
    double d = 4.0;
    int i = 5;
    float f = 6.0f;

    SECTION("Scalar * Scalar") {
        Scalar result = a * b;
        REQUIRE(result.data == Approx(6.0));
    }

    SECTION("Scalar * double") {
        Scalar result = a * d;
        REQUIRE(result.data == Approx(8.0));
    }

    SECTION("double * Scalar") {
        Scalar result = d * a;
        REQUIRE(result.data == Approx(8.0));
    }

    SECTION("Scalar * int") {
        Scalar result = a * i;
        REQUIRE(result.data == Approx(10.0));
    }

    SECTION("int * Scalar") {
        Scalar result = i * a;
        REQUIRE(result.data == Approx(10.0));
    }

    SECTION("Scalar * float") {
        Scalar result = a * f;
        REQUIRE(result.data == Approx(12.0));
    }

    SECTION("float * Scalar") {
        Scalar result = f * a;
        REQUIRE(result.data == Approx(12.0));
    }
}

TEST_CASE("Scalar Subtraction", "[Scalar]") {
    Scalar a(10.0), b(4.0);
    double d = 3.0;
    int i = 2;
    float f = 1.0f;

    SECTION("Scalar - Scalar") {
        Scalar result = a - b;
        REQUIRE(result.data == Approx(6.0));
    }

    SECTION("Scalar - double") {
        Scalar result = a - d;
        REQUIRE(result.data == Approx(7.0));
    }

    SECTION("double - Scalar") {
        Scalar result = d - a;
        REQUIRE(result.data == Approx(-7.0));
    }

    SECTION("Scalar - int") {
        Scalar result = a - i;
        REQUIRE(result.data == Approx(8.0));
    }

    SECTION("int - Scalar") {
        Scalar result = i - a;
        REQUIRE(result.data == Approx(-8.0));
    }

    SECTION("Scalar - float") {
        Scalar result = a - f;
        REQUIRE(result.data == Approx(9.0));
    }

    SECTION("float - Scalar") {
        Scalar result = f - a;
        REQUIRE(result.data == Approx(-9.0));
    }
}

TEST_CASE("Scalar Division", "[Scalar]") {
    Scalar a(20.0), b(4.0);
    double d = 5.0;
    int i = 2;
    float f = 10.0f;

    SECTION("Scalar / Scalar") {
        Scalar result = a / b;
        REQUIRE(result.data == Approx(5.0));
    }

    SECTION("Scalar / double") {
        Scalar result = a / d;
        REQUIRE(result.data == Approx(4.0));
    }

    SECTION("double / Scalar") {
        Scalar result = d / a;
        REQUIRE(result.data == Approx(0.25));
    }

    SECTION("Scalar / int") {
        Scalar result = a / i;
        REQUIRE(result.data == Approx(10.0));
    }

    SECTION("int / Scalar") {
        Scalar result = i / a;
        REQUIRE(result.data == Approx(0.1));
    }

    SECTION("Scalar / float") {
        Scalar result = a / f;
        REQUIRE(result.data == Approx(2.0));
    }

    SECTION("float / Scalar") {
        Scalar result = f / a;
        REQUIRE(result.data == Approx(0.5));
    }
}

TEST_CASE("Scalar Less Than Comparison", "[Scalar]") {
    Scalar a(2.0), b(3.0);
    double d = 4.0;
    int i = 5;
    float f = 6.0f;

    SECTION("Scalar < Scalar") {
        Scalar result = a < b;
        REQUIRE(result.data == 1);  // Assuming true is represented as 1
    }

    SECTION("Scalar < double") {
        Scalar result = a < d;
        REQUIRE(result.data == 1);
    }

    SECTION("double < Scalar") {
        Scalar result = d < a;
        REQUIRE(result.data == 0);
    }

    SECTION("Scalar < int") {
        Scalar result = a < i;
        REQUIRE(result.data == 1);
    }

    SECTION("int < Scalar") {
        Scalar result = i < a;
        REQUIRE(result.data == 0);
    }

    SECTION("Scalar < float") {
        Scalar result = a < f;
        REQUIRE(result.data == 1);
    }

    SECTION("float < Scalar") {
        Scalar result = f < a;
        REQUIRE(result.data == 0);
    }
}

TEST_CASE("Scalar Greater Than Comparison", "[Scalar]") {
    Scalar a(4.0), b(3.0);
    double d = 2.0;
    int i = 1;
    float f = 0.0f;

    SECTION("Scalar > Scalar") {
        Scalar result = a > b;
        REQUIRE(result.data == 1);
    }

    SECTION("Scalar > double") {
        Scalar result = a > d;
        REQUIRE(result.data == 1);
    }

    SECTION("double > Scalar") {
        Scalar result = d > a;
        REQUIRE(result.data == 0);
    }

    SECTION("Scalar > int") {
        Scalar result = a > i;
        REQUIRE(result.data == 1);
    }

    SECTION("int > Scalar") {
        Scalar result = i > a;
        REQUIRE(result.data == 0);
    }

    SECTION("Scalar > float") {
        Scalar result = a > f;
        REQUIRE(result.data == 1);
    }

    SECTION("float > Scalar") {
        Scalar result = f > a;
        REQUIRE(result.data == 0);
    }
}

TEST_CASE("Scalar Equality Comparison", "[Scalar]") {
    Scalar a(3.0), b(3.0);
    double d = 3.0;
    int i = 3;
    float f = 3.0f;

    SECTION("Scalar == Scalar") {
        Scalar result = a == b;
        REQUIRE(result.data == 1);
    }

    SECTION("Scalar == double") {
        Scalar result = a == d;
        REQUIRE(result.data == 1);
    }

    SECTION("double == Scalar") {
        Scalar result = d == a;
        REQUIRE(result.data == 1);
    }

    SECTION("Scalar == int") {
        Scalar result = a == i;
        REQUIRE(result.data == 1);
    }

    SECTION("int == Scalar") {
        Scalar result = i == a;
        REQUIRE(result.data == 1);
    }

    SECTION("Scalar == float") {
        Scalar result = a == f;
        REQUIRE(result.data == 1);
    }

    SECTION("float == Scalar") {
        Scalar result = f == a;
        REQUIRE(result.data == 1);
    }
}

TEST_CASE("Scalar Output Stream", "[Scalar]") {
    std::stringstream ss;
    Scalar a(3.5, 1.5);

    SECTION("Output Stream Format") {
        ss << a;
        REQUIRE(ss.str() == "Scalar(data=3.5, grad=1.5)\n");
    }
}
