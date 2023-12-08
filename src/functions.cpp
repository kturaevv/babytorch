#include <cassert>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "operators.hpp"
#include "scalar.hpp"

namespace functions {

    struct Context;
    struct Scalar;

    template <typename F>
    struct ScalarFunction {
        static Scalar apply() {
            // some additional logic here
            F::forward();
            // return Scalar(); TODO
        }
    };

    // TODO: implement other functions
    struct FunctionAdd {
        static double forward(double self, double other) {
            return operators::add(self, other);
        }
    };

}  // namespace minitorch
