#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ptr.hpp"
#include "scalar.hpp"

namespace autodiff {

    std::vector<sptr<Scalar>> topological_sort(sptr<Scalar> root) {
        //
        std::unordered_set<double> visited;
        std::vector<sptr<Scalar>> order;
        std::stack<sptr<Scalar>> stack;

        stack.push(root);

        while (!stack.empty()) {
            sptr<Scalar> current_scalar = stack.top();
            stack.pop();

            if (visited.contains(current_scalar->id) || current_scalar->is_leaf())
                continue;

            visited.insert(current_scalar->id);
            order.emplace_back(current_scalar);

            for (auto parent : current_scalar->parents())
                if (!visited.contains(parent->id))
                    stack.push(parent);
        }

        return order;
    }

    void backpropagate(sptr<Scalar> variable, double deriv) {
        auto order = topological_sort(variable);

        std::unordered_map<double, double> grads;
        grads[variable->id] = deriv;

        for (auto v : order) {
            double d_out = grads[v->id];

            for (auto [var, grad] : v->chain_rule(d_out))
                if (var->is_leaf())
                    var->accumulate_grad(grad);
                else if (grads.contains(var->id))
                    grads[var->id] += grad;
                else
                    grads[var->id] = grad;
        }

        return;
    }

}
