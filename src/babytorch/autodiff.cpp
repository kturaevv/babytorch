#include <any>
#include <functional>
#include <iostream>
#include <memory>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "scalar.hpp"

namespace autodiff {

    std::vector<std::shared_ptr<Scalar>> topological_sort(
        std::shared_ptr<Scalar> root) {
        //
        std::unordered_set<double> visited;
        std::vector<std::shared_ptr<Scalar>> order;
        std::stack<std::shared_ptr<Scalar>> stack;

        stack.push(root);

        while (!stack.empty()) {
            std::shared_ptr<Scalar> current_scalar = stack.top();
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

    void backpropagate(std::shared_ptr<Scalar> variable, double deriv) {
        auto order = topological_sort(variable);

        std::unordered_map<double, double> grads;
        grads[variable->id] = deriv;

        std::cout << "Topological sort completed!\n";
        for (auto v : order) {
            double d_out = grads[v->id];
            std::cout << v->id;

            for (auto [var, grad] : v->chain_rule(d_out)) {
                std::cout << var << "Gradient: " << grad << std::endl;
                if (var->is_leaf())
                    var->accumulate_grad(grad);
                else if (grads.contains(var->id))
                    grads[var->id] += grad;
                else
                    grads[var->id] = grad;
            }
        }

        return;
    }

}