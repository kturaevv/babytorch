#include <any>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensor.hpp"

namespace tensor_autodiff {

    using namespace tensor;

    std::vector<Tensor*> topological_sort(Tensor* root) {
        //
        std::unordered_set<size_t> visited;
        std::vector<Tensor*> order;
        std::stack<Tensor*> stack;

        stack.push(root);

        while (!stack.empty()) {
            Tensor* cur_tensor = stack.top();
            stack.pop();

            if (visited.contains(cur_tensor->id) || cur_tensor->is_leaf())
                continue;

            visited.insert(cur_tensor->id);
            order.emplace_back(cur_tensor);

            for (Tensor* parent : cur_tensor->parents())
                if (!visited.contains(parent->id))
                    stack.push(parent);
        }

        return order;
    }

    void backpropagate(Tensor* variable, Tensor* deriv) {
        auto order = topological_sort(variable);

        std::unordered_map<size_t, Tensor*> grad_table;
        grad_table[variable->id] = deriv;

        for (auto v : order) {
            Tensor* d_out = grad_table[v->id];

            for (auto [var, grad] : v->chain_rule(d_out))
                if (var->is_leaf())
                    var->accumulate_grad(grad);
                else if (grad_table.contains(var->id))
                    *grad_table[var->id] += grad;
                else
                    grad_table[var->id] = grad;
        }

        return;
    }

}