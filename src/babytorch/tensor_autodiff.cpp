#include <any>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensor.hpp"
#include "ptr.hpp"


namespace tensor_autodiff {

    using namespace tensor;

    std::vector<sptr<Tensor>> topological_sort(sptr<Tensor> root) {
        //
        std::unordered_set<size_t> visited;
        std::vector<sptr<Tensor>> order;
        std::stack<sptr<Tensor>> stack;

        stack.push(root);

        while (!stack.empty()) {
            sptr<Tensor> cur_tensor = stack.top();
            stack.pop();

            if (visited.contains(cur_tensor->id) || cur_tensor->is_leaf())
                continue;

            visited.insert(cur_tensor->id);
            order.emplace_back(cur_tensor);

            for (sptr<Tensor> parent : cur_tensor->parents())
                if (!visited.contains(parent->id))
                    stack.push(parent);
        }

        return order;
    }

    void backpropagate(sptr<Tensor> variable, sptr<Tensor> deriv) {
        auto order = topological_sort(variable);

        std::unordered_map<size_t, sptr<Tensor>> grad_table;
        grad_table[variable->id] = deriv;

        // for (auto v : order) {
        // sptr<Tensor> d_out = grad_table[v->id];

        // for (auto [var, grad] : std::move(v->chain_rule(d_out)))
        //     if (var->is_leaf())
        //         var->accumulate_grad(grad);
        //     else if (grad_table.contains(var->id))
        //         *grad_table[var->id] += grad;
        // // else
        //     grad_table[var->id] = grad;
        // }

        return;
    }

}