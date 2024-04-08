#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ptr.hpp"
#include "tensor.hpp"

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
                stack.push(parent);
        }

        return order;
    }

    void backpropagate(sptr<Tensor> variable, sptr<Tensor> deriv) {
        auto order = topological_sort(variable);

        std::unordered_map<size_t, sptr<Tensor>> grad_table;
        grad_table[variable->id] = deriv;

        for (auto curr_node : order) {
            sptr<Tensor> d_out = grad_table[curr_node->id];

            for (auto [input, grad] : curr_node->chain_rule(d_out))
                if (input->is_leaf())
                    input->accumulate_grad(std::move(grad));
                else if (!grad_table.contains(input->id))
                    grad_table[input->id] = std::move(grad);
                else if (grad_table.contains(input->id))
                    grad_table[input->id] = grad_table[input->id] + grad;
        }
        return;
    }

}
