#include <any>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tensor {
    struct Tensor;
}

namespace tensor_autodiff {

    using namespace tensor;

    std::vector<Tensor> topological_sort(Tensor& v);

    void backpropagate(Tensor& variable);
    void backpropagate(Tensor& variable, Tensor& deriv);

    struct Context {
        std::vector<Tensor> saved_values;

        template <typename... Args>
        void save_for_backwards(Args&... args) {
            (saved_values.push_back(args), ...);
            return;
        }
    };

    // std::vector<Tensor> topological_sort(Tensor root) {
    //     //
    //     std::unordered_set<Tensor&> visited;
    //     std::vector<Tensor> order;
    //     std::stack<Tensor> stack;

    //     stack.push(root);

    //     while (!stack.empty()) {
    //         Tensor cur_tensor = stack.top();
    //         stack.pop();

    //         if (visited.contains(cur_tensor->id) || cur_tensor->is_leaf())
    //             continue;

    //         visited.insert(cur_tensor->id);
    //         order.emplace_back(cur_tensor);

    //         for (auto parent : cur_tensor->parents())
    //             if (!visited.contains(parent->id))
    //                 stack.push(parent);
    //     }

    //     return order;
    // }

    // void backpropagate(Tensor& variable, Tensor& deriv) {
    //     auto order = topological_sort(variable);

    //     std::unordered_map<size_t, Tensor> grads;
    //     grads[variable->id] = deriv;

    //     for (auto v : order) {
    //         Tensor d_out = grads[v->id];

    //         for (auto [var, grad] : v->chain_rule(d_out))
    //             if (var->is_leaf())
    //                 var->accumulate_grad(grad);
    //             else if (grads.contains(var->id))
    //                 grads[var->id] += grad;
    //             else
    //                 grads[var->id] = grad;
    //     }

    //     return;
    // }

}