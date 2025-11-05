//
// Created by tomer on 11/5/2025.
//

#ifndef NEURALNEURON_LAYER_H
#define NEURALNEURON_LAYER_H

#include "activations.h"
#include "algebra.h"
#include <functional>
#include <utility>
#include <vector>

using value_t = long double;


namespace nn {

class layer {



public:
    std::shared_ptr<nn::matrix> weights;
    std::shared_ptr<vector> biases;

    activation act;

    static std::shared_ptr<nn::vector> eval_f(const std::shared_ptr<nn::vector>& y_v, const fn_t& f) {
        auto ret = std::make_shared<vector>(y_v->n);
        for (size_t i = 0; i < y_v->get_n(); ++i) ret->at(i) = f(y_v->at(i));
        return ret;
    }
    layer(const size_t num_inputs, const size_t neurons, const activation& act)
        : weights(std::make_shared<nn::matrix>(neurons, num_inputs)), biases(std::make_shared<nn::vector>(neurons)),
            act(act) {


        for (auto& x: weights->a) x = act.init(num_inputs, neurons);
        for (auto& x: biases->a) x = act.init(num_inputs, neurons);

    }

    [[nodiscard]] std::shared_ptr<nn::vector> calc_z(const std::shared_ptr<nn::vector>& inputs) const {
        return (weights * inputs) + (biases);
    }
    [[nodiscard]] std::shared_ptr<nn::vector> eval(const std::shared_ptr<nn::vector>& inputs) const {
        return eval_f(calc_z(inputs), act.act);
    }
    [[nodiscard]] std::shared_ptr<nn::vector> der_eval(const std::shared_ptr<nn::vector>& inputs) const {
        return eval_f(inputs, act.der_act);
    }

};

inline std::shared_ptr<nn::vector> predict(const std::vector<layer>& net, const std::shared_ptr<nn::vector>& inputs) {
    auto prev_y_v = inputs;

    for (const auto & i : net) {
        prev_y_v = i.calc_z(prev_y_v);
        prev_y_v = layer::eval_f(prev_y_v, i.act.act);
    } return prev_y_v;
}

inline metric_t train(std::vector<layer>& net,
                      const std::shared_ptr<nn::vector>& inputs,
                      const std::shared_ptr<nn::vector>& outputs,
                      const value_t lr = 0.92)
{
    const std::size_t layers = net.size();
    std::vector<std::shared_ptr<nn::vector>> a(layers + 1), z(layers), da(layers+1);
    a[0] = inputs;

    for (size_t i = 0; i < layers; ++i) {
        z[i] = net[i].calc_z(a[i]);
        a[i + 1] = layer::eval_f(z[i], net[i].act.act);
    }

    const auto loss = mse_loss(a.back(), outputs);
    const auto grad_loss = grad_mse_loss(a.back(), outputs);

    da[layers] = grad_loss;


    for (std::size_t i = layers; i >0; --i) {
        auto da_ft_z = da[i] * layer::eval_f(z[i-1], net[i-1].act.der_act);
        da[i-1] = trans(net[i-1].weights) * da_ft_z;
        net[i-1].biases -= lr * da_ft_z;

        net[i-1].weights -= lr * (da_ft_z * trans(a[i-1]));

    }

    return loss;
}




} // neuron

#endif //NEURALNEURON_NEURON_H