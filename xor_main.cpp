#include "activations.h"
#include "algebra.h"
#include "neuron.h"
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>
using v_ptr = std::shared_ptr<nn::vector>;

int main() {
    auto make_v = [](size_t n, const std::vector<value_t> &values) {
        auto ret = std::make_shared<nn::vector>(n);
        for (size_t i = 0; i < n; ++i) { ret->at(i) = values.at(i); }
        return ret;
    };
    auto io_pair = [make_v](const std::vector<value_t> &in, const std::vector<value_t> &out) {
        return std::make_pair(make_v(in.size(), in), make_v(out.size(), out));
    };
    std::vector dataset = {
        io_pair({0, 0}, {0}),
        io_pair({0, 1}, {1}),
        io_pair({1, 0}, {1}),
        io_pair({1, 1}, {0}),
    };

    constexpr size_t num_steps = 100ull;

    std::vector<nn::layer> network = {
        nn::layer(2, 2, nn::lrelu(0.05l)),
        nn::layer(2, 1, nn::sigmoid()),


    };


    // for (size_t step = 0; step < num_steps; ++step) {
    constexpr nn::metric_t epsilon = 1e-5;

    nn::metric_t sum = 1e18;
    while (sum > dataset.size() * epsilon) {
        sum = 0.0l;
        for (auto &[X, Y]: dataset) {
            constexpr value_t lr = 1l;
            const auto loss = nn::train(network, X, Y, lr);
            sum += loss;
        }
        std::cout << "Average Loss: " << (sum / dataset.size()) << std::endl;
    }
    value_t x, y;
    while (true) {
        std::cout << "Please enter the values for testing (x,y): " << std::endl;
        std::cin >> x >> y;
        auto val = nn::predict(network, make_v(2, {x, y}))->at(0);
        std::cout << "The result of the network is: " << std::fixed
                << std::setprecision(14) << val << std::endl;
        std::cout << "This means for threshold 0.5 that answer is: " << (val > 0.5) << std::endl;
    }
}
