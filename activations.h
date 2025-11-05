//
// Created by tomer on 11/4/2025.
//

#ifndef NEURALNEURON_ACTIVATIONS_H
#define NEURALNEURON_ACTIVATIONS_H
#include <random>
#include <functional>
using value_t = long double;

namespace nn
{
    struct random_generator {
        std::mt19937 gen;
        std::uniform_real_distribution<> dis;

        explicit random_generator(const uint32_t seed = 1337) : gen(seed), dis(0, 1) {}
        value_t roll(const value_t min, const value_t max) { return min + (max-min) * dis(gen); }
    };

    inline value_t random_double(const value_t min, const value_t max) {
        static random_generator gen;
        return gen.roll(min, max);
    }

    using fn_t = std::function<value_t(const value_t x)>;
    using fn2_t = std::function<value_t(const value_t x, const value_t y)>;

    class activation {
    public:
        fn_t act;
        fn_t der_act;
        fn2_t init;
    };

    inline value_t xavier(const value_t x, const value_t y) {
        const auto r = sqrtl(6.0l / (x + y));
        return random_double(-r , r);
    }
    inline value_t he(const value_t x, const value_t y) {
        const auto r = sqrtl(6.0l / x);
        return random_double(-r , r);
    }

    inline activation sigmoid() {
        activation a;
        auto sigf = [](const value_t x)->value_t { return 1.0 / (1.0 + expl(-x)); };
        a.act = sigf;
        a.der_act = [sigf](const value_t x)->value_t {
            const auto sig = sigf(x);
            return sig * (1.0 - sig);
        };
        a.init = xavier;
        return a;
    }

    inline activation tanh() {
        activation a;
        a.act = [](const value_t x)->value_t { return tanhl(x); };
        a.der_act = [](const value_t x)->value_t {
            const value_t va= tanhl(x);
            return 1 - va*va;
        };
        a.init = xavier;
        return a;
    }

    inline activation relu() {
        activation a;
        a.act = [](const value_t x)->value_t { return std::max(0.0l, x); };
        a.der_act = [](const value_t x)->value_t {return (x>0.0l) ? 1.0l : 0.0l; };
        a.init = he;
        return a;
    }

    inline activation lrelu(const value_t alpha) {
        activation a;
        a.act = [alpha](const value_t x)->value_t { return x >= 0 ? x : alpha * x; };
        a.der_act = [alpha](const value_t x)->value_t {return x >= 0.0 ? 1.0l : alpha; };
        a.init = he;
        return a;
    }


}


#endif //NEURALNEURON_ACTIVATIONS_H