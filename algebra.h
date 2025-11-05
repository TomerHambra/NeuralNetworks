//
// Created by tomer on 11/4/2025.
//

#ifndef NEURALNEURON_ALGEBRA_H
#define NEURALNEURON_ALGEBRA_H

#include <memory>
#include <cassert>
#include <vector>
#include <functional>
#include "activations.h"

using value_t = long double;

namespace nn {

class vector;
class matrix {
protected:
    size_t n, m;
    std::vector<value_t> a;
public:
    virtual ~matrix() = default;

    matrix(const size_t n, const size_t m) : n(n), m(m), a(n * m, 0) { }

    friend class layer;
    [[nodiscard]] size_t get_n() const { return n; }
    [[nodiscard]] size_t get_m() const { return m; }

    virtual value_t &at(const size_t i, const size_t j) {
        assert(i < n && j < m);
        return a[i * m + j];
    }
    virtual value_t operator()(const size_t i, const size_t j) const {
        assert(i < n && j < m);
        return a[i * m + j];
    }
    friend std::shared_ptr<matrix> operator*(const std::shared_ptr<matrix>& lhs, const std::shared_ptr<matrix>& rhs) {
        assert(lhs->m == rhs->n);
        auto ret = std::make_shared<matrix>(lhs->n, rhs->m);
        for (size_t i = 0; i < lhs->n; i++)
            for (size_t j = 0; j < rhs->m; j++)
                for (size_t k = 0; k < lhs->m; k++)
                    ret->at(i, j) += lhs->operator()(i, k) * rhs->operator()(k, j);


        return ret;
    }
    friend std::shared_ptr<vector> operator*(const std::shared_ptr<matrix>& lhs, const std::shared_ptr<vector>& rhs);

    friend std::shared_ptr<matrix> operator+(const std::shared_ptr<matrix>& lhs, const std::shared_ptr<matrix>& rhs) {
        assert(lhs->m == rhs->m && lhs->n == rhs->n);
        auto ret = std::make_shared<matrix>(lhs->n, rhs->m);
        for (size_t i = 0; i < lhs->n; i++)
            for (size_t j = 0; j < rhs->m; j++)
                ret->at(i, j) = lhs->operator()(i, j) + rhs->operator()(i, j);

        return ret;
    }
    friend std::shared_ptr<matrix> operator-(const std::shared_ptr<matrix>& lhs, const std::shared_ptr<matrix>& rhs) {
        assert(lhs->m == rhs->m && lhs->n == rhs->n);
        auto ret = std::make_shared<matrix>(lhs->n, rhs->m);
        for (size_t i = 0; i < lhs->n; i++)
            for (size_t j = 0; j < rhs->m; j++)
                ret->at(i, j) = lhs->operator()(i, j) - rhs->operator()(i, j);

        return ret;
    }
    friend std::shared_ptr<matrix> operator-=(std::shared_ptr<matrix> lhs, const std::shared_ptr<matrix>& rhs) {
        assert(lhs->m == rhs->m && lhs->n == rhs->n);
        for (size_t i = 0; i < lhs->n; i++)
            for (size_t j = 0; j < rhs->m; j++)
                lhs->at(i, j) -= rhs->operator()(i, j);

        return lhs;
    }
    friend std::shared_ptr<matrix> trans(const std::shared_ptr<matrix>& rhs) {
        auto ret = std::make_shared<matrix>(rhs->m, rhs->n);
        for (size_t i = 0; i < rhs->m; i++)
            for (size_t j = 0; j < rhs->n; j++)
                ret->at(i, j) = rhs->operator()(j, i);
        return ret;
    }
    friend std::shared_ptr<matrix> operator*(const value_t lhs, const std::shared_ptr<matrix>& rhs) {
        auto ret = std::make_shared<matrix>(rhs->n, rhs->m);
        for (size_t i = 0; i < rhs->n; i++)
            for (size_t j = 0; j < rhs->m; j++)
                ret->at(i, j) = lhs * rhs->operator()(i, j);
        return ret;
    }
};

class vector : public matrix {
    bool is_row;
public:
    explicit vector(const size_t n) : matrix(n, 1), is_row(false) {}

    virtual value_t &at(const size_t i) {
        if (is_row) return matrix::at(0, i);
        return matrix::at(i, 0);
    }
    virtual value_t operator()(const size_t i) const {
        if (is_row) return matrix::operator()(0, i);
        return matrix::operator()(i, 0);
    }
    // void transpose() { is_row = !is_row, std::swap(n, m); }   @wip


    friend std::shared_ptr<vector> operator*(const std::shared_ptr<matrix>& lhs, const std::shared_ptr<vector>& rhs) {
        assert(lhs->m == rhs->n);
        auto ret = std::make_shared<vector>(lhs->n);
        for (size_t i = 0; i < lhs->n; i++) {
            for (size_t j = 0; j < rhs->m; j++) {
                for (size_t k = 0; k < lhs->m; k++) {
                    ret->at(i) += lhs->operator()(i, k) * rhs->operator()(k);
                }
            }
        } return ret;
    }

    friend std::shared_ptr<vector> operator+(const std::shared_ptr<vector>& lhs, const std::shared_ptr<vector>& rhs) {
        assert(lhs->m == rhs->m && lhs->n == rhs->n);
        auto ret = std::make_shared<vector>(lhs->n);
        for (size_t i = 0; i < lhs->n; i++)
            ret->at(i) = lhs->operator()(i) + rhs->operator()(i);
        return ret;
    }
    friend std::shared_ptr<vector> operator-(const std::shared_ptr<vector>& lhs, const std::shared_ptr<vector>& rhs) {
        assert(lhs->m == rhs->m && lhs->n == rhs->n);
        auto ret = std::make_shared<vector>(lhs->n);
        for (size_t i = 0; i < lhs->n; i++)
            ret->at(i) = lhs->operator()(i) - rhs->operator()(i);
        return ret;
    }
    friend std::shared_ptr<vector> operator*(const std::shared_ptr<vector>& lhs, const std::shared_ptr<vector>& rhs) {
        assert(lhs->m == rhs->m && lhs->n == rhs->n);
        auto ret = std::make_shared<vector>(lhs->n);
        for (size_t i = 0; i < lhs->n; i++)
            ret->at(i) = lhs->operator()(i) * rhs->operator()(i);
        return ret;
    }
};
using metric_t = long double;

inline metric_t mse_loss(const std::shared_ptr<vector>& result, const std::shared_ptr<vector>& expected) {
    metric_t sum = 0.0l;
    assert(result->get_n() == expected->get_n());
    for (size_t i = 0; i < result->get_n(); i++) {
        const value_t diff = result->at(i) - expected->at(i);
        sum += diff * diff;
    } return sum / 2.0l;
}

inline std::shared_ptr<vector> grad_mse_loss(const std::shared_ptr<vector>& result, const std::shared_ptr<vector>& expected) {
    return result - expected;
}

}
#endif //NEURALNEURON_ALGEBRA_H