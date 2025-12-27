#ifndef __NMFD_DENSE_VECTOR_SPACE_H__
#define __NMFD_DENSE_VECTOR_SPACE_H__

#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/memory/host.h>

#include <nmfd/operations/dense_operations_base.h>

namespace nmfd
{
namespace operations
{

template<class Backend, class Type, class VectorTraits> class dense_vector_space
{

    using vector_type = typename VectorTraits::vector_type;
    using scalar_type = typename VectorTraits::scalar_type;

public:
    size_t get_size(const vector_type& x) const
    {
        return vt_.get_size(x);
    }

    void init_vector(vector_type& vec) const {}
    template<class... Args> void init_vectors(Args&&... args) const
    {
        std::initializer_list<int>{((void)init_vector(std::forward<Args>(args)), 0)...};
    }
    void free_vector(vector_type& vec) const
    {
        vec.free();
    }
    template<class... Args> void free_vectors(Args&&... args) const
    {
        std::initializer_list<int>{((void)free_row_vector(std::forward<Args>(args)), 0)...};
    }
    template<class... Args> void start_use_vectors(Args&&... args) const {}
    void stop_use_vector(vector_type& x) const {}
    template<class... Args> void stop_use_vectors(Args&&... args) const {}
    bool check_is_valid_number(const vector_type& x) const
    {
        for (size_t i = 0; i < get_size(x); ++i)
        {
            if (std::isinf(x[i]))
            {
                return false;
            }
        }

        return true;
    }
    [[nodiscard]] scalar_type scalar_prod(const vector_type& x, const vector_type& y) const
    {
        scalar_type res(0.f);
        for (int i = 0; i < get_size(x); ++i)
        {
            res += x[i] * y[i];
        }
        return res;
    }
    [[nodiscard]] scalar_type scalar_prod_l2(const vector_type& x, const vector_type& y) const
    {
        return scalar_prod(x, y);
    }

    [[nodiscard]] scalar_type norm(const vector_type& x) const
    {
        return std::sqrt(scalar_prod(x, x));
    }
    [[nodiscard]] scalar_type norm_sq(const vector_type& x) const
    {
        return scalar_prod(x, x);
    }
    [[nodiscard]] scalar_type norm2_sq(const vector_type& x) const
    {
        return norm_sq(x);
    }
    [[nodiscard]] scalar_type norm2(const vector_type& x) const
    {
        return std::sqrt(scalar_prod(x, x));
    }
    [[nodiscard]] scalar_type norm_l2_sq(const vector_type& x) const
    {
        return norm_sq(x);
    }
    [[nodiscard]] scalar_type norm_l2(const vector_type& x) const
    {
        return std::sqrt(scalar_prod(x, x));
    }
    /// asum(x)
    [[nodiscard]] scalar_type norm1(const vector_type& x) const
    {
        return asum(x);
    }
    /// returns some weighted L1/l1 norm (problem dependent)
    [[nodiscard]] scalar_type norm_l1(const vector_type& x) const
    {
        return norm1(x);
    }
    [[nodiscard]] scalar_type norm_inf(const vector_type& x) const
    {
        scalar_type max_val = 0.0;
        for (size_t j = 0; j < get_size(x); j++)
        {
            max_val = (max_val < std::abs(x[j])) ? std::abs(x[j]) : max_val;
        }
        return max_val;
    }
    /// returns maximum of all elements absolute weighted values (problem dependent)
    [[nodiscard]] scalar_type norm_l_inf(const vector_type& x) const
    {
        return norm_inf(x);
    }


    [[nodiscard]] scalar_type sum(const vector_type& x) const
    {
        scalar_type res = 0.0;
        for (size_t j = 0; j < get_size(x); j++)
        {
            res += x[j];
        }
        return res;
    }
    [[nodiscard]] scalar_type asum(const vector_type& x) const
    {
        scalar_type res = 0.0;
        for (size_t j = 0; j < get_size(x); j++)
        {
            res += std::abs(x[j]);
        }
        return res;
    }

    scalar_type normalize(vector_type& x) const
    {
        auto norm_x = norm(x);
        if (norm_x > 0.0)
        {
            scale(static_cast<scalar_type>(1.0) / norm_x, x);
        }
        return norm_x;
    }

    void set_value_at_point(scalar_type val_x, size_t at, vector_type& x) const
    {
        x[at] = val_x;
    }
    scalar_type get_value_at_point(size_t at, const vector_type& x) const
    {
        return x[at];
    }
    // calc: x := <vector_type with all elements equal to given scalar value>
    void assign_scalar(const scalar_type scalar, vector_type& x) const
    {
        for (size_t i = 0; i < get_size(x); ++i)
            x[i] = scalar;
    }
    // calc: x := mul_x*x + <vector_type of all scalar value>
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x) const
    {
        for (size_t i = 0; i < get_size(x); ++i)
            x[i] = mul_x * x[i] + scalar;
    }
    void scale(scalar_type scale, vector_type& x) const
    {
        add_mul_scalar(static_cast<scalar_type>(0.0), scale, x);
    }
    // copy: y := x
    void assign(const vector_type& x, vector_type& y) const
    {
        for (int i = 0; i < get_size(x); ++i)
        {
            y[i] = x[i];
        }
    }
    // calc: y := mul_x*x
    void assign_lin_comb(scalar_type mul_x, const vector_type& x, vector_type& y) const
    {
        for (int i = 0; i < get_size(x); ++i)
        {
            y[i] = mul_x * x[i];
        }
    }

    // calc: z := mul_x*x + mul_y*y
    void assign_lin_comb(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y,
                         vector_type& z) const
    {
        for (int i = 0; i < get_size(x); ++i)
            z[i] = mul_x * x[i] + mul_y * y[i];
    }
    // calc: result := mul_x*x + mul_y*y (alias for assign_lin_comb with different argument order)
    void assign_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y,
                    vector_type& result) const
    {
        assign_lin_comb(mul_x, x, mul_y, y, result);
    }
    // calc: y := mul_x*x + y
    void add_lin_comb(scalar_type mul_x, const vector_type& x, vector_type& y) const
    {
        for (int i = 0; i < get_size(x); ++i)
            y[i] += mul_x * x[i];
    }
    // calc: y := mul_x*x + mul_y*y
    void add_lin_comb(scalar_type mul_x, const vector_type& x, scalar_type mul_y, vector_type& y) const
    {
        for (int i = 0; i < get_size(x); ++i)
            y[i] = mul_x * x[i] + mul_y * y[i];
    }
    // calc: z := mul_x*x + mul_y*y + mul_z*z
    void add_lin_comb(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y,
                      scalar_type mul_z, vector_type& z) const
    {
        for (int i = 0; i < get_size(x); ++i)
            z[i] = mul_x * x[i] + mul_y * y[i] + mul_z * z[i];
    }

    void make_abs_copy(const vector_type& x, vector_type& y) const
    {
        for (size_t j = 0; j < get_size(x); j++)
        {
            y[j] = std::abs(x[j]);
        }
    }
    void make_abs(vector_type& x) const
    {
        for (size_t j = 0; j < get_size(x); j++)
        {
            auto xa = std::abs(x[j]);
            x[j] = xa;
        }
    }
    // y_j = max(x_j,y_j,sc)
    void max_pointwise(const scalar_type sc, const vector_type& x, vector_type& y) const
    {
        for (size_t j = 0; j < get_size(x); j++)
        {
            y[j] = (x[j] > y[j]) ? ((x[j] > sc) ? x[j] : sc) : ((y[j] > sc) ? y[j] : sc);
        }
    }
    void max_pointwise(const scalar_type sc, vector_type& y) const
    {
        for (size_t j = 0; j < get_size(y); j++)
        {
            y[j] = (y[j] > sc) ? y[j] : sc;
        }
    }
    // y_j = min(x_j,y_j,sc)
    void min_pointwise(const scalar_type sc, const vector_type& x, vector_type& y) const
    {
        for (size_t j = 0; j < get_size(x); j++)
        {
            y[j] = (x[j] < y[j]) ? ((x[j] < sc) ? x[j] : sc) : ((y[j] < sc) ? y[j] : sc);
        }
    }
    void min_pointwise(const scalar_type sc, vector_type& y) const
    {
        for (size_t j = 0; j < get_size(y); j++)
        {
            y[j] = (y[j] < sc) ? y[j] : sc;
        }
    }
    // calc: x := x*mul_y*y
    void mul_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y) const
    {
        for (size_t j = 0; j < get_size(x); j++)
        {
            x[j] *= mul_y * y[j];
        }
    }
    // calc: z := mul_x*x*mul_y*y
    void mul_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y,
                       vector_type& z) const
    {
        for (size_t j = 0; j < get_size(x); j++)
        {
            z[j] = (mul_x * x[j]) * (mul_y * y[j]);
        }
    }
    // calc: z := (mul_x*x)/(mul_y*y)
    void div_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y,
                       vector_type& z) const
    {
        for (size_t j = 0; j < get_size(x); j++)
        {
            z[j] = (mul_x * x[j]) / (mul_y * y[j]);
        }
    }
    // calc: x := x/(mul_y*y)
    void div_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y) const
    {
        for (size_t j = 0; j < get_size(x); j++)
        {
            x[j] *= static_cast<scalar_type>(1.0) / (mul_y * y[j]);
        }
    }

    // TODO:!
    /*std::pair<scalar_type, size_t> max_argmax_element(vector_type& y) const
    {
        auto max_iterator = std::max_element(y.begin(), y.end());
        size_t argmax = std::distance(y.begin(), max_iterator);

        return {*max_iterator, argmax};
    }

    scalar_type max_element(vector_type& x)const
    {
        auto ret = max_argmax_element(x);
        return ret.first;
    }

    size_t argmax_element(vector_type& x)const
    {
        auto ret = max_argmax_element(x);
        return ret.second;
    }*/

    void assign_slices(const vector_type& x, const std::vector<std::pair<size_t, size_t>> slices, vector_type& y) const
    {
        size_t index_y = 0;
        for (const auto& slice : slices)
        {
            size_t begin = slice.first;
            size_t end = slice.second;
            if (end > get_size(x))
            {
                throw std::logic_error(
                    "static_vector_space::assign_slice: provided slice size is greater than input vector size.");
            }
            for (size_t j = begin; j < end; j++)
            {
                y[index_y++] = x[j];
            }
        }
    }

    void assign_skip_slices(const vector_type& x, const std::vector<std::pair<size_t, size_t>> skip_slices,
                            vector_type& y) const
    {
        size_t index_y = 0;
        for (size_t j = 0; j < get_size(x); j++)
        {
            for (const auto& slice : skip_slices)
            {
                size_t begin = slice.first;
                size_t end = slice.second;
                if ((j <= begin) || (j > end))
                {
                    y[index_y++] = x[j];
                }
            }
        }
    }

protected:
    VectorTraits vt_;
};

} // namespace operations

} // namespace nmfd

#endif
