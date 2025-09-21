#ifndef __NMFD_PAIR_VECTOR_SPACE_H__
#define __NMFD_PAIR_VECTOR_SPACE_H__

#include <cmath>
#include <array>
#include <iterator>
#include <algorithm>

/// TODO systemize size_t behaviour - mb make some Vector dependent type?
/// TODO multivector - add some generic wrap for multivector (std::vector) and use it here

namespace nmfd
{
namespace operations
{

template <class VectorSpace1,class VectorSpace2>
struct pair_vector_space
{
    using vector1_type = typename VectorSpace1::vector_type;
    using vector2_type = typename VectorSpace2::vector_type;
    using vector_type = std::pair<vector1_type,vector2_type>;

    using scalar1_type = typename VectorSpace1::scalar_type;
    using scalar2_type = typename VectorSpace2::scalar_type;
    using scalar_type = std::pair<scalar1_type, scalar2_type>;

    using at_type = std::pair<size_t, size_t>;

private:
    VectorSpace1 vs1;
    VectorSpace2 vs2;

public:
    size_t size()const
    {
        return vs1.size() + vs2.size();
    }
    size_t get_size(const vector_type& x)const
    {
        return vs1.get_size(x.first) + vs2.get_size(x.second);
    }


    void init_vector(vector_type& x)const
    {
    }
    template<class ...Args>
    void init_vectors(Args&&...args)const
    {
    }
    void free_vector(vector_type& x)const
    {
    }
    template<class ...Args>
    void free_vectors(Args&&...args) const
    {
    }
    void start_use_vector(vector_type& x)const
    {
    }
    template<class ...Args>
    void start_use_vectors(Args&&...args)const
    {
    }
    void stop_use_vector(vector_type& x)const
    {
    }
    template<class ...Args>
    void stop_use_vectors(Args&&...args)const
    {
    }
    bool check_is_valid_number(const vector_type &x)const
    {
        return vs1.check_is_valid_number(x.first) && vs2.check_is_valid_number(x.second);
    }


    scalar1_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        scalar1_type res = vs1.scalar_prod(x.first, y.first);
        res += static_cast<scalar1_type>(vs2.scalar_prod(x.second, y.second));

        return res;
    }

    scalar1_type norm(const vector_type &x)const
    {
        return std::sqrt(scalar_prod(x, x));
    }
    scalar1_type norm_sq(const vector_type &x)const
    {
        return scalar_prod(x, x);
    }
    scalar1_type norm_inf(const vector_type& x)const
    {
        scalar1_type max_val1 = vs1.norm_inf(x.first);
        scalar1_type max_val2 = static_cast<scalar1_type>(vs2.norm_inf(x.second));
        scalar1_type max_val = (max_val1<max_val2)?max_val2:max_val1;

        return max_val;
    }
    scalar1_type norm2_sq(const vector_type& x)const
    {
        return norm_sq(x);
    }
    scalar1_type sum(const vector_type &x)
    {
        return 0;
    }
    scalar1_type asum(const vector_type &x)
    {
        return 0;
    }

    scalar1_type normalize(vector_type& x)const
    {
        scalar_type norm_x = norm(x);
        if (norm_x > 0.0)
        {
            scale(static_cast<scalar_type>(1.0) / norm_x, x);
        }

        return norm_x;
    }

    // TODO: How to assign scalars?
    void set_value_at_point(scalar1_type val_x, at_type at, vector_type& x) const
    {
        if (at.first == 0)
        {
            vs1.set_value_at_point(static_cast<scalar1_type>(val_x), at.second, x.first);
        }
        else if (at.first == 1)
        {
            vs2.set_value_at_point(static_cast<scalar2_type>(val_x), at.second, x.second);
        }
        else
        {
            throw std::logic_error("pair_vector_space::set_value_at_point: at index is out of range");
        }
    }

    // void set_value_at_point(scalar2_type val_x, at_type at, vector_type& x) const
    // {
    //     if (at.first == 0)
    //     {
    //         vs1.set_value_at_point(static_cast<scalar1_type>(val_x), at.second, x.first);
    //     }
    //     else if (at.first == 1)
    //     {
    //         vs2.set_value_at_point(static_cast<scalar2_type>(val_x), at.second, x.second);
    //     }
    //     else
    //     {
    //         throw std::logic_error("pair_vector_space::set_value_at_point: at index is out of range");
    //     }
    // }



    scalar1_type get_value_at_point(at_type at, const vector_type& x) const
    {
        if (at.first == 0)
        {
            return vs1.get_value_at_point(at.second, x.first);
        }
        else if (at.first == 1)
        {
            return static_cast<scalar1_type>(vs2.get_value_at_point(at.second, x.second));
        }
        else
        {
            throw std::logic_error("pair_vector_space::get_value_at_point: at index is out of range");
        }
    }

    //calc: x := <vector_type with all elements equal to given scalar value>
    void assign_scalar(const scalar_type scalar, vector_type& x)const
    {
        vs1.assign_scalar(scalar.first, x.first);
        vs2.assign_scalar(scalar.second, x.second);
    }

    //calc: x := mul_x*x + <vector_type of all scalar value>
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x)const
    {
        vs1.add_mul_scalar(scalar.first, mul_x.first, x.first);
        vs2.add_mul_scalar(scalar.second, mul_x.second, x.second);
    }

    //calc: x := scale*x
    void scale(scalar_type scale, vector_type &x)const
    {
        add_mul_scalar(static_cast<scalar_type>(0.0), scale, x);
    }

    //copy: y := x
    void assign(const vector_type& x, vector_type& y)const
    {
        vs1.assign(x.first, y.first);
        vs2.assign(x.second, y.second);
    }
    //calc: y := mul_x*x
    void assign_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        vs1.assign_mul(mul_x.first, x.first, y.first);
        vs2.assign_mul(mul_x.second, x.second, y.second);
    }
    //calc: z := mul_x*x + mul_y*y
    void assign_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, vector_type& z)const
    {
        vs1.assign_mul(mul_x.first, x.first, mul_y.first, y.first, z.first);
        vs2.assign_mul(mul_x.second, x.second, mul_y.second, y.second, z.second);
    }

    //calc: y := mul_x*x + y
    void add_mul(scalar_type mul_x, const vector_type& x, vector_type& y)const
    {
        vs1.add_mul(mul_x.first, x.first, y.first);
        vs2.add_mul(mul_x.second, x.second, y.second);
    }
    //calc: y := mul_x*x + mul_y*y
    void add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, vector_type& y)const
    {
        vs1.add_mul(mul_x.first, x.first, mul_y.first, y.first);
        vs2.add_mul(mul_x.second, x.second, mul_y.second, y.second);
    }
    //calc: z := mul_x*x + mul_y*y + mul_z*z
    void add_mul(scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y,
                            scalar_type mul_z, vector_type& z)const
    {
        vs1.add_mul(mul_x.first, x.first, mul_y.first, y.first, mul_z.first, z.first);
        vs2.add_mul(mul_x.second, x.second, mul_y.second, y.second, mul_z.second, z.second);
    }

    void make_abs_copy(const vector_type& x, vector_type& y)const
    {
        vs1.make_abs_copy(x.first, y.first);
        vs2.make_abs_copy(x.second, y.second);
    }
    void make_abs(vector_type& x)const
    {
        vs1.make_abs(x.first);
        vs2.make_abs(x.second);
    }

    // y_j = max(x_j,y_j,sc)
    void max_pointwise(const scalar_type sc, const vector_type& x, vector_type& y)const
    {
        vs1.max_pointwise(sc.first, x.first, y.first);
        vs2.max_pointwise(sc.second, x.second, y.second);
    }
    void max_pointwise(const scalar_type sc, vector_type& y)const
    {
        vs1.max_pointwise(sc.first, y.first);
        vs2.max_pointwise(sc.second, y.second);
    }
    // y_j = min(x_j,y_j,sc)
    void min_pointwise(const scalar_type sc, const vector_type& x, vector_type& y)const
    {
        vs1.min_pointwise(sc.first, x.first, y.first);
        vs2.min_pointwise(sc.second, x.second, y.second);
    }
    void min_pointwise(const scalar_type sc, vector_type& y)const
    {
        vs1.min_pointwise(sc.first, y.first);
        vs2.min_pointwise(sc.second, y.second);
    }

    // //calc: x := x*mul_y*y
    void mul_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y)const
    {
        vs1.mul_pointwise(x.first, mul_y.first, y.first);
        vs2.mul_pointwise(x.second, mul_y.second, y.second);
    }
    //calc: z := mul_x*x*mul_y*y
    void mul_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y,
                        vector_type& z)const
    {
        vs1.mul_pointwise(mul_x.first, x.first, mul_y.first, y.first, z.first);
        vs2.mul_pointwise(mul_x.second, x.second, mul_y.second, y.second, z.second);
    }
    //calc: z := (mul_x*x)/(mul_y*y)
    void div_pointwise(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, const vector_type& y,
                        vector_type& z)const
    {
        vs1.div_pointwise(mul_x.first, x.first, mul_y.first, y.first, z.first);
        vs2.div_pointwise(mul_x.second, x.second, mul_y.second, y.second, z.second);
    }
    //calc: x := x/(mul_y*y)
    void div_pointwise(vector_type& x, const scalar_type mul_y, const vector_type& y)const
    {
        vs1.div_pointwise(x.first, mul_y.first, y.first);
        vs2.div_pointwise(x.second, mul_y.second, y.second);
    }

    // //TODO:!
    // /*std::pair<scalar_type, size_t> max_argmax_element(vector_type& y) const
    // {
    //     auto max_iterator = std::max_element(y.begin(), y.end());
    //     size_t argmax = std::distance(y.begin(), max_iterator);

    //     return {*max_iterator, argmax};
    // }

    // scalar_type max_element(vector_type& x)const
    // {
    //     auto ret = max_argmax_element(x);
    //     return ret.first;
    // }

    // size_t argmax_element(vector_type& x)const
    // {
    //     auto ret = max_argmax_element(x);
    //     return ret.second;
    // }*/

    //TODO
    // void assign_slices(const vector_type& x, const std::vector< std::pair<size_t,size_t> > slices, vector_type&y)const
    // {
    //     size_t index_y = 0;
    //     for(auto& slice: slices)
    //     {
    //         size_t begin = slice.first;
    //         size_t end = slice.second;
    //         if(end>Dim)
    //         {
    //             throw std::logic_error("static_vector_space::assign_slice: provided slice size is greater than input vector size.");
    //         }
    //         for(size_t j = begin; j<end;j++)
    //         {
    //             y[index_y++] = x[j];
    //         }
    //     }
    // }

    // void assign_skip_slices(const vector_type& x, const std::vector< std::pair<size_t,size_t> > skip_slices, vector_type&y)const
    // {
    //     size_t index_y = 0;
    //     for(size_t j = 0; j<Dim;j++)
    //     {
    //         for(auto& slice: skip_slices)
    //         {
    //             size_t begin = slice.first;
    //             size_t end = slice.second;
    //             if((j<=begin)||(j>end))
    //             {
    //                 y[index_y++] = x[j];
    //             }
    //         }
    //     }
    // }

};

} // namespace operations
} // namespace nmfd

#endif
