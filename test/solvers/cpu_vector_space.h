#ifndef __NMFD_CPU_VECTOR_SPACE_H__
#define __NMFD_CPU_VECTOR_SPACE_H__

#include <cmath>
#include <exception>
#include <cstdlib>
#include <new>
#include <nmfd/operations/vector_space_base.h>
#include <nmfd/operations/default_multivector_space_base.h>

namespace nmfd
{
namespace detail
{

template<class Ord, class Type, class VectorType>
class reductions
{
    Ord sz_;
    VectorType helper_vector_;
    Ord pow2_;
public:
    reductions(Ord sz, VectorType& helper_vector):
    sz_(sz), helper_vector_(helper_vector)
    {
        pow2_ = static_cast<Ord>(std::ceil( std::log2(sz_) ));
    }
    ~reductions(){};

    Type naive_dot(const VectorType &x, const VectorType &y) const
    {
        for(Ord j=0;j<sz_;j++)
        {
            helper_vector_[j] = x[j]*y[j];
        }
        Type res = 0;
        for(Ord j=0;j<sz_;j++)
        {
            res+=helper_vector_[j];
        }        
        return res;
    }

    Type dot(const VectorType &x, const VectorType &y) const
    {
        for(Ord j=0;j<sz_;j++)
        {
            helper_vector_[j] = x[j]*y[j];
        }
        return sum_pow2_helper();

    }

    Type sum(const VectorType &x) const
    {
        for(Ord j=0;j<sz_;j++)
        {
            helper_vector_[j] = x[j];
        }        
        return sum_pow2_helper();
    }
    Type asum(const VectorType &x) const
    {
        for(Ord j=0;j<sz_;j++)
        {
            helper_vector_[j] = std::abs(x[j]);
        }        
        return sum_pow2_helper();
    }    

private:
    Type sum_pow2_helper() const
    {
        Ord sz_reduction = 2 << (pow2_-2); // 2^(pow2_-1)
        while (sz_reduction>1)
        {
            for(int j=0;j<sz_reduction;j++)
            {
                if(j+sz_reduction>=sz_)
                {
                    break; //this is valid in the first sum
                }
                helper_vector_[j]+=helper_vector_[j+sz_reduction];
            }
            sz_reduction /= 2;
        }
        return helper_vector_[0]+helper_vector_[1];        
    }

};

// template<class Ord>
// class size_mapper
// {
//     template<class Vec>
//     std::size_t addressof_l(const Vec& vec) const
//     {
//         return reinterpret_cast<Ord>(std::addressof(vec));
//     }
//     std::size_t sz_;
//     mutable std::map<Ord, Ord> mvec_table_;

// public:
//     size_mapper(Ord sz):sz_(sz){}

//     template<class Vec>
//     void add(const Vec& x, Ord m) const
//     {
//         mvec_table_.insert({addressof_l(x), m});
//     }
//     template<class Vec>
//     void remove(const Vec& x) const
//     {
//         mvec_table_.erase( addressof_l(x) );
//     }
    
//     template<class Vec>
//     Ord size(const Vec& x) const
//     {
//         std::size_t m_l = 1;
//         m_l = mvec_table_[addressof_l(x)];
//         m_l = m_l>0?m_l:1;
//         return m_l;
//     }

// };

}

template
<
    class Type, class VectorType, class Log, class Ordinal = std::ptrdiff_t
>
class cpu_vector_space : 
  //public nmfd::operations::vector_space_base<Type, VectorType, VectorType, Ordinal>
  public nmfd::operations::default_multivector_space_base
  <
    cpu_vector_space<Type, VectorType, Log, Ordinal>,
    Type, VectorType, Ordinal
  >
{
    /*using parent_t = 
        nmfd::operations::vector_space_base
        <
            Type, VectorType, VectorType, Ordinal
        >;*/
    using parent_t = 
        nmfd::operations::default_multivector_space_base
        <
            cpu_vector_space<Type, VectorType, Log, Ordinal>,
            Type, VectorType, Ordinal
        >;

public:
    using Ord = typename parent_t::Ord;
    using vector_type = typename parent_t::vector_type;
    //using MultiVectorType = vector_type;
    //using multivector_type = MultiVectorType;
    using MultiVectorType = typename parent_t::multivector_type;
    using multivector_type = typename parent_t::multivector_type;
    using scalar_type = typename parent_t::scalar_type;
    
private:
    using reductions_t = detail::reductions<Ord, scalar_type, vector_type>;

    Ord sz_;
    Ord size_of_mem_;
    reductions_t* reductions_;
    VectorType helper_vector_;
public:


    cpu_vector_space(Ord sz, bool use_high_precision = false) : 
        parent_t(use_high_precision),sz_(sz)
    {
        commmon_constructor_operations();
    }

private:
    void commmon_constructor_operations()
    {
        size_of_mem_ = sizeof(Type)*sz_;
        helper_vector_ = reinterpret_cast<VectorType>(std::malloc(size_of_mem_));
        if(helper_vector_ == nullptr)
        {
            throw std::bad_alloc();
        }
        reductions_ = new reductions_t(sz_, helper_vector_);   
    }

public:    
    ~cpu_vector_space()
    {
        free(helper_vector_);
        helper_vector_ = nullptr;
        delete reductions_;

    }

    [[nodiscard]] Ord size() const
    {
        return sz_;
    }

    void init_vector(vector_type& x) const
    {
        x = reinterpret_cast<VectorType>(std::malloc(size_of_mem_));
        if(x == nullptr)
        {
            throw std::bad_alloc();
        }
    }
    void free_vector(vector_type& x) const
    {
        free(x);
    }
    void start_use_vector(vector_type& x) const
    {}
    void stop_use_vector(vector_type& x) const
    {}

    /*void init_multivector(multivector_type& x, Ord m) const
    {
        x = reinterpret_cast<MultiVectorType>(std::malloc(size_of_mem_*m) );
        if(x == nullptr)
        {
            throw std::bad_alloc();
        }
    }
    void free_multivector(multivector_type& x, Ord m) const
    {
        free(x);
    }
    void start_use_multivector(multivector_type& x, Ord m) const
    {}
    void stop_use_multivector(multivector_type& x, Ord m) const
    {}*/
    
    /*[[nodiscard]] vector_type at(const multivector_type& x, Ord m, Ord k_) const
    {
        if (k_ < 0 || k_>=m  ) 
        {
            throw std::out_of_range("cpu_vector_space: multivector.at");
        }
        return &x[ sz_*k_ ];
    }
    /// multivector interface
    void assign(const multivector_type& mx, Ord m, Ord k_, vector_type& x) const
    {
        //throw std::logic_error("no mv interface");
        assign(at(mx,m,k_), x);
    }
    void assign(const vector_type& x, multivector_type& mx, Ord m, Ord k_) const
    {
        //throw std::logic_error("no mv interface");
        auto mx_k = at(mx,m,k_);
        assign(x, mx_k);
    }
    [[nodiscard]] scalar_type scalar_prod(const multivector_type& mx, Ord m, Ord k_, const vector_type &y)const
    {
        //throw std::logic_error("no mv interface");
        return scalar_prod(at(mx,m,k_), y);
    }
    [[nodiscard]] scalar_type scalar_prod_l2(const multivector_type& mx, Ord m, Ord k_, const vector_type &y)const
    {
        //throw std::logic_error("no mv interface");
        return scalar_prod_l2(at(mx,m,k_), y);
    }
    void add_lin_comb(const scalar_type mul_x, const multivector_type& mx, Ord m, Ord k_, const scalar_type mul_y, vector_type& y) const
    {
        //throw std::logic_error("no mv interface");
        add_lin_comb(mul_x, at(mx,m,k_), mul_y, y);
    }*/
    ///multivector interface from parent restore because of overshadowing    
    using parent_t::assign;
    using parent_t::scalar_prod;
    using parent_t::scalar_prod_l2;
    using parent_t::add_lin_comb;

    [[nodiscard]] bool is_valid_number(const vector_type &x) const
    {
        bool res = true;
        for(Ord j=0;j<sz_;j++ )
        {
            if( !std::isfinite(x[j]) )
            {
                res = false;
                break;
            }
        }
        return res;
    }

    [[nodiscard]] scalar_type scalar_prod(const vector_type &x, const vector_type &y)const
    {
        return reductions_->dot(x, y);
    }
    [[nodiscard]] scalar_type scalar_prod_l2(const vector_type &x, const vector_type &y)const
    {
        return reductions_->dot(x, y);
    }

    [[nodiscard]] scalar_type sum(const vector_type &x)const
    {
        return reductions_->sum(x);
    }
    
    [[nodiscard]] scalar_type asum(const vector_type &x)const
    {
        return reductions_->asum(x);
    }

    [[nodiscard]] scalar_type norm(const vector_type &x) const
    {
        return std::sqrt(scalar_prod(x,x));
    }
    [[nodiscard]] scalar_type norm2(const vector_type &x) const
    {
        return std::sqrt( scalar_prod(x,x)/static_cast<Type>(sz_) );
    }
    [[nodiscard]] scalar_type norm_sq(const vector_type &x) const
    {
        return scalar_prod(x,x);
    }

    [[nodiscard]] scalar_type norm2_sq(const vector_type &x) const
    {
        return scalar_prod(x,x)/static_cast<Type>(sz_);
    }
    void assign_scalar(const scalar_type scalar, vector_type& x) const
    {
        for(Ord j=0;j<sz_;j++ )
        {
            x[j] = scalar;
        }
    }
    void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x) const
    {
        for(Ord j=0;j<sz_;j++ )
        {
            x[j] = mul_x*x[j] + scalar;
        }           
    }
    void scale(const scalar_type scale, vector_type &x) const
    {
        for(Ord j=0;j<sz_;j++ )
        {
            x[j] *= scale;
        }           
    }
    void assign(const vector_type& x, vector_type& y) const
    {
        for(Ord j=0;j<sz_;j++ )
        {
            y[j] = x[j];
        }          
    }
    void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const
    {
        for(Ord j=0;j<sz_;j++ )
        {
            y[j] = mul_x*x[j];
        }        
    }
    void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const
    {
        for(Ord j=0;j<sz_;j++ )
        {
            y[j] = mul_x*x[j] + mul_y*y[j];
        }         
    }


};

}


#endif
