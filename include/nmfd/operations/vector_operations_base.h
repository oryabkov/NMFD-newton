#ifndef __NMFD_VECTOR_OPERATIONS_BASE_H__
#define __NMFD_VECTOR_OPERATIONS_BASE_H__

#include <map>
#include <utility>
#include <scfd/utils/logged_obj_base.h>


namespace nmfd
{
namespace operations
{

// namespace detail
// {
// template<class Vec>
// class multivector
// {

//     std::size_t addressof_l(const Vec& vec)
//     {
//         return reinterpret_cast<std::size_t>(std::addressof(vec));
//     }
//     std::size_t sz_;
//     std::map<std::size_t, std::size_t> mvec_table_;

// public:
//     multivector(std::size_t sz):
//     sz_(sz)
//     {}
    
//     void init_vector(Vec& x, std::size_t m = 1)
//     {
//         x = Vec(1000*m);
//         if(m>1)
//         {
//             mvec_table_.insert({addressof_l(x), m});
//         }
//     }
    
//     std::pair<std::size_t, std::size_t> size(Vec& x)
//     {
//         std::size_t m_l = 1;
//         m_l = mvec_table_[addressof_l(x)];
//         m_l = m_l>0?m_l:1;
//         return {sz_, m_l};
//     }

// };
// }

template<class Type, class VectorType, class MultiVectorType/*, class Log, class Cardinal = std::size_t*/, class Ordinal = std::ptrdiff_t>
class vector_operations_base /*: public scfd::utils::logged_obj_base<Log>*/
{

public:
    using vector_type = VectorType;
    using multivector_type = MultiVectorType;
    using scalar_type = Type;
    //using Card = Cardinal;
    using Ord = Ordinal;    
    using ordinal_type = Ord;
    using big_ordinal_type = Ord;  

private:
    using T = scalar_type;

protected:
    //Ord sz_;
    bool use_high_precision_;

public:
    vector_operations_base(/*Ord sz,*/  bool use_high_precision = false):
    /*sz_(sz),*/ use_high_precision_(use_high_precision)
    {}

    virtual ~vector_operations_base() = default;

    void set_high_precision()
    {
        use_high_precision_ = true;
    }
    void set_regular_precision()
    {
        use_high_precision_ = false;
    }

    /*[[nodiscard]] Ord size() const
    {
        return sz_;
    }*/
    
    [[nodiscard]] virtual vector_type at(multivector_type& x, Ord m, Ord k_) const = 0;

    [[nodiscard]] virtual bool is_valid_number(const vector_type &x) const = 0;
    //reduction operations:
    [[nodiscard]] virtual scalar_type scalar_prod(const vector_type &x, const vector_type &y)const = 0;
    [[nodiscard]] virtual scalar_type sum(const vector_type &x)const = 0;
    
    [[nodiscard]] virtual scalar_type asum(const vector_type &x)const = 0;

    //standard vector norm:=sqrt(sum(x^2))
    [[nodiscard]] virtual scalar_type norm(const vector_type &x) const = 0;
    //L2 emulation for the vector norm2:=sqrt(sum(x^2)/sz_)
    [[nodiscard]] virtual scalar_type norm2(const vector_type &x) const = 0;
    //standard vector norm_sq:=sum(x^2)
    [[nodiscard]] virtual scalar_type norm_sq(const vector_type &x) const = 0;
    //L2 emulation for the vector norm2_sq:=sum(x^2)/sz_
    [[nodiscard]] virtual scalar_type norm2_sq(const vector_type &x) const = 0;

    //calc: x := <vector_type with all elements equal to given scalar value> 
    virtual void assign_scalar(const scalar_type scalar, vector_type& x) const = 0;
    //calc: x := mul_x*x + <vector_type of all scalar value> 
    //ISSUE rename into add_scalar?
    virtual void add_mul_scalar(const scalar_type scalar, const scalar_type mul_x, vector_type& x) const = 0;
    virtual void scale(const scalar_type scale, vector_type &x) const = 0;
    //copy: y := x
    virtual void assign(const vector_type& x, vector_type& y) const = 0;
    //calc: y := mul_x*x
    //ISSUE rename into assign_lin_comb?
    virtual void assign_lin_comb(const scalar_type mul_x, const vector_type& x, vector_type& y) const = 0;
    //calc: y := mul_x*x + mul_y*y
    //ISSUE rename into add_lin_comb or add_mul_lin_comb?
    virtual void add_lin_comb(const scalar_type mul_x, const vector_type& x, const scalar_type mul_y, vector_type& y) const = 0;


};

}
}

#endif
