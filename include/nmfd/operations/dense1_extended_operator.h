#ifndef __NMFD_DENSE1_EXTENDED_OPERATOR_H__
#define __NMFD_DENSE1_EXTENDED_OPERATOR_H__

#include <cmath>
#include <array>
#include <iterator>
#include <algorithm>
#include <vector>
#include <memory>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/operations/pair_vector_space.h>

/// TODO systemize size_t behaviour - mb make some Vector dependent type?
/// TODO multivector - add some generic wrap for multivector (std::vector) and use it here

namespace nmfd
{
namespace operations
{

template<class OrigOperator, class OrigVectorSpace>
class dense1_extended_operator
/*
Class represents operator of form
    (A   u)
    (v^t w)
where A is some origin operator extended with vector u row v^t and number w.
*/
{
    ///TODO add static assert for scalar_type and vector_type coincidence
    static_assert(std::is_same_v<typename OrigOperator::scalar_type, typename OrigVectorSpace::scalar_type>, "scalar_type in OrigOperator and OrigVectorSpace must be the same");
    static_assert(std::is_same_v<typename OrigOperator::vector_type, typename OrigVectorSpace::vector_type>, "vector_type in OrigOperator and OrigVectorSpace must be the same");

    using scalar_type = typename OrigOperator::scalar_type;

    using orig_vector_space_type = OrigVectorSpace; // vector space for u and v^t
    using orig_vector_type = typename OrigOperator::vector_type; // vector type of u and v^t

    using scalar_space_type = operations::static_vector_space<scalar_type,1>; // vector space for w
    using scalar_vector_type = typename scalar_space_type::vector_type; // vector type of w

    using vector_space_type = operations::pair_vector_space<OrigVectorSpace,scalar_space_type>;
    using vector_type = typename vector_space_type::vector_type;

private:
    std::shared_ptr<const orig_vector_space_type> orig_vec_space;
    std::shared_ptr<const scalar_space_type> scalar_space;

    std::shared_ptr<const OrigOperator> A;
    orig_vector_type u;
    orig_vector_type v;
    scalar_vector_type w;

public:
    // TODO: What this constructor does?
    // dense1_extended_operator(std::shared_ptr<OrigVectorSpace> orig_vec_space, std::shared_ptr<const OrigOperator> orig_op = nullptr);

    dense1_extended_operator(std::shared_ptr<OrigVectorSpace> orig_vec_space, std::shared_ptr<const OrigOperator> orig_op, const orig_vector_type &_u, const orig_vector_type &_v, const scalar_vector_type &_w):
        orig_vec_space(orig_vec_space),
        scalar_space(std::make_shared<scalar_space_type>()),
        A(orig_op),
        u(_u),
        v(_v),
        w(_w)
    {}

    /// Use to set or reset origin A operator
    void set_orig_operator(std::shared_ptr<const OrigOperator> orig_op)
    {
        A = orig_op;
    }

    // TODO: Add const versions of u() v() w()?
    ///These can be used to reset dense extension values
    // vector_type &u();
    // vector_type &v();
    // scalar_type &w();
    ///Issue add const versions of u() v() w()?

    void apply(const vector_type &in, vector_type &out)
    {
        const scalar_type scalar_in_second = scalar_space->get_value_at_point(0, in.second);
        const scalar_type scalar_w = scalar_space->get_value_at_point(0, w);

        // out.first() = A.apply(in.first()) + in.second()*u
        A->apply(in.first, out.first);
        orig_vec_space->add_lin_comb(scalar_in_second, u, out.first);

        // out.second() = v^t * in.first() + w * in.second()
        scalar_type scalar_v_in_first = orig_vec_space->scalar_prod(v, in.first);
        scalar_type scalar_w_in_second = scalar_space->scalar_prod(w, in.second);
        scalar_space->set_value_at_point(scalar_v_in_first + scalar_w_in_second, 0, out.second);
    }

    std::shared_ptr<vector_space_type> get_im_space() const
    {
        return orig_vec_space;
    }

    std::shared_ptr<vector_space_type> get_dom_space() const
    {
        return orig_vec_space;
    }

};

} // namespace operations
} // namespace nmfd

#endif
