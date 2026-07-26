#ifndef __PROLONGATOR_H__
#define __PROLONGATOR_H__

#include <memory>

#include "kernels/prolongator.h"

namespace tests
{

template <
    class VectorSpace, class Log,
    /**********************************************/
    class Backend = typename VectorSpace::backend_type>
class prolongator
{
public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;
    using scalar_type           = typename VectorSpace::scalar_type;
    using vector_type           = typename VectorSpace::vector_type;
    using vector_space_type     = VectorSpace; // defines Vector Space working in
    using ordinal_type          = typename VectorSpace::ordinal_type;

    using Ord = ordinal_type;

    using vector_space_ptr = std::shared_ptr<VectorSpace>;
    using idx_nd_type      = typename VectorSpace::idx_nd_type;

    using boundary_cond_type = boundary_cond<dim, tensor_dim>;

    using for_each_nd_type = typename Backend::template for_each_nd_type<dim>;

public: // Especially for SYCL
    using prolongator_kernel =
        kernels::prolongator_kernel<idx_nd_type, ordinal_type, vector_type, tensor_dim, boundary_cond_type>;
    // using prolongator_kernel =
    //     kernels::prolongator_kernel<idx_nd_type, ordinal_type, vector_type, tensor_dim, scalar_type>;


public:
    prolongator( idx_nd_type range, boundary_cond_type b_cond ) : range_( range ), b_cond_( b_cond ) // in im space
    {
        for ( int i = 0; i < idx_nd_type::dim; ++i )
        {
            if ( range[i] % 2u != 0 )
                throw std::logic_error(
                    "nmfd::prolongator: encountered odd value in vector_space range_! not supported case"
                );
        }
    }

    idx_nd_type get_size() const noexcept
    {
        return range_;
    }

    vector_space_ptr get_dom_space() const
    {
        return vector_space_ptr( range_ / Ord{ 2u } );
    }

    vector_space_ptr get_im_space() const
    {
        return vector_space_ptr( range_ );
    }

    // domain -> (prolongate) -> image
    void apply( vector_type &from, vector_type &to ) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst( prolongator_kernel{ from, to, b_cond_ }, range_ );
    };

private:
    idx_nd_type        range_; // in im space
    boundary_cond_type b_cond_;
};

} // namespace tests

#endif
