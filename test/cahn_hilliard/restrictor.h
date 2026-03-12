#ifndef __RESTRICTOR_H__
#define __RESTRICTOR_H__

#include <memory>

#include "kernels/restrictor.h"
#include "include/boundary.h" // for boundary conditions
#include <nmfd/detail/vector_wrap.h>

namespace tests
{

template <
    class VectorSpace, class Log,
    /**********************************************/
    class Backend = typename VectorSpace::backend_type>
class restrictor
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
    using restrictor_kernel =
        kernels::restrictor_kernel<idx_nd_type, ordinal_type, vector_type, tensor_dim, boundary_cond_type>;

public:
    restrictor( idx_nd_type range, boundary_cond_type b_cond )
        : range_( range ), b_cond_( b_cond ), // in dom space
          vspace_( std::make_shared<vector_space_type>( range ) ),
          lin_vector_wrap_( *vspace_ )
    {
        for ( int i = 0; i < idx_nd_type::dim; ++i )
        {
            if ( range[i] % 2u != 0 )
                throw std::logic_error(
                    "nmfd::restrictor: encountered odd value in vector_space range_! not supported case"
                );
        }
        vspace_->assign_scalar( 0.0, *lin_vector_wrap_ );
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

    void set_linearization_point( const vector_type &p )
    {
        vspace_->assign( p, *lin_vector_wrap_ );
    }

    vector_type get_lin_vector() const
    {
        return *lin_vector_wrap_;
    }

    // domain -> (restrict) -> image
    void apply( vector_type &from, vector_type &to ) const
    {
        auto             half_r = range_ / Ord{ 2u };
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst( restrictor_kernel{ from, to, *lin_vector_wrap_, b_cond_, from.rect_nd() }, half_r );
    };

private:
    idx_nd_type        range_; // in dom space
    boundary_cond_type b_cond_;

    vector_space_ptr vspace_;
    using vector_wrap_t = nmfd::detail::vector_wrap<VectorSpace, true, true>;
    vector_wrap_t    lin_vector_wrap_;


};

} // namespace tests

#endif
