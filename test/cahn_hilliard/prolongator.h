#ifndef __PROLONGATOR_H__
#define __PROLONGATOR_H__

#include <memory>

#include "kernels/prolongator.h"

namespace tests
{

template <
    class VectorSpace,
    class Log,
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

    using for_each_nd_type = typename Backend::template for_each_nd_type<dim>;

public: // Especially for SYCL
    using prolongator_kernel = kernels::prolongator_kernel<idx_nd_type, ordinal_type, vector_type, tensor_dim>;

public:
    prolongator( idx_nd_type range ) : range_( range ) // in im space
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

    std::shared_ptr<vector_space_type> get_dom_space() const
    {
        return std::make_shared<vector_space_type>( range_ / Ord{ 2u } );
    }
    std::shared_ptr<vector_space_type> get_im_space() const
    {
        return std::make_shared<vector_space_type>( range_ );
    }

    // domain -> (prolongate) -> image
    void apply( vector_type &from, vector_type &to ) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst( prolongator_kernel{ from, to }, range_ );
    };

private:
    idx_nd_type range_; // in im space
};

} // namespace tests

#endif
