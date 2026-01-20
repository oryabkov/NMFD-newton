#ifndef __IDENTITY_OP_H__
#define __IDENTITY_OP_H__

#include <memory>

#include "kernels/identity.h"

namespace tests
{

template <
    class LinearOperator,
    class VectorSpace,
    class Log,
    /**********************************************/
    class Backend = typename VectorSpace::backend_type>
class identity_op
{
    using lin_op_t = LinearOperator;

public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;
    using scalar_type           = typename VectorSpace::scalar_type;
    using vector_type           = typename VectorSpace::vector_type;
    using vector_space_type     = VectorSpace; // defines Vector Space working in
    using ordinal_type          = typename VectorSpace::ordinal_type;

    using vector_space_ptr = std::shared_ptr<VectorSpace>;
    using idx_nd_type      = typename VectorSpace::idx_nd_type;

    using for_each_nd_type = typename Backend::template for_each_nd_type<dim>;

public: // Especially for SYCL
    using identity_kernel = kernels::identity_kernel<idx_nd_type, vector_type, tensor_dim>;

public:
    struct params
    {
    };
    using params_hierarchy = params;
    struct utils
    {
    };
    using utils_hierarchy = utils;

    identity_op( const utils_hierarchy &u, const params_hierarchy &p )
    {
    }

    identity_op( idx_nd_type range = {} ) : range_( range )
    {
    }

public:
    idx_nd_type get_size() const noexcept
    {
        return range_;
    }

    void set_operator( std::shared_ptr<const lin_op_t> op )
    {
        range_ = op->get_size();
    }

    std::shared_ptr<vector_space_type> get_dom_space() const
    {
        return std::make_shared<vector_space_type>( range_ );
    }
    std::shared_ptr<vector_space_type> get_im_space() const
    {
        return std::make_shared<vector_space_type>( range_ );
    }

    void apply( vector_type &from, vector_type &to ) const
    {
        for_each_nd_type for_each_nd_inst;
        for_each_nd_inst( identity_kernel{ from, to }, range_ );
    };

private:
    idx_nd_type range_;
};

} // namespace tests

#endif
