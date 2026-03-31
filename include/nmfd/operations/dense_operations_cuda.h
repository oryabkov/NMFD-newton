#ifndef __NMFD_DENSE_OPERATIONS_CUDA_H__
#define __NMFD_DENSE_OPERATIONS_CUDA_H__

#include <scfd/external_libraries/cublas_wrap.h>
#include <scfd/backend/cuda.h>

#include <nmfd/operations/dense_operations_base.h>

namespace nmfd
{
namespace operations
{

template <class Type, class Ordinal = std::ptrdiff_t>
class dense_operations_cuda : public dense_operations<Type, scfd::backend::cuda, Ordinal>
{
public:
    using parent_t = dense_operations<Type, scfd::backend::cuda, Ordinal>;

    using matrix_type = typename parent_t::matrix_type;
    using vector_type = typename parent_t::vector_type;
    using scalar_type = typename parent_t::scalar_type;
    using cublas_t    = scfd::cublas_wrap;

public:
    dense_operations_cuda() = default;

    template <typename... Args>
    dense_operations_cuda( Args &&...args ) : parent_t( std::forward<Args>( args )... )
    {
    }

    void add_matrix_vector_prod(
        const scalar_type alpha, const matrix_type &mat, const vector_type &x, const scalar_type beta, vector_type &y
    ) const
    {
        auto      &cublas = scfd::cublas_wrap::inst();
        const auto sz     = mat.size_nd();
        const auto rows   = sz[0];
        const auto cols   = sz[1];
        cublas.gemv(
            'N', rows, mat.raw_ptr(), cols, rows, alpha, parent_t::vt_.get_raw_ptr( x ), beta,
            parent_t::vt_.get_raw_ptr( y )
        );
    }

    [[nodiscard]] std::shared_ptr<matrix_type>
    matrix_matrix_prod( const matrix_type &mat_a, const matrix_type &mat_b ) const
    {
        auto &cublas = scfd::cublas_wrap::inst();

        const auto sz_a = mat_a.size_nd();
        const auto sz_b = mat_b.size_nd();

        const auto rows_a        = sz_a[0];
        const auto cols_a_rows_b = sz_a[1];
        const auto cols_b        = sz_b[1];

        auto result = std::make_shared<matrix_type>();
        result->init( rows_a, cols_b );
        parent_t::assign_zero_matrix( *result );

        cublas.gemm(
            'N', 'N', rows_a, cols_b, cols_a_rows_b, scalar_type{ 1 }, mat_a.raw_ptr(), rows_a, mat_b.raw_ptr(),
            cols_a_rows_b, scalar_type{ 0 }, result->raw_ptr(), rows_a
        );

        return result;
    }
};

}

}

#endif
