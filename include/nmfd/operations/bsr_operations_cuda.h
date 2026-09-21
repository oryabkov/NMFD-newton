#ifndef __NMFD_BSR_OPERATIONS_CUDA_H__
#define __NMFD_BSR_OPERATIONS_CUDA_H__

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <scfd/arrays/array.h>
#include <scfd/memory/cuda.h>
#include <scfd/utils/cuda_safe_call.h>

#include <nmfd/operations/bsr_matrix.h>

namespace nmfd
{
namespace operations
{

namespace detail
{

inline void cusparse_check( cusparseStatus_t status, const char *expr, const char *file, int line )
{
    if ( status != CUSPARSE_STATUS_SUCCESS )
    {
        std::ostringstream oss;
        oss << file << ":" << line << ": cuSPARSE call failed: " << expr << " (status " << static_cast<int>( status )
            << ")";
        throw std::runtime_error( oss.str() );
    }
}

#define NMFD_CUSPARSE_SAFE_CALL( expr )                                                                                \
    ::nmfd::operations::detail::cusparse_check( ( expr ), #expr, __FILE__, __LINE__ )

template <class T>
struct cusparse_value_type;

template <>
struct cusparse_value_type<float>
{
    static constexpr cudaDataType value = CUDA_R_32F;
};

template <>
struct cusparse_value_type<double>
{
    static constexpr cudaDataType value = CUDA_R_64F;
};

/**
 * cuSPARSE index type matching the size of the ordinal type. Only 32- and
 * 64-bit ordinals are supported by cuSPARSE.
 */
template <class Ord>
struct cusparse_index_type
{
    static_assert( std::is_integral<Ord>::value, "ordinal_type must be an integral type" );
    static_assert(
        sizeof( Ord ) == sizeof( std::int32_t ) || sizeof( Ord ) == sizeof( std::int64_t ),
        "ordinal_type must be 32 or 64 bits wide to be used with cuSPARSE"
    );

    static constexpr cusparseIndexType_t value =
        sizeof( Ord ) == sizeof( std::int32_t ) ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
};

/// RAII owner for a cuSPARSE handle.
struct cusparse_handle_owner
{
    cusparseHandle_t handle = nullptr;

    cusparse_handle_owner()                                           = default;
    cusparse_handle_owner( const cusparse_handle_owner & )            = delete;
    cusparse_handle_owner &operator=( const cusparse_handle_owner & ) = delete;

    ~cusparse_handle_owner()
    {
        if ( handle != nullptr )
            cusparseDestroy( handle );
    }
};

/// RAII owner for a cuSPARSE sparse matrix descriptor.
struct cusparse_spmat_owner
{
    cusparseSpMatDescr_t descr = nullptr;

    cusparse_spmat_owner()                                          = default;
    cusparse_spmat_owner( const cusparse_spmat_owner & )            = delete;
    cusparse_spmat_owner &operator=( const cusparse_spmat_owner & ) = delete;

    ~cusparse_spmat_owner()
    {
        if ( descr != nullptr )
            cusparseDestroySpMat( descr );
    }
};

/// RAII owner for a read-only cuSPARSE sparse matrix descriptor.
struct cusparse_const_spmat_owner
{
    cusparseConstSpMatDescr_t descr = nullptr;

    cusparse_const_spmat_owner()                                                = default;
    cusparse_const_spmat_owner( const cusparse_const_spmat_owner & )            = delete;
    cusparse_const_spmat_owner &operator=( const cusparse_const_spmat_owner & ) = delete;

    ~cusparse_const_spmat_owner()
    {
        if ( descr != nullptr )
            cusparseDestroySpMat( descr );
    }
};

/// RAII owner for a cuSPARSE SpGEMM descriptor.
struct cusparse_spgemm_owner
{
    cusparseSpGEMMDescr_t descr = nullptr;

    cusparse_spgemm_owner()                                           = default;
    cusparse_spgemm_owner( const cusparse_spgemm_owner & )            = delete;
    cusparse_spgemm_owner &operator=( const cusparse_spgemm_owner & ) = delete;

    ~cusparse_spgemm_owner()
    {
        if ( descr != nullptr )
            cusparseSpGEMM_destroyDescr( descr );
    }
};

/// RAII owner for a raw device buffer.
struct cuda_buffer_owner
{
    void *ptr = nullptr;

    cuda_buffer_owner()                                       = default;
    cuda_buffer_owner( const cuda_buffer_owner & )            = delete;
    cuda_buffer_owner &operator=( const cuda_buffer_owner & ) = delete;

    ~cuda_buffer_owner()
    {
        if ( ptr != nullptr )
            cudaFree( ptr );
    }

    void allocate( std::size_t size )
    {
        if ( size != 0 )
            CUDA_SAFE_CALL( cudaMalloc( &ptr, size ) );
    }
};

template <class T, class Ord>
__global__ void fill_one_kernel( Ord n, T *data )
{
    const Ord i = static_cast<Ord>( blockIdx.x ) * blockDim.x + threadIdx.x;

    if ( i < n )
        data[i] = T( 1 );
}

template <class T, class Ord>
void fill_one( Ord n, T *data )
{
    if ( n == 0 )
        return;

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>( ( n + block_size - 1 ) / block_size );

    fill_one_kernel<T, Ord><<<grid_size, block_size>>>( n, data );

    CUDA_SAFE_CALL( cudaGetLastError() );
}

/**
 * One thread per scalar row of y:
 *
 *   y[row] = sum_k B_k(local_row, :) * x[col_k * block_sz_c : ...]
 *
 * Blocks are stored row-major, so the local row of a block is contiguous
 * in memory, which makes the inner loop a unit-stride dot product.
 */
template <class T, class Ord>
__global__ void bsr_mat_vec_prod_kernel(
    Ord nrows, Ord bsr, Ord bsc, const Ord *row_ptrs, const Ord *col_inds, const T *vals, const T *x, T *y
)
{
    const Ord scalar_rows = nrows * bsr;
    const Ord row         = static_cast<Ord>( blockIdx.x ) * blockDim.x + threadIdx.x;

    if ( row >= scalar_rows )
        return;

    const Ord block_row = row / bsr;
    const Ord local_row = row % bsr;

    T sum = T( 0 );

    for ( Ord k = row_ptrs[block_row]; k < row_ptrs[block_row + 1]; ++k )
    {
        const Ord col   = col_inds[k];
        const T  *block = vals + k * bsr * bsc + local_row * bsc;
        const T  *x_seg = x + col * bsc;

        for ( Ord c = 0; c < bsc; ++c )
        {
            sum += block[c] * x_seg[c];
        }
    }

    y[row] = sum;
}

/**
 * One thread per block row: stores the block row index of every
 * nonzero block of C, so the numeric kernel does not have to search for it.
 */
template <class Ord>
__global__ void bsr_block_row_kernel( Ord nrows, const Ord *row_ptrs, Ord *block_row )
{
    const Ord row = static_cast<Ord>( blockIdx.x ) * blockDim.x + threadIdx.x;

    if ( row >= nrows )
        return;

    for ( Ord k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k )
    {
        block_row[k] = row;
    }
}

/**
 * One thread per scalar element of every nonzero block of C:
 *
 *   C_k(r, c) = sum_p A(i, p)(r, :) * B(p, j)(:, c),  j = col_inds_C[k]
 *
 * where i is the block row of k and p runs over the common block columns of A
 * and block rows of B. B(p, j) is located by a linear scan of B's row p.
 */
template <class T, class Ord>
__global__ void bsr_mat_mat_prod_kernel(
    Ord nnzb_c, Ord c_bsr, Ord c_bsc, Ord shared, const Ord *c_ci, const Ord *c_block_row, const Ord *a_rp,
    const Ord *a_ci, const T *a_vals, const Ord *b_rp, const Ord *b_ci, const T *b_vals, T *c_vals
)
{
    const Ord block_elems = c_bsr * c_bsc;
    const Ord total       = nnzb_c * block_elems;
    const Ord t           = static_cast<Ord>( blockIdx.x ) * blockDim.x + threadIdx.x;

    if ( t >= total )
        return;

    const Ord k = t / block_elems;
    const Ord e = t % block_elems;
    const Ord r = e / c_bsc;
    const Ord c = e % c_bsc;

    const Ord block_row = c_block_row[k];
    const Ord col       = c_ci[k];

    T sum = T( 0 );

    for ( Ord ka = a_rp[block_row]; ka < a_rp[block_row + 1]; ++ka )
    {
        const Ord p = a_ci[ka];

        for ( Ord kb = b_rp[p]; kb < b_rp[p + 1]; ++kb )
        {
            if ( b_ci[kb] != col )
                continue;

            const T *a_block = a_vals + ka * c_bsr * shared + r * shared;
            const T *b_block = b_vals + kb * shared * c_bsc;

            for ( Ord q = 0; q < shared; ++q )
            {
                sum += a_block[q] * b_block[q * c_bsc + c];
            }
        }
    }

    c_vals[k * block_elems + e] = sum;
}

} // namespace detail

/**
 * BSR matrix-vector product on CUDA: y = A * x.
 *
 * A: (nrows * block_sz_r) x (ncols * block_sz_c)
 * x: ncols * block_sz_c vector
 * y: nrows * block_sz_r vector
 *
 * Supports rectangular blocks and uses the native ordinal type of the
 * matrix, so no conversion of the sparsity pattern is required.
 */
template <class T, class Ord, class Memory>
void bsr_mat_vec_prod_cuda(
    const bsr_matrix<T, Ord, Memory> &A, const scfd::arrays::array<T, Memory> &x,
    scfd::arrays::array<T, Memory> &y
)
{
    using ordinal_type = typename bsr_matrix<T, Ord, Memory>::ordinal_type;

    if ( x.size() != A.scalar_cols() )
        throw std::invalid_argument( "bsr_mat_vec_prod_cuda: x has incompatible size" );

    if ( y.size() != A.scalar_rows() )
        throw std::invalid_argument( "bsr_mat_vec_prod_cuda: y has incompatible size" );

    const ordinal_type scalar_rows = A.scalar_rows();

    if ( scalar_rows == 0 )
        return;

    constexpr int block_size = 256;
    const int     grid_size  = static_cast<int>( ( scalar_rows + block_size - 1 ) / block_size );

    detail::bsr_mat_vec_prod_kernel<T, ordinal_type><<<grid_size, block_size>>>(
        A.nrows(), A.block_sz_r(), A.block_sz_c(), A.row_ptrs_data(), A.col_inds_data(), A.vals_data(), x.raw_ptr(),
        y.raw_ptr()
    );

    CUDA_SAFE_CALL( cudaGetLastError() );
}

/**
 * Compute the sparsity pattern of C = A * B on CUDA.
 *
 * cuSPARSE has no SpGEMM for BSR, so the block-level sparsity pattern of A
 * and B is treated as a plain CSR matrix (with artificial all-ones values)
 * and multiplied with cusparseSpGEMM. The resulting CSR pattern is exactly
 * the block sparsity pattern of C, which is then stored into C. The whole
 * symbolic phase runs on the device, and C is left ready for
 * bsr_mat_mat_prod_cuda().
 *
 * Note: cusparseSpGEMM supports 64-bit indices only since CUDA 13; with an
 * older cuSPARSE a 32-bit ordinal type must be used.
 */
template <class T, class Ord, class Memory>
void bsr_mat_mat_prod_skeleton_cuda(
    const bsr_matrix<T, Ord, Memory> &A, const bsr_matrix<T, Ord, Memory> &B,
    bsr_matrix<T, Ord, Memory> &C
)
{
    using ordinal_type = typename bsr_matrix<T, Ord, Memory>::ordinal_type;
    using array_val_t  = scfd::arrays::array<T, Memory>;

    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value,
        "bsr_mat_mat_prod_skeleton_cuda supports only float or double"
    );

#if defined( CUSPARSE_VER_MAJOR ) && ( CUSPARSE_VER_MAJOR < 13 )
    static_assert(
        sizeof( ordinal_type ) == sizeof( std::int32_t ),
        "cuSPARSE < 13 supports only 32-bit indices in cusparseSpGEMM; "
        "use a 32-bit ordinal_type or CUDA >= 13"
    );
#endif

    if ( A.ncols() != B.nrows() )
        throw std::invalid_argument( "bsr_mat_mat_prod_skeleton_cuda: A.ncols() must equal B.nrows()" );

    if ( A.block_sz_c() != B.block_sz_r() )
        throw std::invalid_argument( "bsr_mat_mat_prod_skeleton_cuda: A.block_sz_c() must equal B.block_sz_r()" );

    const ordinal_type nrows      = A.nrows();
    const ordinal_type ncols      = B.ncols();
    const ordinal_type block_sz_r = A.block_sz_r();
    const ordinal_type block_sz_c = B.block_sz_c();
    const ordinal_type nnzb_a     = A.nnzb();
    const ordinal_type nnzb_b     = B.nnzb();

    if ( nnzb_a == 0 || nnzb_b == 0 )
    {
        C.init( nrows, ncols, 0, block_sz_r, block_sz_c );
        return;
    }

    // Artificial all-ones values: only the block pattern of A and B is used.
    array_val_t a_pattern_vals;
    array_val_t b_pattern_vals;
    a_pattern_vals.init( nnzb_a );
    b_pattern_vals.init( nnzb_b );

    detail::fill_one<T, ordinal_type>( nnzb_a, a_pattern_vals.raw_ptr() );
    detail::fill_one<T, ordinal_type>( nnzb_b, b_pattern_vals.raw_ptr() );

    const cusparseIndexType_t index_type = detail::cusparse_index_type<ordinal_type>::value;
    const cudaDataType        value_type = detail::cusparse_value_type<T>::value;

    detail::cusparse_handle_owner handle;
    NMFD_CUSPARSE_SAFE_CALL( cusparseCreate( &handle.handle ) );

    detail::cusparse_const_spmat_owner mat_a;
    detail::cusparse_const_spmat_owner mat_b;
    detail::cusparse_spmat_owner       mat_c;

    NMFD_CUSPARSE_SAFE_CALL( cusparseCreateConstCsr(
        &mat_a.descr, static_cast<std::int64_t>( nrows ), static_cast<std::int64_t>( A.ncols() ),
        static_cast<std::int64_t>( nnzb_a ), A.row_ptrs_data(), A.col_inds_data(), a_pattern_vals.raw_ptr(), index_type,
        index_type, CUSPARSE_INDEX_BASE_ZERO, value_type
    ) );

    NMFD_CUSPARSE_SAFE_CALL( cusparseCreateConstCsr(
        &mat_b.descr, static_cast<std::int64_t>( B.nrows() ), static_cast<std::int64_t>( ncols ),
        static_cast<std::int64_t>( nnzb_b ), B.row_ptrs_data(), B.col_inds_data(), b_pattern_vals.raw_ptr(), index_type,
        index_type, CUSPARSE_INDEX_BASE_ZERO, value_type
    ) );

    NMFD_CUSPARSE_SAFE_CALL( cusparseCreateCsr(
        &mat_c.descr, static_cast<std::int64_t>( nrows ), static_cast<std::int64_t>( ncols ), 0, nullptr, nullptr,
        nullptr, index_type, index_type, CUSPARSE_INDEX_BASE_ZERO, value_type
    ) );

    detail::cusparse_spgemm_owner spgemm;
    NMFD_CUSPARSE_SAFE_CALL( cusparseSpGEMM_createDescr( &spgemm.descr ) );

    const T alpha = T( 1 );
    const T beta  = T( 0 );

    std::size_t buffer_size1 = 0;
    NMFD_CUSPARSE_SAFE_CALL( cusparseSpGEMM_workEstimation(
        handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a.descr,
        mat_b.descr, &beta, mat_c.descr, value_type, CUSPARSE_SPGEMM_DEFAULT, spgemm.descr, &buffer_size1, nullptr
    ) );

    detail::cuda_buffer_owner buffer1;
    buffer1.allocate( buffer_size1 );

    NMFD_CUSPARSE_SAFE_CALL( cusparseSpGEMM_workEstimation(
        handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a.descr,
        mat_b.descr, &beta, mat_c.descr, value_type, CUSPARSE_SPGEMM_DEFAULT, spgemm.descr, &buffer_size1, buffer1.ptr
    ) );

    std::size_t buffer_size2 = 0;
    NMFD_CUSPARSE_SAFE_CALL( cusparseSpGEMM_compute(
        handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a.descr,
        mat_b.descr, &beta, mat_c.descr, value_type, CUSPARSE_SPGEMM_DEFAULT, spgemm.descr, &buffer_size2, nullptr
    ) );

    detail::cuda_buffer_owner buffer2;
    buffer2.allocate( buffer_size2 );

    NMFD_CUSPARSE_SAFE_CALL( cusparseSpGEMM_compute(
        handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a.descr,
        mat_b.descr, &beta, mat_c.descr, value_type, CUSPARSE_SPGEMM_DEFAULT, spgemm.descr, &buffer_size2, buffer2.ptr
    ) );

    std::int64_t c_nrows = 0;
    std::int64_t c_ncols = 0;
    std::int64_t c_nnz   = 0;
    NMFD_CUSPARSE_SAFE_CALL( cusparseSpMatGetSize( mat_c.descr, &c_nrows, &c_ncols, &c_nnz ) );

    const ordinal_type nnzb_c = static_cast<ordinal_type>( c_nnz );

    C.init( nrows, ncols, nnzb_c, block_sz_r, block_sz_c );

    if ( nnzb_c == 0 )
    {
        CUDA_SAFE_CALL(
            cudaMemset( C.row_ptrs_data(), 0, static_cast<std::size_t>( nrows + 1 ) * sizeof( ordinal_type ) )
        );
        return;
    }

    array_val_t c_pattern_vals;
    c_pattern_vals.init( nnzb_c );

    NMFD_CUSPARSE_SAFE_CALL(
        cusparseCsrSetPointers( mat_c.descr, C.row_ptrs_data(), C.col_inds_data(), c_pattern_vals.raw_ptr() )
    );

    NMFD_CUSPARSE_SAFE_CALL( cusparseSpGEMM_copy(
        handle.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_a.descr,
        mat_b.descr, &beta, mat_c.descr, value_type, CUSPARSE_SPGEMM_DEFAULT, spgemm.descr
    ) );
}

/**
 * Compute values of C = A * B on CUDA.
 *
 * C must already contain the sparsity pattern produced by
 * bsr_mat_mat_prod_skeleton_cuda().
 */
template <class T, class Ord, class Memory>
void bsr_mat_mat_prod_cuda(
    const bsr_matrix<T, Ord, Memory> &A, const bsr_matrix<T, Ord, Memory> &B,
    bsr_matrix<T, Ord, Memory> &C
)
{
    using ordinal_type = typename bsr_matrix<T, Ord, Memory>::ordinal_type;

    if ( A.ncols() != B.nrows() )
        throw std::invalid_argument( "bsr_mat_mat_prod_cuda: A.ncols() must equal B.nrows()" );

    if ( A.block_sz_c() != B.block_sz_r() )
        throw std::invalid_argument( "bsr_mat_mat_prod_cuda: A.block_sz_c() must equal B.block_sz_r()" );

    if ( C.nrows() != A.nrows() || C.ncols() != B.ncols() )
        throw std::invalid_argument( "bsr_mat_mat_prod_cuda: C has incompatible matrix dimensions" );

    if ( C.block_sz_r() != A.block_sz_r() || C.block_sz_c() != B.block_sz_c() )
        throw std::invalid_argument( "bsr_mat_mat_prod_cuda: C has incompatible block dimensions" );

    const ordinal_type nnzb = C.nnzb();

    if ( nnzb == 0 )
        return;

    scfd::arrays::array<ordinal_type, Memory> block_row;
    block_row.init( nnzb );

    constexpr int block_size = 256;

    {
        const int grid_size = static_cast<int>( ( C.nrows() + block_size - 1 ) / block_size );

        detail::bsr_block_row_kernel<ordinal_type>
            <<<grid_size, block_size>>>( C.nrows(), C.row_ptrs_data(), block_row.raw_ptr() );
    }

    const ordinal_type total = nnzb * C.block_sz_r() * C.block_sz_c();

    {
        const int grid_size = static_cast<int>( ( total + block_size - 1 ) / block_size );

        detail::bsr_mat_mat_prod_kernel<T, ordinal_type><<<grid_size, block_size>>>(
            nnzb, C.block_sz_r(), C.block_sz_c(), A.block_sz_c(), C.col_inds_data(), block_row.raw_ptr(),
            A.row_ptrs_data(), A.col_inds_data(), A.vals_data(), B.row_ptrs_data(), B.col_inds_data(), B.vals_data(),
            C.vals_data()
        );
    }

    CUDA_SAFE_CALL( cudaGetLastError() );
}

#undef NMFD_CUSPARSE_SAFE_CALL

} // namespace operations
} // namespace nmfd

#endif // __NMFD_BSR_OPERATIONS_CUDA_H__
