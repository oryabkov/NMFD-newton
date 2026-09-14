#ifndef __NMFD_BSR_OPERATIONS_H__
#define __NMFD_BSR_OPERATIONS_H__

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <nmfd/operations/bsr_matrix.h>

namespace nmfd
{
namespace operations
{

/**
 * BSR matrix-vector product: y = A * x.
 *
 * A: (nrows * block_sz_r) x (ncols * block_sz_c)
 * x: ncols * block_sz_c vector
 * y: nrows * block_sz_r vector
 */
template <class T, class Ord, class Memory>
void bsr_mat_vec_prod(
    const bsr_matrix<T, Ord, Memory> &A, const scfd::arrays::array<T, Memory> &x, scfd::arrays::array<T, Memory> &y
)
{
    using ordinal_type = typename bsr_matrix<T, Ord, Memory>::ordinal_type;

    const ordinal_type bsr = A.block_sz_r();
    const ordinal_type bsc = A.block_sz_c();

    if ( x.size() != A.scalar_cols() )
        throw std::invalid_argument( "bsr_mat_vec_prod: x has incompatible size" );

    if ( y.size() != A.scalar_rows() )
        throw std::invalid_argument( "bsr_mat_vec_prod: y has incompatible size" );

    const ordinal_type *row_ptrs = A.row_ptrs_data();
    const ordinal_type *col_inds = A.col_inds_data();
    scfd::backend::for_each<ordinal_type>()(
        [=] __DEVICE_TAG__( ordinal_type row ) {
            for ( ordinal_type r = 0; r < bsr; ++r )
                y( row * bsr + r ) = T( 0 );

            for ( ordinal_type k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k )
            {
                const ordinal_type col = col_inds[k];

                for ( ordinal_type r = 0; r < bsr; ++r )
                {
                    for ( ordinal_type c = 0; c < bsc; ++c )
                    {
                        y( row * bsr + r ) += A.vals( k, r, c ) * x( col * bsc + c );
                    }
                }
            }
        },
        A.nrows()
    );
    scfd::backend::for_each<ordinal_type>().wait();
}


/**
 * Compute sparsity pattern of C = A * B.
 *
 * C is initialized with the resulting block dimensions
 * and sparsity pattern. \n
 * Reference: \n
 * F. G. Gustavson,
 * "Two Fast Algorithms for Sparse Matrices:
 * Multiplication and Permuted Transposition",
 * ACM Trans. Math. Softw., 4(3), 1978, pp. 250-269.
*/

template <class T, class Ord, class Memory>
void bsr_mat_mat_prod_skeleton(
    const bsr_matrix<T, Ord, Memory> &A, const bsr_matrix<T, Ord, Memory> &B, bsr_matrix<T, Ord, Memory> &C
)
{
    using ordinal_type = typename bsr_matrix<T, Ord, Memory>::ordinal_type;
    using marker_type  = std::ptrdiff_t;
    if ( A.ncols() != B.nrows() )
    {
        throw std::invalid_argument( "bsr_mat_mat_prod_skeleton: A.ncols() must equal B.nrows()" );
    }

    if ( A.block_sz_c() != B.block_sz_r() )
    {
        throw std::invalid_argument( "bsr_mat_mat_prod_skeleton: A.block_sz_c() must equal B.block_sz_r()" );
    }

    const ordinal_type *a_rp = A.row_ptrs_data();
    const ordinal_type *a_ci = A.col_inds_data();

    const ordinal_type *b_rp = B.row_ptrs_data();
    const ordinal_type *b_ci = B.col_inds_data();

    const ordinal_type nrows = A.nrows();
    const ordinal_type ncols = B.ncols();

    const ordinal_type block_sz_r = A.block_sz_r();
    const ordinal_type block_sz_c = B.block_sz_c();

    scfd::arrays::array<marker_type, Memory> marker;
    marker.init( ncols );
    marker_type *mk = marker.raw_ptr();
    for ( ordinal_type j = 0; j < ncols; ++j )
    {
        mk[j] = -1;
    }

    scfd::arrays::array<ordinal_type, Memory> row_nnz;
    row_nnz.init( nrows );
    marker_type mark = 0;

    // Symbolic pass:
    // determine the number of nonzero blocks in each row of C.
    for ( ordinal_type i = 0; i < nrows; ++i )
    {
        ++mark;
        ordinal_type count = 0;
        for ( ordinal_type ka = a_rp[i]; ka < a_rp[i + 1]; ++ka )
        {
            const ordinal_type a_col = a_ci[ka];
            for ( ordinal_type kb = b_rp[a_col]; kb < b_rp[a_col + 1]; ++kb )
            {
                const ordinal_type b_col = b_ci[kb];
                if ( mk[b_col] != mark )
                {
                    mk[b_col] = mark;
                    ++count;
                }
            }
        }
        row_nnz( i ) = count;
    }

    // Total number of nonzero blocks in C.
    ordinal_type nnzb = 0;

    for ( ordinal_type i = 0; i < nrows; ++i )
    {
        nnzb += row_nnz( i );
    }

    C.init( nrows, ncols, nnzb, block_sz_r, block_sz_c );

    // Build row pointers.
    C.row_ptr( 0 ) = 0;

    for ( ordinal_type i = 0; i < nrows; ++i )
    {
        C.row_ptr( i + 1 ) = C.row_ptr( i ) + row_nnz( i );
    }

    // Second symbolic pass:
    // fill column indices of C.
    for ( ordinal_type i = 0; i < nrows; ++i )
    {
        ++mark;

        ordinal_type offset = C.row_ptr( i );

        for ( ordinal_type ka = a_rp[i]; ka < a_rp[i + 1]; ++ka )
        {
            const ordinal_type a_col = a_ci[ka];

            for ( ordinal_type kb = b_rp[a_col]; kb < b_rp[a_col + 1]; ++kb )
            {
                const ordinal_type b_col = b_ci[kb];

                if ( mk[b_col] != mark )
                {
                    mk[b_col]             = mark;
                    C.col_ind( offset++ ) = b_col;
                }
            }
        }
    }
}


/**
 * Compute values of C = A * B.
 *
 * C must already contain the sparsity pattern produced by
 * bsr_mat_mat_prod_skeleton(). \n
 * Reference: \n
 * T. A. Davis, "Direct Methods for Sparse Linear Systems",
 * SIAM, 2006, Chapter 2.
 */
template <class T, class Ord, class Memory>
void bsr_mat_mat_prod(
    const bsr_matrix<T, Ord, Memory> &A, const bsr_matrix<T, Ord, Memory> &B, bsr_matrix<T, Ord, Memory> &C
)
{
    using ordinal_type = typename bsr_matrix<T, Ord, Memory>::ordinal_type;
    using marker_type  = std::ptrdiff_t;

    if ( A.ncols() != B.nrows() )
    {
        throw std::invalid_argument( "bsr_mat_mat_prod: A.ncols() must equal B.nrows()" );
    }

    if ( A.block_sz_c() != B.block_sz_r() )
    {
        throw std::invalid_argument( "bsr_mat_mat_prod: A.block_sz_c() must equal B.block_sz_r()" );
    }

    if ( C.nrows() != A.nrows() || C.ncols() != B.ncols() )
    {
        throw std::invalid_argument( "bsr_mat_mat_prod: C has incompatible matrix dimensions" );
    }

    if ( C.block_sz_r() != A.block_sz_r() || C.block_sz_c() != B.block_sz_c() )
    {
        throw std::invalid_argument( "bsr_mat_mat_prod: C has incompatible block dimensions" );
    }

    const ordinal_type a_bsr = A.block_sz_r();
    const ordinal_type a_bsc = A.block_sz_c();
    const ordinal_type b_bsc = B.block_sz_c();

    const ordinal_type *a_rp = A.row_ptrs_data();
    const ordinal_type *a_ci = A.col_inds_data();

    const ordinal_type *b_rp = B.row_ptrs_data();
    const ordinal_type *b_ci = B.col_inds_data();

    const ordinal_type *c_rp = C.row_ptrs_data();
    const ordinal_type *c_ci = C.col_inds_data();

    // marker[col] contains the position of the corresponding
    // block in the current row of C.
    scfd::arrays::array<ordinal_type, Memory> marker;
    marker.init( C.ncols() );

    ordinal_type *mk = marker.raw_ptr();

    // Clear C values.
    for ( ordinal_type k = 0; k < C.nnzb(); ++k )
    {
        for ( ordinal_type r = 0; r < a_bsr; ++r )
        {
            for ( ordinal_type c = 0; c < b_bsc; ++c )
            {
                C.vals( k, r, c ) = T( 0 );
            }
        }
    }

    // Numeric phase.
    for ( ordinal_type i = 0; i < C.nrows(); ++i )
    {
        // Build mapping:
        //
        //     column of C -> block index in C
        //
        // for the current block row.
        for ( ordinal_type k = c_rp[i]; k < c_rp[i + 1]; ++k )
        {
            mk[c_ci[k]] = k;
        }

        // Compute C(i,:) = A(i,:) * B.
        for ( ordinal_type ka = a_rp[i]; ka < a_rp[i + 1]; ++ka )
        {
            const ordinal_type a_col = a_ci[ka];

            for ( ordinal_type kb = b_rp[a_col]; kb < b_rp[a_col + 1]; ++kb )
            {
                const ordinal_type b_col = b_ci[kb];

                const ordinal_type kc = mk[b_col];

                for ( ordinal_type r = 0; r < a_bsr; ++r )
                {
                    for ( ordinal_type c = 0; c < b_bsc; ++c )
                    {
                        T sum = T( 0 );

                        for ( ordinal_type p = 0; p < a_bsc; ++p )
                        {
                            sum += A.vals( ka, r, p ) * B.vals( kb, p, c );
                        }

                        C.vals( kc, r, c ) += sum;
                    }
                }
            }
        }
    }
}

} // namespace operations
} // namespace nmfd

#endif