#ifndef __NMFD_BSR_OPERATIONS_H__
#define __NMFD_BSR_OPERATIONS_H__

#include <nmfd/operations/bsr_matrix.h>

namespace nmfd
{
namespace operations
{

/**
 * BSR matrix-vector product: y = A * x (host).
 *
 * A: (nrows * block_sz_r) x (ncols * block_sz_c)
 * x: ncols * block_sz_c vector
 * y: nrows * block_sz_r vector
 */
template <class T, class Memory>
void bsr_mat_vec_prod(
    const bsr_matrix<T, Memory>& A,
    const T* x, T* y)
{
    using ordinal_type = typename bsr_matrix<T, Memory>::ordinal_type;
    const ordinal_type bsr = A.block_sz_r;
    const ordinal_type bsc = A.block_sz_c;
    const ordinal_type* rp = A.row_ptrs.raw_ptr();
    const ordinal_type* ci = A.col_inds.raw_ptr();

    for (ordinal_type i = 0; i < A.nrows * bsr; ++i)
        y[i] = T(0);

    for (ordinal_type row = 0; row < A.nrows; ++row)
    {
        for (ordinal_type k = rp[row]; k < rp[row + 1]; ++k)
        {
            ordinal_type col = ci[k];
            const T* block = A.block_ptr(k);

            for (ordinal_type r = 0; r < bsr; ++r)
                for (ordinal_type c = 0; c < bsc; ++c)
                    y[row * bsr + r] += block[r * bsc + c] * x[col * bsc + c];
        }
    }
}

/**
 * Compute sparsity pattern (skeleton) of C = A * B.
 * Allocates C.row_ptrs and C.col_inds, but NOT C.vals.
 */
template <class T, class Memory>
void bsr_mat_mat_prod_skeleton(
    const bsr_matrix<T, Memory>& A,
    const bsr_matrix<T, Memory>& B,
    bsr_matrix<T, Memory>& C)
{
    using ordinal_type = typename bsr_matrix<T, Memory>::ordinal_type;
    const ordinal_type* a_rp = A.row_ptrs.raw_ptr();
    const ordinal_type* a_ci = A.col_inds.raw_ptr();
    const ordinal_type* b_rp = B.row_ptrs.raw_ptr();
    const ordinal_type* b_ci = B.col_inds.raw_ptr();

    C.nrows      = A.nrows;
    C.ncols      = B.ncols;
    C.block_sz   = A.block_sz;
    C.block_sz_r = A.block_sz_r;
    C.block_sz_c = B.block_sz_c;

    C.row_ptrs.init(C.nrows + 1);
    ordinal_type* c_rp = C.row_ptrs.raw_ptr();

    scfd::arrays::array<ordinal_type, Memory> marker;
    marker.init(C.ncols);
    ordinal_type* mk = marker.raw_ptr();

    for (ordinal_type j = 0; j < C.ncols; ++j)
        mk[j] = -1;

    c_rp[0] = 0;
    ordinal_type nnz_total = 0;

    for (ordinal_type i = 0; i < A.nrows; ++i)
    {
        ordinal_type row_nnz = 0;

        for (ordinal_type ka = a_rp[i]; ka < a_rp[i + 1]; ++ka)
        {
            ordinal_type a_col = a_ci[ka];

            for (ordinal_type kb = b_rp[a_col]; kb < b_rp[a_col + 1]; ++kb)
            {
                ordinal_type b_col = b_ci[kb];

                if (mk[b_col] != i)
                {
                    mk[b_col] = i;
                    ++row_nnz;
                }
            }
        }

        nnz_total += row_nnz;
        c_rp[i + 1] = nnz_total;
    }

    C.nnz = nnz_total;
    C.col_inds.init(C.nnz);
    ordinal_type* c_ci = C.col_inds.raw_ptr();

    for (ordinal_type j = 0; j < C.ncols; ++j)
        mk[j] = -1;

    for (ordinal_type i = 0; i < A.nrows; ++i)
    {
        ordinal_type offset = c_rp[i];

        for (ordinal_type ka = a_rp[i]; ka < a_rp[i + 1]; ++ka)
        {
            ordinal_type a_col = a_ci[ka];

            for (ordinal_type kb = b_rp[a_col]; kb < b_rp[a_col + 1]; ++kb)
            {
                ordinal_type b_col = b_ci[kb];

                if (mk[b_col] != i)
                {
                    mk[b_col] = i;
                    c_ci[offset++] = b_col;
                }
            }
        }
    }
}

/**
 * Compute values of C = A * B (host).
 * C must already have its skeleton computed (row_ptrs, col_inds, nnz, block_sz).
 */
template <class T, class Memory>
void bsr_mat_mat_prod(
    const bsr_matrix<T, Memory>& A,
    const bsr_matrix<T, Memory>& B,
    bsr_matrix<T, Memory>& C)
{
    using ordinal_type = typename bsr_matrix<T, Memory>::ordinal_type;
    const ordinal_type bs = A.block_sz;
    const ordinal_type* a_rp = A.row_ptrs.raw_ptr();
    const ordinal_type* a_ci = A.col_inds.raw_ptr();
    const ordinal_type* b_rp = B.row_ptrs.raw_ptr();
    const ordinal_type* b_ci = B.col_inds.raw_ptr();
    const ordinal_type* c_rp = C.row_ptrs.raw_ptr();
    const ordinal_type* c_ci = C.col_inds.raw_ptr();

    C.vals.init(C.nnz * bs * bs);
    T* cv = C.vals.raw_ptr();

    for (ordinal_type i = 0; i < C.nnz * bs * bs; ++i)
        cv[i] = T(0);

    for (ordinal_type i = 0; i < C.nrows; ++i)
    {
        for (ordinal_type ka = a_rp[i]; ka < a_rp[i + 1]; ++ka)
        {
            ordinal_type a_col = a_ci[ka];
            const T* ab = A.block_ptr(ka);

            for (ordinal_type kb = b_rp[a_col]; kb < b_rp[a_col + 1]; ++kb)
            {
                ordinal_type b_col = b_ci[kb];
                const T* bb = B.block_ptr(kb);

                ordinal_type kc = -1;
                for (ordinal_type k = c_rp[i]; k < c_rp[i + 1]; ++k)
                {
                    if (c_ci[k] == b_col) { kc = k; break; }
                }
                assert(kc >= 0);

                T* cb = cv + kc * bs * bs;

                for (ordinal_type r = 0; r < bs; ++r)
                    for (ordinal_type c = 0; c < bs; ++c)
                    {
                        T sum = T(0);
                        for (ordinal_type p = 0; p < bs; ++p)
                            sum += ab[r * bs + p] * bb[p * bs + c];
                        cb[r * bs + c] += sum;
                    }
            }
        }
    }
}

} // namespace operations
} // namespace nmfd

#endif // __NMFD_BSR_OPERATIONS_H__
