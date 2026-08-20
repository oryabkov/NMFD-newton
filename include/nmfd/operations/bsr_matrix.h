#ifndef __NMFD_BSR_MATRIX_H__
#define __NMFD_BSR_MATRIX_H__

#include <scfd/arrays/array.h>
#include <scfd/memory/host.h>
#include <cassert>

namespace nmfd
{
namespace operations
{

/**
 * Block Sparse Row (BSR) matrix — local (host) version.
 *
 * Memory layout for vals (for square block_sz):
 *   For non-zero block k (row_ptrs(row) <= k < row_ptrs(row+1)):
 *     vals( k * block_sz * block_sz + r * block_sz + c ) = block(r, c)
 *
 * For rectangular blocks: use block_sz_r and block_sz_c.
 *
 * Uses raw_ptr() for direct pointer access and operator() for element access.
 */
template <class T, class Memory = scfd::memory::host, class Ord = std::ptrdiff_t>
struct bsr_matrix
{
    using ordinal_type = Ord;
    using array_t      = scfd::arrays::array<Ord, Memory>;
    using vals_array_t = scfd::arrays::array<T, Memory>;

    ordinal_type nrows;      ///< number of block rows
    ordinal_type ncols;      ///< number of block columns
    ordinal_type nnz;        ///< number of non-zero blocks
    ordinal_type block_sz;   ///< block size (square blocks)
    ordinal_type block_sz_r; ///< block rows (for rectangular blocks)
    ordinal_type block_sz_c; ///< block columns (for rectangular blocks)

    array_t    row_ptrs;  ///< CSR row pointers, size nrows + 1
    array_t    col_inds;  ///< column indices of non-zero blocks, size nnz
    vals_array_t vals;    ///< block values, size nnz * block_sz_r * block_sz_c

    bsr_matrix() = default;

    void init(ordinal_type nrows, ordinal_type ncols, ordinal_type nnz, ordinal_type block_sz)
    {
        this->nrows      = nrows;
        this->ncols      = ncols;
        this->nnz        = nnz;
        this->block_sz   = block_sz;
        this->block_sz_r = block_sz;
        this->block_sz_c = block_sz;

        row_ptrs.init(nrows + 1);
        col_inds.init(nnz);
        vals.init(nnz * block_sz * block_sz);
    }

    void init(ordinal_type nrows, ordinal_type ncols, ordinal_type nnz,
              ordinal_type block_sz_r, ordinal_type block_sz_c)
    {
        this->nrows      = nrows;
        this->ncols      = ncols;
        this->nnz        = nnz;
        this->block_sz   = 0; // not square
        this->block_sz_r = block_sz_r;
        this->block_sz_c = block_sz_c;

        row_ptrs.init(nrows + 1);
        col_inds.init(nnz);
        vals.init(nnz * block_sz_r * block_sz_c);
    }

    /// Access element (r, c) of non-zero block k
    T& block_val(ordinal_type k, ordinal_type r, ordinal_type c)
    {
        return vals.raw_ptr()[k * block_sz_r * block_sz_c + r * block_sz_c + c];
    }
    const T& block_val(ordinal_type k, ordinal_type r, ordinal_type c) const
    {
        return vals.raw_ptr()[k * block_sz_r * block_sz_c + r * block_sz_c + c];
    }

    /// Raw pointer to values of block k
    T* block_ptr(ordinal_type k)
    {
        return vals.raw_ptr() + k * block_sz_r * block_sz_c;
    }
    const T* block_ptr(ordinal_type k) const
    {
        return vals.raw_ptr() + k * block_sz_r * block_sz_c;
    }
};

} // namespace operations
} // namespace nmfd

#endif // __NMFD_BSR_MATRIX_H__
