#ifndef __NMFD_BSR_MATRIX_H__
#define __NMFD_BSR_MATRIX_H__

#include <scfd/arrays/tensorN_array.h>
#include <scfd/arrays/array.h>
#include <scfd/memory/host.h>
#include <cassert>

namespace nmfd
{
namespace operations
{

/**
 * Block Sparse Row (BSR) matrix.
 *
 * The matrix is stored in block-CSR format.
 *
 * Matrix dimensions:
 *   nrows * block_sz_r  scalar rows
 *   ncols * block_sz_c  scalar columns
 *
 * For each block row:
 *   row_ptrs(i) <= k < row_ptrs(i + 1)
 *
 * gives the indices of its non-zero blocks. The corresponding
 * block column is stored in col_ind(k).
 *
 * For square blocks, block_sz_r == block_sz_c.
 */
template <class T, class Ord, class Memory>
class bsr_matrix
{
public:
    using ordinal_type = Ord;
    using array_t      = scfd::arrays::array<ordinal_type, Memory>;
    using vals_t       = scfd::arrays::tensor2_array<
        T, Memory, scfd::arrays::dyn_dim, scfd::arrays::dyn_dim, scfd::arrays::last_index_fast_arranger>;
    using row_ptrs_view_t = typename array_t::view_type;
    using col_inds_view_t = typename array_t::view_type;
    using vals_view_t     = typename vals_t::view_type;
    bsr_matrix() : nrows_( 0 ), ncols_( 0 ), nnzb_( 0 ), block_sz_r_( 0 ), block_sz_c_( 0 )
    {
    }

    /// Initialize BSR matrix with square blocks
    void init( ordinal_type nrows, ordinal_type ncols, ordinal_type nnzb, ordinal_type block_sz )
    {
        init( nrows, ncols, nnzb, block_sz, block_sz );
    }

    /// Initialize BSR matrix with rectangular blocks
    void
    init( ordinal_type nrows, ordinal_type ncols, ordinal_type nnzb, ordinal_type block_sz_r, ordinal_type block_sz_c )
    {
        nrows_      = nrows;
        ncols_      = ncols;
        nnzb_       = nnzb;
        block_sz_r_ = block_sz_r;
        block_sz_c_ = block_sz_c;

        row_ptrs_.init( nrows_ + 1 );
        col_inds_.init( nnzb_ );
        vals_.init( nnzb_, block_sz_r_, block_sz_c_ );
    }

    /// Number of block rows
    __DEVICE_TAG__ ordinal_type nrows() const
    {
        return nrows_;
    }

    /// Number of block columns
    __DEVICE_TAG__ ordinal_type ncols() const
    {
        return ncols_;
    }

    /// Number of non-zero blocks
    __DEVICE_TAG__ ordinal_type nnzb() const
    {
        return nnzb_;
    }

    /// Number of rows in a block
    __DEVICE_TAG__ ordinal_type block_sz_r() const
    {
        return block_sz_r_;
    }

    /// Number of columns in a block
    __DEVICE_TAG__ ordinal_type block_sz_c() const
    {
        return block_sz_c_;
    }

    /// Number of values in one block
    __DEVICE_TAG__ ordinal_type block_size() const
    {
        return block_sz_r_ * block_sz_c_;
    }

    /// Number of scalar rows
    __DEVICE_TAG__ ordinal_type scalar_rows() const
    {
        return nrows_ * block_sz_r_;
    }

    /// Number of scalar columns
    __DEVICE_TAG__ ordinal_type scalar_cols() const
    {
        return ncols_ * block_sz_c_;
    }

    /// CSR row pointers, size nrows + 1
    ordinal_type *row_ptrs_data()
    {
        return row_ptrs_.raw_ptr();
    }

    const ordinal_type *row_ptrs_data() const
    {
        return row_ptrs_.raw_ptr();
    }
    T *vals_data()

    {

        return vals_.raw_ptr();
    }

    const T *vals_data() const

    {

        return vals_.raw_ptr();
    }

    /// Column indices of non-zero blocks, size nnzb
    ordinal_type *col_inds_data()
    {
        return col_inds_.raw_ptr();
    }
    const ordinal_type *col_inds_data() const

    {
        return col_inds_.raw_ptr();
    }

    /// Create a host view of the row pointers (optionally synced from the array)
    row_ptrs_view_t create_row_ptrs_view( bool sync_from_array_ = true ) const
    {
        return row_ptrs_.create_view( sync_from_array_ );
    }

    /// Create a host view of the column indices (optionally synced from the array)
    col_inds_view_t create_col_inds_view( bool sync_from_array_ = true ) const
    {
        return col_inds_.create_view( sync_from_array_ );
    }

    /// Create a host view of the block values (optionally synced from the array)
    vals_view_t create_vals_view( bool sync_from_array_ = true ) const
    {
        return vals_.create_view( sync_from_array_ );
    }

    /// Block values
    __DEVICE_TAG__ T &vals( ordinal_type k, ordinal_type r, ordinal_type c ) &
    {
        return vals_( k, r, c );
    }
    /// Block values
    __DEVICE_TAG__ const T &vals( ordinal_type k, ordinal_type r, ordinal_type c ) const &
    {
        return vals_( k, r, c );
    }

    /// Access row pointer
    __DEVICE_TAG__ ordinal_type &row_ptr( ordinal_type i ) &
    {
        return row_ptrs_( i );
    }

    /// Access row pointer
    __DEVICE_TAG__ ordinal_type row_ptr( ordinal_type i ) const &
    {
        return row_ptrs_( i );
    }

    /// Access column index
    __DEVICE_TAG__ ordinal_type &col_ind( ordinal_type i ) &
    {
        return col_inds_( i );
    }

    /// Access column index
    __DEVICE_TAG__ ordinal_type col_ind( ordinal_type i ) const &
    {
        return col_inds_( i );
    }

private:
    ordinal_type nrows_;      ///< number of block rows
    ordinal_type ncols_;      ///< number of block columns
    ordinal_type nnzb_;       ///< number of non-zero blocks
    ordinal_type block_sz_r_; ///< block rows
    ordinal_type block_sz_c_; ///< block columns

    array_t row_ptrs_; ///< CSR row pointers, size nrows + 1
    array_t col_inds_; ///< column indices of non-zero blocks, size nnzb
    vals_t  vals_;     ///< block values, size nnzb * block_sz_r * block_sz_c
};

} // namespace operations
} // namespace nmfd

#endif // __NMFD_BSR_MATRIX_H__
