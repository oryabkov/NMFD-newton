#ifndef __BALANCER_H__
#define __BALANCER_H__

#include <stdexcept>
#include <vector>
#include <scfd/static_vec/vec.h>
#include <scfd/static_vec/rect.h>

namespace tests
{

template <int Dim, class Ord, class BigOrd, int TensorDim>
class balancer
{
public:
    using ord_vec_t      = scfd::static_vec::vec<Ord, Dim>;
    using big_ord_vec_t  = scfd::static_vec::vec<BigOrd, Dim>;
    using big_ord_rect_t = scfd::static_vec::rect<BigOrd, Dim>;
    using bool_vec_t     = scfd::static_vec::vec<bool, Dim>;

    // Number of blocks along each axis for the given (power-of-two) process count.
    static ord_vec_t grid_dims( int num_procs )
    {
        if ( num_procs < 1 || ( num_procs & ( num_procs - 1 ) ) != 0 )
        {
            throw std::logic_error( "balancer: num_procs must be a positive power of two" );
        }

        int k = 0; // k = log2(num_procs)
        while ( ( num_procs >> ( k + 1 ) ) != 0 ) { ++k; }

        int       base = k / Dim, rem = k % Dim;
        ord_vec_t dims;
        for ( int j = 0; j < Dim; ++j )
        {
            int e   = base + ( j < rem ? 1 : 0 ); // leading axes take the extra split first
            dims[j] = Ord( 1 ) << e;              // 2^e blocks along axis j
        }
        return dims;
    }

    // Decompose `dom_size` into num_procs blocks.
    void balance(
        const big_ord_vec_t &dom_size, int num_procs, int myid, const int global_left_bc[Dim][TensorDim],
        const int global_right_bc[Dim][TensorDim], std::vector<big_ord_rect_t> &proc_rects,
        big_ord_rect_t &my_glob_rect, int loc_left_bc[Dim][TensorDim], int loc_right_bc[Dim][TensorDim],
        bool_vec_t &periodic_flags ) const
    {
        ord_vec_t dims = grid_dims( num_procs );

        big_ord_vec_t step;
        for ( int j = 0; j < Dim; ++j )
        {
            if ( dom_size[j] % dims[j] != 0 )
            {
                throw std::logic_error( "balancer: domain size not divisible by the split count on some axis" );
            }

            step[j] = dom_size[j] / dims[j];
        }


        proc_rects.resize( num_procs );
        for ( int p = 0; p < num_procs; ++p )
        {
            proc_rects[p] = block_rect( p, dims, step );
        }
        my_glob_rect = proc_rects[myid];

        ord_vec_t my_pc = block_coords( myid, dims );
        for ( int j = 0; j < Dim; ++j )
        {
            bool on_low  = ( my_pc[j] == 0 );
            bool on_high = ( my_pc[j] == dims[j] - 1 );
            for ( int c = 0; c < TensorDim; ++c )
            {
                loc_left_bc[j][c]  = on_low ? global_left_bc[j][c] : 0;
                loc_right_bc[j][c] = on_high ? global_right_bc[j][c] : 0;
            }
            periodic_flags[j] = ( global_left_bc[j][0] == 0 ) || ( global_right_bc[j][0] == 0 );
        }
    }

private:
    static ord_vec_t block_coords( int p, const ord_vec_t &dims )
    {
        ord_vec_t pc;
        int       rem = p;
        for ( int j = 0; j < Dim; ++j )
        {
            pc[j] = rem % dims[j];
            rem /= dims[j];
        }
        return pc;
    }

    static big_ord_rect_t block_rect( int p, const ord_vec_t &dims, const big_ord_vec_t &step )
    {
        ord_vec_t     pc = block_coords( p, dims );
        big_ord_vec_t i1, i2;
        for ( int j = 0; j < Dim; ++j )
        {
            i1[j] = BigOrd( pc[j] ) * step[j];
            i2[j] = i1[j] + step[j];
        }
        return big_ord_rect_t( i1, i2 );
    }
};

} // namespace tests

#endif
