#ifndef __TESTS_SOLUTION_IO_H__
#define __TESTS_SOLUTION_IO_H__

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace tests
{

/**
 * Writes a 3D tensor field solution to a binary file.
 * Format: [dims (3x int32)] [n_components (int32)] [data (float32)]
 */
template <class VectorSpace, class Part, template <class> class BinaryFile>
class solution_writer
{
public:
    using vector_type = typename VectorSpace::vector_type;

    solution_writer( const Part &part, int tensor_dim ) : part_( part ), tensor_dim_( tensor_dim )
    {
    }

    void write( const vector_type &v, const std::string &filename ) const
    {
        auto glob_rect = part_.get_own_rect();
        auto loc_rect  = part_.get_own_loc_rect();
        auto loc_size  = loc_rect.calc_size();

        using big_ordinal = typename Part::big_ordinal;
        big_ordinal Nx = part_.dom_size[0];
        big_ordinal Ny = part_.dom_size[1];
        big_ordinal Nz = part_.dom_size[2];

        BinaryFile<float> file( part_.comm_info, filename );
        file.size( 4 + Nx * Ny * Nz * static_cast<big_ordinal>( tensor_dim_ ) );

        if ( part_.comm_info.myid == 0 )
        {
            int32_t header[4] = { static_cast<int32_t>( Nx ), static_cast<int32_t>( Ny ), static_cast<int32_t>( Nz ),
                                   static_cast<int32_t>( tensor_dim_ ) };
            float   hdr[4];
            std::memcpy( hdr, header, sizeof( hdr ) );
            file.write_at( 0, hdr, 4 );
        }

        typename vector_type::view_type view( v, true );

        std::vector<float> buf( static_cast<std::size_t>( loc_size[0] ) * tensor_dim_ );
        for ( int k = 0; k < loc_size[2]; ++k )
        {
            for ( int j = 0; j < loc_size[1]; ++j )
            {
                for ( int i = 0; i < loc_size[0]; ++i )
                {
                    for ( int t = 0; t < tensor_dim_; ++t )
                    {
                        buf[i * tensor_dim_ + t] = static_cast<float>( view( i, j, k, t ) );
                    }
                }

                big_ordinal z_global  = k + glob_rect.i1[2];
                big_ordinal y_global  = j + glob_rect.i1[1];
                big_ordinal x0_global = glob_rect.i1[0];
                big_ordinal offset =
                    4 + ( z_global * Ny * Nx + y_global * Nx + x0_global ) * static_cast<big_ordinal>( tensor_dim_ );
                file.write_at( offset, buf.data(), loc_size[0] * tensor_dim_ );
            }
        }

        view.release();
    }

private:
    Part part_;
    int  tensor_dim_;
};

} // namespace tests

#endif // __TESTS_SOLUTION_IO_H__
