#ifndef __TESTS_SOLUTION_IO_H__
#define __TESTS_SOLUTION_IO_H__

#include <fstream>
#include <string>

namespace tests
{

/**
 * Saves a 3D tensor field solution to a binary file.
 * Format: [dims (3x int32)] [n_components (int32)] [data (doubles)]
 */
template <class Vector, class IdxType>
void save_solution_binary( const Vector &vec, const std::string &filename, int grid_size, int tensor_dim )
{
    std::ofstream out( filename, std::ios::binary );

    // Write header: dimensions and number of components
    int32_t dims[3]      = { grid_size, grid_size, grid_size };
    int32_t n_components = tensor_dim;
    out.write( reinterpret_cast<const char *>( dims ), sizeof( dims ) );
    out.write( reinterpret_cast<const char *>( &n_components ), sizeof( n_components ) );

    // Write data for each grid point
    for ( int k = 0; k < grid_size; ++k )
    {
        for ( int j = 0; j < grid_size; ++j )
        {
            for ( int i = 0; i < grid_size; ++i )
            {
                IdxType idx{ i, j, k };
                auto    v = vec.get_vec( idx );
                for ( int t = 0; t < tensor_dim; ++t )
                {
                    double val = static_cast<double>( v[t] );
                    out.write( reinterpret_cast<const char *>( &val ), sizeof( val ) );
                }
            }
        }
    }
    out.close();
}

} // namespace tests

#endif // __TESTS_SOLUTION_IO_H__
