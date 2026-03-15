#ifndef __NMFD_DENSE_OPERATIONS_BASE_H__
#define __NMFD_DENSE_OPERATIONS_BASE_H__

#include <memory>

#include <scfd/arrays/array_nd.h>

#include <nmfd/operations/detail/scfd_array_traits.h>
#include <nmfd/operations/dense_vector_space.h>
#include <nmfd/operations/kernels/dense_operations.h>

namespace nmfd
{
namespace operations
{

template <class Type, class Backend, class Ordinal = std::ptrdiff_t>
class dense_operations
    : public dense_vector_operations<detail::scfd_array_traits<Type, typename Backend::memory_type>, Backend, Ordinal>
{
    static constexpr int Dim = 2;

    using traits_type = detail::scfd_array_traits<Type, typename Backend::memory_type>;
    using parent_t    = dense_vector_operations<traits_type, Backend, Ordinal>;

public:
    using arr_ord           = scfd::arrays::ordinal_type;
    using scalar_type       = Type;
    using vector_type       = typename traits_type::vector_type;
    using memory_type       = typename Backend::memory_type;
    using for_each_nd_type  = typename Backend::template for_each_nd_type<Dim, arr_ord>;
    using reduce_type       = typename Backend::reduce_type;
    using idx_nd_type       = scfd::static_vec::vec<arr_ord, Dim>;
    using matrix_type       = scfd::arrays::array_nd<scalar_type, Dim, memory_type>;
    using multivector_type  = typename std::vector<vector_type>;
    using vector_space_type = dense_vector_space<traits_type, Backend, Ordinal>;

    using matrix_transpose_2d_kernel     = kernels::matrix_transpose_2d<matrix_type>;
    using matrix_sum_2d_kernel           = kernels::matrix_sum_2d<scalar_type, matrix_type>;
    using matrix_sq_2d_kernel            = kernels::matrix_sq_2d<matrix_type>;
    using matrix_extract_diag_2d_kernel  = kernels::matrix_extract_diag_2d<scalar_type, matrix_type>;
    using matrix_diag_from_vec_2d_kernel = kernels::matrix_diag_from_vec_2d<scalar_type, matrix_type>;
    using matrix_scalar_diag_2d_kernel   = kernels::matrix_scalar_diag_2d<scalar_type, matrix_type>;
    using matrix_diag_extract_kernel     = kernels::matrix_diag_extract<scalar_type, matrix_type>;

public:
    dense_operations() = default;

    template <typename... Args>
    dense_operations( Args &&...args ) : parent_t( std::forward<Args>( args )... )
    {
    }

    // matrix_vector_operations

    //calc: y := alpha*mat*x + beta*y
    void add_matrix_vector_prod(
        const scalar_type alpha, const matrix_type &mat, const vector_type &x, const scalar_type beta, vector_type &y
    ) const
    {
        const auto mat_view = mat.create_view( true );
        const auto x_view   = x.create_view( true );
        auto       y_view   = y.create_view( true );

        auto sz   = mat.size_nd();
        auto rows = sz[0];
        auto cols = sz[1];

        for ( size_t i = 0; i < rows; ++i )
        {
            scalar_type sum = scalar_type{ 0 };
            for ( size_t j = 0; j < cols; ++j )
            {
                sum += mat_view( i, j ) * x_view( j );
            }
            y_view( i ) = alpha * sum + beta * y_view( i );
        }
        y_view.release( true );
    }
    //calc: z := alpha*mat*x + beta*y
    void assign_matrix_vector_prod(
        const scalar_type alpha, const matrix_type &mat, const vector_type &x, const scalar_type beta,
        const vector_type &y, vector_type &z
    ) const
    {
        parent_t::assign( y, z );
        add_matrix_vector_prod( alpha, mat, x, beta, z );
    }

    [[nodiscard]] std::shared_ptr<vector_space_type> get_matrix_im_space( const matrix_type &mat ) const
    {
        auto rows = mat.size_nd()[0];
        return std::make_shared<vector_space_type>( static_cast<size_t>( rows ) );
    }

    [[nodiscard]] std::shared_ptr<vector_space_type> get_matrix_dom_space( const matrix_type &mat ) const
    {
        auto cols = mat.size_nd()[1];
        return std::make_shared<vector_space_type>( static_cast<size_t>( cols ) );
    }


    // matrix_operations

    /// C = A^T
    [[nodiscard]] std::shared_ptr<matrix_type> matrix_transpose( const matrix_type &mat ) const
    {
        auto sz   = mat.size_nd();
        auto rows = sz[0];
        auto cols = sz[1];

        auto result = std::make_shared<matrix_type>();
        result->init( cols, rows );

        for_each_nd_inst_( matrix_transpose_2d_kernel{ mat, *result }, mat.size_nd() );
        return result;
    }

    /// C = A * B
    [[nodiscard]] std::shared_ptr<matrix_type>
    matrix_matrix_prod( const matrix_type &mat_a, const matrix_type &mat_b ) const
    {
        auto sz_a = mat_a.size_nd();
        auto sz_b = mat_b.size_nd();
        auto m    = sz_a[0];
        auto k    = sz_a[1];
        auto n    = sz_b[1];

        auto result = std::make_shared<matrix_type>();
        result->init( m, n );

        const auto a_view = mat_a.create_view( true );
        const auto b_view = mat_b.create_view( true );
        auto       c_view = result->create_view( false );

        for ( size_t i = 0; i < m; ++i )
        {
            for ( size_t j = 0; j < n; ++j )
            {
                scalar_type sum = scalar_type{ 0 };
                for ( size_t p = 0; p < k; ++p )
                    sum += a_view( i, p ) * b_view( p, j );
                c_view( i, j ) = sum;
            }
        }

        c_view.release( true );
        return result;
    }

    /// C = alpha * A + beta * B
    [[nodiscard]] std::shared_ptr<matrix_type> matrix_matrix_sum(
        const scalar_type alpha, const matrix_type &mat_a, const scalar_type beta, const matrix_type &mat_b
    ) const
    {
        auto sz   = mat_a.size_nd();
        auto rows = sz[0];
        auto cols = sz[1];

        auto result = std::make_shared<matrix_type>();
        result->init( rows, cols );

        for_each_nd_inst_( matrix_sum_2d_kernel{ alpha, mat_a, beta, mat_b, *result }, mat_a.size_nd() );
        return result;
    }

    /// ||A||_F = sqrt( sum_{i,j} A(i,j)^2 )
    [[nodiscard]] scalar_type matrix_norm_fro( const matrix_type &mat ) const
    {
        auto sz    = mat.size_nd();
        auto total = sz[0] * sz[1];

        verify_helper_matrix_size( sz[0], sz[1] );

        for_each_nd_inst_( matrix_sq_2d_kernel{ mat, mat_helper_ }, mat.size_nd() );
        return std::sqrt( parent_t::reduce_inst_( total, mat_helper_.raw_ptr(), scalar_type{ 0 } ) );
    }

    /// Returns diagonal matrix: D(i,i) = mat(i,i), optionally inverted
    [[nodiscard]] std::shared_ptr<matrix_type> matrix_diag( const matrix_type &mat, bool invert = false ) const
    {
        auto sz   = mat.size_nd();
        auto rows = sz[0];
        auto cols = sz[1];

        auto result = std::make_shared<matrix_type>();
        result->init( rows, cols );

        for_each_nd_inst_( matrix_extract_diag_2d_kernel{ mat, *result, invert }, mat.size_nd() );
        return result;
    }

    /// Returns diagonal of the matrix as vector (vector must already be allocated and have corresponding partitioning)
    void matrix_diag( const matrix_type &mat, vector_type &x, bool invert = false ) const
    {
        auto sz      = mat.size_nd();
        auto cols    = sz[1];
        auto min_dim = sz[0] < cols ? sz[0] : cols;

        parent_t::for_each_inst_( matrix_diag_extract_kernel{ mat, parent_t::vt_.get_raw_ptr( x ), invert }, min_dim );
    }
    /// Creates a matrix with diagonal structure and values on its diagonal from vector x
    [[nodiscard]] std::shared_ptr<matrix_type> diag_matrix_from_vector( const vector_type &x ) const
    {
        auto n = static_cast<arr_ord>( parent_t::get_loc_size( x ) );

        auto result = std::make_shared<matrix_type>();
        result->init( n, n );

        for_each_nd_inst_(
            matrix_diag_from_vec_2d_kernel{ parent_t::vt_.get_raw_ptr( x ), *result }, result->size_nd()
        );
        return result;
    }

    /// Creates a matrix with diagonal structure and scalar value on its diagonal val
    /// Parallel structure defined by vector x
    /// TODO make something better then explicit vector!!
    [[nodiscard]] std::shared_ptr<matrix_type> scalar_matrix( const vector_type &x, scalar_type val ) const
    {
        auto n = static_cast<arr_ord>( parent_t::get_loc_size( x ) );

        auto result = std::make_shared<matrix_type>();
        result->init( n, n );

        for_each_nd_inst_( matrix_scalar_diag_2d_kernel{ val, *result }, result->size_nd() );
        return result;
    }


    void write_matrix_to_mm_file( const std::string &file_name, const matrix_type &mat ) const
    {
        SCFD_TODO( "Implement write_matrix_to_mm_file" );
    }

    void write_matrix_to_mm_file( const std::string &file_name, std::shared_ptr<matrix_type> mat ) const
    {
        SCFD_TODO( "Implement write_matrix_to_mm_file" );
    }

private:
    void verify_helper_matrix_size( arr_ord rows, arr_ord cols ) const
    {
        if ( !mat_helper_.is_free() && mat_helper_.size() > rows * cols )
        {
            return;
        }
        if ( !mat_helper_.is_free() )
        {
            mat_helper_.free();
        }
        mat_helper_.init( rows, cols );
    }

    mutable matrix_type      mat_helper_;
    mutable for_each_nd_type for_each_nd_inst_;
};

}

}


// TODO: rework dense_operations_base.

/*

#include <cmath>
#include <utility>
#include <iostream>
#include <initializer_list>
#include <iomanip>
#include <memory>
#include <random>

#include <scfd/memory/host.h>
#include <scfd/arrays/tensor_array_nd.h>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/arrays/last_index_fast_arranger.h>

namespace nmfd
{
namespace operations
{

template <
    class Card, class Type, class DenseVectorType = scfd::arrays::tensor0_array_nd<Type, 1, scfd::memory::host>,
    class DenseMatrixType = scfd::arrays::tensor0_array_nd<Type, 2, scfd::memory::host>>
class dense_operations_base
{
private:
    using T            = Type;
    mutable Card rows_ = 0, cols_ = 0;

    void check_init() const
    {
        if ( ( rows_ == 0 ) && ( cols_ == 0 ) )
        {

            throw std::logic_error( "nmfd:lin_solvers:detail:dense_operations_base:: methods called withought class "
                                    "delayed initialization." );
        }
    }

public:
    using scalar_type = Type;
    using vector_type = DenseVectorType;
    using matrix_type = DenseMatrixType;

private:
    std::random_device                       rd;
    mutable std::mt19937                     gen;
    mutable std::uniform_real_distribution<> dis;

public:
    dense_operations_base( Card rows, Card cols ) : rows_( rows ), cols_( cols )
    {
        common_constructor_operations();
    }

    dense_operations_base() : rows_( 0 ), cols_( 0 )
    {
        common_constructor_operations();
    }

    void common_constructor_operations()
    {
        gen = std::mt19937( rd() );
        dis = std::uniform_real_distribution<>( -1.0, 1.0 );
    }

    ~dense_operations_base() = default;

    void init( Card rows, Card cols ) const
    {
        rows_ = rows;
        cols_ = cols;
    }


    void init_row_vector( vector_type &vec ) const
    {
        check_init();
        vec.init( cols_ );
    }
    template <class... Args>
    void init_row_vectors( Args &&...args ) const
    {
        check_init();
        std::initializer_list<int>{ ( (void)init_row_vector( std::forward<Args>( args ) ), 0 )... };
    }

    void init_col_vector( vector_type &vec ) const
    {
        check_init();
        vec.init( rows_ );
    }
    template <class... Args>
    void init_col_vectors( Args &&...args ) const
    {
        check_init();
        std::initializer_list<int>{ ( (void)init_col_vector( std::forward<Args>( args ) ), 0 )... };
    }

    void init_matrix( matrix_type &mat ) const
    {
        check_init();
        mat.init( rows_, cols_ );
    }
    template <class... Args>
    void init_matrices( Args &&...args ) const
    {
        check_init();
        std::initializer_list<int>{ ( (void)init_matrix( std::forward<Args>( args ) ), 0 )... };
    }
    void free_row_vector( vector_type &vec ) const
    {
        vec.free();
    }
    template <class... Args>
    void free_row_vectors( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)free_row_vector( std::forward<Args>( args ) ), 0 )... };
    }
    void free_col_vector( vector_type &vec ) const
    {
        vec.free();
    }
    template <class... Args>
    void free_col_vectors( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)free_col_vector( std::forward<Args>( args ) ), 0 )... };
    }
    void free_matrix( matrix_type &mat ) const
    {
        mat.free();
    }
    template <class... Args>
    void free_matrices( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)free_matrix( std::forward<Args>( args ) ), 0 )... };
    }
    std::pair<Card, Card> size() const
    {
        return { rows_, cols_ };
    }
    bool is_valid_row_vector( const vector_type &vec ) const
    {
        check_init();
        bool res = true;
        for ( Card j = 0; j < cols_; j++ )
        {
            if ( !std::isfinite( vec( j ) ) )
            {
                res = false;
                break;
            }
        }
        return res;
    }
    bool is_valid_col_vector( const vector_type &vec ) const
    {
        check_init();
        bool res = true;
        for ( Card j = 0; j < rows_; j++ )
        {
            if ( !std::isfinite( vec( j ) ) )
            {
                res = false;
                break;
            }
        }
        return res;
    }
    bool is_valid_matrix( const matrix_type &mat ) const
    {
        check_init();
        bool res = true;
        for ( Card j = 0; j < rows_; j++ )
        {
            for ( Card k = 0; k < cols_; k++ )
            {
                if ( !std::isfinite( mat( j, k ) ) )
                {
                    res = false;
                    break;
                }
            }
            if ( !res )
            {
                break;
            }
        }
        return res;
    }
    void assign_scalar_row_vector( const scalar_type scalar, vector_type &vec ) const
    {
        check_init();
        for ( Card j = 0; j < cols_; j++ )
        {
            vec( j ) = scalar;
        }
    }
    void assign_scalar_col_vector( const scalar_type scalar, vector_type &vec )
    {
        check_init();
        for ( Card j = 0; j < rows_; j++ )
        {
            vec( j ) = scalar;
        }
    }
    void assign_scalar_matrix( const scalar_type scalar, matrix_type &mat ) const
    {
        check_init();
        for ( Card j = 0; j < rows_; j++ )
        {
            for ( Card k = 0; k < cols_; k++ )
            {
                mat( j, k ) = scalar;
            }
        }
    }
    void assign_row_vector( const vector_type &x, vector_type &y ) const
    {
        check_init();
        for ( Card j = 0; j < cols_; j++ )
        {
            y( j ) = x( j );
        }
    }
    void assign_col_vector( const vector_type &x, vector_type &y ) const
    {
        check_init();
        for ( Card j = 0; j < rows_; j++ )
        {
            y( j ) = x( j );
        }
    }
    void assign_matrix( const matrix_type &A, matrix_type &B ) const
    {
        check_init();
        for ( Card j = 0; j < rows_; j++ )
        {
            for ( Card k = 0; k < cols_; k++ )
            {
                B( j, k ) = A( j, k );
            }
        }
    }
    Type &matrix_at( const matrix_type &A, const Card row, const Card col )
    {
        return A( row, col );
    }
    Type &vector_at( const vector_type &x, const Card j )
    {
        return x( j );
    }

    void matrix_set_column( const vector_type &col_vec, const Card col, matrix_type &mat ) const
    {
        check_init();
        for ( Card j = 0; j < rows_; j++ )
        {
            mat( j, col ) = col_vec( j );
        }
    }
    void matrix_set_row( const vector_type &row_vec, const Card row, matrix_type &mat ) const
    {
        check_init();
        for ( Card j = 0; j < cols_; j++ )
        {
            mat( row, j ) = row_vec( j );
        }
    }


    void set_random_row_vector( vector_type &vec, const T &from = 0, const T &to = 1 ) const
    {
        for ( Card j = 0; j < rows_; j++ )
        {
            T val    = dis( gen );
            val      = ( to - from ) * ( 0.5 * ( val + 1 ) ) + from;
            vec( j ) = val;
        }
    }
    void set_random_col_vector( vector_type &vec, const T &from = 0, const T &to = 1 ) const
    {
        for ( Card j = 0; j < cols_; j++ )
        {
            T val    = dis( gen );
            val      = ( to - from ) * ( 0.5 * ( val + 1 ) ) + from;
            vec( j ) = val;
        }
    }
    void set_random_matrix( matrix_type &mat, const T &from = 0, const T &to = 1 ) const
    {
        for ( Card j = 0; j < rows_; j++ )
        {
            for ( Card k = 0; k < cols_; k++ )
            {
                T val       = dis( gen );
                val         = ( to - from ) * ( 0.5 * ( val + 1 ) ) + from;
                mat( j, k ) = val;
            }
        }
    }

    T norm_col_vector( const vector_type &vec ) const
    {
        T norm = 0;
        for ( Card j = 0; j < cols_; j++ )
        {
            norm += vec( j ) * vec( j );
        }
        return std::sqrt( norm );
    }

    T norm_row_vector( const vector_type &vec ) const
    {
        T norm = 0;
        for ( Card j = 0; j < rows_; j++ )
        {
            norm += vec( j ) * vec( j );
        }
        return std::sqrt( norm );
    }

    void solve_upper_triangular_subsystem( const matrix_type &A, vector_type &x, const Card ind ) const
    {
        for ( Card j = ind; j-- > 0; )
        {
            x( j ) /= A( j, j );
            for ( Card k = 0; k < j; ++k )
            {
                x( k ) -= A( k, j ) * x( j );
            }
        }
    }
    void
    solve_upper_triangular_subsystem( const matrix_type &A, const vector_type &b, vector_type &x, const Card ind ) const
    {
        if ( ind > 0 )
        {
            assign_col_vector( b, x );
            solve_upper_triangular_subsystem( A, x, ind );
        }
    }

    void apply_plane_rotation( scalar_type &dx, scalar_type &dy, const scalar_type &cs, const scalar_type &sn ) const
    {
        T temp = cs * dx + sn * dy;
        dy     = -sn * dx + cs * dy;
        dx     = temp;
    }
    void generate_plane_rotation( const scalar_type &dx, const scalar_type &dy, scalar_type &cs, scalar_type &sn ) const
    {
        if ( dy == static_cast<T>( 0 ) )
        {
            cs = static_cast<T>( 1 );
            sn = static_cast<T>( 0 );
        }
        else if ( std::abs( dy ) > std::abs( dx ) )
        {
            T tmp = dx / dy;
            sn    = static_cast<T>( 1 ) / std::sqrt( static_cast<T>( 1 ) + tmp * tmp );
            cs    = tmp * sn;
        }
        else
        {
            T tmp = dy / dx;
            cs    = static_cast<T>( 1 ) / std::sqrt( static_cast<T>( 1 ) + tmp * tmp );
            sn    = tmp * cs;
        }
    }
    void plane_rotation_col( matrix_type &H, vector_type &cs_, vector_type &sn_, vector_type &s, const Card col ) const
    {
        for ( int k = 0; k < col; k++ )
        {
            apply_plane_rotation( H( k, col ), H( k + 1, col ), cs_( k ), sn_( k ) );
        }

        generate_plane_rotation( H( col, col ), H( col + 1, col ), cs_( col ), sn_( col ) );
        apply_plane_rotation( H( col, col ), H( col + 1, col ), cs_( col ), sn_( col ) );
        H( col + 1, col ) = static_cast<T>( 0 );
        apply_plane_rotation( s( col ), s( col + 1 ), cs_( col ), sn_( col ) );
    }

    void print_col_vector( const vector_type &vec, int prec = 2 )
    {
        if ( prec > 2 )
            std::cout << std::setprecision( prec ) << std::scientific;
        else
            std::cout << std::setprecision( 2 ) << std::fixed;
        for ( Card j = 0; j < rows_; j++ )
        {
            std::cout << vec( j ) << std::endl;
        }
    }
    void print_row_vector( const vector_type &vec, int prec = 2 )
    {
        if ( prec > 2 )
            std::cout << std::setprecision( prec ) << std::scientific;
        else
            std::cout << std::setprecision( 2 ) << std::fixed;

        for ( Card j = 0; j < cols_; j++ )
        {
            std::cout << vec( j ) << " ";
        }
    }
    void print_matrix( const matrix_type &H, int prec = 2 )
    {
        {
            if ( prec > 2 )
                std::cout << std::setprecision( prec ) << std::scientific;
            else
                std::cout << std::setprecision( 2 ) << std::fixed;

            for ( Card j = 0; j < rows_; j++ )
            {
                for ( Card k = 0; k < cols_; k++ )
                {
                    std::cout << H( j, k ) << " ";
                }
                std::cout << std::endl;
            }
        }
    }
};

} // namespace operations
} // namespace nmfd

*/

#endif
