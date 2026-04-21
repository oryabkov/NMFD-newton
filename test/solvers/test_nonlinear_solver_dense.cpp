#include <memory>
#include <cmath>

#include <scfd/utils/device_tag.h>

#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/solvers/monitor_krylov.h>
#include <nmfd/solvers/gmres.h>
#include <nmfd/backend/backend.h>
#include <nmfd/operations/matrix_operator.h>

#ifndef USE_DOUBLE_PRECISION
using scalar                               = float;
inline constexpr scalar eps                = 1e-5f;
inline constexpr scalar newton_lin_rel_tol = 1e-4f;
#else
using scalar                               = double;
inline constexpr scalar eps                = 1e-10;
inline constexpr scalar newton_lin_rel_tol = 1e-6;
#endif

namespace
{

template <class Scalar>
struct residual_kernel
{
    const Scalar *u;
    Scalar       *f;

    template <class Idx>
    __DEVICE_TAG__ void operator()( const Idx idx )
    {
        f[idx] = ( idx == Idx( 0 ) ) ? ( u[0] * u[0] + u[1] * u[1] - Scalar{ 2 } ) : ( u[0] * u[1] - Scalar{ 1 } );
    }
};

template <class Scalar, class MatrixType>
struct jacobi_2d_kernel
{
    const Scalar *u;
    MatrixType    J;

    template <class IdxND>
    __DEVICE_TAG__ void operator()( const IdxND &idx )
    {
        const auto i = idx[0];
        const auto j = idx[1];
        J( idx )     = ( i == 0 ) ? ( Scalar{ 2 } * u[j] ) : ( j == 0 ? u[1] : u[0] );
    }
};

template <class DenseOperationsBase>
struct nonlinear_dense_ops : DenseOperationsBase
{
    using DenseOperationsBase::DenseOperationsBase;
    using DenseOperationsBase::for_each_inst_;
    using DenseOperationsBase::for_each_nd_inst_;
    using DenseOperationsBase::vt_;
    using typename DenseOperationsBase::matrix_type;
    using typename DenseOperationsBase::scalar_type;
    using typename DenseOperationsBase::vector_type;

    void apply_nonlinear_residual( const vector_type &u, vector_type &f ) const
    {
        const scalar_type *pu = vt_.get_raw_ptr( u );
        scalar_type       *pf = vt_.get_raw_ptr( f );
        for_each_inst_( residual_kernel<scalar_type>{ pu, pf }, 2 );
    }

    void fill_exact_jacobian( matrix_type &J, const vector_type &u ) const
    {
        const scalar_type *pu = vt_.get_raw_ptr( u );
        for_each_nd_inst_( jacobi_2d_kernel<scalar_type, matrix_type>{ pu, J }, J.size_nd() );
    }
};

} // namespace

template <class VecSpace>
scalar eq_residual( const VecSpace &vec_sp, const typename VecSpace::vector_type &x )
{
    const scalar x0 = vec_sp.get_value_at_point( 0, x );
    const scalar x1 = vec_sp.get_value_at_point( 1, x );
    const scalar f0 = x0 * x0 + x1 * x1 - scalar( 2 );
    const scalar f1 = x0 * x1 - scalar( 1 );
    return std::sqrt( f0 * f0 + f1 * f1 );
}

int main( int argc, char const *args[] )
{
    using T            = scalar;
    using backend_type = nmfd::backend::current<T>;
    using log_t        = typename backend_type::log_type;

    using dense_ops_base_t  = typename backend_type::dense_operations_type;
    using dense_ops_t       = nonlinear_dense_ops<dense_ops_base_t>;
    using vec_space_t       = typename backend_type::vector_space_type;
    using vector_type       = typename dense_ops_t::vector_type;
    using matrix_type       = typename dense_ops_t::matrix_type;
    using matrix_operator_t = nmfd::operations::matrix_operator<dense_ops_t>;
    using monitor_t         = nmfd::solvers::monitor_krylov<vec_space_t, log_t>;
    using gmres_t           = nmfd::solvers::gmres<vec_space_t, monitor_t, log_t, matrix_operator_t>;

    struct system_op_t
    {
        using scalar_type          = T;
        using jacobi_operator_type = matrix_operator_t;

        explicit system_op_t( std::shared_ptr<dense_ops_t> ops ) : ops_( std::move( ops ) )
        {
            matrix_type M;
            ops_->init_matrix( 2, 2, M );
            ops_->assign_zero_matrix( M );
            jacobi_ = std::make_shared<matrix_operator_t>( ops_, std::move( M ) );
        }

        void apply( const vector_type &x, vector_type &f ) const
        {
            ops_->apply_nonlinear_residual( x, f );
        }

        void set_linearization_point( const vector_type &x )
        {
            ops_->fill_exact_jacobian( *jacobi_, x );
        }

        [[nodiscard]] std::shared_ptr<const matrix_operator_t> get_jacobi_operator() const
        {
            return jacobi_;
        }

        std::shared_ptr<dense_ops_t>       ops_;
        std::shared_ptr<matrix_operator_t> jacobi_;
    };

    using newton_iteration_t = nmfd::solvers::newton_iteration<vec_space_t, system_op_t, gmres_t>;
    using newton_solver_t    = nmfd::solvers::nonlinear_solver<vec_space_t, log_t, system_op_t, newton_iteration_t>;

    int          error = 0;
    backend_type backend;
    log_t       &log = backend.log();

    auto ops    = std::make_shared<dense_ops_t>( 2 );
    auto vec_sp = std::make_shared<vec_space_t>( 2 );

    {
        log.info( "test newton with backend construction" );
        newton_solver_t::params_hierarchy prm;
        prm.iteration_operator.lin_solver.monitor.rel_tol = newton_lin_rel_tol;
        std::shared_ptr<newton_solver_t> newton_solver    = std::make_shared<newton_solver_t>(
            newton_solver_t::utils_hierarchy( backend, vec_sp ), prm );
        newton_solver->convergence_strategy()->set_tolerance( eps );
        system_op_t system_op( ops );

        vector_type x;
        vec_sp->init_vector( x );
        vec_sp->start_use_vector( x );
        vec_sp->set_value_at_point( scalar( 10 ), 0, x );
        vec_sp->set_value_at_point( scalar( 2 ), 1, x );

        newton_solver->solve( &system_op, nullptr, nullptr, x );
        log.info_f(
            "result vector x: %0.15f %0.15f", vec_sp->get_value_at_point( 0, x ),
            vec_sp->get_value_at_point( 1, x )
        );
        scalar resid = eq_residual( *vec_sp, x );
        log.info_f( "residual norm: %0.15e", resid );
        if ( resid > eps )
        {
            log.error( "Failed to converge!!" );
            error++;
        }

        vec_sp->stop_use_vector( x );
        vec_sp->free_vector( x );
    }

    if ( error > 0 )
    {
        log.error_f( "Got error = %d.", error );
    }
    else
    {
        log.info( "No errors." );
    }

    return error;
}
