#include <memory>
#include <cmath>
#include <scfd/utils/log.h>
#include <scfd/static_vec/vec.h>
#include <scfd/static_mat/mat.h>
#include <nmfd/backend/single_node_cpu.h>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/solvers/newton_iteration.h>

template<class Mat,class VectorSpace>
struct linsolver
{
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;
    using operator_type = Mat;

    linsolver() = default;
    NMFD_ALGO_ALL_EMPTY_DEFINE(linsolver)
    

    Mat a_inv;
    void set_operator(const std::shared_ptr<const Mat> &a)
    {
        a_inv = inv(*a);
    }
    bool solve(const vector_type &rhs, vector_type &x)const
    {
        x = a_inv*rhs;
        return true;
    }
};

int main(int argc, char const *args[])
{
    using backend_t = nmfd::backend::single_node_cpu<>;
    using log_t = backend_t::log_type;
    using T = double;
    static const int dim = 2;
    using vec_t = scfd::static_vec::vec<T,dim>;
    using mat_t = scfd::static_mat::mat<T,dim,dim>;
    //using T_mvec = double*;
    using vec_sp_t = nmfd::operations::static_vector_space<T, dim, vec_t>;
    using linsolver_t = linsolver<mat_t,vec_sp_t>;
    struct system_op_t
    {
        using scalar_type = T;
        using vector_type = vec_t;
        using jacobi_operator_type = mat_t;

        system_op_t() : jacobi(std::make_shared<mat_t>())
        {
        }

        void apply(const vec_t &x, vec_t &f)const
        {
            f[0] = x[0]*x[0] + x[1]*x[1] - 2.;
            f[1] = x[0]*x[1] - 1.;
        }

        void set_linearization_point(const vec_t &x)
        {
            (*jacobi)(0,0) = 2*x[0]; (*jacobi)(0,1) = 2*x[1];
            (*jacobi)(1,0) =   x[1]; (*jacobi)(1,1) =   x[0];
        }
        std::shared_ptr<const mat_t> get_jacobi_operator()const
        {
            return jacobi;
        }

        std::shared_ptr<mat_t> jacobi;
    };
    using newton_iteration_t = nmfd::solvers::newton_iteration<vec_sp_t,system_op_t,linsolver_t>;
    using newton_solver_t = nmfd::solvers::nonlinear_solver<vec_sp_t, log_t, system_op_t, newton_iteration_t>;



    int error = 0;
    backend_t backend;
    log_t &log = backend.log();
    std::shared_ptr<vec_sp_t> vec_sp = std::make_shared<vec_sp_t>();
    {
        log.info("test newton with manual construction");
        std::shared_ptr<linsolver_t> lin_solver = std::make_shared<linsolver_t>();
        std::shared_ptr<newton_iteration_t> newton_iteration = std::make_shared<newton_iteration_t>(vec_sp, lin_solver);
        std::shared_ptr<newton_solver_t> newton_solver = std::make_shared<newton_solver_t>(vec_sp, &log, newton_iteration);
        system_op_t system_op;
        vec_t x(10.,2.);
        newton_solver->solve(&system_op, nullptr, nullptr, x);
        log.info_f("result vector x: %f %f", x[0], x[1]);
    }

    {
        log.info("test newton with hierarchic construction");
        std::shared_ptr<newton_solver_t> newton_solver = 
            std::make_shared<newton_solver_t>(
                newton_solver_t::utils_hierarchy{
                    {{},vec_sp},    /// iteration_operator
                    {vec_sp,&log},  /// convergence_strategy
                    vec_sp,&log     /// newton utils itself
                }
            );
        system_op_t system_op;
        vec_t x(10.,2.);
        newton_solver->solve(&system_op, nullptr, nullptr, x);
        log.info_f("result vector x: %f %f", x[0], x[1]);
    }

    {
        log.info("test newton with backend construction");
        std::shared_ptr<newton_solver_t> newton_solver = 
            std::make_shared<newton_solver_t>(
                newton_solver_t::utils_hierarchy(backend,vec_sp)
            );
        system_op_t system_op;
        vec_t x(10.,2.);
        newton_solver->solve(&system_op, nullptr, nullptr, x);
        log.info_f("result vector x: %f %f", x[0], x[1]);
    }



    if(error > 0)
    {
        log.error_f("Got error = %e.", error ) ;
    }
    else
    {
        log.info("No errors.") ;
    }

    return error;
}
