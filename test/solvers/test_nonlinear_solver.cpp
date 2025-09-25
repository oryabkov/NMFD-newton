#include <memory>
#include <cmath>
#include <scfd/utils/log.h>
#include <scfd/static_vec/vec.h>
#include <scfd/static_mat/mat.h>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/solvers/newton_iteration.h>


int main(int argc, char const *args[])
{
    using log_t = scfd::utils::log_std;
    using T = double;
    static const int dim = 2;
    using vec_t = scfd::static_vec::vec<T,dim>;
    using mat_t = scfd::static_mat::mat<T,dim,dim>;
    //using T_mvec = double*;
    using vec_sp_t = nmfd::operations::static_vector_space<T, dim, vec_t>;
    struct linsolver_t
    {
        using vector_type = vec_t;
        using operator_type = mat_t;

        linsolver_t() = default;
        NMFD_ALGO_ALL_EMPTY_DEFINE(linsolver_t)
        linsolver_t(  
            const utils_hierarchy& utils,
            const params_hierarchy& prm = params_hierarchy()      
        )
        {
        }

        mat_t a_inv;
        void set_operator(const std::shared_ptr<const mat_t> &a)
        {
            a_inv = inv(*a);
        }
        bool solve(const vec_t &rhs, vec_t &x)const
        {
            x = a_inv*rhs;
            return true;
        }
    };
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
    log_t log;
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
        newton_solver_t::utils_hierarchy u_h{
            {{},vec_sp},    /// iteration_operator
            {vec_sp,&log},  /// convergence_strategy
            vec_sp,&log     /// newton utils itself
        };
        std::shared_ptr<newton_solver_t> newton_solver = std::make_shared<newton_solver_t>(u_h);
        system_op_t system_op;
        vec_t x(10.,2.);
        newton_solver->solve(&system_op, nullptr, nullptr, x);
        log.info_f("result vector x: %f %f", x[0], x[1]);
    }

    //testing left and right preconditioners
    /*{
        std::size_t N = N_with_preconds;
        vec_ops = std::make_shared<vec_ops_t>(N);
        auto prec_diff = std::make_shared<prec_diff_t>(vec_ops, 15);
        auto prec_adv = std::make_shared<prec_adv_t>(vec_ops, 1);

        T tau = 1.0;
        T a = 1.0;
        T_vec x,y,resid;
        vec_ops->init_vector(x);
        vec_ops->init_vector(y);
        vec_ops->start_use_vector(x);
        vec_ops->start_use_vector(y);

        log.info_f("=>diffusion with size %i, timestep %.02f.", vec_ops->size(), tau );
        auto lin_op_diff = std::make_shared<lin_op_diff_t>(*vec_ops, tau); //with time step 5


        for(int j=0;j<N;j++)
        {
            y[j] = std::sin(1.0*j/(N-1)*M_PIl);
            // x[j] = 0.1*std::sin(1.0*j/(N-1)*M_PIl);
        }
        y[0] = y[N-1] = 0;

        gmres_diff_t::params params_diff;
        params_diff.monitor.rel_tol = 1.0e-10;
        params_diff.monitor.max_iters_num = 300;
        params_diff.basis_size = 25;
        {
            log.info("left preconditioner");
            params_diff.preconditioner_side = 'L';
            gmres_diff_t gmres(lin_op_diff, vec_ops, &log, params_diff, prec_diff);

            vec_ops->assign_scalar(0.0, x);
            bool res = gmres.solve(y, x);
            error += (!res);

            log.info_f("pLgmres res: %s", res?"true":"false");
            log.info(" reusing the solution...");
            vec_ops->add_mul_scalar(0.0, 0.99999, x);
            res = gmres.solve(y, x);
            error += (!res);
            log.info_f("pLgmres res with x0: %s", res?"true":"false");
            get_residual(*lin_op_diff, x, y);
        }
        {
            log.info("right preconditioner");
            params_diff.preconditioner_side = 'R';
            gmres_diff_t gmres(lin_op_diff, vec_ops, &log, params_diff, prec_diff);

            vec_ops->assign_scalar(0.0, x);
            bool res = gmres.solve(y, x);
            error += (!res);
            log.info_f("pRgmres res: %s", res?"true":"false");
            log.info(" reusing the solution...");
            vec_ops->add_mul_scalar(0.0, 0.99999, x);
            res = gmres.solve(y, x);
            error += (!res);
            log.info_f("pRgmres res with x0: %s", res?"true":"false");
            get_residual(*lin_op_diff, x, y);
        }

        log.info_f("=>advection with size %i, speed %.02f, timestep %.02f.", vec_ops->size(), a, tau );
        gmres_adv_t::params params_adv;
        params_adv.monitor.rel_tol = 1.0e-10;
        params_adv.monitor.max_iters_num = 300;
        params_adv.basis_size = 15;
        auto lin_op_adv = std::make_shared<lin_op_adv_t>(*vec_ops, a, tau);
        {
            gmres_adv_t gmres(lin_op_adv, vec_ops, &log, params_adv, prec_adv);
            log.info("left preconditioner");
            params_adv.preconditioner_side = 'L';
            vec_ops->assign_scalar(0.0, x);
            bool res = gmres.solve(y, x);
            error += (!res);
            log.info_f("pLgmres res: %s", res?"true":"false");
            log.info("reusing the solution...");
            vec_ops->add_mul_scalar(0.0, 0.9999999, x);
            res = gmres.solve(y, x);
            error += (!res);
            log.info_f("pLgmres res with x0: %s", res?"true":"false");
            get_residual(*lin_op_adv, x, y);
        }
        {
            gmres_adv_t gmres(lin_op_adv, vec_ops, &log, params_adv, prec_adv);
            log.info("left preconditioner");
            params_adv.preconditioner_side = 'R';
            vec_ops->assign_scalar(0.0, x);
            bool res = gmres.solve(y, x);
            error += (!res);
            log.info_f("pRgmres res: %s", res?"true":"false");
            log.info("reusing the solution...");
            vec_ops->add_mul_scalar(0.0, 0.9999999, x);
            res = gmres.solve(y, x);
            error += (!res);
            log.info_f("pRgmres res with x0: %s", res?"true":"false");
            get_residual(*lin_op_adv, x, y);
            // for(int j=0;j<N;j++)
            // {
            //     std::cout << (1.0*j+0.5)/N << "," << x[j] << std::endl;
            // }

        }

        vec_ops->stop_use_vector(x);
        vec_ops->stop_use_vector(y);
        vec_ops->free_vector(x);
        vec_ops->free_vector(y);
    }*/



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
