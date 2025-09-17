#ifndef __NMFD_NONLINEAR_SOLVER_H__
#define __NMFD_NONLINEAR_SOLVER_H__

/**
 * General form for nonlinear iterative solver.
 * Depending on particular IterationOperator may be Newton solver, Picard simple iteration solver,
 * arc-length continuation solver in extended (x,lambda) space or something problem specific.
 **/

/// NOTE originally taken from deflated_continuation master branch source/numerical_algos/newton_solvers/newton_solver_extended.h 22.07.2025

#include <string>
#include <memory>
#include <stdexcept>
#include <nmfd/operations/ident_operator.h>
#include <nmfd/operations/zero_functional.h>
#include "../detail/str_source_helper.h"
#include "../detail/vector_wrap.h"
#include "default_convergence_strategy.h"


namespace nmfd
{
namespace solvers
{

template
<
    class VectorSpace, 
    class Log, 
    class NonlinearOperator, 
    class IterationOperator, 
    class ProjectOperator = operations::ident_operator<VectorSpace>, 
    class QualityFunctor = operations::zero_functional<VectorSpace>,
    class ConvergenceStrategy = default_convergence_strategy<VectorSpace, Log, NonlinearOperator, ProjectOperator, QualityFunctor>
>
class nonlinear_solver
{
    typedef typename VectorSpace::scalar_type  T;
public:
    typedef typename VectorSpace::scalar_type  scalar_type;
    typedef typename VectorSpace::vector_type  vector_type;

    nonlinear_solver(
        std::shared_ptr<VectorSpace> vec_ops, 
        std::shared_ptr<IterationOperator> iter_op, 
        std::shared_ptr<ConvergenceStrategy> conv_strat
    ) :
      vec_ops_(std::move(vec_ops)),
      iter_op_(std::move(iter_op)),
      conv_strat_(std::move(conv_strat)),
      delta_x_(*vec_ops_)
    {
        //vec_ops_->init_vector(delta_x_); vec_ops_->start_use_vector(delta_x_);
    }
    
    ~nonlinear_solver()
    {
        //vec_ops_->stop_use_vector(delta_x_); vec_ops_->free_vector(delta_x_); 
    }

    //inplace
    bool solve(NonlinearOperator *nonlin_op, ProjectOperator *project_op, QualityFunctor *quality_func, vector_type& x)
    {
        vec_ops_->assign_scalar(T(0.0), *delta_x_);
        bool converged = false;
        conv_strat_->reset_iterations(); //reset iteration count, newton wight and iteration history
        while(!conv_strat_->check_convergence(nonlin_op, project_op, quality_func, x, *delta_x_))
        {
            //reset iterational vectors??!
            vec_ops_->assign_scalar(T(0.0), *delta_x_);     

            bool linsolver_converged = iter_op_->solve(*nonlin_op, x, *delta_x_);

            /// TODO some how react to non-converged linsolver maybe??
            /// Think to add parameter to calibrate this behaviour
        }
        if(result_status==0)
        {
            converged = true;
        }
        if( (result_status == 2)||(result_status == 3) ) //inf or nan
        {
            throw std::runtime_error(std::string("nonlinear_solver: " __FILE__ " " __STR(__LINE__) " invalid number returned from update.") );            
        }

        return converged;
    }

    /// ISSUE i would rather have ability to add vector as sort of RHS instead of outofplace solver
    /// Do we actually need it??
    bool solve(NonlinearOperator *nonlin_op, ProjectOperator *project_op, QualityFunctor *quality_func, const vector_type& x0, vector_type& x)
    {
        vec_ops_->assign(x0, x);
        bool converged = false;
        converged = solve(nonlin_op, project_op, quality_func, x);
        if(!converged)
        {
            vec_ops_->assign(x0, x);
        }
        return converged;
    }   

    ///ISSUE???
    ConvergenceStrategy* get_convergence_strategy_handle()
    {
        return conv_strat->get();
    }

    const std::shared_ptr<ConvergenceStrategy> &convergence_strategy()
    {
        return conv_strat;
    }

private:
    using vec_wrap_t = detail::vector_wrap<VectorSpace,true,true>;

    std::shared_ptr<VectorSpace> vec_ops_;
    std::shared_ptr<IterationOperator> iter_op_;
    std::shared_ptr<ConvergenceStrategy> conv_strat_;
    vec_wrap_t delta_x_;


};



}
}

#endif