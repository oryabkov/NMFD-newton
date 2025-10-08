// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of NMFD.

// NMFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// NMFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with NMFD.  If not, see <http://www.gnu.org/licenses/>.

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
#include <nmfd/detail/algo_hierarchy_macro.h>
#include <nmfd/detail/algo_hierarchy_creator.h>
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
    using T = typename VectorSpace::scalar_type;
    using logged_obj_t = scfd::utils::logged_obj_base<Log>;
    using logged_obj_params_t = typename logged_obj_t::params;
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;

    struct params : public logged_obj_t::params
    {
        params(
            const std::string &log_pefix = "", const std::string &log_name = "nonlinear_solver::"
        ) : logged_obj_t::params(0, log_pefix + log_name)
        {
        }
        /// TODO add json
    };
    struct utils
    {
        std::shared_ptr<VectorSpace> vec_space;
        Log *log;
        utils() = default;
        utils(
            std::shared_ptr<VectorSpace> vec_space_, Log *log_ = nullptr
        ) : 
            vec_space(vec_space_), log(log_)
        {
        }
        template<class Backend>
        utils(Backend &backend, std::shared_ptr<VectorSpace> vec_space) : utils(vec_space, &backend.log())
        {
        }
    };
    NMFD_ALGO_HIERARCHY_TYPES_DEFINE(nonlinear_solver,IterationOperator,iteration_operator,ConvergenceStrategy,convergence_strategy)

    nonlinear_solver(
        std::shared_ptr<VectorSpace> vec_ops, Log *log,
        std::shared_ptr<IterationOperator> iter_op, 
        std::shared_ptr<ConvergenceStrategy> conv_strat = nullptr
    ) :
      vec_ops_(std::move(vec_ops)),
      iter_op_(std::move(iter_op)),
      conv_strat_(std::move(conv_strat)),
      delta_x_(*vec_ops_)
    {
        if (!conv_strat_)
        {
            /// TODO this only should work for default convergence strategy - make creator!
            conv_strat_ = std::make_shared<ConvergenceStrategy>(vec_ops_,log);
        }
        //vec_ops_->init_vector(delta_x_); vec_ops_->start_use_vector(delta_x_);
    }
    nonlinear_solver(  
        const utils_hierarchy& utils,
        const params_hierarchy& prm = params_hierarchy()      
    ) : 
        nonlinear_solver(  
            utils.vec_space, utils.log, 
            nmfd::detail::algo_hierarchy_creator<IterationOperator>::get(utils.iteration_operator,prm.iteration_operator),
            nmfd::detail::algo_hierarchy_creator<ConvergenceStrategy>::get(utils.convergence_strategy,prm.convergence_strategy)
        )
    {
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
        if(conv_strat_->get_result_status()==0)
        {
            converged = true;
        }
        if( (conv_strat_->get_result_status() == 2)||(conv_strat_->get_result_status() == 3) ) //inf or nan
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
        return conv_strat_->get();
    }

    const std::shared_ptr<ConvergenceStrategy> &convergence_strategy()
    {
        return conv_strat_;
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