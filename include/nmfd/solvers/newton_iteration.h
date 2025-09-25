#ifndef __NMFD_NEWTON_ITERATION_H__
#define __NMFD_NEWTON_ITERATION_H__

#include <nmfd/detail/algo_hierarchy_macro.h>
#include <nmfd/detail/algo_hierarchy_creator.h>

/// NOTE originally taken from deflated_continuation master branch source/deflation/system_operator_deflation.h 22.07.2025

namespace nmfd
{
namespace solvers
{

template<class VectorSpace, class NonlinearOperator, class LinearSolver>
class newton_iteration
{
    typedef typename VectorSpace::scalar_type  T;
public:
    typedef typename VectorSpace::scalar_type  scalar_type;
    typedef typename VectorSpace::vector_type  vector_type;
    typedef typename NonlinearOperator::jacobi_operator_type  linear_operator;
    
    NMFD_ALGO_EMPTY_PARAMS_TYPE_DEFINE(newton_iteration)
    struct utils
    {
        std::shared_ptr<VectorSpace> vec_space;
        utils() = default;
        utils(
            std::shared_ptr<VectorSpace> vec_space_
        ) : 
            vec_space(vec_space_)
        {
        }
    };
    NMFD_ALGO_HIERARCHY_TYPES_DEFINE(newton_iteration,LinearSolver,lin_solver)

    newton_iteration(std::shared_ptr<VectorSpace> vec_ops, std::shared_ptr<LinearSolver> lin_solver):
      vec_ops_(std::move(vec_ops)),
      lin_solver_(std::move(lin_solver))
    {
        vec_ops_->init_vector(f_); 
    }
    newton_iteration(  
        const utils_hierarchy& utils,
        const params_hierarchy& prm = params_hierarchy()      
    ) : 
        newton_iteration(  
            utils.vec_space,
            nmfd::detail::algo_hierarchy_creator<LinearSolver>::get(utils.lin_solver,prm.lin_solver)
        )
    {
    }
    ~newton_iteration()
    {
        vec_ops_->free_vector(f_);
    }

    bool solve(NonlinearOperator &nonlin_op, const vector_type& x, vector_type& d_x)
    {
        vec_ops_->start_use_vector(f_);
        nonlin_op.set_linearization_point(x);
        nonlin_op.apply(x, f_); // f = F(x)
        vec_ops_->scale(T(-1), f_);
        lin_solver_->set_operator(nonlin_op.get_jacobi_operator());
        bool flag_lin_solver = lin_solver_->solve(f_, d_x);
        vec_ops_->stop_use_vector(f_); 
        return flag_lin_solver;
    }
private:
    std::shared_ptr<VectorSpace> vec_ops_;
    std::shared_ptr<LinearSolver> lin_solver_;
    vector_type f_;

};

}
}

#endif