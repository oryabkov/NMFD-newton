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

#ifndef __NMFD_NEWTON_ITERATION_H__
#define __NMFD_NEWTON_ITERATION_H__

#include <nmfd/detail/algo_hierarchy_macro.h>
#include <nmfd/detail/algo_hierarchy_creator.h>
#include <nmfd/detail/vector_wrap.h>

/// NOTE originally taken from deflated_continuation master branch source/deflation/system_operator_deflation.h 22.07.2025

namespace nmfd
{
namespace solvers
{

template<class VectorSpace, class NonlinearOperator, class LinearSolver>
class newton_iteration
{
    using T             = typename VectorSpace::scalar_type;
    using vec_wrap_type = nmfd::detail::vector_wrap<VectorSpace, false, false>;

public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;
    using linear_operator = typename NonlinearOperator::jacobi_operator_type;

    NMFD_ALGO_EMPTY_PARAMS_TYPE_DEFINE(newton_iteration)
    struct utils
    {
        std::shared_ptr<VectorSpace> vec_space;
        utils() = default;
        utils(std::shared_ptr<VectorSpace> vec_space_) : vec_space(std::move(vec_space_))
        {
        }
        template<class Backend>
        utils(Backend &backend, std::shared_ptr<VectorSpace> vec_space_) : utils(vec_space_)
        {
        }
    };
    NMFD_ALGO_HIERARCHY_TYPES_DEFINE(newton_iteration,LinearSolver,lin_solver)

    newton_iteration( std::shared_ptr<VectorSpace> vec_ops, std::shared_ptr<LinearSolver> lin_solver )
        : vec_ops_( std::move( vec_ops ) ), lin_solver_( std::move( lin_solver ) ), f_( *vec_ops_ )
    {
        f_.init();
        f_.start_use();
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
    ~newton_iteration() = default;

    bool solve( NonlinearOperator &nonlin_op, const vector_type &x, vector_type &d_x )
    {
        nonlin_op.set_linearization_point( x );
        nonlin_op.apply( x, *f_ ); // f = F(x)
        vec_ops_->scale( T( -1 ), *f_ );
        lin_solver_->set_operator( nonlin_op.get_jacobi_operator() );
        return lin_solver_->solve( *f_, d_x );
    }

private:
    std::shared_ptr<VectorSpace> vec_ops_;
    std::shared_ptr<LinearSolver> lin_solver_;
    vec_wrap_type                 f_;
};

}
}

#endif
