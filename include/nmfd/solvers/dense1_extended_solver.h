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

#ifndef __NMFD_DENSE1_EXTENDED_SOLVER_H__
#define __NMFD_DENSE1_EXTENDED_SOLVER_H__

#include <string>
#include <memory>
#include <stdexcept>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/operations/pair_vector_space.h>
#include <nmfd/operations/dense1_extended_operator.h>
#include "nmfd/detail/vector_wrap.h"


namespace nmfd
{
namespace solvers
{

/**
General solver for the block linear system of size (n+1 \times n+1) (rank one update):

    AX=B

where:
A=[LinearOperator u;v^T w];
X=[x y];
B=[b beta];
**/

template <class OriginalSolver, class OrigOperator, class OrigVectorSpace>
class dense1_extended_solver
{
    using scalar_type = typename OrigVectorSpace::scalar_type;

    using orig_space_type = OrigVectorSpace;
    using orig_vector_type = typename OrigVectorSpace::vector_type;

    using scalar_space_type = operations::static_vector_space<scalar_type, 1>;
    using scalar_vector_type = typename scalar_space_type::vector_type;

    using vector_space_type = operations::pair_vector_space<OrigVectorSpace, scalar_space_type>;
    using vector_type = typename vector_space_type::vector_type;

    using orig_solver_type = OriginalSolver;
    using operator_type = operations::dense1_extended_operator<OrigOperator, OrigVectorSpace>;

public:
    dense1_extended_solver(std::shared_ptr<orig_space_type> orig_vec_space,
                           std::shared_ptr<orig_solver_type> orig_solver,
                           std::shared_ptr<const operator_type> op = nullptr):
        orig_vec_space_(orig_vec_space),
        scalar_space_(std::make_shared<scalar_space_type>()),
        pair_space_(std::make_shared<vector_space_type>(orig_vec_space_, scalar_space_)),
        orig_solver_(orig_solver),
        operator_(op),
        r_wrap_(*orig_vec_space_),
        vx_wrap_(*scalar_space_),
        vr_wrap_(*scalar_space_)
    {
        if (orig_solver_ && operator_) {
            orig_solver_->set_operator(operator_->get_orig_operator());
        }
    }

    void set_solver(std::shared_ptr<orig_solver_type> orig_solver)
    {
        orig_solver_ = orig_solver;
        if (operator_) {
            orig_solver_->set_operator(operator_->get_orig_operator());
        }
    }

    void set_operator(std::shared_ptr<const operator_type> op)
    {
        operator_ = op;
        if (orig_solver_) {
            orig_solver_->set_operator(operator_->get_orig_operator());
        }
    }

    bool solve(const vector_type &rhs, vector_type &res)
    {
        // Variables
        orig_vector_type &x = res.first;
        scalar_vector_type &y = res.second;

        // Rhs parts
        const orig_vector_type &b = rhs.first;
        const scalar_vector_type &beta = rhs.second;

        // Operator
        const orig_vector_type &u = operator_->u();
        const orig_vector_type &v = operator_->v();
        const scalar_vector_type &w = operator_->w();

        // 1. Solving two systems
        bool flag1 = orig_solver_->solve(b, x); // x := A^-1*b
        bool flag2 = orig_solver_->solve(u, *r_wrap_); // r_ := A^-1*u

        scalar_space_->assign_scalar(orig_vec_space_->scalar_prod(v, x), *vx_wrap_); // vx := v^T*x
        scalar_space_->assign_scalar(orig_vec_space_->scalar_prod(v, *r_wrap_), *vr_wrap_); // vr := v^T*r_

        // 2. Calculating y = (beta - v^T*A^-1*b) / (w - v^T*A^-1*u)
        scalar_space_->assign_lin_comb(1.0, beta, -1, *vx_wrap_, y); // y := 1.0*beta - 1.0*vx
        scalar_space_->add_lin_comb(1.0, w, -1, *vr_wrap_); // vr := 1.0*w - 1.0*vr
        scalar_space_->div_pointwise(y, 1.0, *vr_wrap_); // y := y / 1.0*vr

        // 3. Calculating x = A^-1*b - A^-1*u*y
        orig_vec_space_->add_lin_comb(-scalar_space_->get_value_at_point(0, y), *r_wrap_, x); // x := -y[0]*r_ + x

        // 4. Returning result
        orig_vec_space_->assign(x, res.first);
        scalar_space_->assign(y, res.second);

        return flag1 && flag2;
    }

private:
    using orig_vector_wrap_t = detail::vector_wrap<orig_space_type, true, true>;
    using scalar_vector_wrap_t = detail::vector_wrap<scalar_space_type, true, true>;

    std::shared_ptr<orig_space_type> orig_vec_space_;
    std::shared_ptr<scalar_space_type> scalar_space_;
    std::shared_ptr<vector_space_type> pair_space_;

    std::shared_ptr<orig_solver_type> orig_solver_;
    std::shared_ptr<const operator_type> operator_;

    orig_vector_wrap_t r_wrap_;
    scalar_vector_wrap_t vx_wrap_;
    scalar_vector_wrap_t vr_wrap_;
};

}
}

#endif
