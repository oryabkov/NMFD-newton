#include <memory>
#include <cmath>
#include <scfd/utils/log.h>
#include <scfd/static_vec/vec.h>
#include <nmfd/backend/single_node_cpu.h>
#include <nmfd/operations/static_vector_space.h>
#include <nmfd/operations/pair_vector_space.h>
#include <nmfd/solvers/nonlinear_solver.h>
#include <nmfd/solvers/newton_iteration.h>
#include <nmfd/operations/dense1_extended_operator.h>
#include <nmfd/solvers/dense1_extended_solver.h>
#include "nmfd/detail/vector_wrap.h"


template<class VectorSpace>
class TestOperator
{
public:
    using scalar_type = typename VectorSpace::scalar_type;
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;

    TestOperator(std::shared_ptr<vector_space_type> vector_space) :
        vector_space_(vector_space)
    {}

    void apply(const vector_type &x, vector_type &f)const
    {
        vector_space_->assign_lin_comb(value_, x, f);
    }

    void set_value(scalar_type value)
    {
        value_ = value;
    }

    scalar_type get_value() const
    {
        return value_;
    }

private:
    std::shared_ptr<vector_space_type> vector_space_;
    scalar_type value_;
};


template<class Operator, class VectorSpace>
class linsolver
{
public:
    using vector_type = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;
    using operator_type = Operator;

    // linsolver() = default;
    linsolver(std::shared_ptr<vector_space_type> vector_space) :
        vector_space_(vector_space)
    {}

    void set_operator(std::shared_ptr<const operator_type> a)
    {
        operator_ = a;
    }

    bool solve(const vector_type &rhs, vector_type &x) const
    {
        if (operator_ == nullptr)
        {
            return false;
        }

        vector_space_->div_pointwise(1.0, rhs, 1.0, operator_->get_value(), x);
        return true;
    }

private:
    std::shared_ptr<const operator_type> operator_ = nullptr;
    std::shared_ptr<vector_space_type> vector_space_;
};



int main(int argc, char const *args[])
{
    using backend_t = nmfd::backend::single_node_cpu<>;
    using log_t = backend_t::log_type;
    using T = double;
    static const int Dim = 1;

    using orig_vector_type = scfd::static_vec::vec<T, Dim>;
    using orig_vector_space_type = nmfd::operations::static_vector_space<T, Dim, orig_vector_type>;

    using scalar_vector_type = std::array<T, 1>;
    using scalar_vector_space_type = nmfd::operations::static_vector_space<T, 1, scalar_vector_type>;

    using vector_type = std::pair<orig_vector_type, scalar_vector_type>;
    using vector_space_type = nmfd::operations::pair_vector_space<orig_vector_space_type, scalar_vector_space_type>;

    using operator_type = TestOperator<orig_vector_space_type>;
    using linsolver_type = linsolver<operator_type, orig_vector_space_type>;
    using dense1_extended_operator_type = nmfd::operations::dense1_extended_operator<operator_type, orig_vector_space_type>;
    using dense1_extended_solver_type = nmfd::solvers::dense1_extended_solver<linsolver_type, operator_type, orig_vector_space_type>;


    class system_with_circle_constraint
    {
    public:
        using scalar_type = T;
        using vector_type = vector_type;
        using jacobi_operator_type = dense1_extended_operator_type;
        using orig_operator_type = operator_type;

        system_with_circle_constraint(std::shared_ptr<vector_space_type> vector_space, std::shared_ptr<orig_vector_space_type> orig_vec_space, std::shared_ptr<scalar_vector_space_type> scalar_space) :
            jacobi_(std::make_shared<dense1_extended_operator_type>(orig_vec_space)),
            orig_operator_(std::make_shared<orig_operator_type>(orig_vec_space)),
            vector_space_(vector_space),
            orig_vector_space_(orig_vec_space),
            scalar_vector_space_(scalar_space),
            u_wrap_(*orig_vec_space),
            v_wrap_(*orig_vec_space),
            w_wrap_(*scalar_space),
            x0_wrap_(*vector_space),
            delta_x_wrap_(*vector_space),
            eps_(0.1)
        {}

        // Set the previous point for the distance constraint
        void set_previous_point(const vector_type &x0)
        {
            vector_space_->assign(x0, *x0_wrap_);
        }


        void set_previous_tangent(const vector_type &grad_x0) {}

        void set_eps(T eps) { eps_ = eps; }
        T get_eps() const { return eps_; }

        // Compute the extended system residual:
        // f[0] = x^2 + lambda^2 - 1
        // f[1] = (x - x_0)^2 + (lambda - lambda_0)^2 - eps^2
        void apply(const vector_type &x, vector_type &f)
        {
            // f[0] = x^2 + lambda^2 - 1
            orig_vector_space_->assign_scalar(vector_space_->norm_sq(x) - 1., f.first);

            // f[1] = (x - x_0)^2 + (lambda - lambda_0)^2 - eps^2
            vector_space_->assign_lin_comb(1.0, x, -1.0, *x0_wrap_, *delta_x_wrap_);
            scalar_vector_space_->assign_scalar(vector_space_->norm_sq(*delta_x_wrap_) - eps_ * eps_, f.second);
        }

        void set_linearization_point(const vector_type &x)
        {
            vector_space_->assign_lin_comb(1.0, x, -1.0, *x0_wrap_, *delta_x_wrap_);

            // A = dF/dx = 2x
            orig_operator_->set_value(2 * x.first[0]);

            // u = dF/d(lambda) = 2 * lambda
            orig_vector_space_->assign_scalar(2 * x.second[0], *u_wrap_);

            // v = dN/dx = 2 * (x - x_0)
            orig_vector_space_->assign_lin_comb(2.0, (*delta_x_wrap_).first, *v_wrap_);

            // w = dN/d(lambda) = 2 * (lambda - lambda_0)
            scalar_vector_space_->assign_lin_comb(2.0, (*delta_x_wrap_).second, *w_wrap_);

            jacobi_->set_orig_operator(orig_operator_);
            orig_vector_space_->assign(*u_wrap_, jacobi_->u());
            orig_vector_space_->assign(*v_wrap_, jacobi_->v());
            scalar_vector_space_->assign(*w_wrap_, jacobi_->w());
        }

        std::shared_ptr<const dense1_extended_operator_type> get_jacobi_operator() const
        {
            return jacobi_;
        }

    private:
        std::shared_ptr<dense1_extended_operator_type> jacobi_;
        std::shared_ptr<orig_operator_type> orig_operator_;
        std::shared_ptr<vector_space_type> vector_space_;
        std::shared_ptr<orig_vector_space_type> orig_vector_space_;
        std::shared_ptr<scalar_vector_space_type> scalar_vector_space_;

        using vector_wrap_t = nmfd::detail::vector_wrap<vector_space_type, true, true>;
        using orig_vector_wrap_t = nmfd::detail::vector_wrap<orig_vector_space_type, true, true>;
        using scalar_vector_wrap_t = nmfd::detail::vector_wrap<scalar_vector_space_type, true, true>;

        orig_vector_wrap_t u_wrap_;
        orig_vector_wrap_t v_wrap_;
        scalar_vector_wrap_t w_wrap_;
        vector_wrap_t x0_wrap_;
        vector_wrap_t delta_x_wrap_;

        scalar_type eps_;
    };

    class system_with_tangent_constraint
    {
    public:
        using scalar_type = T;
        using vector_type = vector_type;
        using jacobi_operator_type = dense1_extended_operator_type;
        using orig_operator_type = operator_type;

        system_with_tangent_constraint(std::shared_ptr<vector_space_type> vector_space, std::shared_ptr<orig_vector_space_type> orig_vec_space, std::shared_ptr<scalar_vector_space_type> scalar_space) :
            jacobi_(std::make_shared<dense1_extended_operator_type>(orig_vec_space)),
            orig_operator_(std::make_shared<orig_operator_type>(orig_vec_space)),
            vector_space_(vector_space),
            orig_vector_space_(orig_vec_space),
            scalar_vector_space_(scalar_space),
            u_wrap_(*orig_vec_space),
            v_wrap_(*orig_vec_space),
            w_wrap_(*scalar_space),
            grad_x0_wrap_(*vector_space),
            x0_wrap_(*vector_space),
            delta_x_wrap_(*vector_space),
            eps_(0.1)
        {}

        // Set the previous point for the tangent constraint
        void set_previous_point(const vector_type &x0)
        {
            vector_space_->assign(x0, *x0_wrap_);
            // Tangent direction at x0 is (-lambda_0, x_0) for counter-clockwise motion
            // vector_space_->set_value_at_point(0, -vector_space_->get_value_at_point(1, x0), *grad_x0_wrap_);
            // vector_space_->set_value_at_point(1, vector_space_->get_value_at_point(0, x0), *grad_x0_wrap_);
        }

        void set_previous_tangent(const vector_type &grad_x0)
        {
            vector_space_->assign(grad_x0, *grad_x0_wrap_);
        }

        void set_eps(T eps) { eps_ = eps; }
        T get_eps() const { return eps_; }

        // Compute the extended system residual:
        void apply(const vector_type &x, vector_type &f)
        {
            // f[0] = x^2 + lambda^2 - 1
            orig_vector_space_->assign_scalar(vector_space_->norm_sq(x) - 1., f.first);

            // f[1] = tangent · (x - x_0) - eps^2 = (-lambda_0) * (x - x_0) + x_0 * (lambda - lambda_0) - eps^2
            vector_space_->assign_lin_comb(1.0, x, -1.0, *x0_wrap_, *delta_x_wrap_);
            scalar_vector_space_->assign_scalar(vector_space_->scalar_prod(*grad_x0_wrap_, *delta_x_wrap_) - eps_ * eps_, f.second);
        }

        void set_linearization_point(const vector_type &x)
        {
            vector_space_->assign_lin_comb(1.0, x, -1.0, *x0_wrap_, *delta_x_wrap_);

            // A = dF/dx = 2x
            orig_operator_->set_value(2 * x.first[0]);

            // u = dF/d(lambda) = 2 * lambda
            orig_vector_space_->assign_scalar(2 * x.second[0], *u_wrap_);

            // v = dN/dx = -lambda_0
            orig_vector_space_->assign_scalar(vector_space_->get_value_at_point(0, *grad_x0_wrap_), *v_wrap_);

            // w = dN/d(lambda) = x_0
            scalar_vector_space_->assign_scalar(vector_space_->get_value_at_point(1, *grad_x0_wrap_), *w_wrap_);

            jacobi_->set_orig_operator(orig_operator_);
            orig_vector_space_->assign(*u_wrap_, jacobi_->u());
            orig_vector_space_->assign(*v_wrap_, jacobi_->v());
            scalar_vector_space_->assign(*w_wrap_, jacobi_->w());
        }

        std::shared_ptr<const dense1_extended_operator_type> get_jacobi_operator() const
        {
            return jacobi_;
        }

    private:
        std::shared_ptr<dense1_extended_operator_type> jacobi_;
        std::shared_ptr<orig_operator_type> orig_operator_;
        std::shared_ptr<vector_space_type> vector_space_;
        std::shared_ptr<orig_vector_space_type> orig_vector_space_;
        std::shared_ptr<scalar_vector_space_type> scalar_vector_space_;

        using vector_wrap_t = nmfd::detail::vector_wrap<vector_space_type, true, true>;
        using orig_vector_wrap_t = nmfd::detail::vector_wrap<orig_vector_space_type, true, true>;
        using scalar_vector_wrap_t = nmfd::detail::vector_wrap<scalar_vector_space_type, true, true>;

        orig_vector_wrap_t u_wrap_;
        orig_vector_wrap_t v_wrap_;
        scalar_vector_wrap_t w_wrap_;
        vector_wrap_t x0_wrap_;
        vector_wrap_t grad_x0_wrap_;
        vector_wrap_t delta_x_wrap_;

        scalar_type eps_;
    };

    class star_shaped_system_with_tangent_constraint
    {
    public:
        using scalar_type = T;
        using vector_type = vector_type;
        using jacobi_operator_type = dense1_extended_operator_type;
        using orig_operator_type = operator_type;

        star_shaped_system_with_tangent_constraint(std::shared_ptr<vector_space_type> vector_space, std::shared_ptr<orig_vector_space_type> orig_vec_space, std::shared_ptr<scalar_vector_space_type> scalar_space) :
            jacobi_(std::make_shared<dense1_extended_operator_type>(orig_vec_space)),
            orig_operator_(std::make_shared<orig_operator_type>(orig_vec_space)),
            vector_space_(vector_space),
            orig_vector_space_(orig_vec_space),
            scalar_vector_space_(scalar_space),
            u_wrap_(*orig_vec_space),
            v_wrap_(*orig_vec_space),
            w_wrap_(*scalar_space),
            grad_x0_wrap_(*vector_space),
            x0_wrap_(*vector_space),
            delta_x_wrap_(*vector_space),
            eps_(0.1)
        {}

        // Set the previous point for the tangent constraint
        void set_previous_point(const vector_type &x0)
        {
            vector_space_->assign(x0, *x0_wrap_);
            // Tangent direction at x0 is (-lambda_0, x_0) for counter-clockwise motion
            // vector_space_->set_value_at_point(0, -vector_space_->get_value_at_point(1, x0), *grad_x0_wrap_);
            // vector_space_->set_value_at_point(1, vector_space_->get_value_at_point(0, x0), *grad_x0_wrap_);
        }

        void set_previous_tangent(const vector_type &grad_x0)
        {
            vector_space_->assign(grad_x0, *grad_x0_wrap_);
        }

        void set_eps(T eps) { eps_ = eps; }
        T get_eps() const { return eps_; }

        // Compute the extended system residual:
        // F(x,y) = sqrt(x^2 + y^2) - 1 - 0.2 * 4xy(x^2 - y^2) / (x^2 + y^2)^2
        // N(x,y) = tangent · (x - x_0) - eps^2
        void apply(const vector_type &x, vector_type &f)
        {
            // f[0] = sqrt(x^2 + y^2) - 1 - 0.2 * 4xy(x^2 - y^2) / (x^2 + y^2)^2
            //      = sqrt(r2) - 1 - 0.8 * xy * (x^2 - y^2) / r2^2
            orig_vector_space_->assign_scalar(
                std::sqrt(vector_space_->norm_sq(x)) - 1.0
                - 0.8 * x.first[0] * x.second[0]
                      * (x.first[0] * x.first[0] - x.second[0] * x.second[0])
                      / (vector_space_->norm_sq(x) * vector_space_->norm_sq(x)),
                f.first);

            // f[1] = tangent · (x - x_0) - eps^2
            vector_space_->assign_lin_comb(1.0, x, -1.0, *x0_wrap_, *delta_x_wrap_);
            scalar_vector_space_->assign_scalar(
                vector_space_->scalar_prod(*grad_x0_wrap_, *delta_x_wrap_) - eps_ * eps_,
                f.second);
        }

        void set_linearization_point(const vector_type &x)
        {
            vector_space_->assign_lin_comb(1.0, x, -1.0, *x0_wrap_, *delta_x_wrap_);

            // Shorthand for x and y values
            const scalar_type& xv = x.first[0];
            const scalar_type& yv = x.second[0];
            const scalar_type r2 = xv * xv + yv * yv;
            const scalar_type r2_cubed = r2 * r2 * r2;
            const scalar_type r2_5_2 = std::pow(r2, 2.5);

            // A = dF/dx = 0.8 * (x^4*y - 6*x^2*y^3 + y^5 + 1.25*x*r^5) / r^6
            orig_operator_->set_value(
                0.8 * (xv*xv*xv*xv * yv - 6.0 * xv*xv * yv*yv*yv + yv*yv*yv*yv*yv
                       + 1.25 * xv * r2_5_2) / r2_cubed);

            // u = dF/dy = -0.8 * (x^5 - 6*x^3*y^2 + x*y^4 - 1.25*y*r^5) / r^6
            orig_vector_space_->assign_scalar(
                -0.8 * (xv*xv*xv*xv*xv - 6.0 * xv*xv*xv * yv*yv + xv * yv*yv*yv*yv
                        - 1.25 * yv * r2_5_2) / r2_cubed,
                *u_wrap_);

            // v = dN/dx = grad_x0[0] (tangent direction x-component)
            orig_vector_space_->assign_scalar(
                vector_space_->get_value_at_point(0, *grad_x0_wrap_), *v_wrap_);

            // w = dN/dy = grad_x0[1] (tangent direction y-component)
            scalar_vector_space_->assign_scalar(
                vector_space_->get_value_at_point(1, *grad_x0_wrap_), *w_wrap_);

            jacobi_->set_orig_operator(orig_operator_);
            orig_vector_space_->assign(*u_wrap_, jacobi_->u());
            orig_vector_space_->assign(*v_wrap_, jacobi_->v());
            scalar_vector_space_->assign(*w_wrap_, jacobi_->w());
        }

        std::shared_ptr<const dense1_extended_operator_type> get_jacobi_operator() const
        {
            return jacobi_;
        }

    private:
        std::shared_ptr<dense1_extended_operator_type> jacobi_;
        std::shared_ptr<orig_operator_type> orig_operator_;
        std::shared_ptr<vector_space_type> vector_space_;
        std::shared_ptr<orig_vector_space_type> orig_vector_space_;
        std::shared_ptr<scalar_vector_space_type> scalar_vector_space_;

        using vector_wrap_t = nmfd::detail::vector_wrap<vector_space_type, true, true>;
        using orig_vector_wrap_t = nmfd::detail::vector_wrap<orig_vector_space_type, true, true>;
        using scalar_vector_wrap_t = nmfd::detail::vector_wrap<scalar_vector_space_type, true, true>;

        orig_vector_wrap_t u_wrap_;
        orig_vector_wrap_t v_wrap_;
        scalar_vector_wrap_t w_wrap_;
        vector_wrap_t x0_wrap_;
        vector_wrap_t grad_x0_wrap_;
        vector_wrap_t delta_x_wrap_;

        scalar_type eps_;
    };

    const T tol = 1e-6;
    const T step = 0.1;
    const int max_num_steps = 100;  // Number of continuation steps

    size_t passed_counter = 0;
    size_t failed_counter = 0;
    backend_t backend;
    log_t &log = backend.log();
    std::shared_ptr<orig_vector_space_type> orig_vector_space = std::make_shared<orig_vector_space_type>();
    std::shared_ptr<scalar_vector_space_type> scalar_vector_space = std::make_shared<scalar_vector_space_type>();
    std::shared_ptr<vector_space_type> vector_space = std::make_shared<vector_space_type>(orig_vector_space, scalar_vector_space);
    std::shared_ptr<linsolver_type> lin_solver = std::make_shared<linsolver_type>(orig_vector_space);
    std::shared_ptr<dense1_extended_solver_type> extended_solver = std::make_shared<dense1_extended_solver_type>(orig_vector_space, lin_solver);


    // ====================================================================
    // GROUP 1: Circle Constraint
    // ====================================================================
    {
        log.info("test continuation around a circle with circle constraint");

        using system_op_type = system_with_circle_constraint;
        using newton_iteration_t = nmfd::solvers::newton_iteration<vector_space_type, system_op_type, dense1_extended_solver_type>;
        using newton_solver_t = nmfd::solvers::nonlinear_solver<vector_space_type, log_t, system_op_type, newton_iteration_t>;

        std::shared_ptr<newton_iteration_t> newton_iteration = std::make_shared<newton_iteration_t>(vector_space, extended_solver);
        std::shared_ptr<newton_solver_t> newton_solver = std::make_shared<newton_solver_t>(vector_space, &log, newton_iteration);
        newton_solver->convergence_strategy()->set_tolerance(tol);

        system_op_type system_op(vector_space, orig_vector_space, scalar_vector_space);
        system_op.set_eps(step);

        // Starting point on the circle: (x, lambda) = (1, 0)
        vector_type x(orig_vector_type{1.0}, scalar_vector_type{0.0});
        vector_type x_prev(orig_vector_type{1.0}, scalar_vector_type{0.0});
        vector_type grad_x(orig_vector_type{0.0}, scalar_vector_type{0.0});
        vector_type f(orig_vector_type{0.0}, scalar_vector_type{0.0});
        log.info_f("Step %2d: (x, lambda) = (%+0.6f, %+0.6f), |F| = %.2e",
            0, x.first[0], x.second[0], 0.0);

        T phi = 0.0;

        // Continuation loop: walk around the circle
        for (int step_idx = 0; step_idx < max_num_steps; ++step_idx)
        {
            // Set the previous point for the constraint and update the previous point
            system_op.set_previous_point(x);
            vector_space->assign(x, x_prev);

            // Predictor step: move along tangent direction (perpendicular to gradient of F)
            // grad_x = (-lambda, x);
            vector_space->set_value_at_point(-vector_space->get_value_at_point(1, x), 0, grad_x);
            vector_space->set_value_at_point(vector_space->get_value_at_point(0, x), 1, grad_x);
            vector_space->add_mul_scalar(0.0, step / vector_space->norm2(x), grad_x);

            // x += step * grad_x / norm(grad_x)
            vector_space->add_lin_comb(1.0, grad_x, x);

            system_op.set_previous_tangent(grad_x);


            // Corrector step: Newton iteration to find intersection of F=0 and N=0
            newton_solver->solve(&system_op, nullptr, nullptr, x);

            system_op.apply(x, f);
            T circle_residual = std::abs(vector_space->get_value_at_point(0, f));
            log.info_f("Step %2d: (x, lambda) = (%+0.6f, %+0.6f), |F| = %.2e",
                       step_idx + 1, x.first[0], x.second[0], circle_residual);

            if (circle_residual > tol * 100)
            {
                log.error_f("Step %d: Failed to stay on circle! |F| = %e", step_idx + 1, circle_residual);
                failed_counter++;
                break;
            }

            // Calculate the rotation angle
            phi += std::asin((
                vector_space->get_value_at_point(0, x_prev) * vector_space->get_value_at_point(1, x) -
                vector_space->get_value_at_point(1, x_prev) * vector_space->get_value_at_point(0, x)
            ) / (vector_space->norm2(x_prev) * vector_space->norm2(x)));

            if (phi > 2 * M_PI)
            {
                log.info_f("phi = %f is greater than 2 * M_PI, stopping the continuation", phi);
                passed_counter++;
                break;
            }
        }

        // Final log
        log.info_f("Final position: (x, lambda) = (%0.6f, %0.6f)", x.first[0], x.second[0]);
        log.info_f("Total rotation angle: %0.4f rad = %0.1f deg", phi, phi * 180.0 / M_PI);
    }


    // ====================================================================
    // GROUP 2: Tangent Constraint
    // ====================================================================
    {
        log.info("test continuation around a circle with tangent constraint");

        using system_op_type = system_with_tangent_constraint;
        // using system_op_type = star_shaped_system_with_tangent_constraint;
        using newton_iteration_t = nmfd::solvers::newton_iteration<vector_space_type, system_op_type, dense1_extended_solver_type>;
        using newton_solver_t = nmfd::solvers::nonlinear_solver<vector_space_type, log_t, system_op_type, newton_iteration_t>;

        std::shared_ptr<newton_iteration_t> newton_iteration = std::make_shared<newton_iteration_t>(vector_space, extended_solver);
        std::shared_ptr<newton_solver_t> newton_solver = std::make_shared<newton_solver_t>(vector_space, &log, newton_iteration);
        newton_solver->convergence_strategy()->set_tolerance(tol);

        system_op_type system_op(vector_space, orig_vector_space, scalar_vector_space);
        system_op.set_eps(step);

        // Starting point on the circle: (x, lambda) = (1, 0)
        vector_type x(orig_vector_type{1.0}, scalar_vector_type{0.0});
        vector_type x_prev(orig_vector_type{1.0}, scalar_vector_type{0.0});
        vector_type grad_x(orig_vector_type{0.0}, scalar_vector_type{0.0});
        vector_type f(orig_vector_type{0.0}, scalar_vector_type{0.0});
        log.info_f("Step %2d: (x, lambda) = (%+0.6f, %+0.6f), |F| = %.2e",
            0, x.first[0], x.second[0], 0.0);

        T phi = 0.0;

        // Continuation loop: walk around the circle
        for (int step_idx = 0; step_idx < max_num_steps; ++step_idx)
        {
            // Set the previous point for the constraint and update the previous point
            system_op.set_previous_point(x);
            vector_space->assign(x, x_prev);

            // Predictor step: move along tangent direction (perpendicular to gradient of F)
            // grad_x = (-lambda, x);
            vector_space->set_value_at_point(-vector_space->get_value_at_point(1, x), 0, grad_x);
            vector_space->set_value_at_point(vector_space->get_value_at_point(0, x), 1, grad_x);
            vector_space->add_mul_scalar(0.0, step / vector_space->norm2(x), grad_x);

            // x += step * grad_x / norm(grad_x)
            vector_space->add_lin_comb(1.0, grad_x, x);

            system_op.set_previous_tangent(grad_x);


            // Corrector step: Newton iteration to find intersection of F=0 and N=0
            newton_solver->solve(&system_op, nullptr, nullptr, x);

            system_op.apply(x, f);
            T circle_residual = std::abs(vector_space->get_value_at_point(0, f));
            log.info_f("Step %2d: (x, lambda) = (%+0.6f, %+0.6f), |F| = %.2e",
                       step_idx + 1, x.first[0], x.second[0], circle_residual);

            if (circle_residual > tol * 100)
            {
                log.error_f("Step %d: Failed to stay on circle! |F| = %e", step_idx + 1, circle_residual);
                failed_counter++;
                break;
            }

            // Calculate the rotation angle
            phi += std::asin((
                vector_space->get_value_at_point(0, x_prev) * vector_space->get_value_at_point(1, x) -
                vector_space->get_value_at_point(1, x_prev) * vector_space->get_value_at_point(0, x)
            ) / (vector_space->norm2(x_prev) * vector_space->norm2(x)));

            if (phi > 2 * M_PI)
            {
                log.info_f("phi = %f is greater than 2 * M_PI, stopping the continuation", phi);
                passed_counter++;
                break;
            }
        }

        // Final log
        log.info_f("Final position: (x, lambda) = (%0.6f, %0.6f)", x.first[0], x.second[0]);
        log.info_f("Total rotation angle: %0.4f rad = %0.1f deg", phi, phi * 180.0 / M_PI);
    }


    // ====================================================================
    // GROUP 3: Star-Shaped Curve Constraint
    // ====================================================================
    {
        log.info("test continuation around a star-shaped curve with tangent constraint");

        using system_op_type = star_shaped_system_with_tangent_constraint;
        using newton_iteration_t = nmfd::solvers::newton_iteration<vector_space_type, system_op_type, dense1_extended_solver_type>;
        using newton_solver_t = nmfd::solvers::nonlinear_solver<vector_space_type, log_t, system_op_type, newton_iteration_t>;

        std::shared_ptr<newton_iteration_t> newton_iteration = std::make_shared<newton_iteration_t>(vector_space, extended_solver);
        std::shared_ptr<newton_solver_t> newton_solver = std::make_shared<newton_solver_t>(vector_space, &log, newton_iteration);
        newton_solver->convergence_strategy()->set_tolerance(tol);

        system_op_type system_op(vector_space, orig_vector_space, scalar_vector_space);
        system_op.set_eps(step);

        // Starting point on the circle: (x, lambda) = (1, 0)
        vector_type x(orig_vector_type{1.0}, scalar_vector_type{0.0});
        vector_type x_prev(orig_vector_type{1.0}, scalar_vector_type{0.0});
        vector_type grad_x(orig_vector_type{0.0}, scalar_vector_type{0.0});
        vector_type f(orig_vector_type{0.0}, scalar_vector_type{0.0});
        log.info_f("Step %2d: (x, lambda) = (%+0.6f, %+0.6f), |F| = %.2e",
            0, x.first[0], x.second[0], 0.0);

        T phi = 0.0;

        // Continuation loop: walk around the circle
        for (int step_idx = 0; step_idx < max_num_steps; ++step_idx)
        {
            // Set the previous point for the constraint and update the previous point
            system_op.set_previous_point(x);
            vector_space->assign(x, x_prev);

            // Predictor step: move along tangent direction (perpendicular to gradient of F)
            // grad_x = (-lambda, x);
            vector_space->set_value_at_point(-vector_space->get_value_at_point(1, x), 0, grad_x);
            vector_space->set_value_at_point(vector_space->get_value_at_point(0, x), 1, grad_x);


            // Shorthand for x and y values
            const T& xv = x.first[0];
            const T& yv = x.second[0];
            const T r2 = xv * xv + yv * yv;
            const T r2_cubed = r2 * r2 * r2;
            const T r2_5_2 = std::pow(r2, 2.5);

            vector_space->set_value_at_point(
                0.8 * (xv*xv*xv*xv * yv - 6.0 * xv*xv * yv*yv*yv + yv*yv*yv*yv*yv
                       + 1.25 * xv * r2_5_2) / r2_cubed, 1, grad_x);

            vector_space->set_value_at_point(
                0.8 * (xv*xv*xv*xv*xv - 6.0 * xv*xv*xv * yv*yv + xv * yv*yv*yv*yv
                        - 1.25 * yv * r2_5_2) / r2_cubed, 0, grad_x);

            vector_space->add_mul_scalar(0.0, step / vector_space->norm2(x), grad_x);

            // x += step * grad_x / norm(grad_x)
            vector_space->add_lin_comb(1.0, grad_x, x);

            system_op.set_previous_tangent(grad_x);


            // Corrector step: Newton iteration to find intersection of F=0 and N=0
            newton_solver->solve(&system_op, nullptr, nullptr, x);

            system_op.apply(x, f);
            T star_shaped_curve_residual = std::abs(vector_space->get_value_at_point(0, f));
            log.info_f("Step %2d: (x, lambda) = (%+0.6f, %+0.6f), |F| = %.2e",
                       step_idx + 1, x.first[0], x.second[0], star_shaped_curve_residual);

            if (star_shaped_curve_residual > tol * 100)
            {
                log.error_f("Step %d: Failed to stay on star-shaped curve! |F| = %e", step_idx + 1, star_shaped_curve_residual);
                failed_counter++;
                // break;
            }

            // Calculate the rotation angle
            phi += std::asin((
                vector_space->get_value_at_point(0, x_prev) * vector_space->get_value_at_point(1, x) -
                vector_space->get_value_at_point(1, x_prev) * vector_space->get_value_at_point(0, x)
            ) / (vector_space->norm2(x_prev) * vector_space->norm2(x)));

            if (phi > 2 * M_PI)
            {
                log.info_f("phi = %f is greater than 2 * M_PI, stopping the continuation", phi);
                passed_counter++;
                break;
            }
        }

        // Final log
        log.info_f("Final position: (x, lambda) = (%0.6f, %0.6f)", x.first[0], x.second[0]);
        log.info_f("Total rotation angle: %0.4f rad = %0.1f deg", phi, phi * 180.0 / M_PI);
    }


    // ====================================================================
    // FINAL SUMMARY
    // ====================================================================
    log.info("================================================");
    log.info("=== TEST SUMMARY ===");
    log.info("✓ Passed: " + std::to_string(passed_counter));
    log.info("✗ Failed: " + std::to_string(failed_counter));
    log.info("Total tests: " + std::to_string(passed_counter + failed_counter));

    if (failed_counter == 0) {
        log.info("🎉 All tests passed successfully!");
    } else {
        log.info("⚠️  Some tests failed. Please review the output above.");
    }
    log.info("================================================");

    return (failed_counter == 0) ? 0 : 1;
}
