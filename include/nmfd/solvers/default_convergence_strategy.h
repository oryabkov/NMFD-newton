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

#ifndef __DEFAULT_CONVERGENCE_STRATEGY_H__
#define __DEFAULT_CONVERGENCE_STRATEGY_H__
/**
convergence rules for Newton iterator for continuation process
*/
#include <cmath>
#include <vector>
#include <algorithm> // std::min_element
#include <iterator>  // std::begin, std::end
#include <utility>
#include <stdexcept>
#include <limits>
#include <memory>
#include <scfd/utils/logged_obj_base.h>
#include <nmfd/operations/ident_operator.h>
#include <nmfd/operations/zero_functional.h>
#include <nmfd/detail/algo_hierarchy_macro.h>
#include <nmfd/detail/algo_hierarchy_creator.h>
#include <nmfd/detail/vector_wrap.h>

/// TODO check it!
/// NOTE originally taken from deflated_continuation timesteppers branch source/continuation/convergence_strategy.h 22.07.2025

namespace nmfd
{
namespace solvers
{
// Override only residual assessment; iteration, backtracking and rollback stay shared.
struct nonlinear_residual_norm
{
    template <class Space, class Operator>
    static auto
    evaluate( Space &space, Operator &op, const typename Space::vector_type &state, typename Space::vector_type &work )
    {
        op.apply( state, work );
        return space.norm_l2( work );
    }
};

template <
    class VectorSpace, class Log, class NonlinearOperator,
    class ProjectOperator = operations::ident_operator<VectorSpace>,
    class QualityFunctor = operations::zero_functional<VectorSpace>, class ResidualNorm = nonlinear_residual_norm>
class default_convergence_strategy : public scfd::utils::logged_obj_base<Log>
{
private:
    using T             = typename VectorSpace::scalar_type;
    using T_vec         = typename VectorSpace::vector_type;
    using vec_wrap_type = nmfd::detail::vector_wrap<VectorSpace, false, false>;

public:
    using vector_space_type = VectorSpace;
    using logged_obj_type   = scfd::utils::logged_obj_base<Log>;

    struct params : public logged_obj_type::params
    {
        unsigned int stagnation_max          = 10;
        unsigned int max_backtracking_trials = 64;
        //unsigned int maximum_iterations = 100;
        unsigned int max_iters_num = 100;
        //T tolerance = T(1.0e-6);
        //T tolerance_0;
        T    rel_tol                 = T( 0 );
        T    abs_tol                 = T( 1.0e-6 );
        T    maximum_norm_increase   = 0.0;
        T    newton_weight_threshold = 1.0e-12;
        T    newton_weight_initial   = T( 1 );
        T    newton_weight_mul       = T( 0.5 );
        T    relax_tolerance_factor  = T( 1 );
        bool verbose = true, store_norms_history = false;
        bool perform_full_check = true;


        params( const std::string &log_pefix = "", const std::string &log_name = "default_convergence_strategy::" )
            : logged_obj_type::params( 0, log_pefix + log_name )
        {
        }
#ifdef NMFD_ENABLE_NLOHMANN
        void from_json( const nlohmann::json &j )
        {
            params updated        = *this;
            updated.max_iters_num = read_unsigned_parameter( j, "max_iters_num", max_iters_num );
            updated.max_backtracking_trials =
                read_unsigned_parameter( j, "max_backtracking_trials", max_backtracking_trials );
            updated.stagnation_max          = read_unsigned_parameter( j, "stagnation_max", stagnation_max );
            updated.rel_tol                 = j.value( "rel_tol", rel_tol );
            updated.abs_tol                 = j.value( "abs_tol", abs_tol );
            updated.maximum_norm_increase   = j.value( "maximum_norm_increase", maximum_norm_increase );
            updated.newton_weight_threshold = j.value( "newton_weight_threshold", newton_weight_threshold );
            updated.newton_weight_initial   = j.value( "newton_weight_initial", newton_weight_initial );
            updated.newton_weight_mul       = j.value( "newton_weight_mul", newton_weight_mul );
            updated.relax_tolerance_factor  = j.value( "relax_tolerance_factor", relax_tolerance_factor );
            updated.verbose                 = j.value( "verbose", verbose );
            updated.perform_full_check      = j.value( "perform_full_check", perform_full_check );
            updated.store_norms_history     = j.value( "store_norms_history", store_norms_history );
            validate_parameters( updated );
            *this = std::move( updated );
        }
        nlohmann::json to_json() const
        {
            return nlohmann::json{
                { "max_iters_num", max_iters_num },
                { "max_backtracking_trials", max_backtracking_trials },
                //{"min_iters_num", min_iters_num},
                { "stagnation_max", stagnation_max },
                { "rel_tol", rel_tol },
                { "abs_tol", abs_tol },
                { "maximum_norm_increase", maximum_norm_increase },
                { "newton_weight_threshold", newton_weight_threshold },
                { "newton_weight_initial", newton_weight_initial },
                { "newton_weight_mul", newton_weight_mul },
                { "relax_tolerance_factor", relax_tolerance_factor },
                //{"out_min_resid_norm", out_min_resid_norm},
                { "verbose", verbose },
                { "perform_full_check", perform_full_check },
                { "store_norms_history", store_norms_history }
                //{"divide_out_norms_by_rel_base", divide_out_norms_by_rel_base}
            };
        }
#endif
    };
    struct utils
    {
        std::shared_ptr<VectorSpace> vec_space;
        Log                         *log = nullptr;
        utils()                          = default;
        utils( std::shared_ptr<VectorSpace> vec_space_, Log *log_ = nullptr )
            : vec_space( std::move( vec_space_ ) ), log( log_ )
        {
        }
        template <class Backend>
        utils( Backend &backend, std::shared_ptr<VectorSpace> vec_space )
            : utils( std::move( vec_space ), &backend.log() )
        {
        }
        template <class Backend>
        utils( std::shared_ptr<Backend> backend, std::shared_ptr<VectorSpace> vec_space )
            : utils( std::move( vec_space ) )
        {
            if ( !backend )
                throw std::invalid_argument( "Missing backend" );
            log = &backend->log();
        }
    };
    NMFD_ALGO_HIERARCHY_TYPES_DEFINE( default_convergence_strategy )

    default_convergence_strategy( std::shared_ptr<VectorSpace> vec_space, Log *log, params prm = params() )
        : logged_obj_type( log, prm ), prm_( prm ),
          vec_space_(
              vec_space ? std::move( vec_space )
                        : throw std::invalid_argument( "Newton convergence requires a vector space" )
          ),
          iterations( 0 ), x1( *vec_space_ ), x1_storage( *vec_space_ ), Fx( *vec_space_ )
    {
        validate_parameters( prm_ );
        x1.init();
        x1.start_use();
        x1_storage.init();
        x1_storage.start_use();
        Fx.init();
        Fx.start_use();
        if ( prm_.store_norms_history )
        {
            norms_evolution.reserve( prm_.max_iters_num );
        }
    }
    default_convergence_strategy( const utils_hierarchy &utils, const params_hierarchy &prm = params_hierarchy() )
        : default_convergence_strategy( utils.vec_space, utils.log, prm )
    {
    }
    ~default_convergence_strategy() = default;

    T rel_tol() const
    {
        return prm_.rel_tol;
    }
    T abs_tol() const
    {
        return prm_.abs_tol;
    }
    T rel_tol_base() const
    {
        return Fx_initial_norm_;
    }
    T tol() const
    {
        return abs_tol() + rel_tol() * rel_tol_base();
    }


    void set_convergence_constants(
        T tolerance_, int maximum_iterations_, T relax_tolerance_factor_, int relax_tolerance_steps_,
        T newton_weight_ = T( 1 ), bool store_norms_history_ = false, bool verbose_ = true,
        unsigned int stagnation_max_ = 10, T maximum_norm_increase_p = 0.0, T newton_weight_threshold_p = 1.0e-12,
        bool use_abs_tol = true, bool perform_full_check = true
    )
    {
        if ( maximum_iterations_ < 1 )
        {
            throw std::invalid_argument( "maximum_iterations must be positive" );
        }
        params updated                  = prm_;
        updated.abs_tol                 = use_abs_tol ? tolerance_ : T( 0 );
        updated.rel_tol                 = use_abs_tol ? T( 0 ) : tolerance_;
        updated.max_iters_num           = static_cast<unsigned int>( maximum_iterations_ );
        updated.newton_weight_initial   = newton_weight_;
        updated.store_norms_history     = store_norms_history_;
        updated.verbose                 = verbose_;
        updated.stagnation_max          = stagnation_max_;
        updated.relax_tolerance_factor  = relax_tolerance_factor_;
        updated.perform_full_check      = perform_full_check;
        updated.maximum_norm_increase   = maximum_norm_increase_p;
        updated.newton_weight_threshold = newton_weight_threshold_p;
        validate_parameters( updated );
        if ( updated.store_norms_history )
        {
            norms_evolution.reserve( updated.max_iters_num );
        }
        prm_          = std::move( updated );
        newton_weight = prm_.newton_weight_initial;
        logged_obj_type::info_f(
            "check: relax_tolerance_factor = %le, maximum_norm_increase = %le, newton_weight_threshold = %le",
            (double)prm_.relax_tolerance_factor, (double)prm_.maximum_norm_increase,
            double( prm_.newton_weight_threshold )
        );
    }
    void set_tolerance( T abs_tol, T rel_tol = T( 0 ) )
    {
        validate_tolerances( abs_tol, rel_tol );
        prm_.abs_tol = abs_tol;
        prm_.rel_tol = rel_tol;
    }
    void increment_ok( bool increment_ok_flag )
    {
        increment_ok_ = increment_ok_flag;
    }
    bool check_convergence(
        NonlinearOperator *nonlin_op, ProjectOperator *project_op, QualityFunctor *quality_func, T_vec &x,
        T_vec &delta_x
    )
    {
        bool finish = false; //states that the newton process should stop.
        // result_status defines on how this process is stoped.
        newton_weight = prm_.newton_weight_initial;
        if ( prm_.perform_full_check )
        {
            bool x_ok = vec_space_->is_valid_number( x );
            if ( !x_ok )
            {
                result_status = 11;
                return true;
            }
        }
        T normFx = ResidualNorm::evaluate( *vec_space_, *nonlin_op, x, *Fx );
        if ( !std::isfinite( normFx ) ) //set result_status = 2 if the provided vector is inconsistent
        {
            result_status = 2;
            //finish = true;
            return true;
        }
        if ( normFx <= tol() ) //do nothing is my kind of problem =)
        {
            result_status = 0;
            logged_obj_type::info_f(
                "iteration %u, residuals n: %le < tol(): %le => finished.", iterations, (double)normFx, (double)tol()
            );
            return true;
        }
        if ( !increment_ok_ )
        {
            result_status = 6;
            return true;
        }
        unsigned int backtracking_trials = 0;
        if ( iterations == 0 )
        {
            /// stores initial solution and norm
            vec_space_->assign( x, *x1_storage );
            Fx1_storage_norm_ = normFx;
            Fx_initial_norm_  = normFx;
            /// postulate continue (other cases checked earlier) and continue
            result_status = 1;
            iterations++;
            return false;
        }
        if ( prm_.perform_full_check )
        {
            if ( !vec_space_->is_valid_number( delta_x ) )
            {
                result_status = 7;
                return true;
            }
        }
        T normFx1;
        do
        {
            // increment trail counter
            ++backtracking_trials;
            result_status = 3;
            //update solution
            const auto update_ret = update_solution( nonlin_op, project_op, x, delta_x, *x1 );
            normFx1               = update_ret.first;
            // if ( !update_ret.second )
            // {
            //     finish = true;
            //     break;
            // }
            logged_obj_type::info_f(
                "increase threshold: %.01f, weight update from %le to %le with weight: %le, weight threshold: %le, "
                "returned_ok: %s",
                static_cast<double>( prm_.maximum_norm_increase ), static_cast<double>( normFx ),
                static_cast<double>( normFx1 ), static_cast<double>( newton_weight ),
                static_cast<double>( prm_.newton_weight_threshold ), parse_bool( update_ret.second ).c_str()
            );
            if ( update_ret.second && std::isfinite( normFx1 ) )
            {
                result_status = 1;
                if ( normFx1 <= tol() ) //converged
                {
                    //norms_storage.push_back(normFx1);
                    vec_space_->assign( *x1, x );
                    result_status = 0;
                    finish        = true;
                    break;
                }
                if ( ( normFx1 - normFx ) <= prm_.maximum_norm_increase * normFx )
                {
                    vec_space_->assign( *x1, x );
                    if ( std::abs( normFx1 - normFx ) < 1.0e-6 * normFx )
                    {
                        stagnation++;
                    }
                    else
                    {
                        stagnation = 0;
                    }
                    if ( stagnation > prm_.stagnation_max )
                    {
                        result_status = 5;
                        finish        = true;
                    }
                    break;
                }
            }
            if ( backtracking_trials >= prm_.max_backtracking_trials )
            {
                if ( ( result_status != 3 ) && ( result_status != 10 ) )
                {
                    result_status = 8;
                }
                finish = true;
                break;
            }
            newton_weight *= prm_.newton_weight_mul;
            if ( newton_weight < prm_.newton_weight_threshold )
            {
                if ( result_status != 3 && result_status != 10 )
                    result_status = 4;
                finish = true;
                break;
            }
        } while ( true );
        //store norm only if the step is successfull
        if ( prm_.store_norms_history )
        {
            norms_evolution.push_back( normFx1 );
        }

        //auto min_value = *std::min_element(norms_storage.begin(),norms_storage.end());
        //norms_storage.push_back(normFx1);
        iterations++;
        if ( ( iterations > prm_.max_iters_num ) && ( !finish ) )
        {
            result_status = 9; //maximum iterations reached.
            finish        = true;
        }
        auto result_status_string = parse_result_status( result_status );
        auto finish_string        = parse_bool( finish );
        logged_obj_type::info_f(
            "iteration: %u, max_iterations: %u, residuals n: %le, n+1: %le, min_value: %le, result_status: %i => %s, "
            "is_finished = %s, newton_weight = %le, stagnation = %u ",
            iterations, prm_.max_iters_num, static_cast<double>( normFx ), static_cast<double>( normFx1 ),
            static_cast<double>( Fx1_storage_norm_ ), result_status, result_status_string.c_str(),
            finish_string.c_str(), static_cast<double>( newton_weight ), stagnation
        );

        // store this solution point if the norm is the smalles of all
        if ( ( ( !std::isfinite( Fx1_storage_norm_ ) ) || ( Fx1_storage_norm_ >= normFx1 ) ) &&
             ( ( result_status == 1 ) || ( result_status == 4 ) ) )
        {
            vec_space_->assign( *x1, *x1_storage );
            Fx1_storage_norm_ = normFx1;
        }


        if ( finish )
        {
            //this sets minimum norm solution that is correct finite solution before solution algorithm stops
            if ( ( ( std::isfinite( Fx1_storage_norm_ ) ) && ( Fx1_storage_norm_ < normFx1 ) ) )
            {
                /// NOTE x1 is actually not needed anymore just for convinience (mb delete it?)
                vec_space_->assign( *x1_storage, *x1 );
                vec_space_->assign( *x1_storage, x );
                normFx1 = Fx1_storage_norm_;
                //signal that relaxed tolerance converged and put it into vector of signals
                if ( normFx1 <= tol() * prm_.relax_tolerance_factor )
                {
                    logged_obj_type::warning_f(
                        "Newton is setting relaxed tolerance = %le,  solution with norm = %le",
                        double( tol() * prm_.relax_tolerance_factor ), (double)normFx1
                    );
                    result_status = 0;
                }
            }

            /*bool relaxed_tolerance_reached_max = *std::max_element(relaxed_tolerance_reached.begin(),relaxed_tolerance_reached.end());
            if( relaxed_tolerance_reached_max&&(result_status>0)&&(relaxed_tolerance_reached.size()>0) )
            {
                auto min_value = *std::min_element(norms_storage.begin(),norms_storage.end());
                size_t soluton_num = 0;
                for(int jjj = 0;jjj<relaxed_tolerance_reached.size();jjj++)
                {
                    log->warning_f("continuation::convergence: solution %i: norm = %le, flag = %s, relaxed_tol = %le", soluton_num++, norms_storage[jjj], (relaxed_tolerance_reached[jjj]?"true":"false"), tolerance*relax_tolerance_factor  );
                }
            }*/
            //this signals that we couldn't set up the solution with the relaxed tolerance
            if ( result_status > 0 )
            {
                logged_obj_type::warning_f(
                    "newton step failed to finish: result_status = %i, ||x| = %le, relaxed_tol = %le", result_status,
                    static_cast<double>( vec_space_->norm_l2( x ) ),
                    static_cast<double>( tol() * prm_.relax_tolerance_factor )
                );
            }

            if ( quality_func )
            {
                //checks whaterver is needed for nans, errors or whaterver is considered a quality solution in the nonlinear operator.
                T solution_quality = quality_func->calc( x );
                logged_obj_type::info_f(
                    "Newton obtained solution quality = %le.", static_cast<double>( solution_quality )
                );
            }
        }

        return finish;
    }

    unsigned int get_number_of_iterations()
    {
        return iterations;
    }
    int get_result_status()
    {
        return result_status;
    }
    void reset_iterations()
    {
        iterations = 0;
        //reset_weight();
        norms_evolution.clear();
        stagnation       = 0;
        increment_ok_    = true; //increment ok
        Fx_initial_norm_ = 0;
    }
    /*void reset_weight()
    {

    }*/
    std::vector<T> *get_norms_history_handle()
    {
        return &norms_evolution;
    }

private:
    static void validate_tolerances( T abs_tol, T rel_tol )
    {
        if ( !std::isfinite( abs_tol ) || !std::isfinite( rel_tol ) || abs_tol < T( 0 ) || rel_tol < T( 0 ) )
        {
            throw std::invalid_argument( "Newton tolerances must be finite and nonnegative" );
        }
    }

    static void validate_parameters( const params &prm )
    {
        validate_tolerances( prm.abs_tol, prm.rel_tol );
        // Match the signed setter and leave room for the terminal iteration count.
        if ( prm.max_iters_num == 0 ||
             prm.max_iters_num > static_cast<unsigned int>( std::numeric_limits<int>::max() ) )
        {
            throw std::invalid_argument( "max_iters_num must be in [1, INT_MAX]" );
        }
        if ( !std::isfinite( prm.relax_tolerance_factor ) || prm.relax_tolerance_factor <= T( 0 ) )
        {
            throw std::invalid_argument( "relax_tolerance_factor must be finite and positive" );
        }
        if ( !std::isfinite( prm.maximum_norm_increase ) )
        {
            throw std::invalid_argument( "maximum_norm_increase must be finite" );
        }
        if ( !std::isfinite( prm.newton_weight_mul ) || prm.newton_weight_mul <= T( 0 ) ||
             prm.newton_weight_mul >= T( 1 ) )
        {
            throw std::invalid_argument( "Backtracking requires 0 < newton_weight_mul < 1" );
        }
        validate_backtracking_limits(
            prm.newton_weight_initial, prm.newton_weight_threshold, prm.max_backtracking_trials
        );
    }

#ifdef NMFD_ENABLE_NLOHMANN
    static unsigned int read_unsigned_parameter( const nlohmann::json &j, const char *key, unsigned int current )
    {
        const auto value = j.find( key );
        if ( value == j.end() )
        {
            return current;
        }
        if ( !value->is_number_integer() || *value < 0 || *value > std::numeric_limits<unsigned int>::max() )
        {
            throw std::invalid_argument( std::string( key ) + " must be an unsigned integer in range" );
        }
        return value->template get<unsigned int>();
    }
#endif

    static void validate_backtracking_limits( T initial_weight, T threshold, unsigned int max_trials )
    {
        if ( max_trials == 0 )
        {
            throw std::invalid_argument( "max_backtracking_trials must be positive" );
        }
        if ( !std::isfinite( initial_weight ) || !std::isfinite( threshold ) || threshold <= T( 0 ) ||
             initial_weight < threshold )
        {
            throw std::invalid_argument( "Backtracking requires finite weights with 0 < threshold <= initial_weight" );
        }
    }

    params prm_;

    std::shared_ptr<VectorSpace> vec_space_;

    unsigned int   iterations;
    unsigned int   stagnation = 0;
    int            result_status;
    bool           increment_ok_ = true;
    vec_wrap_type  x1, x1_storage, Fx;
    T              Fx_initial_norm_ = 0;
    T              Fx1_storage_norm_;
    T              newton_weight;
    std::vector<T> norms_evolution;

    /*int relax_tolerance_steps;
    T d_step;
    int current_relax_step;*/

    //updates a solution with a newton weight value provided
    std::pair<T, bool> inline update_solution(
        NonlinearOperator *nonlin_op, ProjectOperator *project_op, T_vec &x, T_vec &delta_x, T_vec &x1
    )
    {
        vec_space_->assign_lin_comb( static_cast<T>( 1.0 ), x, newton_weight, delta_x, x1 );
        bool returned_ok = true;

        if ( prm_.perform_full_check )
        {
            returned_ok = vec_space_->is_valid_number( x1 );
            if ( !returned_ok )
            {
                result_status = 3;
                return { std::numeric_limits<T>::infinity(), returned_ok };
            }
        }
        if ( project_op )
        {
            project_op->apply(
                x1
            ); // project to invariant solution subspace. Should be blank if nothing is needed to be projected.
            if ( prm_.perform_full_check )
            {
                returned_ok = vec_space_->is_valid_number( x1 );
                if ( !returned_ok )
                {
                    result_status = 10;
                    return { std::numeric_limits<T>::infinity(), returned_ok };
                }
            }
        }
        T normFx1 = ResidualNorm::evaluate( *vec_space_, *nonlin_op, x1, *Fx );
        return { normFx1, returned_ok };
    }


    std::string parse_result_status( int result_status )
    {
        switch ( result_status )
        {
        case 0:
            return { "converged" };
            break;
        case 1:
            return { "in progress" };
            break;
        case 2:
            return { "not finite input F(n)" };
            break;
        case 3:
            return { "not finite update n+1" };
            break;
        case 4:
            return { "too small update weight" };
            break;
        case 5:
            return { "stagnation" };
            break;
        case 6:
            return { "supplied increment was inconsistent" };
            break;
        case 7:
            return { "not finite du update" };
            break;
        case 8:
            return { "maximum backtracking attempts reached" };
            break;
        case 9:
            return { "maximum iterations reached" };
            break;
        case 10:
            return { "not finite projection returned" };
            break;
        case 11:
            return { "not finite input n" };
            break;
        default:
            return { "unknown state!" };
            break;
        }
    }

    std::string parse_bool( bool val )
    {
        return ( val ? "true" : "false" );
    }
};


}
}

#endif
