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
#include <scfd/utils/logged_obj_base.h>
#include <algorithm> // std::min_element
#include <iterator>  // std::begin, std::end
#include <nmfd/operations/ident_operator.h>
#include <nmfd/operations/zero_functional.h>
#include <nmfd/detail/algo_hierarchy_macro.h>
#include <nmfd/detail/algo_hierarchy_creator.h>

/// TODO check it!
/// NOTE originally taken from deflated_continuation timesteppers branch source/continuation/convergence_strategy.h 22.07.2025

namespace nmfd
{
namespace solvers
{


template
<
    class VectorSpace, 
    class Log, 
    class NonlinearOperator, 
    class ProjectOperator = operations::ident_operator<VectorSpace>, 
    class QualityFunctor = operations::zero_functional<VectorSpace>
>
class default_convergence_strategy
{
private:
    using T = typename VectorSpace::scalar_type;
    using T_vec = typename VectorSpace::vector_type;
    using logged_obj_t = scfd::utils::logged_obj_base<Log>;

public:    
    using vector_space_type = VectorSpace;

    struct params : public logged_obj_t::params
    {
        unsigned int stagnation_max = 10;  
        unsigned int maximum_iterations = 100;
        T tolerance = T(1.0e-6);
        T tolerance_0;
        T maximum_norm_increase = 0.0;
        T newton_weight_threshold = 1.0e-12;
        T newton_weight_initial = T(1);
        T newton_weight_mul = T(0.5);
        T relax_tolerance_factor;
        bool verbose = true, store_norms_history = false;

        params(
            const std::string &log_pefix = "", const std::string &log_name = "default_convergence_strategy::"
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
    NMFD_ALGO_HIERARCHY_TYPES_DEFINE(default_convergence_strategy)

    default_convergence_strategy(std::shared_ptr<VectorSpace> vec_space, Log* log_, params prm = params()) :
      prm_(prm),
      vec_space_(std::move(vec_space)),
      log(log_),
      iterations(0)
    {
        vec_space_->init_vector(x1); vec_space_->start_use_vector(x1);
        vec_space_->init_vector(x1_storage); vec_space_->start_use_vector(x1_storage);
        vec_space_->init_vector(Fx); vec_space_->start_use_vector(Fx);
        if(prm_.store_norms_history)
        {
            norms_evolution.reserve(prm_.maximum_iterations);
        }
    }
    default_convergence_strategy(  
        const utils_hierarchy& utils,
        const params_hierarchy& prm = params_hierarchy()      
    ) : 
        default_convergence_strategy(  
            utils.vec_space, utils.log, prm
        )
    {
    }
    ~default_convergence_strategy()
    {
        vec_space_->stop_use_vector(x1); vec_space_->free_vector(x1);
        vec_space_->stop_use_vector(x1_storage); vec_space_->free_vector(x1_storage);
        vec_space_->stop_use_vector(Fx); vec_space_->free_vector(Fx);
    }

    //T rel_tol()const { return prm_.rel_tol; }
    T abs_tol()const { return prm_.tolerance; }
    //T rel_tol_base()const { return rhs_norm(); }
    T tol()const 
    { 
        //return abs_tol() + rel_tol()*rel_tol_base(); 
        return abs_tol(); 
    }

    void set_convergence_constants(T tolerance_, unsigned int maximum_iterations_, T relax_tolerance_factor_, int relax_tolerance_steps_, T newton_weight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true, unsigned int stagnation_max_ = 10, T maximum_norm_increase_p = 0.0, T newton_weight_threshold_p = 1.0e-12)
    {
        prm_.tolerance = tolerance_;
        prm_.tolerance_0 = tolerance_;
        prm_.maximum_iterations = maximum_iterations_;
        newton_weight = newton_weight_;
        prm_.newton_weight_initial = newton_weight_;
        prm_.store_norms_history = store_norms_history_;
        prm_.verbose = verbose_;
        prm_.stagnation_max = stagnation_max_;
        prm_.relax_tolerance_factor = relax_tolerance_factor_;
        // relax_tolerance_steps = relax_tolerance_steps_;
        //current_relax_step = 0;
        if(prm_.store_norms_history)
        {
            norms_evolution.reserve(prm_.maximum_iterations);
        } 
        // T d_step = relax_tolerance_factor/T(relax_tolerance_steps);   
        
        // d_step = std::log10(relax_tolerance_factor)/T(relax_tolerance_steps);
        
        // log->info_f("continuation::convergence: check: relax_tolerance_steps = %i, relax_tolerance_factor = %le, d_step = %le, d_step_exp = %le", relax_tolerance_steps, (double)relax_tolerance_factor, (double)d_step, (double)std::pow<T>(T(10), d_step));
        prm_.maximum_norm_increase = maximum_norm_increase_p;
        prm_.newton_weight_threshold = newton_weight_threshold_p;

        log->info_f("continuation::convergence: check: relax_tolerance_factor = %le, maximum_norm_increase = %le, newton_weight_threshold = %le", (double)prm_.relax_tolerance_factor, (double)prm_.maximum_norm_increase, double(prm_.newton_weight_threshold) );

    }   

    

    bool check_convergence(NonlinearOperator *nonlin_op, ProjectOperator *project_op, QualityFunctor *quality_func, T_vec& x, T_vec& delta_x)
    {
        bool finish = false; //states that the newton process should stop.
        // result_status defines on how this process is stoped.
        newton_weight = prm_.newton_weight_initial;
        nonlin_op->apply(x, Fx);
        T normFx = vec_space_->norm_l2(Fx);
        if(!std::isfinite(normFx)) //set result_status = 2 if the provided vector is inconsistent
        {
            result_status = 2;
            //finish = true; 
            return true;
        }
        if(normFx < tol()) //do nothing is my kind of problem =)
        {
            result_status = 0;
            log->info_f("continuation::convergence: iteration %i, residuals n: %le < tol(): %le => finished.",iterations, (double)normFx, (double)tol() );            
            return true;
        }
        if(iterations == 0)
        {
            /// stores initial solution and norm
            vec_space_->assign(x, x1_storage);
            Fx1_storage_norm_ = normFx;
            /// postulate continue (other cases checked earlier) and continue
            result_status = 1;
            iterations++;
            return false;
        }
        T normFx1;
        result_status = 3;
        do
        {
            //update solution
            normFx1 = update_solution(nonlin_op, project_op, x, delta_x, x1);
            log->info_f("continuation::convergence: increase threshold: %.01f, weight update from %le to %le with weight: %le and weight threshold: %le ", prm_.maximum_norm_increase, normFx, normFx1, newton_weight,  prm_.newton_weight_threshold);
            if(std::isfinite(normFx1))
            {
                result_status = 1;
                //finish = true; 
                if(normFx1 < tol()) //converged
                {
                    //norms_storage.push_back(normFx1);
                    vec_space_->assign(x1, x);
                    result_status = 0;
                    finish = true;
                    break;
                }
                if ((normFx1 - normFx) <= prm_.maximum_norm_increase*normFx)
                {
                    vec_space_->assign(x1, x);
                    if( std::abs(normFx1 - normFx) < 1.0e-6*normFx )
                    {
                        stagnation++;
                    }
                    if(stagnation > prm_.stagnation_max)
                    {
                        result_status = 5;
                        finish = true;
                    }
                    break;
                }
            }
            newton_weight *= prm_.newton_weight_mul;
            if(newton_weight < prm_.newton_weight_threshold)
            {
                if (result_status != 3) result_status = 4;
                finish = true;
                break;
            }
        } while ( true );
        //store norm only if the step is successfull
        if(prm_.store_norms_history)
        {
            norms_evolution.push_back(normFx1);
        }

        //auto min_value = *std::min_element(norms_storage.begin(),norms_storage.end());
        //norms_storage.push_back(normFx1);
        iterations++;
        if(iterations > prm_.maximum_iterations)
        {
            finish = true;         
        }
        auto result_status_string = parse_result_status(result_status);
        auto finish_string = parse_bool(finish);
        log->info_f("continuation::convergence: iteration: %i, max_iterations: %i, residuals n: %le, n+1: %le, min_value: %le, result_status: %i => %s, is_finished = %s, newton_weight = %le, stagnation = %u ",iterations, prm_.maximum_iterations, (double)normFx, (double)normFx1, double(Fx1_storage_norm_), result_status,  result_status_string.c_str(), finish_string.c_str(), newton_weight, stagnation );

        // store this solution point if the norm is the smalles of all
        if( ( (!std::isfinite(Fx1_storage_norm_))||(Fx1_storage_norm_ >= normFx1) )&&( (result_status == 1)||(result_status == 4) ) )
        {
            vec_space_->assign(x1, x1_storage);
            Fx1_storage_norm_ = normFx1;
        }


        if (finish)
        {
            //this sets minimum norm solution that is correct finite solution before solution algorithm stops
            if( ( (std::isfinite(Fx1_storage_norm_))&&(Fx1_storage_norm_ < normFx1) ) )
            {
                /// NOTE x1 is actually not needed anymore just for convinience (mb delete it?)
                vec_space_->assign(x1_storage, x1);
                vec_space_->assign(x1_storage, x);
                normFx1 = Fx1_storage_norm_;
                //signal that relaxed tolerance converged and put it into vector of signals
                if( normFx1 <= tol()*prm_.relax_tolerance_factor  )
                {
                    log->warning_f("continuation::convergence: Newton is setting relaxed tolerance = %le,  solution with norm = %le", double(tol()*prm_.relax_tolerance_factor), (double)normFx1 );
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
            if (result_status>0)
            {
                log->error_f("continuation::convergence: newton step failed to finish: result_status = %i, ||x| = %le, relaxed_tol = %le", result_status, vec_space_->norm_l2(x), tol()*prm_.relax_tolerance_factor );
            }

            if (quality_func)
            {
                //checks whaterver is needed for nans, errors or whaterver is considered a quality solution in the nonlinear operator.
                T solution_quality = quality_func->calc(x);
                log->info_f("continuation::convergence: Newton obtained solution quality = %le.", solution_quality);
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
        stagnation = 0;
    }
    /*void reset_weight()
    {
        
    }*/
    std::vector<T>* get_norms_history_handle()
    {
        return &norms_evolution;
    }

private:
    params prm_;
      
    std::shared_ptr<VectorSpace> vec_space_;
    Log* log;
    
    unsigned int iterations;
    unsigned int stagnation = 0;

    int result_status;

    T_vec x1, x1_storage, Fx;
    T Fx1_storage_norm_;
    T newton_weight;
    std::vector<T> norms_evolution;
    
    /*int relax_tolerance_steps;
    T d_step;
    int current_relax_step;*/

    //updates a solution with a newton weight value provided
    T inline update_solution(NonlinearOperator *nonlin_op, ProjectOperator *project_op, T_vec& x, T_vec& delta_x, T_vec& x1)
    {
        vec_space_->assign_mul(static_cast<T>(1.0), x, newton_weight, delta_x, x1);
        if (project_op)
        {
            project_op->apply(x1); // project to invariant solution subspace. Should be blank if nothing is needed to be projected.
        }
        nonlin_op->apply(x1, Fx);
        T normFx1 = vec_space_->norm_l2(Fx);
        return normFx1;
    }


    std::string parse_result_status(int result_status)
    {
        switch(result_status)
        {
            case 0:
                return{"converged"};
                break;
            case 1:
                return{"in progress"};
                break;                
            case 2:
                return{"not finite input n"};
                break;
            case 3:
                return{"not finite update n+1"};
                break;   
            case 4:
                return{"too small update weight"};
                break;
            case 5:
                return{"stagnation"};
                break;
            default:
                return{"unknown state!"};
                break;
        }
    }

    std::string parse_bool(bool val)
    {
        return (val?"true":"false");
    }


};


}
}

#endif