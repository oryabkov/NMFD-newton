#ifndef __COARSENING_H__
#define __COARSENING_H__

#include <tuple>
#include <memory>

#include "restrictor.h"
#include "prolongator.h"

#include <scfd/communication/trivial_comm.h>
#include <scfd/communication/rect_partitioner.h>
#include <scfd/communication/rect_distributor.h>

namespace tests
{

template <class LinearOperator, class Log>
class coarsening
{
public:
    using operator_type     = LinearOperator;
    using vector_space_type = typename operator_type::vector_space_type;
    using vector_type       = typename vector_space_type::vector_type;
    using restrictor_type   = restrictor<vector_space_type, Log>;
    using prolongator_type  = prolongator<vector_space_type, Log>;

    using ordinal_type = typename vector_space_type::ordinal_type;
    using idx_nd_type  = typename vector_space_type::idx_nd_type;
    using scalar_type  = typename vector_space_type::scalar_type;
    using grid_step_type = typename restrictor_type::grid_step_type;

    static const int dim        = operator_type::dim;
    static const int tensor_dim = operator_type::tensor_dim;

    using dist_type      = typename operator_type::dist_type;
    using dist_ptr       = typename operator_type::dist_ptr;
    using part_type      = typename dist_type::rect_partitioner_t;
    using comm_type      = typename dist_type::comm_type;
    using bool_vec_t     = typename dist_type::bool_vec_t;
    using big_ord_vec_t    = typename part_type::big_ord_vec_t;
    using big_ord_rect_t   = typename part_type::big_ord_rect_t;
    using big_ordinal_type = typename part_type::big_ordinal;

public:
    struct params
    {
    };
    using params_hierarchy = params;
    struct utils
    {
        part_type      part;
        bool_vec_t     periodic_flags;
        ordinal_type   stencil           = ordinal_type( 1 );
        int            max_stencil_order = 1;
    };
    using utils_hierarchy = utils;

    coarsening( const utils_hierarchy &u, const params_hierarchy &p ) : utils_( u )
    {
    }

    std::tuple<std::shared_ptr<restrictor_type>, std::shared_ptr<prolongator_type>> next_level( const operator_type &op)
    {
        auto fine_step   = op.get_h();
        auto coarse_step = fine_step * scalar_type( 2 );

        auto res = std::make_shared<restrictor_type>( op.get_size(), fine_step, op.get_b_cond(), utils_.part.comm_info );
        auto pro = std::make_shared<prolongator_type>( op.get_size(), coarse_step, op.get_b_cond(), utils_.part.comm_info );

        auto fine_lin = op.get_lin_vector();
        res->set_linearization_point( fine_lin );

        return std::make_tuple( res, pro );
    }

    std::shared_ptr<operator_type>
    coarse_operator( const operator_type &op, const restrictor_type &restrictor, prolongator_type &prolongator )
    {
        using Ord    = ordinal_type;
        using Scalar = scalar_type;

        auto coarse_size = op.get_size() / Ord{ 2 };
        auto coarse_h    = op.get_h() * Scalar{ 2 };

        scalar_type C = 4;
        Scalar max_h = coarse_h[0];
        for ( int i = 0; i < coarse_h.dim; ++i )
        {
            max_h = std::max(coarse_h[i], max_h);
        }
        // Scalar new_gamma = op.get_gamma();
        // Scalar new_gamma = 4 * op.get_gamma();
        Scalar new_gamma = std::max( op.get_gamma(), C*max_h*max_h );

        auto b_cond = op.get_b_cond();
        b_cond.set_gamma( new_gamma );


        const part_type &finest_part  = utils_.part;
        auto             finest_block = finest_part.get_own_rect().calc_size();

        // Compute current level of mg
        big_ord_vec_t scale;
        for ( int j = 0; j < dim; ++j )
        {
            scale[j] = finest_block[j] / big_ordinal_type( coarse_size[j] );
        }

        // Cut the rects of the partitioner for the next level
        part_type coarse_part = finest_part;
        for ( int j = 0; j < dim; ++j )
        {
            coarse_part.dom_size[j] = finest_part.dom_size[j] / scale[j];
        }
        for ( auto &r : coarse_part.proc_rects )
        {
            for ( int j = 0; j < dim; ++j )
            {
                r.i1[j] /= scale[j];
                r.i2[j] /= scale[j];
            }
        }

        auto coarse_dist = std::make_shared<dist_type>();
        coarse_dist->init_for_tensors(
            tensor_dim, coarse_part, utils_.periodic_flags, utils_.stencil, utils_.max_stencil_order );

        // Coarse vector space must carry the same stencil as the fine one
        auto coarse_vspace = std::make_shared<vector_space_type>(
            coarse_size, finest_part.comm_info, false, utils_.stencil, utils_.max_stencil_order );

        auto coarse_op = std::make_shared<operator_type>(
            coarse_vspace, coarse_h, b_cond, coarse_dist, op.get_time_derivative() );

        coarse_op->set_mobility( op.get_mobility() );
        coarse_op->set_gamma( new_gamma );

        // Restrict the linearization point from fine to coarse level
        vector_type fine_vector = op.get_lin_vector();
        vector_type coarse_vector( coarse_size );
        restrictor.apply( fine_vector, coarse_vector, false );
        coarse_op->set_linearization_point( coarse_vector );

        // Set the prolongator's b_cond and linearization point (both coarse-level)
        prolongator.set_b_cond( b_cond );
        prolongator.set_linearization_point( coarse_vector );

        return coarse_op;
    }

    bool coarse_enough( const operator_type &op ) const
    {
        auto range = op.get_size();
        for ( int i = 0; i < range.dim; ++i )
            if ( range[i] <= 2 )
                return true;
        return false;
    }

private:
    utils utils_;
};

} // namespace tests

#endif
