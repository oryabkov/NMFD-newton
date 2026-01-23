#ifndef __COARSENING_H__
#define __COARSENING_H__

#include <tuple>

#include "restrictor.h"
#include "prolongator.h"

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

public:
    struct params
    {
    };
    using params_hierarchy = params;
    struct utils
    {
    };
    using utils_hierarchy = utils;

    coarsening( const utils_hierarchy &u, const params_hierarchy &p )
    {
    }

    std::tuple<std::shared_ptr<restrictor_type>, std::shared_ptr<prolongator_type>> next_level( const operator_type &op
    )
    {
        return std::make_tuple(
            std::make_shared<restrictor_type>( op.get_size() ), std::make_shared<prolongator_type>( op.get_size() )
        );
    }

    std::shared_ptr<operator_type>
    coarse_operator( const operator_type &op, const restrictor_type &restrictor, const prolongator_type &prolongator )
    {
        using Ord    = ordinal_type;
        using Scalar = scalar_type;

        auto coarse_size = op.get_size() / Ord{ 2 };
        auto coarse_h    = op.get_h() * Scalar{ 2 };

        // Create coarse operator with same time_derivative parameters, D and gamma
        auto coarse_op =
            std::make_shared<operator_type>( coarse_size, coarse_h, op.get_b_cond(), op.get_time_derivative() );

        // Restrict the linearization point from fine to coarse level
        vector_type fine_vector = op.get_vector();
        vector_type coarse_vector( coarse_size );
        restrictor.apply( fine_vector, coarse_vector );
        coarse_op->set_vector( coarse_vector );

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
};

} // namespace tests

#endif
