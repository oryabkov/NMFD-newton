#ifndef __TIME_DERIVATIVE_H__
#define __TIME_DERIVATIVE_H__

#include <cmath>
#include <memory>
#include <scfd/utils/device_tag.h>
#include <scfd/static_vec/vec.h>
#include <nmfd/detail/vector_wrap.h>

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
#define PI M_PI

namespace tests
{

template <class VectorSpace, class TensorType>
class time_derivative
{
public:
    static const int dim        = VectorSpace::dim;
    static const int tensor_dim = VectorSpace::tensor_dim;

    using scalar_type       = typename VectorSpace::scalar_type;
    using tensor_type       = scfd::static_vec::vec<scalar_type, tensor_dim>;
    using vector_type       = typename VectorSpace::vector_type;
    using vector_space_type = VectorSpace;
    using vector_space_ptr  = std::shared_ptr<VectorSpace>;
    using idx_nd_type       = typename VectorSpace::idx_nd_type;

public:
    time_derivative( idx_nd_type range )
        : vspace_( std::make_shared<vector_space_type>( range ) ), previous_state_wrap_( *vspace_ )
    {
        vspace_->assign_scalar( 0.0, *previous_state_wrap_ );
    }

    time_derivative( const vector_space_type &vspace ) : time_derivative( vspace.get_size() )
    {
    }

    vector_space_ptr get_space() const
    {
        return vspace_;
    }

    vector_type get_previous_state() const
    {
        return *previous_state_wrap_;
    }

    void set_previous_state( const vector_type &state )
    {
        vspace_->assign( state, *previous_state_wrap_ );
    }

    scalar_type get_dt_inf() const
    {
        return dt_inf_;
    }

    void set_dt_inf( scalar_type dt_inf )
    {
        dt_inf_ = dt_inf;
    }


private:
    vector_space_ptr vspace_;
    scalar_type      dt_inf_ = scalar_type( 0.0 );

    using vector_wrap_t = nmfd::detail::vector_wrap<VectorSpace, true, true>;
    vector_wrap_t previous_state_wrap_;
};

} // namespace tests

#endif
