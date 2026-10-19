#ifndef NMFD_TEST_CH_LOCAL_COMMUNICATION_H
#define NMFD_TEST_CH_LOCAL_COMMUNICATION_H

#include <memory>
#include <scfd/communication/trivial_platform.h>
#include <scfd/communication/rect_distributor.h>

namespace tests
{

template <class VectorSpace>
struct local_communication
{
    using vector_space_type      = VectorSpace;
    using backend_type           = typename VectorSpace::backend_type;
    using ordinal_type           = typename VectorSpace::ordinal_type;
    using memory_type            = typename VectorSpace::memory_type;
    using comm_type              = typename VectorSpace::comm_type;
    using idx_type               = typename VectorSpace::idx_nd_type;
    static constexpr int dim     = VectorSpace::dim;
    static constexpr int stencil = 2;
    using dist_type              = scfd::communication::rect_distributor<
                     typename VectorSpace::scalar_type, dim, memory_type, typename backend_type::template for_each_nd_type<dim>,
                     ordinal_type, long, comm_type>;
    using part_type     = typename dist_type::rect_partitioner_t;
    using big_idx_type  = scfd::static_vec::vec<long, dim>;
    using big_rect_type = scfd::static_vec::rect<long, dim>;

    scfd::communication::trivial_platform<memory_type> platform{ 0, nullptr };
    part_type                                          part;
    typename dist_type::bool_vec_t                     periodic_flags = dist_type::bool_vec_t::make_zero();
    std::shared_ptr<dist_type>                         distributor    = std::make_shared<dist_type>();
    std::shared_ptr<VectorSpace>                       space;

    local_communication(
        idx_type range, const int ( &left )[dim][VectorSpace::tensor_dim],
        const int ( &right )[dim][VectorSpace::tensor_dim]
    )
        : part( platform.comm_world(), big_idx_type( range ) )
    {
        for ( int axis = 0; axis < dim; ++axis )
            for ( int component = 0; component < VectorSpace::tensor_dim; ++component )
                periodic_flags[axis] =
                    periodic_flags[axis] || left[axis][component] == 0 || right[axis][component] == 0;
        part.proc_rects = { big_rect_type( big_idx_type::make_zero(), big_idx_type( range ) ) };
        distributor->init_for_tensors( VectorSpace::tensor_dim, part, periodic_flags, stencil, dim );
        space = std::make_shared<VectorSpace>( range, platform.comm_world(), false, stencil, dim );
    }

    typename VectorSpace::vector_type vector() const
    {
        typename VectorSpace::vector_type result;
        space->init_vector( result );
        return result;
    }
};

}

#endif
