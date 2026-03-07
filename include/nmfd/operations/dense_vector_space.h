#ifndef __NMFD_DENSE_VECTOR_SPACE_H__
#define __NMFD_DENSE_VECTOR_SPACE_H__

#include <scfd/utils/todo.h>

#include <nmfd/operations/kernels/dense_vector_space.h>
#include <nmfd/operations/dense_vector_operations.h>

namespace nmfd
{
namespace operations
{


template <class Type, class VectorTraits, class Backend, class Ordinal = std::ptrdiff_t>
class dense_vector_space : public dense_vector_operations<Type, VectorTraits, Backend, Ordinal>
{
public:
    using vector_type = typename VectorTraits::vector_type;

public:
    dense_vector_space() = default;

    template <typename... Args>
    dense_vector_space( Args &&...args )
        : dense_vector_operations<Type, VectorTraits, Backend, Ordinal>( std::forward<Args>( args )... )
    {
    }

    void init_vector( vector_type &vec ) const
    {
        const VectorTraits &vt = this->get_vector_traits();
        vt.alloc( vt.loc_size(), vec );
    }
    template <class... Args>
    void init_vectors( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)init_vector( std::forward<Args>( args ) ), 0 )... };
    }

    void free_vector( vector_type &vec ) const
    {
        const VectorTraits &vt = this->get_vector_traits();
        vt.dealloc( vec );
    }
    template <class... Args>
    void free_vectors( Args &&...args ) const
    {
        std::initializer_list<int>{ ( (void)free_row_vector( std::forward<Args>( args ) ), 0 )... };
    }

    void start_use_vector( vector_type &x ) const
    {
    }
    template <class... Args>
    void start_use_vectors( Args &&...args ) const
    {
    }

    void stop_use_vector( vector_type &x ) const
    {
    }
    template <class... Args>
    void stop_use_vectors( Args &&...args ) const
    {
    }
};

} // namespace operations

} // namespace nmfd

#endif
