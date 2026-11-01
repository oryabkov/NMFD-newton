

#ifndef __NMFD_OPERATIONS_DETAIL_BACKEND_TRAITS_H__
#define __NMFD_OPERATIONS_DETAIL_BACKEND_TRAITS_H__


#include <type_traits>

namespace scfd
{
namespace backend
{
struct cuda;
}
}


namespace nmfd
{
namespace operations
{
namespace detail
{

template <class Backend>
struct backend_has_cusolver_like_wrap : std::false_type
{
};

template <>
struct backend_has_cusolver_like_wrap<scfd::backend::cuda> : std::true_type
{
};

template <class Backend>
struct backend_has_cublas_like_wrap : std::false_type
{
};

template <>
struct backend_has_cublas_like_wrap<scfd::backend::cuda> : std::true_type
{
};

}
}
}

#endif
