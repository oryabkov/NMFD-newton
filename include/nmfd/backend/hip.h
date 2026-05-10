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

#ifndef __NMFD_BACKEND_CUDA_H__
#define __NMFD_BACKEND_CUDA_H__

#include <scfd/external_libraries/hipblas_wrap.h>
#include <scfd/utils/log_std.h>
#include <scfd/backend/hip.h>

#include <nmfd/operations/dense_operations_cuda_hip.h>
#include <nmfd/operations/dense_vector_space.h>

namespace nmfd
{
namespace backend
{

template <class Type, class Log = scfd::utils::log_std>
class hip : public scfd::backend::hip
{
    using traits_type = operations::detail::scfd_array_traits<Type, memory_type>;

public:
    using log_type              = Log;
    using scfd_backend_type     = scfd::backend::hip;
    using vector_space_type     = operations::dense_vector_space<traits_type, scfd_backend_type>;
    using dense_operations_type = operations::dense_operations_cuda_hip<Type, scfd_backend_type>;
    using cublas_t              = scfd::hipblas_wrap;
    using cusolver_t            = scfd::hipsolver_wrap;

public:
    log_type &log()
    {
        return log_;
    }

    cublas_t &cublas()
    {
        return scfd::hipblas_wrap::inst();
    }

    cusolver_t &cusolver()
    {
        return scfd::hipsolver_wrap::inst();
    }

protected:
    scfd::hipblas_wrap   hipblas_{ true };
    scfd::hipsolver_wrap hipsolver_{ &hipblas_, true };
    log_type             log_;
};

} /// namespace backend
} /// namespace nmfd

#endif
