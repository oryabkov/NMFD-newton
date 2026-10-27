// Copyright © 2016-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch, Sorokin Ivan Antonovich

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

#ifndef __NMFD_PLATFORM_H__
#define __NMFD_PLATFORM_H__

#include <scfd/utils/log_std.h>

#if defined( PLATFORM_SERIAL_CPU )
#    include "serial_cpu.h"

namespace nmfd
{
namespace platform
{

template <class Type, class Log = scfd::utils::log_std>
using current = serial_cpu<Type, Log>;

} // namespace platform
} // namespace nmfd

#elif defined( PLATFORM_CUDA )
#    include "cuda.h"

namespace nmfd
{
namespace platform
{

template <class Type, class Log = scfd::utils::log_std>
using current = cuda<Type, Log>;

} // namespace platform
} // namespace nmfd

#elif defined( PLATFORM_HIP )
#    include "hip.h"

namespace nmfd
{
namespace platform
{

template <class Type, class Log = scfd::utils::log_std>
using current = hip<Type, Log>;

} // namespace platform
} // namespace nmfd

#elif defined( PLATFORM_OMP )
#    include "omp.h"

namespace nmfd
{
namespace platform
{

template <class Type, class Log = scfd::utils::log_std>
using current = omp<Type, Log>;

} // namespace platform
} // namespace nmfd

#else
#    error "No platform has been chosen (define PLATFORM_SERIAL_CPU or PLATFORM_CUDA or PLATFORM_HIP or PLATFORM_OMP)."
#endif

namespace nmfd
{
namespace platform
{

template <class Type, class Log = scfd::utils::log_std>
using vector_space_type = typename current<Type, Log>::vector_space_type;

template <class Type, class Log = scfd::utils::log_std>
using dense_operations_type = typename current<Type, Log>::dense_operations_type;

} // namespace platform
} // namespace nmfd

#endif // __NMFD_PLATFORM_H__
