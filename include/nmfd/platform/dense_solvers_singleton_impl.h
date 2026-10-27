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

#ifndef __NMFD_PLATFORM_DENSE_SOLVERS_SINGLETON_IMPL_H__
#define __NMFD_PLATFORM_DENSE_SOLVERS_SINGLETON_IMPL_H__

#if defined( PLATFORM_CUDA )
#    include <scfd/external_libraries/cublas_wrap_singleton_impl.h>
#    include <scfd/external_libraries/cusolver_wrap_singleton_impl.h>
#elif defined( PLATFORM_HIP )
#    include <scfd/external_libraries/hipblas_wrap_singleton_impl.h>
#    include <scfd/external_libraries/hipsolver_wrap_singleton_impl.h>
#endif

#endif
