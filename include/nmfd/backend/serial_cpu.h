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

#ifndef __NMFD_BACKEND_SERIAL_CPU_H__
#define __NMFD_BACKEND_SERIAL_CPU_H__

#include <scfd/utils/log_std.h>
#include <scfd/backend/serial_cpu.h>

#include <nmfd/operations/dense_operations_base.h>
#include <nmfd/operations/dense_vector_space.h>

namespace nmfd
{
namespace backend
{

template <class Type, class Log = scfd::utils::log_std>
class serial_cpu : public scfd::backend::serial_cpu
{
    using traits_type = operations::detail::scfd_array_traits<Type, memory_type>;

public:
    using log_type              = Log;
    using scfd_backend_type     = scfd::backend::serial_cpu;
    using vector_space_type     = operations::dense_vector_space<traits_type, scfd_backend_type>;
    using dense_operations_type = operations::dense_operations<Type, scfd_backend_type>;

public:
    log_type &log()
    {
        return log_;
    }

protected:
    log_type log_;
};

} /// namespace backend
} /// namespace nmfd

#endif
