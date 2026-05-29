// Copyright © 2016-2021 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of SimpleCFD.

// SimpleCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SimpleCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SimpleCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCFD_MESH_DEVICE_MESH_CONST_DATA_ACCESS_H__
#define __SCFD_MESH_DEVICE_MESH_CONST_DATA_ACCESS_H__

#include "scfd/utils/device_tag.h"
namespace scfd
{
namespace mesh
{
namespace detail
{

template<class T,class Memory,int Dim,class Ord>
__DEVICE_TAG__ const device_mesh<T,Memory,Dim,Ord> &get_mesh()
{
    //TODO some kind of error
}

template<class T>
__DEVICE_TAG__ const gmsh_mesh_elem_reference<T> &get_elem_ref()
{
    //TODO some kind of error
}

}  /// namespace detail
}  /// namespace mesh
}  /// namespace scfd

#endif
