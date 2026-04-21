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

#ifndef __SCFD_DEVICE_DG_QUBATURE_REF_CONST_DATA_ACCESS_H__
#define __SCFD_DEVICE_DG_QUBATURE_REF_CONST_DATA_ACCESS_H__

#include <nmfd/mesh/gmsh_mesh_elem_reference.h>
#include <nmfd/mesh/cubature_reference.h>
#include "device_dg_basis_reference.h"
#include <nmfd/mesh/device_mesh.h>
#include <scfd/dg/device_dg_qubature_ref.h>

namespace scfd
{
namespace dg
{
namespace detail
{

template<class T>
__DEVICE_TAG__ const mesh::gmsh_mesh_elem_reference<T> &get_elem_ref()
{
    //TODO some kind of error
}

template<class T>
__DEVICE_TAG__ const mesh::cubature_reference<SCFD_DG_MAX_POLY_ORDER*2+1,T> &get_cub_ref()
{
    //TODO some kind of error
}

template<class T, int Dim, int MaxPolyDeg>
__DEVICE_TAG__ const device_dg_basis_reference<T,Dim,SCFD_DG_MAX_POLY_ORDER> &get_basis_ref()
{
    //TODO some kind of error
}

template<class T,class Memory,int Dim,class Ord>
__DEVICE_TAG__ const mesh::device_mesh<T,Memory,Dim,Ord> &get_mesh()
{
    //TODO some kind of error
}

template<class T,class Memory,int Dim>
__DEVICE_TAG__ const device_dg_qubature_ref<T,Memory,Dim> &get_dg_qubature_ref()
{
    //TODO some kind of error
}

}  /// namespace detail
}  /// namespace mesh
}  /// namespace scfd

#endif
