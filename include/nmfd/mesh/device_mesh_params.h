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

#ifndef __SCFD_MESH_DEVICE_MESH_PARAMS_H__
#define __SCFD_MESH_DEVICE_MESH_PARAMS_H__

namespace scfd
{
namespace mesh
{

/// TODO do we need to add ALL data structures (like elements and faces types) here?
/// mb, add some global switches like has_elems_data, that turn on/off whole blocks of data?

struct device_mesh_params
{
    /// Elements data

    bool has_elems_centers_data = true;
    bool has_elems_neighbours0_centers_data = true;
    bool has_elems_faces_centers_data = true;
    bool has_elems_vertexes_data = true;
    bool has_elems_neighbours0_data = true;
    bool has_elems_neighbours0_loc_face_i_data = true;
    bool has_elems_faces_group_ids_data = true;
    bool has_elems_group_ids_data = true;
    bool has_elems_faces_norms_data = true;
    bool has_elems_faces_areas_data = true;
    bool has_elems_vols_data = true;
    bool has_elems_nodes_data = true;
    bool has_elems_prim_nodes_data = true;
    bool has_elems_faces_data = true;
    bool has_elems_virt_faces_data = true;
    bool has_elems_virt_faces_virt_pairs_data = true;

    /// Nodes data 

    //TODO

    /// Faces data

    bool has_faces_group_ids_data = true;
    bool has_faces_centers_data = true;
    bool has_faces_vertexes_data = true;
    bool has_faces_nodes_data = true;
    bool has_faces_prim_nodes_data = true;
    /// has_faces_elems_ids_data includes faces_elems_nums, faces_elems_ids and faces_elems_inelem_face_inds
    bool has_faces_elems_ids_data = true;
    bool has_virt_faces_elems_ids_data = true;
    bool has_faces_virt_master_ids_data = true;
};

}  /// namespace mesh
}  /// namespace scfd

#endif
