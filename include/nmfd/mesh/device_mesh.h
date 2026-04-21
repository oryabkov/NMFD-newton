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

#ifndef __SCFD_MESH_DEVICE_MESH_H__
#define __SCFD_MESH_DEVICE_MESH_H__

#include <scfd/arrays/array.h>
#include <scfd/arrays/tensorN_array.h>
#include "device_mesh_params.h"
#include "host_mesh.h"

namespace scfd
{
namespace mesh
{

template<class Ord>
struct index_range_descr
{
    Ord i0, n;

    __DEVICE_TAG__ Ord i1()const
    {
        return i0+n;
    }
};

namespace detail
{
/// Forward declaration; it is needed only for type specialization inside device_mesh class scope
template<class T,class Memory,int Dim,class Ord>
struct device_mesh_funcs;
}

//map_n (nodes map) and map_e (element map) during all calls must be the same
//ISSUE perhaps, better to make copy of maps inside (but what to do with gpu in this case?)

//ISSUE should not we make own flag?? (because this structure is copied to const memory)

//TODO for now Ord = SCFD_ARRAYS_ORDINAL_TYPE is only supported (arrays don't have Ord template parameter for now)

//TODO is it really safe?
using namespace arrays;

template<class T,class Memory,int Dim = 3,class Ord = SCFD_ARRAYS_ORDINAL_TYPE>
struct device_mesh
{
    using scalar_type = T;
    using memory_type = Memory;
    using ordinal_type = Ord;
    using elem_type_ordinal_type = int;
    using index_range_descr_type = index_range_descr<Ord>;

    static const int          dim = Dim;
    static const ordinal_type special_id = std::numeric_limits<ordinal_type>::max();

    using device_mesh_funcs_t = detail::device_mesh_funcs<T,Memory,Dim,Ord>;


    device_mesh_params  params;

    //using namespace detail;

    //ISSUE neither n_cv
    //but i0, n_cv_all somehow does

    ordinal_type        max_faces_n, max_prim_nodes_n, max_nodes_n;
    ordinal_type        faces_max_prim_nodes_n, faces_max_nodes_n;

    /// Elements data part

    index_range_descr_type                          elems_range,
                                                    own_elems_range;

    //vars buffers CONTAINS space for elements in [i0, i0+n_cv_all)
    //Ord                                             i0, n_cv_all;
    //gpu OWN elements in [0,n_cv)
    //Ord                                             n_cv;
    //if is_homogeneous == true then all elements in mesh have the same elem_type
    bool                                            is_homogeneous;
    elem_type_ordinal_type                          homogeneous_elem_type;  //valid only if is_homogeneous == true
    array<elem_type_ordinal_type,Memory>            elems_types;            //valid only if is_homogeneous == false
    tensor1_array<T,Memory,Dim>                     elems_centers;
    tensor2_array<T,Memory,dyn_dim,Dim>             elems_neighbours0_centers;
    tensor2_array<T,Memory,dyn_dim,Dim>             elems_virt_neighbours0_centers;
    tensor2_array<T,Memory,dyn_dim,Dim>             elems_faces_centers;
    //TODO replace with var array
    tensor2_array<T,Memory,dyn_dim,Dim>             elems_vertexes;
    tensor1_array<Ord,Memory,dyn_dim>               elems_neighbours0;
    tensor1_array<Ord,Memory,dyn_dim>               elems_neighbours0_loc_face_i;
    tensor1_array<Ord,Memory,dyn_dim>               elems_virt_neighbours0;
    tensor1_array<Ord,Memory,dyn_dim>               elems_virt_neighbours0_loc_face_i;
    tensor1_array<Ord,Memory,dyn_dim>               elems_faces_group_ids;
    array<Ord,Memory>                               elems_group_ids;
    tensor2_array<T,Memory,dyn_dim,Dim>             elems_faces_norms;
    tensor1_array<T,Memory,dyn_dim>                 elems_faces_areas;
    tensor1_array<T,Memory,1>                       elems_vols;
    /// Elements to nodes graphs
    tensor1_array<Ord,Memory,dyn_dim>               elems_prim_nodes_ids;
    //TODO replace with var array
    tensor1_array<Ord,Memory,dyn_dim>               elems_nodes_ids;
    /// Elements to faces graphs
    tensor1_array<Ord,Memory,dyn_dim>               elems_faces_ids;
    tensor1_array<Ord,Memory,dyn_dim>               elems_faces_inface_elem_inds;
    tensor1_array<Ord,Memory,dyn_dim>               elems_virt_faces_ids;
    tensor1_array<Ord,Memory,dyn_dim>               elems_virt_faces_inface_elem_inds;
    /// Only faces with actual virt pairs have values here - other ones have special_id here
    /// ISSUE this interface is excessive - we may use corresponding array for faces only
    tensor1_array<Ord,Memory,dyn_dim>               elems_virt_faces_virt_pairs_ids;

    /// Nodes data part

    index_range_descr_type                          nodes_range,
                                                    own_nodes_range;
    //Ord                                             i0_nodes, n_nodes_all;
    //Ord                                             n_nodes;
    tensor1_array<T,Memory,Dim>                     nodes_coords;
    tensor0_array<Ord,Memory>                       nodes_group_ids;
    //tensor0_array<Ord,Memory>                       node_bnd_id;

    //nodes to elements graph part
    Ord                                             node_2_elem_graph_sz;
    tensor1_array<Ord,Memory,2>                     node_2_elem_graph_refs;
    tensor0_array<Ord,Memory>                       node_2_elem_graph_elem_ids;
    tensor0_array<Ord,Memory>                       node_2_elem_graph_node_ids;

    /// Faces data part

    index_range_descr_type                          faces_range,
                                                    own_faces_range;

    /// Elements types corresponding for faces
    ///TODO there are issues with this - we cannot just use is_homogeneous flag - we need separate flag for faces
    ///TODO for now homogeneous_face_type is not valid
    elem_type_ordinal_type                          homogeneous_face_type;  //valid only if is_homogeneous == true
    array<elem_type_ordinal_type,Memory>            faces_types;            //valid only if is_homogeneous == false
    array<Ord,Memory>                               faces_group_ids;
    tensor1_array<T,Memory,Dim>                     faces_centers;
    //TODO replace with var array
    tensor2_array<T,Memory,dyn_dim,Dim>             faces_vertexes;
    /// Faces to nodes graphs
    tensor1_array<Ord,Memory,dyn_dim>               faces_prim_nodes_ids;
    //TODO replace with var array
    tensor1_array<Ord,Memory,dyn_dim>               faces_nodes_ids;

    array<Ord,Memory>                               faces_elems_nums;
    tensor1_array<Ord,Memory,2>                     faces_elems_ids;
    tensor1_array<Ord,Memory,2>                     faces_elems_inelem_face_inds;

    array<Ord,Memory>                               virt_faces_elems_nums;
    tensor1_array<Ord,Memory,2>                     virt_faces_elems_ids;
    tensor1_array<Ord,Memory,2>                     virt_faces_elems_inelem_face_inds;

    array<Ord,Memory>                               faces_virt_master_ids;

    /// Virtual pairs data part
    tensor2_array<T,Memory,Dim,Dim>                 virt_pairs_mats;
    tensor2_array<T,Memory,Dim,Dim>                 virt_pairs_inv_mats;
    tensor1_array<T,Memory,Dim>                     virt_pairs_vecs;

    __DEVICE_TAG__ elem_type_ordinal_type  get_elem_type(Ord i)const
    {
        if (is_homogeneous) return homogeneous_elem_type; else return elems_types(i);
    }
    __DEVICE_TAG__ elem_type_ordinal_type  get_face_type(Ord i)const
    {
        //if (is_homogeneous) return homogeneous_face_type; else return faces_types(i);
        ///TODO there are issues with this - we cannot just use is_homogeneous flag - we need separate flag for faces
        return faces_types(i);
    }

    //TODO change bane
    /*template<class MapElems,class BasicMesh>
    void    dump_elems_geom_data(const MapElems &map_e, host_mesh<BasicMesh> &cpu_mesh)const
    {
        auto                    center_view = elems_centers.create_view(true);
        for (Ord i = center_view.begin();i < center_view.end();i++)
        {
            int i_glob = map_e.loc2glob(i);
            center_view.getv(i, cpu_mesh.cv[i_glob].elems_centers);
        }
        center_view.release(false);

        auto        center_neighbour_view = elems_neighbours0_centers.create_view(true);
        for (Ord i = center_neighbour_view.begin();i < center_neighbour_view.end();i++)
        {
            int i_glob = map_e.loc2glob(i);
            for (Ord j = 0;j < cpu_mesh.cv[i_glob].faces_n;++j)
                if (cpu_mesh.cv[i_glob].neighbours[j] != -1)
                    center_neighbour_view.getv(i,j,cpu_mesh.cv[cpu_mesh.cv[i_glob].neighbours[j]].elems_centers);
        }
        center_neighbour_view.release(false);

        auto        center_faces_view = elems_faces_centers.create_view(true);
        for (Ord i = center_faces_view.begin();i < center_faces_view.end();i++)
        {
            int i_glob = map_e.loc2glob(i);
            for (Ord j = 0;j < cpu_mesh.cv[i_glob].faces_n;++j)
                center_faces_view.getv(i,j,cpu_mesh.cv[i_glob].face_centers[j]);
        }
        center_faces_view.release(false);

        auto         vertexes_view = elems_vertexes.create_view(true);
        for (Ord i = vertexes_view.begin();i < vertexes_view.end();i++)
        {
            int i_glob = map_e.loc2glob(i);
            for (Ord j = 0;j < cpu_mesh.cv[i_glob].vert_n;++j)
                vertexes_view.getv(i,j,cpu_mesh.cv[i_glob].elems_vertexes[j]);
        }
        vertexes_view.release(false);

        auto                      vol_view = elems_vols.create_view(true);
        for (Ord i = vol_view.begin();i < vol_view.end();i++)
        {
            int i_glob = map_e.loc2glob(i);
            cpu_mesh.cv[i_glob].vol = vol_view(i,0);
        }
        vol_view.release(false);

        auto            faces_S_view = elems_faces_areas.create_view(true);
        for (Ord i = faces_S_view.begin();i < faces_S_view.end();i++)
        {
            int i_glob = map_e.loc2glob(i);
            for (Ord j = 0;j < cpu_mesh.cv[i_glob].faces_n;++j)
                cpu_mesh.cv[i_glob].S[j] = faces_S_view(i,j);
        }
        faces_S_view.release(false);

        //TODO add cv index-accessors

        auto        norm_view = elems_faces_norms.create_view(true);
        for(Ord i = norm_view.begin();i < norm_view.end();i++)
        {
            int i_glob = map_e.loc2glob(i);
            for (Ord j = 0;j < cpu_mesh.cv[i_glob].faces_n;++j)
                norm_view.getv(i,j,cpu_mesh.cv[i_glob].norms[j]);
        }
        norm_view.release(false);
    }*/
    /*template<class MapNodes,class BasicMesh>
    void    dump_nodes_geom_data(const MapNodes &map_n, host_mesh<BasicMesh> &cpu_mesh)const
    {
        auto                    node_coords_view = nodes_coords.create_view(true);
        for(Ord i_ = 0;i_ < map_n.get_size();++i_)
        {
            int     i_glob = map_n.own_glob_ind(i_),
                i_loc = map_n.own_loc_ind(i_);
            node_coords_view.getv(i_loc, cpu_mesh.nodes[i_glob].c);
        }
        node_coords_view.release(false);
    }*/

    ~device_mesh()
    {
    }

    template<class BasicMesh,class MapElems,class MapFaces,class MapNodes,class ForEach>
    void    init_elems_data
    (
        const host_mesh<BasicMesh> &cpu_mesh,
        const MapElems &map_e, const MapFaces &map_f, const MapNodes &map_n,
        const ForEach &for_each = ForEach()
    );
    template<class BasicMesh,class MapElems,class MapFaces,class MapNodes,class ForEach>
    void    init_nodes_data
    (
        const host_mesh<BasicMesh> &cpu_mesh,
        const MapElems &map_e, const MapFaces &map_f, const MapNodes &map_n,
        const ForEach &for_each = ForEach()
    );
    template<class BasicMesh,class MapElems,class MapFaces,class MapNodes,class ForEach>
    void    init_faces_data
    (
        const host_mesh<BasicMesh> &cpu_mesh,
        const MapElems &map_e, const MapFaces &map_f, const MapNodes &map_n,
        const ForEach &for_each = ForEach()
    );
};

}  /// namespace mesh
}  /// namespace scfd

#endif
