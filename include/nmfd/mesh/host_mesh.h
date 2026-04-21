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

#ifndef __SCFD_MESH_HOST_MESH_H__
#define __SCFD_MESH_HOST_MESH_H__

#include <memory>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <scfd/static_vec/vec.h>
#include "detail/face_key.h"
#include "detail/ranges_sparse_arr.h"

//TODO building face_key for element face fully repeated 3 times - move into separate method
//TODO walkthrough part + stencil repeats several times - mb, create some method with functor parameter?
//TODO graphs (elems_to_faces_graph_ and faces_to_elems_graph_) are build for more elements then intended
//(last ghost level) including incorrect ones (outer faces). Not a problem if no one will use them but some
//inconsistency presents.

namespace scfd
{
namespace mesh
{

//ISSUE what to use: inheritance or aglommeration?
template<class BasicMesh>
class host_mesh : public BasicMesh
{
public:
    using parent_type = BasicMesh;
    using basic_mesh_type = BasicMesh;
    using scalar_type = typename basic_mesh_type::scalar_type;
    using ordinal_type = typename basic_mesh_type::ordinal_type;
    using elem_type_ordinal_type = typename basic_mesh_type::elem_type_ordinal_type;
    using partitioner_type = typename basic_mesh_type::partitioner_type;

    static constexpr int          dim = basic_mesh_type::dim;
    static constexpr ordinal_type special_id = basic_mesh_type::special_id;

public:
    /// Need default contructable BasicMesh
    host_mesh()
    {
        /// Initialization
    }
    /// No 'empty' state
    /*host_mesh(std::shared_ptr<const BasicMesh>  basic_mesh) :
        basic_mesh_(basic_mesh)
    {
        /// Initialization
    }*/

    /// See gmsh_mesh_wrap.h for PartElems description
    void enlarge_stencil(ordinal_type ghost_level)
    {
        parent_type::enlarge_stencil(ghost_level);
        //TODO enlarge_stancil must be reusable - make build_faces reusable (delete old faces, perhaps?)
        build_faces(ghost_level);
    }

    /// Suppose we need to duplicate BasicMesh interface here?
    /// In case of inheritance we get it at once

    /// Part of nodes interface on this level (just get coords with static_vec result)
    static_vec::vec<scalar_type,dim> get_node_coords(ordinal_type i)const
    {
        static_vec::vec<scalar_type,dim>  res;
        parent_type::get_node_coords(i,res.d);
        return res;
    }

    /// Faces interface
    /// TODO not sure if we need to leave this method here
    /// because in general case calculation of total_faces_num is hard and
    /// is performed in faces_partitioner anyway
    /*ordinal_type get_total_faces_num()const
    {
        return total_faces_num_;
    }*/

    elem_type_ordinal_type get_face_type(ordinal_type face_id)const
    {
        return faces_types_[face_id];
    }

    ordinal_type get_elem_faces_num(ordinal_type i)const
    {
        return elems_to_faces_graph_.get_range_size(i);
    }
    void         get_elem_faces(ordinal_type i, ordinal_type *faces)const
    {
        auto it_range = elems_to_faces_graph_.get_range(i);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            faces[j] = it->first;
        }
    }
    void         get_elem_faces_inface_elem_inds(ordinal_type i,ordinal_type *inface_elem_inds)const
    {
        auto it_range = elems_to_faces_graph_.get_range(i);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            inface_elem_inds[j] = it->second;
        }
    }
    ordinal_type get_face_virt_master_id(ordinal_type i)const
    {
        if (!faces_virt_master_ids_arr_.has(i)) return i;
        return faces_virt_master_ids_arr_[i];
    }
    /// NOTE each face can only have maximum 1 virt_pair
    //ISSUE this interface is done as analogy to nodes interface but it seems unappropriate here
    bool         check_face_has_virt_pair_face_id(ordinal_type i, ordinal_type virt_pair_i)const
    {
        auto it1 = faces_virt_pair_face_ids_.find(i);
        if (it1 == faces_virt_pair_face_ids_.end()) return false;
        auto face_virt_pairs = it1->second;
        auto it2 = face_virt_pairs.find(virt_pair_i);
        if (it2 == face_virt_pairs.end()) return false;
        return true;
    }
    ordinal_type get_face_virt_pair_face_id(ordinal_type i, ordinal_type virt_pair_i)const
    {
        return faces_virt_pair_face_ids_.at(i).at(virt_pair_i);
    }
    /// No need to return number of virt_faces - use get_elem_faces_num
    void         get_elem_virt_faces(ordinal_type i, ordinal_type *virt_faces)const
    {
        auto it_range = elems_to_virt_faces_graph_.get_range(i);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            virt_faces[j] = it->first;
        }
    }
    void         get_elem_virt_faces_inface_elem_inds(ordinal_type i,ordinal_type *inface_elem_inds)const
    {
        auto it_range = elems_to_virt_faces_graph_.get_range(i);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            inface_elem_inds[j] = it->second;
        }
    }
    /// Actually it can return either 1 or 2
    ordinal_type get_face_elems_num(ordinal_type i)const
    {
        //TODO some check in debug mode for result (1 or 2)
        return faces_to_elems_graph_.get_range_size(i);
    }
    void         get_face_elems(ordinal_type i,ordinal_type elems[2])const
    {
        auto it_range = faces_to_elems_graph_.get_range(i);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            elems[j] = it->first;
        }
    }
    void         get_face_elems_inelem_face_inds(ordinal_type i,ordinal_type inelem_face_inds[2])const
    {
        auto it_range = faces_to_elems_graph_.get_range(i);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            inelem_face_inds[j] = it->second;
        }
    }
    /// If i is not face_virt_master_id (i.e. i is not virt face),
    /// returns data for corresponding virt face (i.e. for get_face_virt_master_id(i))
    /// Actually it can return either 1 or 2
    ordinal_type get_virt_face_elems_num(ordinal_type i)const
    {
        //TODO some check in debug mode for result (1 or 2)
        return virt_faces_to_elems_graph_.get_range_size(get_face_virt_master_id(i));
    }
    /// If i is not face_virt_master_id (i.e. i is not virt face),
    /// returns data for corresponding virt face (i.e. for get_face_virt_master_id(i))
    void         get_virt_face_elems(ordinal_type i,ordinal_type elems[2])const
    {
        auto it_range = virt_faces_to_elems_graph_.get_range(get_face_virt_master_id(i));
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            elems[j] = it->first;
        }
    }
    /// If i is not face_virt_master_id (i.e. i is not virt face),
    /// returns data for corresponding virt face (i.e. for get_face_virt_master_id(i))
    void         get_virt_face_elems_inelem_face_inds(ordinal_type i,ordinal_type inelem_face_inds[2])const
    {
        auto it_range = virt_faces_to_elems_graph_.get_range(get_face_virt_master_id(i));
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            inelem_face_inds[j] = it->second;
        }
    }

    ordinal_type get_faces_max_prim_nodes_num()const
    {
        return faces_max_prim_nodes_num_;
    }
    ordinal_type get_face_prim_nodes_num(ordinal_type face_id)const
    {
        return faces_to_prim_nodes_graph_.get_range_size(face_id);
    }
    void         get_face_prim_nodes
    (
        ordinal_type face_id, ordinal_type *prim_nodes, ordinal_type *prim_nodes_num = nullptr
    )
    const
    {
        auto it_range = faces_to_prim_nodes_graph_.get_range(face_id);
        if (prim_nodes_num) *prim_nodes_num = get_face_prim_nodes_num(face_id);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            prim_nodes[j] = *it;
        }
    }

    ordinal_type get_faces_max_nodes_num()const
    {
        return faces_max_nodes_num_;
    }
    ordinal_type get_face_nodes_num(ordinal_type face_id)const
    {
        return faces_to_nodes_graph_.get_range_size(face_id);
    }
    void         get_face_nodes
    (
        ordinal_type face_id, ordinal_type *nodes, ordinal_type *nodes_num = nullptr
    )
    const
    {
        auto it_range = faces_to_nodes_graph_.get_range(face_id);
        if (nodes_num) *nodes_num = get_face_nodes_num(face_id);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            nodes[j] = *it;
        }
    }

    bool check_face_has_group_id(ordinal_type face_id)const
    {
        ordinal_type nodes_n;
        ordinal_type prim_nodes[get_faces_max_prim_nodes_num()];
        get_face_prim_nodes(face_id,prim_nodes,&nodes_n);

        return parent_type::check_face_has_group_id(nodes_n,prim_nodes);
    }
    ordinal_type get_face_group_id(ordinal_type face_id)const
    {
        ordinal_type nodes_n;
        ordinal_type prim_nodes[get_faces_max_prim_nodes_num()];
        get_face_prim_nodes(face_id,prim_nodes,&nodes_n);

        return parent_type::get_face_group_id(nodes_n,prim_nodes);
    }

    /// Neighbours0 interface
    /// No need to return number of neighbours0 - use get_elem_faces_num
    void get_elem_neighbours0(ordinal_type i, ordinal_type *elems)const
    {
        auto it_range = elems_to_neighbours0_graph_.get_range(i);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            elems[j] = it->first;
        }
    }
    void get_elem_neighbours0_loc_face_i(ordinal_type i, ordinal_type *loc_face_i)const
    {
        auto it_range = elems_to_neighbours0_graph_.get_range(i);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            loc_face_i[j] = it->second;
        }
    }
    /// No need to return number of virtual neighbours0 - use get_elem_faces_num
    void get_elem_virt_neighbours0(ordinal_type i, ordinal_type *elems)const
    {
        auto it_range = elems_to_virt_neighbours0_graph_.get_range(i);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            elems[j] = it->first;
        }
    }
    void get_elem_virt_neighbours0_loc_face_i(ordinal_type i, ordinal_type *loc_face_i)const
    {
        auto it_range = elems_to_virt_neighbours0_graph_.get_range(i);
        ordinal_type j = 0;
        for (auto it = it_range.first;it != it_range.second;++it,++j)
        {
            loc_face_i[j] = it->second;
        }
    }

private:
    using face_key_t = detail::face_key<ordinal_type>;
    using face_key_equal_func = detail::face_key_equal_func<ordinal_type>;
    using face_key_less_func = detail::face_key_less_func<ordinal_type>;
    using face_key_hash_func = detail::face_key_hash_func<ordinal_type>;

    using faces_types_t = detail::sparse_arr<elem_type_ordinal_type,ordinal_type>;

    /// Here pair's first is id of incident face, second - local element index inside this face (could be 0 or 1)
    using elems_to_faces_graph_t = detail::ranges_sparse_arr<std::pair<ordinal_type,ordinal_type>,ordinal_type>;
    /// Here pair's first is id of incident element, second - local face index inside this element
    using faces_to_elems_graph_t = detail::ranges_sparse_arr<std::pair<ordinal_type,ordinal_type>,ordinal_type>;
    /// Here pair's first is id of incident element, second - local face index inside this element
    using elems_to_neighbours0_graph_t = detail::ranges_sparse_arr<std::pair<ordinal_type,ordinal_type>,ordinal_type>;

    using faces_virt_master_ids_arr_t =  detail::sparse_arr<ordinal_type>;

    //std::shared_ptr<const BasicMesh>  basic_mesh_;

    /// Value here is node_id
    using faces_to_nodes_graph_t = detail::ranges_sparse_arr<ordinal_type,ordinal_type>;

private:
    /// TODO see get_total_faces_num() commneted code
    //ordinal_type                    total_faces_num_;

    ordinal_type                    faces_max_nodes_num_, faces_max_prim_nodes_num_;

    faces_types_t                   faces_types_;

    elems_to_faces_graph_t          elems_to_faces_graph_;
    faces_to_elems_graph_t          faces_to_elems_graph_;
    elems_to_neighbours0_graph_t    elems_to_neighbours0_graph_;

    //TODO hmm))
    std::unordered_map
    <
        ordinal_type,
        std::unordered_map
        <
            ordinal_type,
            ordinal_type
        >
    >                               faces_virt_pair_face_ids_;
    faces_virt_master_ids_arr_t     faces_virt_master_ids_arr_;

    elems_to_faces_graph_t          elems_to_virt_faces_graph_;
    faces_to_elems_graph_t          virt_faces_to_elems_graph_;
    elems_to_neighbours0_graph_t    elems_to_virt_neighbours0_graph_;

    faces_to_nodes_graph_t          faces_to_nodes_graph_, faces_to_prim_nodes_graph_;

protected:

    //TODO virtual?
    void build_faces(ordinal_type ghost_level)
    {
        std::set<ordinal_type> stencil_ids;
        calc_elems_stencil(ghost_level,stencil_ids);
        const auto &part = *parent_type::get_partitioner();

        /// Create faces dict + calc faces_max_nodes_num_, faces_max_prim_nodes_num_

        //std::cout << "create faces dict" << std::endl;
        faces_max_nodes_num_ = 0;
        faces_max_prim_nodes_num_ = 0;
        std::unordered_map<face_key_t,ordinal_type,face_key_hash_func,face_key_equal_func>    faces;
        /// Process own elements
        for (ordinal_type i = 0;i < part.get_size();++i)
        {
            ordinal_type  elem_id = part.own_glob_ind(i);
            build_faces_for_elem(elem_id,faces);
        }
        /// Process stencil elements
        for (auto elem_id : stencil_ids)
        {
            build_faces_for_elem(elem_id,faces);
        }

        //std::cout << "faces virt masters " << std::endl;
        for (auto face_pair : faces)
        {
            face_key_t      face_key = face_pair.first;
            ordinal_type    face_id = face_pair.second;

            bool            has_virt_pair_face = false;
            face_key_t      virt_pair_face_key;
            ordinal_type    virt_pair_face_id;

            for (ordinal_type virt_pair_i = 0;virt_pair_i < parent_type::get_virt_pairs_num();++virt_pair_i)
            {
                if (!face_key.check_has_virt_pair(*this, virt_pair_i))
                {
                    continue;
                }
                else
                {
                    if (has_virt_pair_face)
                        throw
                            std::logic_error("host_mesh::build_faces: face " + std::to_string(face_id) + " has more then one virtual pair");
                    has_virt_pair_face = true;
                    virt_pair_face_key = face_key.create_virt_pair(*this, virt_pair_i);
                    virt_pair_face_id = faces[virt_pair_face_key];
                    faces_virt_pair_face_ids_[face_id][virt_pair_i] = virt_pair_face_id;
                }
            }
            if (has_virt_pair_face)
            {
                faces_virt_master_ids_arr_.add(face_id, virt_pair_face_id);
            }
            else
            {
                //faces_virt_master_ids_arr_.add(face_id, face_id);
            }
        }

        /// Create graphs elems_to_faces_graph_ and faces_to_elems_graph_ based on dict

        //std::cout << "create graphs structure " << std::endl;
        elems_to_faces_graph_.reserve(part.get_size()+stencil_ids.size());
        elems_to_neighbours0_graph_.reserve(part.get_size()+stencil_ids.size());
        faces_to_elems_graph_.reserve(faces.size());

        /// Estmate sizes of graphs elems_to_faces_graph_ and faces_to_elems_graph_
        /// Process own elements
        for (ordinal_type i = 0;i < part.get_size();++i)
        {
            ordinal_type  elem_id = part.own_glob_ind(i);
            reserve_graphs_for_elem(elem_id, faces);
        }
        /// Process stencil elements
        for (auto elem_id : stencil_ids)
        {
            reserve_graphs_for_elem(elem_id, faces);
        }

        /// Complete graphs structures
        elems_to_faces_graph_.complete_structure();
        elems_to_neighbours0_graph_.complete_structure();
        faces_to_elems_graph_.complete_structure();
        elems_to_virt_faces_graph_.complete_structure();
        elems_to_virt_neighbours0_graph_.complete_structure();
        virt_faces_to_elems_graph_.complete_structure();

        //std::cout << "fill graphs" << std::endl;
        /// Fill actual graphs elems_to_faces_graph_ and faces_to_elems_graph_
        /// Process own elements
        for (ordinal_type i = 0;i < part.get_size();++i)
        {
            ordinal_type  elem_id = part.own_glob_ind(i);
            fill_graphs_for_elem(elem_id, faces);
        }
        /// Process stencil elements
        for (auto elem_id : stencil_ids)
        {
            fill_graphs_for_elem(elem_id, faces);
        }

        /// Fill actual graph elems_to_neighbours0_graph_
        /// Process own elements
        for (ordinal_type i = 0;i < part.get_size();++i)
        {
            ordinal_type  elem_id = part.own_glob_ind(i);
            fill_neib_graph_for_elem
            (
                elem_id, elems_to_faces_graph_, faces_to_elems_graph_, elems_to_neighbours0_graph_
            );
        }
        /// Process stencil elements
        for (auto elem_id : stencil_ids)
        {
            fill_neib_graph_for_elem
            (
                elem_id, elems_to_faces_graph_, faces_to_elems_graph_, elems_to_neighbours0_graph_
            );
        }

        /// Fill actual graph elems_to_virt_neighbours0_graph_
        /// Process own elements
        for (ordinal_type i = 0;i < part.get_size();++i)
        {
            ordinal_type  elem_id = part.own_glob_ind(i);
            fill_neib_graph_for_elem
            (
                elem_id, elems_to_virt_faces_graph_, virt_faces_to_elems_graph_, elems_to_virt_neighbours0_graph_
            );
        }
        /// Process stencil elements
        for (auto elem_id : stencil_ids)
        {
            fill_neib_graph_for_elem
            (
                elem_id, elems_to_virt_faces_graph_, virt_faces_to_elems_graph_, elems_to_virt_neighbours0_graph_
            );
        }

        /// Estmate sizes of graphs faces_to_nodes_graph_ and faces_to_prim_nodes_graph_
        for (auto face_pair : faces)
        {
            ordinal_type    face_id = face_pair.second;
            reserve_nodes_graph_for_face(face_id);
        }

        /// Complete graphs structures
        faces_to_nodes_graph_.complete_structure();
        faces_to_prim_nodes_graph_.complete_structure();

        /// Fill actual data for faces_to_nodes_graph_ and faces_to_prim_nodes_graph_ + fills faces_types_
        for (auto face_pair : faces)
        {
            ordinal_type    face_id = face_pair.second;
            fill_nodes_graph_for_face(face_id);
        }

        for (auto face_pair : faces)
        {
            ordinal_type    face_id = face_pair.second;
            fix_faces_virt_pairs_nodes_order(face_id);
        }
    }
    void build_faces_for_elem
    (
        ordinal_type elem_id,
        std::unordered_map<face_key_t,ordinal_type,face_key_hash_func,face_key_equal_func> &faces
    )
    {
        const auto &ref = parent_type::mesh_elem_reference();
        //TODO in general it is not good idea to take it each time (some cached value?)
        ordinal_type glob_max_faces_num = parent_type::get_elems_glob_max_faces_num();
        elem_type_ordinal_type  elem_type = parent_type::get_elem_type(elem_id);
        //ISSUE are there any performance problmes with local allocation here?
        for (ordinal_type j = 0;j < ref.get_faces_n(elem_type);++j)
        {
            ordinal_type face_id = elem_id*glob_max_faces_num + j;
            face_key_t  face_key(*this,elem_id,j);
            /*if (elem_id == 54-45)
            {
                std::cout << "face " << j << std::endl;
                for (ordinal_type jj = 0;jj < face_key.nodes_n();++jj)
                    std::cout << face_key.sorted_prim_node(jj) << std::endl;
                std::cout << std::endl;
                for (ordinal_type jj = 0;jj < face_key.nodes_n();++jj)
                    std::cout << virt_face_key.sorted_prim_node(jj) << std::endl;
                std::cout << std::endl;
            }*/
            auto face_it = faces.find(face_key);
            if (face_it == faces.end())
            {
                faces[face_key] = face_id;
            }
            else
            {
                /// Between two poosible ids for face we choose the minimal one
                /// (i.e. one that was taken from element with minimal id)
                face_it->second = std::min(face_it->second,face_id);
            }

            faces_max_prim_nodes_num_ = std::max(faces_max_prim_nodes_num_,ref.get_face_prim_verts_n(elem_type,j));
            faces_max_nodes_num_ = std::max(faces_max_nodes_num_,ref.get_face_verts_n(elem_type,j));
        }
    }
    void reserve_graphs_for_elem(ordinal_type elem_id, const std::unordered_map<face_key_t,ordinal_type,face_key_hash_func,face_key_equal_func> &faces)
    {
        const auto &ref = parent_type::mesh_elem_reference();
        elem_type_ordinal_type  elem_type = parent_type::get_elem_type(elem_id);
        for (ordinal_type j = 0;j < ref.get_faces_n(elem_type);++j)
        {
            face_key_t face_key(*this,elem_id,j);
            auto face_it = faces.find(face_key);
            if (face_it == faces.end())
                throw std::logic_error("host_mesh::reserve_graphs_for_elem: no face found!");
            ordinal_type    face_id = face_it->second,
                            virt_face_id = get_face_virt_master_id(face_id);

            elems_to_faces_graph_.inc_max_range_size(elem_id,1);
            elems_to_neighbours0_graph_.inc_max_range_size(elem_id,1);
            faces_to_elems_graph_.inc_max_range_size(face_id,1);

            elems_to_virt_faces_graph_.inc_max_range_size(elem_id,1);
            elems_to_virt_neighbours0_graph_.inc_max_range_size(elem_id,1);
            virt_faces_to_elems_graph_.inc_max_range_size(virt_face_id,1);
        }
    }
    void fill_graphs_for_elem(ordinal_type elem_id, const std::unordered_map<face_key_t,ordinal_type,face_key_hash_func,face_key_equal_func> &faces)
    {
        const auto &ref = parent_type::mesh_elem_reference();
        elem_type_ordinal_type  elem_type = parent_type::get_elem_type(elem_id);
        for (ordinal_type j = 0;j < ref.get_faces_n(elem_type);++j)
        {
            face_key_t face_key(*this,elem_id,j);
            auto face_it = faces.find(face_key);
            if (face_it == faces.end())
                throw std::logic_error("host_mesh::reserve_graphs_for_elem: no face found!");
            ordinal_type    face_id = face_it->second,
                            virt_face_id = get_face_virt_master_id(face_id);
            ordinal_type    inface_elem_ind;

            inface_elem_ind =
                faces_to_elems_graph_.add_to_range(face_id, std::pair<ordinal_type,ordinal_type>(elem_id,j))-1;
            elems_to_faces_graph_.add_to_range(elem_id, std::pair<ordinal_type,ordinal_type>(face_id,inface_elem_ind));

            inface_elem_ind =
                virt_faces_to_elems_graph_.add_to_range(virt_face_id,std::pair<ordinal_type,ordinal_type>(elem_id,j))-1;
            elems_to_virt_faces_graph_.add_to_range(elem_id, std::pair<ordinal_type,ordinal_type>(virt_face_id,inface_elem_ind));
        }
    }
    void fill_neib_graph_for_elem
    (
        ordinal_type elem_id,
        const elems_to_faces_graph_t  &curr_elems_to_faces_graph,
        const faces_to_elems_graph_t  &curr_faces_to_elems_graph,
        elems_to_neighbours0_graph_t  &curr_elems_to_neighbours0_graph
    )
    {
        const auto &ref = parent_type::mesh_elem_reference();
        elem_type_ordinal_type  elem_type = parent_type::get_elem_type(elem_id);
        auto it_range = curr_elems_to_faces_graph.get_range(elem_id);
        for (auto it = it_range.first;it != it_range.second;++it)
        {
            ordinal_type face_id = it->first;
            if (curr_faces_to_elems_graph.get_range_size(face_id) != 2)
            {
                curr_elems_to_neighbours0_graph.add_to_range
                (
                    elem_id,
                    std::pair<ordinal_type,ordinal_type>(ordinal_type(special_id),ordinal_type(special_id))
                );
                continue;
            }
            ordinal_type neib0_id;
            ordinal_type loc_face_i;
            auto it_range1 = curr_faces_to_elems_graph.get_range(face_id);
            for (auto it1 = it_range1.first;it1 != it_range1.second;++it1)
            {
                if (it1->first == elem_id) continue;
                neib0_id = it1->first;
                loc_face_i = it1->second;
            }
            //auto it_range2 = curr_elems_to_faces_graph.get_range(neib0_id);
            //TODO think we can use it1->second as loc_face_i
            /*ordinal_type loc_face_i = 0;
            bool         found_loc_face_i = false;
            for (auto it2 = it_range2.first;it2 != it_range2.second;++it2,++loc_face_i)
            {
                ordinal_type face_id2 = *it2;
                if (face_id == face_id2)
                {
                    found_loc_face_i = true;
                    break;
                }
            }
            if (!found_loc_face_i)
                throw
                    std::logic_error
                    (
                        "host_mesh::fill_neib_graph_for_elem: found_loc_face_i is false for "
                        "elem_id = " + std::to_string(elem_id) + " face_id = " + std::to_string(face_id)
                    );*/
            curr_elems_to_neighbours0_graph.add_to_range
            (
                elem_id, std::pair<ordinal_type,ordinal_type>(neib0_id,loc_face_i)
            );
        }
    }
    void reserve_nodes_graph_for_face(ordinal_type face_id)
    {
        auto            it_range = faces_to_elems_graph_.get_range(face_id);
        /// Take 1st elemets index
        ordinal_type    elem_id = it_range.first->first,
                        inelem_face_ind = it_range.first->second;
        const auto      &ref = parent_type::mesh_elem_reference();
        auto            elem_type = parent_type::get_elem_type(elem_id);
        faces_to_nodes_graph_.inc_max_range_size(face_id,ref.get_face_verts_n(elem_type, inelem_face_ind));
        faces_to_prim_nodes_graph_.inc_max_range_size(face_id,ref.get_face_prim_verts_n(elem_type, inelem_face_ind));
    }
    void fill_nodes_graph_for_face(ordinal_type face_id)
    {
        auto            it_range = faces_to_elems_graph_.get_range(face_id);
        /// Take 1st elemets index
        //TODO looks not very good
        ordinal_type    elem_id = it_range.first->first,
                        inelem_face_ind = it_range.first->second;
        const auto      &ref = parent_type::mesh_elem_reference();
        auto            elem_type = parent_type::get_elem_type(elem_id);
        ordinal_type    prim_nodes_num_ = ref.get_face_prim_verts_n(elem_type, inelem_face_ind),
                        nodes_num_ = ref.get_face_verts_n(elem_type, inelem_face_ind);
        ordinal_type    elem_prim_nodes[parent_type::get_elem_prim_nodes_num(elem_id)],
                        elem_nodes[parent_type::get_elem_nodes_num(elem_id)];
        parent_type::get_elem_prim_nodes(elem_id, elem_prim_nodes);
        parent_type::get_elem_nodes(elem_id, elem_nodes);

        ordinal_type    prim_nodes[prim_nodes_num_],
                        nodes[nodes_num_];
        for (ordinal_type face_vert_i = 0;face_vert_i < prim_nodes_num_;++face_vert_i)
        {
            prim_nodes[face_vert_i] = elem_prim_nodes[ref.get_face_prim_vert_i(elem_type,inelem_face_ind,face_vert_i)];
        }
        for (ordinal_type face_vert_i = 0;face_vert_i < nodes_num_;++face_vert_i)
        {
            nodes[face_vert_i] = elem_nodes[ref.get_face_vert_i(elem_type,inelem_face_ind,face_vert_i)];
        }

        for (ordinal_type face_vert_i = 0;face_vert_i < prim_nodes_num_;++face_vert_i)
        {
            faces_to_prim_nodes_graph_.add_to_range
            (
                face_id, prim_nodes[face_vert_i]
            );
        }
        for (ordinal_type face_vert_i = 0;face_vert_i < nodes_num_;++face_vert_i)
        {
            faces_to_nodes_graph_.add_to_range
            (
                face_id, nodes[face_vert_i]
            );
        }

        faces_types_.add(face_id, ref.get_face_elem_type(elem_type,inelem_face_ind));
    }
    /// Called only after faces_to_nodes_graph_ has been built
    void fix_faces_virt_pairs_nodes_order(ordinal_type face_id)
    {
        for (ordinal_type virt_pair_i = 0;virt_pair_i < parent_type::get_virt_pairs_num();++virt_pair_i)
        {
            if (!check_face_has_virt_pair_face_id(face_id, virt_pair_i)) continue;
            ordinal_type face_virt_pair_id = get_face_virt_pair_face_id(face_id, virt_pair_i);

            //auto  it_range = faces_to_prim_nodes_graph_.get_range(face_id);

            ordinal_type    prim_nodes_num_, nodes_num_;
            ordinal_type    prim_nodes[get_faces_max_prim_nodes_num()],
                            nodes[get_faces_max_nodes_num()];
            get_face_prim_nodes( face_id, prim_nodes, &prim_nodes_num_ );
            get_face_nodes( face_id, nodes, &nodes_num_ );

            /// Build dict
            /// nodes_correspondence maps master node id into slave node id
            std::unordered_map<ordinal_type,ordinal_type>   nodes_correspondence;
            for (ordinal_type j = 0;j < nodes_num_;++j)
            {
                /// We assume here that all nodes has virt pairs if face has one
                ordinal_type virt_pair_node_id  = parent_type::get_node_virt_pair_node_id(nodes[j], virt_pair_i);
                nodes_correspondence[virt_pair_node_id] = nodes[j];
            }

            ordinal_type    master_face_prim_nodes[prim_nodes_num_],
                            master_face_nodes[nodes_num_];

            get_face_prim_nodes( face_virt_pair_id, master_face_prim_nodes );
            get_face_nodes( face_virt_pair_id, master_face_nodes );

            ordinal_type j;

            /// Reorder nodes accordind to master nodes order

            auto prim_nodes_it_range = faces_to_prim_nodes_graph_.get_range(face_id);
            j = 0;
            for (auto it = prim_nodes_it_range.first;it != prim_nodes_it_range.second;++it,++j)
            {
                *it = nodes_correspondence[master_face_prim_nodes[j]];
            }

            auto nodes_it_range = faces_to_nodes_graph_.get_range(face_id);
            j = 0;
            for (auto it = nodes_it_range.first;it != nodes_it_range.second;++it,++j)
            {
                *it = nodes_correspondence[master_face_nodes[j]];
            }
        }
    }
    void calc_elems_stencil(ordinal_type ghost_level,std::set<ordinal_type> &stencil_ids)
    {
        const auto &part = *parent_type::get_partitioner();
        stencil_ids.clear();
        std::set<ordinal_type> curr_ids,next_ids;
        for (ordinal_type i = 0;i < part.get_size();++i)
        {
            ordinal_type  elem_id = part.own_glob_ind(i);
            calc_elems_stencil_for_elem(elem_id,stencil_ids,next_ids);
        }
        for (ordinal_type level = 1;level < ghost_level;++level)
        {
            std::swap(curr_ids,next_ids);
            next_ids.clear();
            for (auto elem_id : curr_ids)
            {
                calc_elems_stencil_for_elem(elem_id,stencil_ids,next_ids);
            }
        }
    }
    void calc_elems_stencil_for_elem
    (
        ordinal_type elem_id, std::set<ordinal_type> &stencil_ids, std::set<ordinal_type> &next_ids
    )
    {
        const auto &part = *parent_type::get_partitioner();
        //const auto &ref = parent_type::mesh_elem_reference();
        elem_type_ordinal_type  elem_type = parent_type::get_elem_type(elem_id);
        ordinal_type nodes_n;
        ordinal_type nodes[parent_type::get_elems_max_prim_nodes_num()];
        parent_type::get_elem_prim_nodes(elem_id, nodes, &nodes_n);
        for (ordinal_type node_i = 0;node_i < nodes_n;++node_i)
        {
            ordinal_type incid_elems_n;
            ordinal_type incid_elems[parent_type::get_nodes_max_incident_elems_num()];
            parent_type::get_node_incident_elems(nodes[node_i],incid_elems,&incid_elems_n);
            for (ordinal_type incid_elems_i = 0;incid_elems_i < incid_elems_n;++incid_elems_i)
            {
                ordinal_type incid_elems_id = incid_elems[incid_elems_i];
                /// Simple cutoff (also not very effective)
                if (incid_elems_id == elem_id) continue;
                if (part.check_glob_owned(incid_elems_id)) continue;
                if (stencil_ids.find(incid_elems_id) != stencil_ids.end()) continue;
                stencil_ids.insert(incid_elems_id);
                next_ids.insert(incid_elems_id);
            }
        }
    }

};

}  /// namespace mesh
}  /// namespace scfd

#endif
