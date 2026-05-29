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

#ifndef __SCFD_MESH_FACE_KEY_H__
#define __SCFD_MESH_FACE_KEY_H__

#include <algorithm>
#include <functional>

namespace scfd
{
namespace mesh
{
namespace detail
{

template<class Ord>
struct face_key
{
    //TODO temporal solution (max 4 nodes) but will be enough for most cases
    Ord     nodes_n_;
    Ord     sorted_prim_nodes_[4];

    face_key() = default;
    face_key(Ord nodes_n, Ord prim_nodes[4])
    {
        nodes_n_ = nodes_n;
        for (Ord j = 0;j < nodes_n_;++j)
            sorted_prim_nodes_[j] = prim_nodes[j];
        std::sort(sorted_prim_nodes_,sorted_prim_nodes_+nodes_n_);
    }
    template<class BasicMesh>
    face_key(const BasicMesh &mesh,Ord elem_id,Ord elem_face_i)
    {
        const auto &ref = mesh.mesh_elem_reference();
        auto  elem_type = mesh.get_elem_type(elem_id);
        Ord nodes[mesh.get_elems_max_prim_nodes_num()];
        mesh.get_elem_prim_nodes(elem_id, nodes);
        //TODO temporal solution (max 4 nodes) but will be enough for most cases
        Ord face_nodes[4];
        for (Ord face_vert_i = 0;face_vert_i < ref.get_face_prim_verts_n(elem_type,elem_face_i);++face_vert_i)
        {
            face_nodes[face_vert_i] = nodes[ref.get_face_prim_vert_i(elem_type,elem_face_i,face_vert_i)];
        }
        //TODO looks strange
        *this = face_key(ref.get_face_verts_n(elem_type,elem_face_i), face_nodes);
    }
    template<class BasicMesh>
    bool  check_has_virt_pair(const BasicMesh &mesh, Ord virt_pair_i)const
    {
        for (Ord j = 0;j < nodes_n_;++j)
        {
            if (!mesh.check_node_has_virt_pair_node_id(sorted_prim_nodes_[j], virt_pair_i)) 
                return false;
        }
        return true;
    }
    template<class BasicMesh>
    face_key create_virt_pair(const BasicMesh &mesh, Ord virt_pair_i)const
    {
        face_key res;
        res.nodes_n_ = nodes_n_;
        for (Ord j = 0;j < nodes_n_;++j)
        {
            res.sorted_prim_nodes_[j] = mesh.get_node_virt_pair_node_id(sorted_prim_nodes_[j], virt_pair_i);
        }
        std::sort(res.sorted_prim_nodes_,res.sorted_prim_nodes_+res.nodes_n_);
        return res;
    }
    //Not supported because lower dimension elements are not exposed now
    /*template<class BasicMesh>
    face_key(const BasicMesh &mesh,Ord elem_id)
    {
        const auto &ref = mesh.mesh_elem_reference();
        auto  elem_type = mesh.get_elem_type(elem_id);
        Ord nodes[mesh.get_elems_max_prim_nodes_num()];
        mesh.get_elem_prim_nodes(elem_id, nodes);
        //TODO looks strange
        *this = face_key(ref.get_verts_n(elem_type), nodes);
    }*/

    Ord     nodes_n()const
    {
        return nodes_n_;
    }
    Ord     sorted_prim_node(Ord j)const
    {
        return sorted_prim_nodes_[j];
    }
};

template<class Ord>
struct face_key_equal_func
{
    bool operator()(const face_key<Ord> &f1, const face_key<Ord> &f2)const
    {
        if (f1.nodes_n() != f2.nodes_n()) 
            return false;
        for (int i = 0; i < f1.nodes_n(); i++) 
        {
            if (f1.sorted_prim_node(i) != f2.sorted_prim_node(i)) 
                return false;
        }
        return true;
    }
};

template<class Ord>
struct face_key_less_func
{
    bool operator()(const face_key<Ord> &f1, const face_key<Ord> &f2) const
    {
        if (f1.nodes_n() < f2.nodes_n()) return true;
        if (f1.nodes_n() > f2.nodes_n()) return false;
        for (int i = 0; i < f1.nodes_n(); i++) 
        {
            if (f1.sorted_prim_node(i) < f2.sorted_prim_node(i)) return true;
            if (f1.sorted_prim_node(i) > f2.sorted_prim_node(i)) return false;
        }
        return false;
    }
}; 

template<class Ord>
struct face_key_hash_func
{
    typedef face_key<Ord> argument_type;
    typedef size_t result_type;

    result_type operator()(const argument_type& a) const
    {
        std::hash<Ord> hasher;
        result_type h = 0;
        for (size_t i = 0; i < a.nodes_n(); ++i)
        {
            h = h * 31 + hasher(a.sorted_prim_node(i));
        }
        return h;
    }
};

}  /// namespace detail
}  /// namespace mesh
}  /// namespace scfd

#endif
