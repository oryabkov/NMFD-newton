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

#ifndef __SCFD_MESH_SPARSE_ARR_H__
#define __SCFD_MESH_SPARSE_ARR_H__

#include <cassert>
#include <limits>
#include <vector>
#include <map>

namespace scfd
{
namespace mesh
{
namespace detail
{

template<class T,class Ord = int>
struct sparse_arr
{
    static const Ord special_ind = std::numeric_limits<Ord>::max();

    std::vector<T>       objs_;
    //TODO for dense numeration map can be avoided (something like GMSH internal cache)
    //std::map<Ord,Ord>    glob_ind_to_vec_ind;
    std::vector<Ord>     glob_ind_to_vec_ind;

    void reserve(Ord size)
    {
        objs_.reserve(size);
    }

    Ord             find_elem__(Ord glob_i)const
    {
        //auto       it = glob_ind_to_vec_ind.find(glob_i);
        //assert(it != glob_ind_to_vec_ind.end());
        //return it->second;
        if (glob_i < 0)
            throw std::logic_error("sparse_arr::find_elem__: negative indexes are not supported");
        if (glob_i >= glob_ind_to_vec_ind.size())
            throw std::logic_error("sparse_arr::find_elem__: glob_i exeeds glob_ind_to_vec_ind size");
        if (glob_ind_to_vec_ind.at(glob_i) == special_ind)
            throw std::logic_error("sparse_arr::find_elem__: no elem found");
        return glob_ind_to_vec_ind.at(glob_i);
    }
    T            &operator[](Ord glob_i)
    {
        //ISSUE check if we have glob_i?? i think do it in DEBUG mode
        return objs_[ find_elem__(glob_i) ];
    }
    const T      &operator[](Ord glob_i)const
    {
        //ISSUE check if we have glob_i?? i think do it in DEBUG mode
        return objs_[ find_elem__(glob_i) ];
    }
    //TODO rename has_elem and add_elem into has and and
    bool            has(Ord glob_i)const
    {
        if (glob_i < 0)
            return false;
        if (glob_i >= glob_ind_to_vec_ind.size())
            return false;
        return (glob_ind_to_vec_ind.at(glob_i) != special_ind);

        //return (glob_ind_to_vec_ind.find(glob_i) != glob_ind_to_vec_ind.end());
    }
    void            add(Ord glob_i, const T &e)
    {
        //ISSUE check if we already have glob_i?? i think do it in DEBUG mode
        if (glob_i >= glob_ind_to_vec_ind.size())
        {
            /// *2 is for forehead resizing
            glob_ind_to_vec_ind.resize((glob_i+1)*2, Ord(special_ind));
        }
        glob_ind_to_vec_ind[glob_i] = objs_.size();
        objs_.push_back(e);
    }
};

}  /// namespace detail
}  /// namespace mesh
}  /// namespace scfd

#endif
