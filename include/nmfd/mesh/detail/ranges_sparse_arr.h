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

#ifndef __SCFD_MESH_RANGES_SPARSE_ARR_H__
#define __SCFD_MESH_RANGES_SPARSE_ARR_H__

#include <vector>
#include "sparse_arr.h"

namespace scfd
{
namespace mesh
{
namespace detail
{

template<class T,class Ord = int>
struct ranges_sparse_arr
{
    using sparse_pairs_arr_t = sparse_arr<std::pair<Ord,Ord>,Ord>;
    using indices_iterator_t = typename std::vector<T>::iterator;
    using indices_iterators_range_t = std::pair<indices_iterator_t,indices_iterator_t>;
    using indices_const_iterator_t = typename std::vector<T>::const_iterator;
    using indices_const_iterators_range_t = std::pair<indices_const_iterator_t,indices_const_iterator_t>;

    sparse_pairs_arr_t      sparse_pairs_arr_;
    std::vector<T>          objs_;
    Ord                     max_range_size_;

    void reserve(Ord size)
    {
        sparse_pairs_arr_.reserve(size);
    }
    void inc_max_range_size(Ord i, Ord inc = 1)
    {
        if (!sparse_pairs_arr_.has(i))
            sparse_pairs_arr_.add(i, std::pair<Ord,Ord>(0,0));
        sparse_pairs_arr_[i].first += inc;
    }

    void complete_structure()
    {
        //TODO preffy dirty hack to use internals of sparse_arr. Mb, privatly inherit it instead?
        Ord curr_offset = 0;
        max_range_size_ = 0;
        //for (auto p : sparse_pairs_arr_.glob_ind_to_vec_ind)
        for (auto obj_ind : sparse_pairs_arr_.glob_ind_to_vec_ind)
        {
            //auto &range_pair = sparse_pairs_arr_.objs_[p.second];
            if (obj_ind == sparse_pairs_arr_t::special_ind) continue;
            auto &range_pair = sparse_pairs_arr_.objs_[obj_ind];
            range_pair.second = curr_offset;
            curr_offset += range_pair.first;
            max_range_size_ = std::max(max_range_size_,range_pair.first);
            range_pair.first = 0;
        }
        objs_.resize(curr_offset);
    }

    /// TODO no checks for max limits violation
    /// returns range size after addition of new element
    Ord  add_to_range(Ord i, const T &e)
    {
        auto &range_pair = sparse_pairs_arr_[i];
        objs_[range_pair.second+range_pair.first] = e;
        return ++range_pair.first;
    }
    void has(Ord i)const
    {
        return sparse_pairs_arr_.has(i);
    }
    indices_iterators_range_t get_range(Ord i)
    {
        auto range_pair = sparse_pairs_arr_[i];
        Ord  range_size = range_pair.first,
             range_offset = range_pair.second;
        return 
            indices_iterators_range_t
            (
                objs_.begin() + range_offset,
                objs_.begin() + range_offset + range_size
            );
    }
    indices_const_iterators_range_t get_range(Ord i)const
    {
        auto range_pair = sparse_pairs_arr_[i];
        Ord  range_size = range_pair.first,
             range_offset = range_pair.second;
        return 
            indices_const_iterators_range_t
            (
                objs_.begin() + range_offset,
                objs_.begin() + range_offset + range_size
            );
    }
    Ord get_range_size(Ord i)const
    {
        auto range_pair = sparse_pairs_arr_[i];
        return range_pair.first;
    }
    /// Returns maximum among ranges sizes
    Ord get_max_ranges_size()const
    {
        return max_range_size_;
    }

    
};

}  /// namespace detail
}  /// namespace mesh
}  /// namespace scfd

#endif
