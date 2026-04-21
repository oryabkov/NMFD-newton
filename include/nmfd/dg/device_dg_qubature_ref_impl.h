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

#ifndef __SCFD_DEVICE_DG_QUBATURE_REF_IMPL_H__
#define __SCFD_DEVICE_DG_QUBATURE_REF_IMPL_H__

#include "detail/polynom_indexing.h"
#include "device_dg_qubature_ref.h"
#include "device_dg_qubature_ref_funcs.h"

///SCFD_DG_MAX_POLY_ORDER

//TODO do something with possible simultanious several types INSTANTIATE (mesh and elem_ref names conflict)
//maybe create some kind of DEFINE_TEMPLATE_CONSTANT_BUFFER with explicit template params argumnet
#define SCFD_DEVICE_DG_QUBATURE_REF_INSTANTIATE(T,MEMORY,DIM,ORD,MAX_POLY_ORDER)                        \
    namespace scfd { namespace dg { namespace detail {                                                  \
    DEFINE_CONSTANT_BUFFER((mesh::gmsh_mesh_elem_reference<T>), elem_ref)                               \
    DEFINE_CONSTANT_BUFFER((mesh::cubature_reference<MAX_POLY_ORDER*2+1,T>), cub_ref)                   \
    DEFINE_CONSTANT_BUFFER((device_dg_basis_reference<T,DIM,MAX_POLY_ORDER>), basis_ref)                \
    DEFINE_CONSTANT_BUFFER((mesh::device_mesh<T,MEMORY,DIM,ORD>), mesh)                                 \
    DEFINE_CONSTANT_BUFFER((device_dg_qubature_ref<T,MEMORY,DIM>), dg_qubature_ref)                     \
    template<>                                                                                          \
    void copy_data_to_const_buf<T,MEMORY,DIM,ORD,MAX_POLY_ORDER>                                        \
    (                                                                                                   \
        const mesh::gmsh_mesh_elem_reference<T> &elem_ref_,                                             \
        const mesh::cubature_reference<MAX_POLY_ORDER*2+1,T> &cub_ref_,                                 \
        const device_dg_basis_reference<T,DIM,MAX_POLY_ORDER> &basis_ref_,                              \
        const mesh::device_mesh<T,MEMORY,DIM,ORD> &device_mesh_,                                        \
        const device_dg_qubature_ref<T,MEMORY,DIM> &dg_qubature_ref_                                    \
    )                                                                                                   \
    {                                                                                                   \
        COPY_TO_CONSTANT_BUFFER(elem_ref, elem_ref_);                                                   \
        COPY_TO_CONSTANT_BUFFER(cub_ref, cub_ref_);                                                     \
        COPY_TO_CONSTANT_BUFFER(basis_ref, basis_ref_);                                                 \
        COPY_TO_CONSTANT_BUFFER(mesh, device_mesh_);                                                    \
        COPY_TO_CONSTANT_BUFFER(dg_qubature_ref, dg_qubature_ref_);                                     \
    }                                                                                                   \
    template<>                                                                                          \
    __DEVICE_TAG__ const mesh::gmsh_mesh_elem_reference<T> &get_elem_ref<T>()                           \
    {                                                                                                   \
        return elem_ref();                                                                              \
    }                                                                                                   \
    template<>                                                                                          \
    __DEVICE_TAG__ const mesh::cubature_reference<MAX_POLY_ORDER*2+1,T> &get_cub_ref<T>()               \
    {                                                                                                   \
        return cub_ref();                                                                               \
    }                                                                                                   \
    template<>                                                                                          \
    __DEVICE_TAG__ const device_dg_basis_reference<T,DIM,MAX_POLY_ORDER>&                               \
    get_basis_ref<T,DIM,MAX_POLY_ORDER>()                                                               \
    {                                                                                                   \
        return basis_ref();                                                                             \
    }                                                                                                   \
    template<>                                                                                          \
    __DEVICE_TAG__ const mesh::device_mesh<T,MEMORY,DIM,ORD> &get_mesh<T,MEMORY,DIM,ORD>()              \
    {                                                                                                   \
        return mesh();                                                                                  \
    }                                                                                                   \
    template<>                                                                                          \
    __DEVICE_TAG__ const device_dg_qubature_ref<T,MEMORY,DIM> &get_dg_qubature_ref<T,MEMORY,DIM>()      \
    {                                                                                                   \
        return dg_qubature_ref();                                                                       \
    }                                                                                                   \
    } } }                                                                                               \
    template class scfd::dg::device_dg_qubature_ref<T,MEMORY,DIM>;

namespace scfd
{
namespace dg
{

namespace detail
{

template<class T,class Memory,int Dim,class Ord, int MaxPolyOrder>                                                                                          \
void copy_data_to_const_buf
(
    const mesh::gmsh_mesh_elem_reference<T> &elem_ref_,
    const mesh::cubature_reference<MaxPolyOrder*2+1,T> &cub_ref_,
    const device_dg_basis_reference<T,Dim,MaxPolyOrder> &basis_ref_,
    const mesh::device_mesh<T,Memory,Dim,Ord> &device_mesh_,
    const device_dg_qubature_ref<T,Memory,Dim> &dg_qubature_ref_
)
{
    //TODO some kind of error
}

template<class T, class Memory, int Dim>
struct elem_jacobi_mat_optional_getter<T,Memory,Dim,false>
{
    using dg_ref_t = device_dg_qubature_ref<T,Memory,Dim>;
    using ordinal_t = typename dg_ref_t::ordinal_type;
    
    /// Non-linear elements: no per-element Jacobian cache in this optional.
    __DEVICE_TAG__ static typename elem_jacobi_mat_optional<T,Dim,false>::type
    get(const dg_ref_t &dg_ref, ordinal_t elem_i)
    {
        (void)dg_ref;
        (void)elem_i;
        return typename elem_jacobi_mat_optional<T,Dim,false>::type{};
    }
};

template<class T, class Memory, int Dim>
struct elem_jacobi_mat_optional_getter<T,Memory,Dim,true>
{
    using dg_ref_t = device_dg_qubature_ref<T,Memory,Dim>;
    using ordinal_t = typename dg_ref_t::ordinal_type;

    __DEVICE_TAG__ static typename elem_jacobi_mat_optional<T,Dim,true>::type   
    get(const dg_ref_t &dg_ref, ordinal_t elem_i)
    {
        typename elem_jacobi_mat_optional<T,Dim,true>::type   res;

        /// TODO we need some sort of get_mat as it was in elder versions
        #pragma unroll
        for (int ii1 = 0;ii1 < Dim;++ii1)
        {
            #pragma unroll
            for (int ii2 = 0;ii2 < Dim;++ii2)
            {
                res(ii1,ii2) = dg_ref.coords_jacobs(elem_i,ii1,ii2);
            }
        }

        return res;
    }
};

template<class T, class Memory, int Dim>
struct elem_loc_coords_optional_getter<T,Memory,Dim,false>
{
    using dg_ref_t = device_dg_qubature_ref<T,Memory,Dim>;
    using ordinal_t = typename dg_ref_t::ordinal_type;
    
    /// Non-linear elements: no local coord/Jacobian bundle in this optional.
    __DEVICE_TAG__ static typename elem_loc_coords_optional<T,Dim,false>::type
    get(const dg_ref_t &dg_ref, ordinal_t elem_i)
    {
        (void)dg_ref;
        (void)elem_i;
        return typename elem_loc_coords_optional<T,Dim,false>::type{};
    }
};

template<class T, class Memory, int Dim>
struct elem_loc_coords_optional_getter<T,Memory,Dim,true>
{
    using dg_ref_t = device_dg_qubature_ref<T,Memory,Dim>;
    using ordinal_t = typename dg_ref_t::ordinal_type;

    __DEVICE_TAG__ static typename elem_loc_coords_optional<T,Dim,true>::type   
    get(const dg_ref_t &dg_ref, ordinal_t elem_i)
    {
        typename elem_loc_coords_optional<T,Dim,true>::type   res;

        /// TODO we need some sort of get_mat as it was in elder versions
        #pragma unroll
        for (int ii1 = 0;ii1 < Dim;++ii1)
        {
            #pragma unroll
            for (int ii2 = 0;ii2 < Dim;++ii2)
            {
                res.J(ii1,ii2) = dg_ref.coords_jacobs(elem_i,ii1,ii2);
            }
        }

        dg_ref.coords_shifts.get_vec(res.c0,elem_i);
        
        return res;
    }
};

template<class T, class Memory, int Dim>
struct face_loc_coords_optional_getter<T,Memory,Dim,false>
{
    using dg_ref_t = device_dg_qubature_ref<T,Memory,Dim>;
    using ordinal_t = typename dg_ref_t::ordinal_type;
    
    /// Non-linear faces: no face local coord bundle in this optional.
    __DEVICE_TAG__ static typename face_loc_coords_optional<T,Dim,false>::type
    get(const dg_ref_t &dg_ref, ordinal_t face_i)
    {
        (void)dg_ref;
        (void)face_i;
        return typename face_loc_coords_optional<T,Dim,false>::type{};
    }
};

template<class T, class Memory, int Dim>
struct face_loc_coords_optional_getter<T,Memory,Dim,true>
{
    using dg_ref_t = device_dg_qubature_ref<T,Memory,Dim>;
    using ordinal_t = typename dg_ref_t::ordinal_type;

    __DEVICE_TAG__ static typename face_loc_coords_optional<T,Dim,true>::type   
    get(const dg_ref_t &dg_ref, ordinal_t face_i)
    {
        typename face_loc_coords_optional<T,Dim,true>::type   res;

        /// TODO we need some sort of get_mat as it was in elder versions
        #pragma unroll
        for (int ii1 = 0;ii1 < Dim;++ii1)
        {
            #pragma unroll
            for (int ii2 = 0;ii2 < Dim-1;++ii2)
            {
                res.J(ii1,ii2) = dg_ref.face_coords_jacobs(face_i,ii1,ii2);
            }
        }

        dg_ref.face_coords_shifts.get_vec(res.c0,face_i);
        
        return res;
    }
};

}

template<class T, class Memory, int Dim>
template<class ForEach>
void device_dg_qubature_ref<T,Memory,Dim>::calc_basis_funcs_mass_matrix_diag
(
    basis_array_t res_basis_funcs_mass_matrix_diag,
    const ForEach &for_each
)
{
    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::calc_basis_funcs_mass_matrix_diag
        (
            res_basis_funcs_mass_matrix_diag
        ),
        elems_i0, elems_i1
    );
}

template<class T, class Memory, int Dim>
template<class Func,class ForEach>
void device_dg_qubature_ref<T,Memory,Dim>::calc_l2_local_proj
(
    bool ignore_mass_matrix, basis_array_t basis_funcs_mass_matrix_diag, Func f,
    basis_array_t res_funcs_coeffs,
    const ForEach &for_each
)
{
    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::template calc_l2_local_proj<true,Func>
        (
            ignore_mass_matrix, basis_funcs_mass_matrix_diag, f, res_funcs_coeffs
        ),
        elems_i0, elems_i1
    );
    for_each
    (
        typename device_dg_qubature_ref_funcs_t::template calc_l2_local_proj<false,Func>
        (
            ignore_mass_matrix, basis_funcs_mass_matrix_diag, f, res_funcs_coeffs
        ),
        elems_i0, elems_i1
    );
}

template<class T, class Memory, int Dim>
template<class Func,class ForEach>
void device_dg_qubature_ref<T,Memory,Dim>::calc_func_values
(
    Func f, scalar_func_values_t scalar_func_values,
    const ForEach &for_each
)
{
    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::template calc_func_values<true,Func>
        (
            f, scalar_func_values
        ),
        elems_i0, elems_i1
    );
    for_each
    (
        typename device_dg_qubature_ref_funcs_t::template calc_func_values<false,Func>
        (
            f, scalar_func_values
        ),
        elems_i0, elems_i1
    );
}

template<class T, class Memory, int Dim>
template<class ForEach>
void device_dg_qubature_ref<T,Memory,Dim>::calc_func_values_from_basis_array
(
    basis_array_t func_basis_array, 
    scalar_func_values_t scalar_func_values,
    const ForEach &for_each
)
{
    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::calc_func_values_from_basis_array
        (
            func_basis_array, scalar_func_values
        ),
        elems_i0, elems_i1
    );
}

template<class T, class Memory, int Dim>
template<class ForEach,class Reduce>
T device_dg_qubature_ref<T,Memory,Dim>::calc_l2_norm_by_func_values
(
    scalar_func_values_t scalar_func_values,
    const ForEach &for_each, const Reduce &reduce
)
{
    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::calc_l2_norm_by_func_values
        (
            scalar_func_values, elementwise_array_buf
        ),
        elems_i0, elems_i1
    );

    return reduce(elems_n, elementwise_array_buf.raw_ptr(), T(0));
}

template<class T, class Memory, int Dim>
template<class ForEach,class Reduce>
T device_dg_qubature_ref<T,Memory,Dim>::calc_diff_l2_norm_by_func_values
(
    scalar_func_values_t scalar_func1_values,
    scalar_func_values_t scalar_func2_values,
    const ForEach &for_each, const Reduce &reduce
)
{
    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::calc_diff_l2_norm_by_func_values
        (
            scalar_func1_values, scalar_func2_values, elementwise_array_buf
        ),
        elems_i0, elems_i1
    );

    return std::sqrt(reduce(elems_n, elementwise_array_buf.raw_ptr(), T(0)));
}

template<class T, class Memory, int Dim>
template<class ForEach>
void device_dg_qubature_ref<T,Memory,Dim>::init
(
    const device_mesh_t &device_mesh, 
    int poly_order, int vol_qub_order, int faces_qub_order, 
    const ForEach &for_each
)
{
    if (poly_order > SCFD_DG_MAX_POLY_ORDER)
        throw std::runtime_error("device_dg_qubature_ref::init: poly_order <= SCFD_DG_MAX_POLY_ORDER is violated");
    if (vol_qub_order > poly_order*2)
        throw std::runtime_error("device_dg_qubature_ref::init: vol_qub_order <= poly_order*2 is violated");
    if (faces_qub_order > poly_order*2+1)
        throw std::runtime_error("device_dg_qubature_ref::init: faces_qub_order <= poly_order*2+1 is violated");

    using elem_reference_t = mesh::gmsh_mesh_elem_reference<T>;
    using qub_reference_t = mesh::cubature_reference<SCFD_DG_MAX_POLY_ORDER*2+1,T>;
    using basis_reference_t = detail::device_dg_basis_reference<T,Dim,SCFD_DG_MAX_POLY_ORDER>;

    this->poly_order = poly_order;
    this->vol_qub_order = vol_qub_order;
    this->faces_qub_order = faces_qub_order;
    polylen = detail::polylen(Dim,poly_order);
    max_basis_size = polylen;
    qubature_max_pnts_n = mesh::get_cubature_reference_max_pnts_n(poly_order*2);
    face_qubature_max_pnts_n = mesh::get_face_cubature_reference_max_pnts_n(poly_order*2+1);

    elems_n = device_mesh.own_elems_range.n;
    elems_i0 = device_mesh.own_elems_range.i0;
    elems_i1 = device_mesh.own_elems_range.i1();
    faces_n = device_mesh.own_faces_range.n;
    faces_i0 = device_mesh.own_faces_range.i0;
    faces_i1 = device_mesh.own_faces_range.i1();

    linearity_flags.init(elems_n,elems_i0);
    elem_faces_transforms.init(elems_n, device_mesh.max_faces_n, elems_i0, 0);
    coords_jacobs.init(elems_n,elems_i0);
    coords_inv_jacobs.init(elems_n,elems_i0);
    coords_jacobs_dets.init(elems_n,elems_i0);
    coords_shifts.init(elems_n,elems_i0);

    et_qubature_pnts_n.init(elems_n, elems_i0);
    /// Here tensor dim is qubature_max_pnts_n
    et_qubature_weights.init(elems_n, qubature_max_pnts_n, elems_i0, 0);
    /// Here tensor dim is qubature_max_pnts_n
    et_qubature_pnts.init(elems_n, qubature_max_pnts_n, elems_i0, 0);
    /// Here 1st tensor dim is max_basis_size; 2nd tensor dim is qubature_max_pnts_n
    et_basis_values.init(elems_n, max_basis_size, qubature_max_pnts_n, elems_i0, 0, 0);
    /// Here 1st tensor dim is max_basis_size; 2nd tensor dim is qubature_max_pnts_n
    et_basis_derivatives.init(elems_n, max_basis_size, qubature_max_pnts_n, elems_i0, 0, 0);
    /// Here 1st tensor dim is max_faces_n
    et_qubature_faces_pnts_n.init(elems_n, device_mesh.max_faces_n, elems_i0, 0);
    /// Here 1st tensor dim is max_faces_n; 2nd tensor dim is face_qubature_max_pnts_n
    et_qubature_faces_weights.init(elems_n, device_mesh.max_faces_n, face_qubature_max_pnts_n, elems_i0, 0, 0);
    /// Here 1st tensor dim is max_faces_n; 2nd tensor dim is face_qubature_max_pnts_n
    et_qubature_faces_pnts.init(elems_n, device_mesh.max_faces_n, face_qubature_max_pnts_n, elems_i0, 0, 0);
    /// Here 1st tensor dim is max_basis_size; 2nd tensor dim is max_faces_n; 3rd is face_qubature_max_pnts_n
    et_basis_faces_values.init(elems_n, max_basis_size, device_mesh.max_faces_n, face_qubature_max_pnts_n, elems_i0, 0, 0, 0);
    /// Here 1st tensor dim is max_basis_size; 2nd tensor dim is max_faces_n; 3rd is face_qubature_max_pnts_n
    et_basis_faces_derivatives.init(elems_n, max_basis_size, device_mesh.max_faces_n, face_qubature_max_pnts_n, elems_i0, 0, 0, 0);
    /// Here 1st tensor dim is max_basis_size; 2nd tensor dim is max_nodes_n(TODO or max_prim_nodes_n?)
    //et_basis_vert_values.init(elems_n, max_basis_size, device_mesh.max_nodes_n, elems_i0, 0, 0);
    //tensor2_array<T,Memory,max_basis_size,max_basis_size>                  coeff2_mats.init();
    /// Here 1st tensor dim is max_faces_n+1; 2nd and 3rd tensor dims are max_basis_size
    //coeff2_mats.init(elems_n, device_mesh.max_faces_n+1, max_basis_size, max_basis_size, elems_i0, 0, 0, 0);
    /// Here 1st tensor dim is max_faces_n; 2nd and 3rd tensor dims are max_basis_size
    //basis_transition.init(elems_n, device_mesh.max_faces_n, max_basis_size, max_basis_size, elems_i0, 0, 0, 0);

    face_linearity_flags.init(faces_n,faces_i0);
    face_coords_jacobs.init(faces_n,faces_i0);
    face_coords_integral_muls.init(faces_n,faces_i0);
    face_coords_shifts.init(faces_n,faces_i0);

    et_face_qubature_pnts_n.init(faces_n,faces_i0);
    et_face_qubature_weights.init(faces_n, face_qubature_max_pnts_n, faces_i0, 0);
    et_face_qubature_pnts.init(faces_n, face_qubature_max_pnts_n, faces_i0, 0);

    elementwise_array_buf.init(elems_n, elems_i0);

    elem_reference_t        elem_ref_;
    qub_reference_t         qub_ref_;
    basis_reference_t       basis_ref_;
    detail::copy_data_to_const_buf<T,Memory,Dim,ordinal_type,SCFD_DG_MAX_POLY_ORDER>
    (
        elem_ref_, qub_ref_, basis_ref_, device_mesh, *this
    );

    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::calc_loc_coords(), 
        elems_i0, elems_i1
    );
    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::calc_qubature_weights(), 
        elems_i0, elems_i1
    );

    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::calc_faces_loc_coords(), 
        faces_i0, faces_i1 
    );
    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::calc_faces_quadratures_weights(), 
        faces_i0, faces_i1
    );

    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::init_elem_faces_qubature_pnts(), 
        elems_i0, elems_i1
    );
    
}

template<class T, class Memory, int Dim>
template<class ForEach>
void device_dg_qubature_ref<T,Memory,Dim>::convert_basis_coeffs_into_monom_coeffs
(
    basis_array_t basis_array,monoms_basis_array_t res_monoms_basis_array,
    const ForEach &for_each
)const
{
    for_each
    ( 
        typename device_dg_qubature_ref_funcs_t::convert_basis_coeffs_into_monom_coeffs
        (
            basis_array,res_monoms_basis_array
        ), 
        elems_i0, elems_i1
    );
}

}  /// namespace dg
}  /// namespace scfd

#endif
