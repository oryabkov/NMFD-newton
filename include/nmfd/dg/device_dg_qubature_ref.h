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

#ifndef __SCFD_DEVICE_DG_QUBATURE_REF_H__
#define __SCFD_DEVICE_DG_QUBATURE_REF_H__

#include "scfd/static_mat/mat.h"
#include <scfd/arrays/array.h>
#include <scfd/arrays/tensorN_array.h>

#include <nmfd/mesh/device_mesh.h>

#ifndef SCFD_DG_MAX_POLY_ORDER
#define SCFD_DG_MAX_POLY_ORDER 2
#endif

#ifndef SCFD_DEVICE_DG_BASIS_REFERENCE_ELEM_TYPES_N
#define SCFD_DEVICE_DG_BASIS_REFERENCE_ELEM_TYPES_N 8
#endif

namespace scfd
{
namespace dg
{

//TODO is it really safe?
using namespace arrays;

namespace detail
{
/// Forward declaration; it is needed only for type specialization inside device_mesh class scope
template<class T, class Memory, int Dim, class Ord, int MaxPolyOrder>
struct device_dg_qubature_ref_funcs;

struct empty_struct
{
};

struct empty_loc_coords
{
    /// I know looks strange
    empty_struct    J;

    __DEVICE_TAG__ const empty_struct&
    get_jacobi_mat()const
    {
        return J;
    }
};

template<class T, int Dim>
using elem_jacobi_mat = static_mat::mat<T,Dim,Dim>;

template<class T, int Dim>
struct elem_loc_coords
{
    elem_jacobi_mat<T,Dim>       J;
    static_vec::vec<T,Dim>       c0;

    __DEVICE_TAG__ const elem_jacobi_mat<T,Dim>&
    get_jacobi_mat()const
    {
        return J;
    }
};

template<class T, int Dim>
struct face_loc_coords
{
    static_mat::mat<T,Dim,Dim-1>   J;
    static_vec::vec<T,Dim>         c0;
};

template<class T, int Dim, bool is_needed>
struct elem_jacobi_mat_optional
{
};

template<class T, int Dim>
struct elem_jacobi_mat_optional<T,Dim,false>
{
    using type = empty_struct;
};

template<class T, int Dim>
struct elem_jacobi_mat_optional<T,Dim,true>
{
    using type = elem_jacobi_mat<T,Dim>;
};

template<class T, int Dim, bool is_needed>
struct elem_loc_coords_optional
{
};

template<class T, int Dim>
struct elem_loc_coords_optional<T,Dim,false>
{
    using type = empty_loc_coords;
    /*static const typename elem_jacobi_mat_optional<T,Dim,false>::type&
    get_jacobi_mat(const typename elem_loc_coords_optional<T,Dim,false>::type &c)
    {
        /// NOTE this is kind of strange but works.
        /// We do this bacause we want to return reference not value;
        /// ISSUE is it important?
        return c;
    }*/
};

template<class T, int Dim>
struct elem_loc_coords_optional<T,Dim,true>
{
    using type = elem_loc_coords<T,Dim>;
    /*static const typename elem_jacobi_mat_optional<T,Dim,true>::type&
    get_jacobi_mat(const typename elem_loc_coords_optional<T,Dim,true>::type &c)
    {
        return c.J;
    }*/
};

template<class T, int Dim, bool is_needed>
struct face_loc_coords_optional
{
};

template<class T, int Dim>
struct face_loc_coords_optional<T,Dim,false>
{
    using type = empty_struct;
};

template<class T, int Dim>
struct face_loc_coords_optional<T,Dim,true>
{
    using type = face_loc_coords<T,Dim>;
};

template<class T, class Memory, int Dim, bool is_needed>
struct elem_jacobi_mat_optional_getter
{
};

template<class T, class Memory, int Dim, bool is_needed>
struct elem_loc_coords_optional_getter
{
};

template<class T, class Memory, int Dim, bool is_needed>
struct face_loc_coords_optional_getter
{
};

}

template<class T, class Memory, int Dim = 3>
struct device_dg_qubature_ref
{
    /// TODO add template parameter
    using ordinal_type = int;

    static const int          dim = Dim;

    using vec_t = scfd::static_vec::vec<T,dim>;

    using device_mesh_t = mesh::device_mesh<T,Memory,dim,ordinal_type>;

    /// Here 1st tensor dim is max_basis_size
    using basis_array_t = tensor1_array<T,Memory,dyn_dim>;
    /// Here 1st tensor dim is polylen (in fact, this is the same as monoms_basis_array, just for more readable view)
    using monoms_basis_array_t = tensor1_array<T,Memory,dyn_dim>;

    using elementwise_array_t = array<T,Memory>;

    /// Here 1st tensor dim is qubature_max_pnts_n; values of arbitrary function in elements quadratures
    using scalar_func_values_t = tensor1_array<T,Memory,dyn_dim>;

    using elem_type_ordinal_t = typename device_mesh_t::elem_type_ordinal_type;

    template<bool b>
    using elem_jacobi_mat_optional_t = typename detail::elem_jacobi_mat_optional<T,Dim,b>::type;
    template<bool b>
    using elem_loc_coords_optional_t = typename detail::elem_loc_coords_optional<T,Dim,b>::type;
    template<bool b>
    using face_loc_coords_optional_t = typename detail::face_loc_coords_optional<T,Dim,b>::type;

    using loc_coords_jacobs_t = tensor2_array<T,Memory,dim,dim>;
    using loc_coords_jacobs_dets_t = array<T,Memory>;
    using loc_coords_shifts_t = tensor1_array<T,Memory,dim>;

    using loc_face_coords_jacobs_t = tensor2_array<T,Memory,dim,dim-1>;
    using loc_face_coords_integral_muls_t = array<T,Memory>;
    using loc_face_coords_shifts_t = tensor1_array<T,Memory,dim>;

    /// TODO temporal
    using face_transform_descr_type = int;

    using linearity_flags_t = array<bool,Memory>;
    /// Here 1st tensor dim is max_faces_n
    using faces_transforms_t = tensor1_array<face_transform_descr_type,Memory,dyn_dim>;
    using qubature_pnts_n_t = array<ordinal_type,Memory>;
    /// Here tensor dim is qubature_max_pnts_n (or face_qubature_max_pnts_n for faces)
    using qubature_weights_t = tensor1_array<T,Memory,dyn_dim>;
    /// Here tensor dim is qubature_max_pnts_n (or face_qubature_max_pnts_n for faces)
    using qubature_pnts_t = tensor2_array<T,Memory,dyn_dim,dim>;
    /// Here 1st tensor dim is max_basis_size; 2nd tensor dim is qubature_max_pnts_n
    using basis_values_t = tensor2_array<T,Memory,dyn_dim,dyn_dim>;
    /// Here 1st tensor dim is max_basis_size; 2nd tensor dim is qubature_max_pnts_n
    using basis_derivatives_t = tensor3_array<T,Memory,dyn_dim,dyn_dim,Dim>;
    /// Here 1st tensor dim is max_faces_n
    using qubature_faces_pnts_n_t = tensor1_array<ordinal_type,Memory,dyn_dim>;
    /// Here 1st tensor dim is max_faces_n; 2nd tensor dim is face_qubature_max_pnts_n
    using qubature_faces_weights_t = tensor2_array<T,Memory,dyn_dim,dyn_dim>;
    /// Here 1st tensor dim is max_faces_n; 2nd tensor dim is face_qubature_max_pnts_n
    using qubature_faces_pnts_t = tensor3_array<T,Memory,dyn_dim,dyn_dim,dim>;
    /// Here 1st tensor dim is max_basis_size; 2nd tensor dim is max_faces_n; 3rd is face_qubature_max_pnts_n
    using basis_faces_values_t = tensor3_array<T,Memory,dyn_dim,dyn_dim,dyn_dim>;
    /// Here 1st tensor dim is max_basis_size; 2nd tensor dim is max_faces_n; 3rd is face_qubature_max_pnts_n
    using basis_faces_derivatives_t = tensor4_array<T,Memory,dyn_dim,dyn_dim,dyn_dim,dim>;
    /// Here 1st tensor dim is max_basis_size; 2nd tensor dim is max_nodes_n(TODO or max_prim_nodes_n?)
    using basis_vert_values_t = tensor2_array<T,Memory,dyn_dim,dyn_dim>;
    //using tensor2_array<T,Memory,max_basis_size,max_basis_size>                  coeff2_mats_t;
    /// Here 1st tensor dim is max_faces_n+1; 2nd and 3rd tensor dims are max_basis_size
    using coeff2_mats_t = tensor3_array<T,Memory,dyn_dim,dyn_dim,dyn_dim>;
    /// Here 1st tensor dim is max_faces_n; 2nd and 3rd tensor dims are max_basis_size
    using basis_transition_t = tensor3_array<T,Memory,dyn_dim,dyn_dim,dyn_dim>;

    using loc_coords_jacobs_view_t = typename loc_coords_jacobs_t::view_type;
    using loc_coords_jacobs_dets_view_t = typename loc_coords_jacobs_dets_t::view_type;
    using loc_coords_shifts_view_t = typename loc_coords_shifts_t::view_type;
    using qubature_weights_view_t = typename qubature_weights_t::view_type;
    using basis_values_view_t = typename basis_values_t::view_type;
    using basis_derivatives_view_t = typename basis_derivatives_t::view_type;
    using qubature_faces_weights_view_t = typename qubature_faces_weights_t::view_type;
    using basis_faces_values_view_t = typename basis_faces_values_t::view_type;
    using basis_vert_values_view_t = typename basis_vert_values_t::view_type;
    using coeff2_mats_view_t = typename coeff2_mats_t::view_type;
    using basis_transition_view_t = typename basis_transition_t::view_type;

    using device_dg_qubature_ref_funcs_t =
        detail::device_dg_qubature_ref_funcs<T,Memory,Dim,ordinal_type,SCFD_DG_MAX_POLY_ORDER>;

    /// TODO don't like this but otherwise we don't have access to these params in cpu methods
    /// like convert_basis_coeffs_into_monom_coeffs (copy of device_mesh is lost at this stage!)
    ordinal_type   elems_n, elems_i0, elems_i1,
                   faces_n, faces_i0, faces_i1;

    //typedef t_DG_tml<MaxOrder,T>                                   t_DG;
    int     poly_order;
    int     vol_qub_order;
    int     faces_qub_order;
    int     polylen;
    int     max_basis_size;
    int     qubature_max_pnts_n;
    int     face_qubature_max_pnts_n;

    linearity_flags_t           linearity_flags;
    faces_transforms_t          elem_faces_transforms;
    loc_coords_jacobs_t         coords_jacobs, coords_inv_jacobs;
    loc_coords_jacobs_dets_t    coords_jacobs_dets;
    loc_coords_shifts_t         coords_shifts;

    qubature_pnts_n_t           et_qubature_pnts_n;
    qubature_weights_t          et_qubature_weights;
    qubature_pnts_t             et_qubature_pnts;
    basis_values_t              et_basis_values;
    basis_derivatives_t         et_basis_derivatives;
    qubature_faces_pnts_n_t     et_qubature_faces_pnts_n;
    qubature_faces_weights_t    et_qubature_faces_weights;
    qubature_faces_pnts_t       et_qubature_faces_pnts;
    basis_faces_values_t        et_basis_faces_values;
    basis_faces_derivatives_t   et_basis_faces_derivatives;
    //basis_vert_values_t         et_basis_vert_values;
    //coeff2_mats_t               coeff2_mats;

    linearity_flags_t                   face_linearity_flags;
    loc_face_coords_jacobs_t            face_coords_jacobs;
    loc_face_coords_integral_muls_t     face_coords_integral_muls;
    loc_face_coords_shifts_t            face_coords_shifts;

    qubature_pnts_n_t                   et_face_qubature_pnts_n;
    qubature_weights_t                  et_face_qubature_weights;
    /// Coordinates of face quadratures in global coordinates
    qubature_pnts_t                     et_face_qubature_pnts;

    /// Buffer for different operations like reduce; it doesnot take much - can save it
    elementwise_array_t                 elementwise_array_buf;

    template<bool is_elem_linear>
    __DEVICE_TAG__ elem_jacobi_mat_optional_t<is_elem_linear>
    get_elem_jacobi_mat_optional(ordinal_type elem_i)const
    {
        return detail::elem_jacobi_mat_optional_getter<T,Memory,Dim,is_elem_linear>::get(*this,elem_i);
    }
    template<bool is_elem_linear>
    __DEVICE_TAG__ elem_loc_coords_optional_t<is_elem_linear>
    get_elem_loc_coords_optional(ordinal_type elem_i)const
    {
        return detail::elem_loc_coords_optional_getter<T,Memory,Dim,is_elem_linear>::get(*this,elem_i);
    }
    template<bool is_face_linear>
    __DEVICE_TAG__ face_loc_coords_optional_t<is_face_linear>
    get_face_loc_coords_optional(ordinal_type face_i)const
    {
        return detail::face_loc_coords_optional_getter<T,Memory,Dim,is_face_linear>::get(*this,face_i);
    }

    /// Elements qubatures interface

    __DEVICE_TAG__ ordinal_type get_elem_qubature_pnts_n
    (
        ordinal_type elem_i,
        elem_type_ordinal_t elem_type
    )const
    {
        return et_qubature_pnts_n(elem_i);
    }

    /// lin_jacobi_det is only defined if is_elem_linear==true
    __DEVICE_TAG__ T get_elem_qubature_weight
    (
        ordinal_type elem_i, ordinal_type pnt_i,
        elem_type_ordinal_t elem_type,
        bool is_elem_linear, T lin_jacobi_det
    )const
    {
        return et_qubature_weights(elem_i,pnt_i);
    }

    template<bool has_loc_coords_data>
    __DEVICE_TAG__ vec_t get_elem_qubature_pnt
    (
        ordinal_type elem_i, ordinal_type pnt_i,
        elem_type_ordinal_t elem_type,
        const typename detail::elem_loc_coords_optional<T,Dim,has_loc_coords_data>::type &elem_loc_coords,
        bool is_elem_linear
    )const
    {
        return et_qubature_pnts.get_vec(elem_i,pnt_i);
    }

    /// Elements basis values interface

    __DEVICE_TAG__ T get_basis_value_at_elem_qubature_pnt
    (
        ordinal_type elem_i, ordinal_type basis_i, ordinal_type pnt_i,
        elem_type_ordinal_t elem_type,
        bool is_elem_linear
    )const
    {
        return et_basis_values(elem_i,basis_i,pnt_i);
    }

    template<bool has_loc_coords_data>
    __DEVICE_TAG__ vec_t get_basis_derivative_at_elem_qubature_pnt
    (
        ordinal_type elem_i, ordinal_type basis_i, ordinal_type pnt_i,
        elem_type_ordinal_t elem_type,
        const typename detail::elem_jacobi_mat_optional<T,Dim,has_loc_coords_data>::type &elem_jacobi_mat,
        bool is_elem_linear
    )const
    {
        return et_basis_derivatives.get_vec(elem_i,basis_i,pnt_i);
    }

    /// Element faces qubatures interface

    __DEVICE_TAG__ ordinal_type get_elem_face_qubature_pnts_n
    (
        ordinal_type elem_i, ordinal_type face_i,
        elem_type_ordinal_t elem_type
    )const
    {
        return et_qubature_faces_pnts_n(elem_i,face_i);
    }

    /// lin_jacobi_det is only defined if is_elem_linear==true
    __DEVICE_TAG__ T get_elem_face_qubature_weight
    (
        ordinal_type elem_i, ordinal_type face_i, ordinal_type pnt_i,
        elem_type_ordinal_t elem_type, face_transform_descr_type face_transform,
        bool is_elem_linear, T lin_jacobi_det
    )const
    {
        return et_qubature_faces_weights(elem_i,face_i,pnt_i);
    }

    template<bool has_loc_coords_data>
    __DEVICE_TAG__ vec_t get_elem_face_qubature_pnt
    (
        ordinal_type elem_i, ordinal_type face_i, ordinal_type pnt_i,
        elem_type_ordinal_t elem_type, face_transform_descr_type face_transform,
        const typename detail::elem_loc_coords_optional<T,Dim,has_loc_coords_data>::type &elem_loc_coords,
        bool is_elem_linear
    )const
    {
        return et_qubature_faces_pnts.get_vec(elem_i,face_i,pnt_i);
    }

    /// Element faces basis values interface

    __DEVICE_TAG__ T get_basis_value_at_elem_face_qubature_pnt
    (
        ordinal_type elem_i, ordinal_type face_i, ordinal_type basis_i, ordinal_type pnt_i,
        elem_type_ordinal_t elem_type, face_transform_descr_type face_transform,
        bool is_elem_linear
    )const
    {
        return et_basis_faces_values(elem_i,basis_i,face_i,pnt_i);
    }

    template<bool has_loc_coords_data>
    __DEVICE_TAG__ vec_t get_basis_derivative_at_elem_face_qubature_pnt
    (
        ordinal_type elem_i, ordinal_type face_i, ordinal_type basis_i, ordinal_type pnt_i,
        elem_type_ordinal_t elem_type, face_transform_descr_type face_transform,
        const typename detail::elem_jacobi_mat_optional<T,Dim,has_loc_coords_data>::type &elem_jacobi_mat,
        bool is_elem_linear
    )const
    {
        return et_basis_faces_derivatives.get_vec(elem_i,basis_i,face_i,pnt_i);
    }

    /// Faces qubatures interface

    __DEVICE_TAG__ ordinal_type get_face_qubature_pnts_n
    (
        ordinal_type face_i,
        elem_type_ordinal_t elem_type
    )const
    {
        return et_face_qubature_pnts_n(face_i);
    }

    /// lin_integral_mul is only defined if is_face_linear==true
    __DEVICE_TAG__ T get_face_qubature_weight
    (
        ordinal_type face_i, ordinal_type pnt_i,
        elem_type_ordinal_t elem_type,
        bool is_face_linear, T lin_integral_mul
    )const
    {
        return et_face_qubature_weights(face_i,pnt_i);
    }

    template<bool has_loc_coords_data>
    __DEVICE_TAG__ vec_t get_face_qubature_pnt
    (
        ordinal_type face_i, ordinal_type pnt_i,
        elem_type_ordinal_t elem_type,
        const typename detail::face_loc_coords_optional<T,Dim,has_loc_coords_data>::type &face_loc_coords,
        bool is_face_linear
    )const
    {
        return et_face_qubature_pnts.get_vec(face_i,pnt_i);
    }

    /// poly_order is polynomial order (max order of DG discretization in this case is poly_order+1)
    /// vol_qub_order in general nonlinear case must be equal to poly_order*2
    /// faces_qub_order in general nonlinear case must be equal to poly_order*2+1
    template<class ForEach>
    void init
    (
        const device_mesh_t &device_mesh,
        int poly_order, int vol_qub_order, int faces_qub_order,
        const ForEach &for_each
    );

    template<class ForEach>
    void calc_basis_funcs_mass_matrix_diag
    (
        basis_array_t res_basis_funcs_mass_matrix_diag,
        const ForEach &for_each
    );

    template<class Func,class ForEach>
    void calc_l2_local_proj
    (
        bool ignore_mass_matrix, basis_array_t basis_funcs_mass_matrix_diag,Func f,
        basis_array_t res_basis_funcs_mass_matrix_diag,
        const ForEach &for_each
    );

    template<class Func,class ForEach>
    void calc_func_values
    (
        Func f, scalar_func_values_t scalar_func_values,
        const ForEach &for_each
    );
    template<class ForEach>
    void calc_func_values_from_basis_array
    (
        basis_array_t func_basis_array,
        scalar_func_values_t scalar_func_values,
        const ForEach &for_each
    );
    template<class ForEach,class Reduce>
    T    calc_l2_norm_by_func_values
    (
        scalar_func_values_t scalar_func_values,
        const ForEach &for_each, const Reduce &reduce
    );
    template<class ForEach,class Reduce>
    T    calc_diff_l2_norm_by_func_values
    (
        scalar_func_values_t scalar_func1_values,
        scalar_func_values_t scalar_func2_values,
        const ForEach &for_each, const Reduce &reduce
    );

    template<class ForEach>
    void convert_basis_coeffs_into_monom_coeffs
    (
        basis_array_t basis_array,monoms_basis_array_t res_monoms_basis_array,
        const ForEach &for_each
    )const;
};

}  /// namespace dg
}  /// namespace scfd

#endif
