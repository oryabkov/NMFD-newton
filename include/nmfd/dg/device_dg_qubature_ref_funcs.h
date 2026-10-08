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

#ifndef __SCFD_DEVICE_DG_QUBATURE_REF_FUNCS_H__
#define __SCFD_DEVICE_DG_QUBATURE_REF_FUNCS_H__

#include <nmfd/mesh/cubature_reference.h>
#include "detail/const_data_access.h"
#include "detail/device_dg_basis_reference.h"
#include "detail/polynom_indexing.h"
#include "scfd/utils/scalar_traits.h"

namespace scfd
{
namespace dg
{
namespace detail
{

//TODO just temporal -> use mesh_reference instead
__DEVICE_TAG__ int      get_elem_faces_n(int elem_type)
{
    //ISSUE to make faster access (perhaps, not tested yet) we can make constant array
    if (elem_type == 4) return 4;
    if (elem_type == 5) return 6;
    if (elem_type == 6) return 5;
    if (elem_type == 7) return 5;
    //TODO others, error
    return 0;       //just to remove warning
}

__DEVICE_TAG__ int      get_elem_vert_n(int elem_type)
{
    //ISSUE to make faster access (perhaps, not tested yet) we can make constant array
    if (elem_type == 4) return 4;
    if (elem_type == 5) return 8;
    if (elem_type == 6) return 6;
    if (elem_type == 7) return 5;
    //TODO others, error
    return 0;       //just to remove warning
}

/// Is there any point in MaxPolyOrder if we use global macro anyway?
template<class T, class Memory, int Dim, class Ord, int MaxPolyOrder>
struct device_dg_qubature_ref_funcs
{
    static const int          dim = Dim;

    using scalar = T;
    using ordinal = Ord;
    using sc_tr = scfd::utils::scalar_traits<scalar>;
    using mat_t = scfd::static_mat::mat<scalar,dim,dim>;
    using vec_t = scfd::static_vec::vec<scalar,dim>;
    using elem_reference_t = mesh::gmsh_mesh_elem_reference<T>;
    using qub_reference_t = mesh::cubature_reference<MaxPolyOrder*2+1,T>;
    using basis_elem_reference_t = detail::device_dg_basis_elem_reference<T,Dim,MaxPolyOrder>;
    using basis_reference_t = detail::device_dg_basis_reference<T,Dim,MaxPolyOrder>;
    using device_mesh_t = mesh::device_mesh<T,Memory,Dim,Ord>;
    using device_dg_reference_t = device_dg_qubature_ref<T,Memory,Dim>;
    using elem_type_ord_t = typename device_mesh_t::elem_type_ordinal_type;
    using basis_array_t = typename device_dg_reference_t::basis_array_t;
    using monoms_basis_array_t = typename device_dg_reference_t::monoms_basis_array_t;
    using scalar_func_values_t = typename device_dg_reference_t::scalar_func_values_t;
    using elementwise_array_t = typename device_dg_reference_t::elementwise_array_t;

    static __DEVICE_TAG__ const elem_reference_t &elem_ref()
    {
        return get_elem_ref<T>();
    }
    static __DEVICE_TAG__ const qub_reference_t &cub_ref()
    {
        return get_cub_ref<T>();
    }
    static __DEVICE_TAG__ const basis_reference_t &basis_ref()
    {
        return get_basis_ref<T,Dim,MaxPolyOrder>();
    }
    static __DEVICE_TAG__ const device_mesh_t  &mesh()
    {
        return get_mesh<T,Memory,Dim,Ord>();
    }
    static __DEVICE_TAG__ const device_dg_reference_t  &dg_ref()
    {
        return get_dg_qubature_ref<T,Memory,Dim>();
    }

    struct calc_loc_coords
    {
        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            elem_type_ord_t elem_type = mesh().get_elem_type(i);

            mat_t   J;
            mat_t   J_inv;
            T       det_J;
            vec_t   c0;

            /// TODO planar elements
            if (elem_type == 4) // tetrahedral
            {
                for (int j = 0;j < 3;j++)
                {
                    vec_t tmp = mesh().elems_vertexes.get_vec(i,j+1) - mesh().elems_vertexes.get_vec(i,0);
                    J(0,j) = tmp[0]; J(1,j) = tmp[1]; J(2,j) = tmp[2];
                }
                c0 = mesh().elems_vertexes.get_vec(i,0);
            }
            else if (elem_type == 5)
            {
                vec_t tmp;
                // TODO reference system must coincide with local
                //tmp = mesh().elems_vertexes.get_vec(i,1) - mesh().elems_vertexes.get_vec(i,0);
                tmp = mesh().elems_faces_centers.get_vec(i,3) - mesh().elems_centers.get_vec(i);
                J(0,0) = tmp[0]; J(1,0) = tmp[1]; J(2,0) = tmp[2];
                //tmp = mesh().elems_vertexes.get_vec(i,3) - mesh().elems_vertexes.get_vec(i,0);
                tmp = mesh().elems_faces_centers.get_vec(i,4) - mesh().elems_centers.get_vec(i);
                J(0,1) = tmp[0]; J(1,1) = tmp[1]; J(2,1) = tmp[2];
                //tmp = mesh().elems_vertexes.get_vec(i,4) - mesh().elems_vertexes.get_vec(i,0);
                tmp = mesh().elems_faces_centers.get_vec(i,5) - mesh().elems_centers.get_vec(i);
                J(0,2) = tmp[0]; J(1,2) = tmp[1]; J(2,2) = tmp[2];
                c0 = mesh().elems_centers.get_vec(i);
            }
            else if (elem_type == 6)
            {
                vec_t tmp;
                // TODO reference system must coincide with local
                tmp = mesh().elems_vertexes.get_vec(i,1) - mesh().elems_vertexes.get_vec(i,0);
                J(0,0) = tmp[0]; J(1,0) = tmp[1]; J(2,0) = tmp[2];
                tmp = mesh().elems_vertexes.get_vec(i,2) - mesh().elems_vertexes.get_vec(i,0);
                J(0,1) = tmp[0]; J(1,1) = tmp[1]; J(2,1) = tmp[2];
                //tmp = mesh().elems_vertexes.get_vec(i,3) - mesh().elems_vertexes.get_vec(i,0);
                tmp = (mesh().elems_vertexes.get_vec(i,3) - mesh().elems_vertexes.get_vec(i,0))*T(0.5f);
                J(0,2) = tmp[0]; J(1,2) = tmp[1]; J(2,2) = tmp[2];
                c0 = (mesh().elems_vertexes.get_vec(i,3) + mesh().elems_vertexes.get_vec(i,0))*T(0.5f);
            }
            else
            {
                //TODO error
                //throw std::logic_error("t_DG_tml::jacobian_init: not supported element type yet");
            }

            /// write down results

            J_inv = inv(J);
            det_J = det(J);

            /// TODO only linear elements for now
            dg_ref().linearity_flags(i) = true;

            dg_ref().coords_jacobs_dets(i) = det_J;

            #pragma unroll
            for (int ii1 = 0;ii1 < dim;++ii1)
            {
                #pragma unroll
                for (int ii2 = 0;ii2 < dim;++ii2)
                {
                    dg_ref().coords_jacobs(i,ii1,ii2) = J(ii1,ii2);
                    dg_ref().coords_inv_jacobs(i,ii1,ii2) = J_inv(ii1,ii2);
                }
            }

            #pragma unroll
            for (int ii1 = 0;ii1 < dim;++ii1)
            {
                dg_ref().coords_shifts(i,ii1) = c0[ii1];
            }
        }
    };

    /// This is per element functor
    struct calc_qubature_weights
    {
        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            //const auto      &elem_ref =

            //int                             poly_order = dg_ref().poly_order;
            int                             vol_qub_order = dg_ref().vol_qub_order;
            elem_type_ord_t                 elem_type = mesh().get_elem_type(i);
            T                               det_J = dg_ref().coords_jacobs_dets(i);
            static_mat::mat<T,dim,dim>      J,J_inv;
            vec_t                           c0;

            #pragma unroll
            for (int ii1 = 0;ii1 < dim;++ii1)
            {
                #pragma unroll
                for (int ii2 = 0;ii2 < dim;++ii2)
                {
                    J(ii1,ii2) = dg_ref().coords_jacobs(i,ii1,ii2);
                    J_inv(ii1,ii2) = dg_ref().coords_inv_jacobs(i,ii1,ii2);
                }
            }

            #pragma unroll
            for (int ii1 = 0;ii1 < dim;++ii1)
            {
                c0[ii1] = dg_ref().coords_shifts(i,ii1);
            }

            /// TODO!! here we suppose that reference and local coordinate systems concide!
            const auto      &elem_q_loc = cub_ref().get_elem_cubature(elem_type);
            const auto      &basis_elem_reference = basis_ref().get_elem_basis_ref(elem_type);

            dg_ref().et_qubature_pnts_n(i) = elem_q_loc.get_pnts_n(vol_qub_order);

            /// Calc volumentric et_qubature_weights
            for (int pnt_i = 0; pnt_i < elem_q_loc.get_pnts_n(vol_qub_order); ++pnt_i)
            {
                dg_ref().et_qubature_weights(i,pnt_i) = elem_q_loc.get_weight(vol_qub_order, pnt_i) * det_J;

                static_vec::vec<T,dim>       ref_pnt;
                #pragma unroll
                for (int ii1 = 0;ii1 < dim;++ii1)
                    ref_pnt[ii1] = elem_q_loc.get_pnt(vol_qub_order, pnt_i, ii1);

                static_vec::vec<T,dim>       g_pnt;
                #pragma unroll
                for (int ii1 = 0;ii1 < dim;++ii1)
                {
                    g_pnt[ii1] = c0[ii1];
                    #pragma unroll
                    for (int ii2 = 0;ii2 < dim;++ii2)
                    {
                        g_pnt[ii1] += J(ii1,ii2) * ref_pnt[ii2];
                    }
                }

                #pragma unroll
                for (int ii1 = 0;ii1 < dim;++ii1)
                {
                    dg_ref().et_qubature_pnts(i,pnt_i,ii1) = g_pnt[ii1];
                }
            }

            /// Calc et_basis_values in volumentric qubature points
            for (int basis_i = 0;basis_i < dg_ref().max_basis_size;++basis_i)
            {
                for (int pnt_i = 0; pnt_i < elem_q_loc.get_pnts_n(vol_qub_order); ++pnt_i)
                {
                    vec_t pnt = elem_q_loc.get_pnt(vol_qub_order, pnt_i);
                    dg_ref().et_basis_values(i,basis_i,pnt_i) = basis_elem_reference.funcs[basis_i].eval(pnt.d);
                }
            }

            /// Calc et_basis_derivatives in volumentric qubature points
            for (int basis_i = 0;basis_i < dg_ref().max_basis_size;++basis_i)
            {
                for (int pnt_i = 0; pnt_i < elem_q_loc.get_pnts_n(vol_qub_order); ++pnt_i)
                {
                    vec_t pnt = elem_q_loc.get_pnt(vol_qub_order, pnt_i);
                    vec_t der;
                    for (int j = 0;j < Dim;++j)
                        der[j] = basis_elem_reference.funcs_ders[basis_i][j].eval(pnt.d);
                    der = J_inv.transposed()*der;
                    for (int j = 0;j < Dim;++j)
                        dg_ref().et_basis_derivatives(i,basis_i,pnt_i,j) = der[j];
                }
            }

        }
    };

    /// This is per face (physical face) functor
    struct calc_faces_loc_coords
    {
        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            elem_type_ord_t face_type = mesh().get_face_type(i);

            static_mat::mat<T,dim,dim-1>   J;
            T                              integral_mul_J;
            vec_t                          c0;

            /// TODO 1d elements
            if (face_type == 2) // triangle
            {
                for (int j = 0;j < 2;j++)
                {
                    vec_t tmp = mesh().faces_vertexes.get_vec(i,j+1) - mesh().faces_vertexes.get_vec(i,0);
                    J(0,j) = tmp[0]; J(1,j) = tmp[1]; J(2,j) = tmp[2];
                }
                c0 = mesh().faces_vertexes.get_vec(i,0);
            }
            else if (face_type == 3) // quadrangle
            {
                vec_t tmp;
                tmp = T(0.5)*(mesh().faces_vertexes.get_vec(i,1) - mesh().faces_vertexes.get_vec(i,0));
                J(0,0) = tmp[0]; J(1,0) = tmp[1]; J(2,0) = tmp[2];
                tmp = T(0.5)*(mesh().faces_vertexes.get_vec(i,3) - mesh().faces_vertexes.get_vec(i,0));
                J(0,1) = tmp[0]; J(1,1) = tmp[1]; J(2,1) = tmp[2];
                c0 = mesh().faces_centers.get_vec(i);
            }

            if (dim == 3)
            {
                vec_t     r_u, r_v;
                #pragma unroll
                for (int ii1 = 0;ii1 < dim;++ii1)
                {
                    r_u[ii1] = J(ii1,0);
                    r_v[ii1] = J(ii1,1);
                }

                integral_mul_J = vector_prod(r_u,r_v).norm2();
            }
            else if (dim == 2)
            {
                /// TODO
            }

            /// TODO only linear elements for now
            dg_ref().face_linearity_flags(i) = true;

            dg_ref().face_coords_integral_muls(i) = integral_mul_J;

            #pragma unroll
            for (int ii1 = 0;ii1 < dim;++ii1)
            {
                #pragma unroll
                for (int ii2 = 0;ii2 < dim-1;++ii2)
                {
                    dg_ref().face_coords_jacobs(i,ii1,ii2) = J(ii1,ii2);
                }
            }

            #pragma unroll
            for (int ii1 = 0;ii1 < dim;++ii1)
            {
                dg_ref().face_coords_shifts(i,ii1) = c0[ii1];
            }
        }
    };

    /// This is per face (physical face) functor
    struct calc_faces_quadratures_weights
    {
        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            //const auto      &elem_ref =

            //int             poly_order = dg_ref().poly_order;
            int             faces_qub_order = dg_ref().faces_qub_order;
            elem_type_ord_t face_type = mesh().get_face_type(i);
            T                              mul = dg_ref().face_coords_integral_muls(i);
            static_mat::mat<T,dim,dim-1>   J;
            vec_t                          c0;

            #pragma unroll
            for (int ii1 = 0;ii1 < dim;++ii1)
            {
                #pragma unroll
                for (int ii2 = 0;ii2 < dim-1;++ii2)
                {
                    J(ii1,ii2) = dg_ref().face_coords_jacobs(i,ii1,ii2);
                }
            }

            #pragma unroll
            for (int ii1 = 0;ii1 < dim;++ii1)
            {
                c0[ii1] = dg_ref().face_coords_shifts(i,ii1);
            }

            /// TODO!! here we suppose that reference and local coordinate systems concide!
            const auto      &face_q_loc = cub_ref().get_elem_cubature(face_type);

            dg_ref().et_face_qubature_pnts_n(i) = face_q_loc.get_pnts_n(faces_qub_order);

            for (int pnt_i = 0; pnt_i < face_q_loc.get_pnts_n(faces_qub_order); ++pnt_i)
            {
                dg_ref().et_face_qubature_weights(i,pnt_i) = face_q_loc.get_weight(faces_qub_order, pnt_i) * mul;

                static_vec::vec<T,dim-1>     ref_pnt;
                #pragma unroll
                for (int ii1 = 0;ii1 < dim-1;++ii1)
                    ref_pnt[ii1] = face_q_loc.get_pnt(faces_qub_order, pnt_i, ii1);

                static_vec::vec<T,dim>       g_pnt;
                #pragma unroll
                for (int ii1 = 0;ii1 < dim;++ii1)
                {
                    g_pnt[ii1] = c0[ii1];
                    #pragma unroll
                    for (int ii2 = 0;ii2 < dim-1;++ii2)
                    {
                        g_pnt[ii1] += J(ii1,ii2) * ref_pnt[ii2];
                    }
                }

                #pragma unroll
                for (int ii1 = 0;ii1 < dim;++ii1)
                {
                    dg_ref().et_face_qubature_pnts(i,pnt_i,ii1) = g_pnt[ii1];
                }
            }

        }
    };

    /// This is per element functor
    struct init_elem_faces_qubature_pnts
    {
        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            //int             poly_order = dg_ref().poly_order;
            elem_type_ord_t elem_type = mesh().get_elem_type(i);

            /// TODO!! here we suppose that reference and local coordinate systems concide!
            const auto      &basis_elem_reference = basis_ref().get_elem_basis_ref(elem_type);

            mat_t   J_inv;
            vec_t   c0;

            #pragma unroll
            for (int ii1 = 0;ii1 < dim;++ii1)
            {
                #pragma unroll
                for (int ii2 = 0;ii2 < dim;++ii2)
                {
                    J_inv(ii1,ii2) = dg_ref().coords_inv_jacobs(i,ii1,ii2);
                }
            }
            #pragma unroll
            for (int ii1 = 0;ii1 < dim;++ii1)
            {
                c0[ii1] = dg_ref().coords_shifts(i,ii1);
            }

            for (int face_i = 0;face_i < elem_ref().get_faces_n(elem_type);++face_i)
            {
                Ord face_id = mesh().elems_faces_ids(i,face_i);
                int face_pnts_n = dg_ref().et_face_qubature_pnts_n(face_id);
                dg_ref().et_qubature_faces_pnts_n(i,face_i) = face_pnts_n;
                for (int pnt_i = 0;pnt_i < face_pnts_n;++pnt_i)
                {
                    dg_ref().et_qubature_faces_weights(i,face_i,pnt_i) = dg_ref().et_face_qubature_weights(face_id,pnt_i);
                    for (int j = 0;j < Dim;++j)
                    {
                        dg_ref().et_qubature_faces_pnts(i,face_i,pnt_i,j) =
                            dg_ref().et_face_qubature_pnts(face_id,pnt_i,j);
                    }
                    /// qudrature point in global coordinates
                    vec_t g_pnt = dg_ref().et_face_qubature_pnts.get_vec(face_id,pnt_i);
                    /// qudrature point in element local coordinates
                    vec_t elem_loc_pnt = J_inv*(g_pnt - c0);

                    for (int basis_i = 0;basis_i < dg_ref().max_basis_size;++basis_i)
                    {
                        dg_ref().et_basis_faces_values(i,basis_i,face_i,pnt_i) =
                            basis_elem_reference.funcs[basis_i].eval(elem_loc_pnt.d);
                        vec_t der;
                        for (int j = 0;j < Dim;++j)
                        {
                            der[j] = basis_elem_reference.funcs_ders[basis_i][j].eval(elem_loc_pnt.d);
                        }
                        der = J_inv.transposed()*der;
                        for (int j = 0;j < Dim;++j)
                        {
                            dg_ref().et_basis_faces_derivatives(i,basis_i,face_i,pnt_i,j) = der[j];
                        }
                    }
                }

            }
        }
    };

    /// This is per element functor
    struct calc_basis_funcs_mass_matrix_diag
    {
        FOR_EACH_FUNC_PARAMS_HELP
        (
            calc_basis_funcs_mass_matrix_diag,
            basis_array_t, res_basis_funcs_mass_matrix_diag
        )

        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            bool            is_elem_linear = dg_ref().linearity_flags(i);

            elem_type_ord_t elem_type = mesh().get_elem_type(i);
            Ord             pnts_n = dg_ref().get_elem_qubature_pnts_n(i,elem_type);
            T               lin_jacobi_det = dg_ref().coords_jacobs_dets(i);

            for (Ord basis_i = 0;basis_i < dg_ref().max_basis_size;++basis_i)
            {
                T    res(0);
                for (Ord pnt_i = 0;pnt_i < pnts_n;++pnt_i)
                {
                    //vec_t c = dg_ref().et_qubature_pnts.get_vec(i,pnt_i);
                    T     w =
                        dg_ref().get_elem_qubature_weight
                        (
                            i,pnt_i,elem_type,is_elem_linear,lin_jacobi_det
                        );

                    T     basis_val =
                        dg_ref().get_basis_value_at_elem_qubature_pnt
                        (
                            i,basis_i,pnt_i,elem_type,is_elem_linear
                        );

                    T     f_c = basis_val*basis_val;
                    res += f_c*w;
                }
                res_basis_funcs_mass_matrix_diag(i,basis_i) = res;
            }
        }
    };

    template<bool calc_linear_elems,class FUNC>
    struct calc_l2_local_proj
    {
        FOR_EACH_FUNC_PARAMS_HELP
        (
            calc_l2_local_proj,
            bool, ignore_mass_matrix, basis_array_t, basis_funcs_mass_matrix_diag, FUNC, f,
            basis_array_t, res_funcs_coeffs
        )

        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            bool            is_elem_linear = dg_ref().linearity_flags(i);
            if (is_elem_linear != calc_linear_elems) return;

            elem_type_ord_t elem_type = mesh().get_elem_type(i);
            Ord             pnts_n = dg_ref().get_elem_qubature_pnts_n(i,elem_type);
            T               lin_jacobi_det = dg_ref().coords_jacobs_dets(i);
            auto            elem_loc_coords = dg_ref().template get_elem_loc_coords_optional<calc_linear_elems>(i);

            for (Ord basis_i = 0;basis_i < dg_ref().max_basis_size;++basis_i)
            {
                T    res(0);
                for (Ord pnt_i = 0;pnt_i < pnts_n;++pnt_i)
                {
                    vec_t c =
                        dg_ref().template get_elem_qubature_pnt<calc_linear_elems>
                        (
                            i, pnt_i, elem_type, elem_loc_coords, is_elem_linear
                        );
                    T     w =
                        dg_ref().get_elem_qubature_weight
                        (
                            i,pnt_i,elem_type,is_elem_linear,lin_jacobi_det
                        );
                    T     basis_val =
                        dg_ref().get_basis_value_at_elem_qubature_pnt
                        (
                            i,basis_i,pnt_i,elem_type,is_elem_linear
                        );
                    T     func_val = f(c);
                    T     f_c = basis_val*func_val;
                    res += f_c*w;
                }
                if (!ignore_mass_matrix)
                    res_funcs_coeffs(i,basis_i) = res / basis_funcs_mass_matrix_diag(i,basis_i);
                else
                    res_funcs_coeffs(i,basis_i) = res;
            }
        }
    };

    /// This is per element functor
    template<bool calc_linear_elems,class Func>
    struct calc_func_values
    {
        FOR_EACH_FUNC_PARAMS_HELP
        (
            calc_func_values,
            Func, f, scalar_func_values_t, scalar_func_values
        )

        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            bool            is_elem_linear = dg_ref().linearity_flags(i);
            if (is_elem_linear != calc_linear_elems) return;

            elem_type_ord_t elem_type = mesh().get_elem_type(i);
            Ord             pnts_n = dg_ref().get_elem_qubature_pnts_n(i,elem_type);
            T               lin_jacobi_det = dg_ref().coords_jacobs_dets(i);
            auto            elem_loc_coords = dg_ref().template get_elem_loc_coords_optional<calc_linear_elems>(i);

            for (Ord pnt_i = 0;pnt_i < pnts_n;++pnt_i)
            {
                vec_t c =
                    dg_ref().template get_elem_qubature_pnt<calc_linear_elems>
                    (
                        i,pnt_i,elem_type,elem_loc_coords,is_elem_linear
                    );
                T     func_val = f(c);

                scalar_func_values(i,pnt_i) = func_val;
            }
        }
    };

    /// This is per element functor
    struct calc_func_values_from_basis_array
    {
        FOR_EACH_FUNC_PARAMS_HELP
        (
            calc_func_values_from_basis_array,
            basis_array_t, func_basis_array,
            scalar_func_values_t, scalar_func_values
        )

        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            bool            is_elem_linear = dg_ref().linearity_flags(i);

            elem_type_ord_t elem_type = mesh().get_elem_type(i);
            Ord             pnts_n = dg_ref().get_elem_qubature_pnts_n(i,elem_type);
            T               lin_jacobi_det = dg_ref().coords_jacobs_dets(i);

            for (Ord pnt_i = 0;pnt_i < pnts_n;++pnt_i)
            {
                T    res_val(0);
                for (Ord basis_i = 0;basis_i < dg_ref().max_basis_size;++basis_i)
                {
                    T     basis_val =
                        dg_ref().get_basis_value_at_elem_qubature_pnt
                        (
                            i,basis_i,pnt_i,elem_type,is_elem_linear
                        );
                    res_val += func_basis_array(i,basis_i)*basis_val;
                }

                scalar_func_values(i,pnt_i) = res_val;
            }
        }
    };

    /// This is per element functor
    struct calc_l2_norm_by_func_values
    {
        FOR_EACH_FUNC_PARAMS_HELP
        (
            calc_l2_norm_by_func_values,
            scalar_func_values_t, scalar_func_values,
            elementwise_array_t, local_l2_parts
        )

        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            bool            is_elem_linear = dg_ref().linearity_flags(i);

            elem_type_ord_t elem_type = mesh().get_elem_type(i);
            Ord             pnts_n = dg_ref().get_elem_qubature_pnts_n(i,elem_type);
            T               lin_jacobi_det = dg_ref().coords_jacobs_dets(i);

            T    res(0);
            for (Ord pnt_i = 0;pnt_i < pnts_n;++pnt_i)
            {
                T     w =
                    dg_ref().get_elem_qubature_weight
                    (
                        i, pnt_i, elem_type, is_elem_linear, lin_jacobi_det
                    );
                T     f_c = scalar_func_values(i,pnt_i);
                res += (f_c*f_c)*w;
            }

            local_l2_parts(i) = res;
        }
    };

    /// This is per element functor
    struct calc_diff_l2_norm_by_func_values
    {
        FOR_EACH_FUNC_PARAMS_HELP
        (
            calc_diff_l2_norm_by_func_values,
            scalar_func_values_t, scalar_func1_values,
            scalar_func_values_t, scalar_func2_values,
            elementwise_array_t, local_l2_parts
        )

        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            bool            is_elem_linear = dg_ref().linearity_flags(i);

            elem_type_ord_t elem_type = mesh().get_elem_type(i);
            Ord             pnts_n = dg_ref().get_elem_qubature_pnts_n(i,elem_type);
            T               lin_jacobi_det = dg_ref().coords_jacobs_dets(i);

            T    res(0);
            for (Ord pnt_i = 0;pnt_i < pnts_n;++pnt_i)
            {
                T     w =
                    dg_ref().get_elem_qubature_weight
                    (
                        i, pnt_i, elem_type, is_elem_linear, lin_jacobi_det
                    );
                T     f1_c = scalar_func1_values(i,pnt_i),
                      f2_c = scalar_func2_values(i,pnt_i);
                res += ((f1_c-f2_c)*(f1_c-f2_c))*w;
            }

            local_l2_parts(i) = res;
        }
    };

    /// This is per element functor
    struct convert_basis_coeffs_into_monom_coeffs
    {
        FOR_EACH_FUNC_PARAMS_HELP
        (
            convert_basis_coeffs_into_monom_coeffs,
            basis_array_t, basis_array,
            monoms_basis_array_t, res_monoms_basis_array
        )

        __DEVICE_TAG__ void operator()(const Ord &i)const
        {
            elem_type_ord_t     elem_type = mesh().get_elem_type(i);
            bool                is_elem_linear = dg_ref().linearity_flags(i);
            const auto          &elem_basis_ref = basis_ref().get_elem_basis_ref(elem_type);

            if (!is_elem_linear)
            {
                /// TODO where do we store basis in this case?
                printf("convert_basis_coeffs_into_monom_coeffs: nonlinear case NOT IMPLEMENTED YET!");
                return;
            }

            for (int mon_i = 0;mon_i < dg_ref().polylen;++mon_i)
            {
                T mon_coeff(0);
                for (int basis_i = 0;basis_i < dg_ref().max_basis_size;++basis_i)
                {
                    mon_coeff += elem_basis_ref.funcs[basis_i].coeffs[mon_i]*basis_array(i,basis_i);
                }
                res_monoms_basis_array(i,mon_i) = mon_coeff;
            }
        }
    };

};

}  /// namespace detail
}  /// namespace dg
}  /// namespace scfd

#endif
