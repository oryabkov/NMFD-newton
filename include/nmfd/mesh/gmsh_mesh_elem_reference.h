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

#ifndef __SCFD_MESH_GMSH_MESH_ELEM_REFERENCE_H__
#define __SCFD_MESH_GMSH_MESH_ELEM_REFERENCE_H__

//TODO mesh/device_tag.h is temporal

#include <scfd/utils/device_tag.h>
#include <scfd/static_vec/vec.h>

//NOTE all element types, vertexes and faces enumerations, local coordinate systes are synced with gmsh system
//for more visit http://gmsh.info/doc/texinfo/gmsh.html :
/*

1D:
1    2-node line. 
2D:
2    3-node triangle. 
3    4-node quadrangle. 
3D:
4    4-node tetrahedron. 
5    8-node hexahedron. 
6    6-node prism. 
7    5-node pyramid. 

1D:
8    3-node second order line (2 nodes associated with the vertices and 1 with the edge). 
2D:
9    6-node second order triangle (3 nodes associated with the vertices and 3 with the edges). 
10    9-node second order quadrangle (4 nodes associated with the vertices, 4 with the edges and 1 with the face). 
3D:
11    10-node second order tetrahedron (4 nodes associated with the vertices and 6 with the edges). 
12    27-node second order hexahedron (8 nodes associated with the vertices, 12 with the edges, 6 with the faces and 1 with the volume). 
13    18-node second order prism (6 nodes associated with the vertices, 9 with the edges and 3 with the quadrangular faces). 
14    14-node second order pyramid (5 nodes associated with the vertices, 8 with the edges and 1 with the quadrangular face). 
15    1-node point. 
16    8-node second order quadrangle (4 nodes associated with the vertices and 4 with the edges). 
17    20-node second order hexahedron (8 nodes associated with the vertices and 12 with the edges). 
18    15-node second order prism (6 nodes associated with the vertices and 9 with the edges). 
19    13-node second order pyramid (5 nodes associated with the vertices and 8 with the edges). 
20    9-node third order incomplete triangle (3 nodes associated with the vertices, 6 with the edges) 
21    10-node third order triangle (3 nodes associated with the vertices, 6 with the edges, 1 with the face) 
22    12-node fourth order incomplete triangle (3 nodes associated with the vertices, 9 with the edges) 
23    15-node fourth order triangle (3 nodes associated with the vertices, 9 with the edges, 3 with the face) 
24    15-node fifth order incomplete triangle (3 nodes associated with the vertices, 12 with the edges) 
25    21-node fifth order complete triangle (3 nodes associated with the vertices, 12 with the edges, 6 with the face) 
26    4-node third order edge (2 nodes associated with the vertices, 2 internal to the edge) 
27    5-node fourth order edge (2 nodes associated with the vertices, 3 internal to the edge) 
28    6-node fifth order edge (2 nodes associated with the vertices, 4 internal to the edge) 
29    20-node third order tetrahedron (4 nodes associated with the vertices, 12 with the edges, 4 with the faces) 
30    35-node fourth order tetrahedron (4 nodes associated with the vertices, 18 with the edges, 12 with the faces, 1 in the volume) 
31    56-node fifth order tetrahedron (4 nodes associated with the vertices, 24 with the edges, 24 with the faces, 4 in the volume) 
92    64-node third order hexahedron (8 nodes associated with the vertices, 24 with the edges, 24 with the faces, 8 in the volume) 
93    125-node fourth order hexahedron (8 nodes associated with the vertices, 36 with the edges, 54 with the faces, 27 in the volume) 

*/

namespace scfd
{
namespace mesh
{

/**
* Think that Dim abstraction is not needed (on level of mesh all elements are 3d)
* Ordinal is also not needed (not much elements, int is more then enoght)
*/

//TODO we don't account for these parameters (ElemTypesNum,...) below when initilize this 
//structure (i.e. we initialize data for hexahedra whatever ElemTypesNum equals)
//if these macros somehow will be changed we can get some errors
//on the other hand i don't see any porblems with storing ALL possible element types data

template
<
    class T,
    int ElemTypesNum = 8,
    int MaxFacesNum = 6,
    int MaxVertsNum = 8,
    int FaceMaxVertsNum = 4
>
struct gmsh_mesh_elem_reference
{
    using vec = static_vec::vec<T,3>;

    int             faces_n[ElemTypesNum];
    int             verts_n[ElemTypesNum];
    int             prim_verts_n[ElemTypesNum];
    vec             verts[ElemTypesNum][MaxVertsNum];
    vec             prim_verts[ElemTypesNum][MaxVertsNum];
    int             face_elem_type[ElemTypesNum][MaxFacesNum];
    int             face_verts_n[ElemTypesNum][MaxFacesNum];
    int             face_prim_verts_n[ElemTypesNum][MaxFacesNum];
    int             face_verts[ElemTypesNum][MaxFacesNum][FaceMaxVertsNum];
    int             face_prim_verts[ElemTypesNum][MaxFacesNum][FaceMaxVertsNum];

    __DEVICE_TAG__ int              get_faces_n(int elem_type)const
    {
        return faces_n[elem_type];
        //TODO elem_type error
    }
    __DEVICE_TAG__ int              get_verts_n(int elem_type)const
    {
        return verts_n[elem_type];
        //TODO elem_type error
    }
    __DEVICE_TAG__ int              get_prim_verts_n(int elem_type)const
    {
        return prim_verts_n[elem_type];
        //TODO elem_type error
    }
    __DEVICE_TAG__ const vec        &get_vert(int elem_type,int vert_i)const
    {
        return verts[elem_type][vert_i];
        //TODO elem_type/vert_i error
    }
    __DEVICE_TAG__ void             get_vert(int elem_type,int vert_i, vec &res)const
    {
        res = verts[elem_type][vert_i];
        //TODO elem_type/vert_i error
    }
    __DEVICE_TAG__ T                get_vert(int elem_type,int vert_i, int j)const
    {
        return verts[elem_type][vert_i][j];
        //TODO elem_type/vert_i/j error
    }
    __DEVICE_TAG__ const vec        &get_prim_vert(int elem_type,int vert_i)const
    {
        return prim_verts[elem_type][vert_i];
        //TODO elem_type/vert_i error
    }
    __DEVICE_TAG__ void             get_prim_vert(int elem_type,int vert_i, vec &res)const
    {
        res = prim_verts[elem_type][vert_i];
        //TODO elem_type/vert_i error
    }
    __DEVICE_TAG__ T                get_prim_vert(int elem_type,int vert_i, int j)const
    {
        return prim_verts[elem_type][vert_i][j];
        //TODO elem_type/vert_i/j error
    }
    __DEVICE_TAG__ int              get_face_elem_type(int elem_type, int face_i)const
    {
        return face_elem_type[elem_type][face_i];
    }
    __DEVICE_TAG__ int              get_face_verts_n(int elem_type, int face_i)const
    {
        return face_verts_n[elem_type][face_i];
        //TODO elem_type/face_i error
    }
    __DEVICE_TAG__ int              get_face_prim_verts_n(int elem_type, int face_i)const
    {
        return face_prim_verts_n[elem_type][face_i];
        //TODO elem_type/face_i error
    }
    __DEVICE_TAG__ int              get_face_vert_i(int elem_type,int face_i, int face_vert_i)const
    {
        return face_verts[elem_type][face_i][face_vert_i];
    }
    __DEVICE_TAG__ const vec        &get_face_vert(int elem_type,int face_i, int vert_i)const
    {
        return verts[elem_type][ face_verts[elem_type][face_i][vert_i] ];
    }
    __DEVICE_TAG__ void             get_face_vert(int elem_type,int face_i, int vert_i, vec &res)const
    {
        res = verts[elem_type][ face_verts[elem_type][face_i][vert_i] ];
    }
    __DEVICE_TAG__ T                get_face_vert(int elem_type,int face_i, int vert_i, int j)const
    {
        return verts[elem_type][ face_verts[elem_type][face_i][vert_i] ][j];
    }
    __DEVICE_TAG__ void             get_face_verts(int elem_type,int face_i, vec vertexes[FaceMaxVertsNum])const
    {
        //to make unroll possible
        //TODO excplicit unroll??
        for (int i = 0;i < FaceMaxVertsNum;++i) {
            if (i == get_face_verts_n(elem_type, face_i)) break;
            vertexes[i] = get_face_vert(elem_type,face_i, i);
        }
    }
    __DEVICE_TAG__ int              get_face_prim_vert_i(int elem_type,int face_i, int face_vert_i)const
    {
        return face_prim_verts[elem_type][face_i][face_vert_i];
    }
    __DEVICE_TAG__ const vec        &get_face_prim_vert(int elem_type,int face_i, int vert_i)const
    {
        return prim_verts[elem_type][ face_verts[elem_type][face_i][vert_i] ];
    }
    __DEVICE_TAG__ void             get_face_prim_vert(int elem_type,int face_i, int vert_i, vec &res)const
    {
        res = prim_verts[elem_type][ face_prim_verts[elem_type][face_i][vert_i] ];
    }
    __DEVICE_TAG__ T                get_face_prim_vert(int elem_type,int face_i, int vert_i, int j)const
    {
        return prim_verts[elem_type][ face_prim_verts[elem_type][face_i][vert_i] ][j];
    }
    __DEVICE_TAG__ void             get_face_prim_verts(int elem_type,int face_i, vec vertexes[FaceMaxVertsNum])const
    {
        //to make unroll possible
        //TODO excplicit unroll??
        for (int i = 0;i < FaceMaxVertsNum;++i) {
            if (i == get_face_prim_verts_n(elem_type, face_i)) break;
            vertexes[i] = get_face_prim_vert(elem_type,face_i, i);
        }
    }

    __DEVICE_TAG__ void             get_face_verts_phys(int elem_type, const vec *elem_vertexes,int face_i, vec   vertexes[FaceMaxVertsNum])const
    {
        //to make unroll possible
        //TODO excplicit unroll??
        for (int i = 0;i < FaceMaxVertsNum;++i) {
            if (i == get_face_verts_n(elem_type, face_i)) break;
            ref_to_phys(elem_type, elem_vertexes, get_face_vert(elem_type,face_i, i), vertexes[i]);
        }
    }
    __DEVICE_TAG__ bool             is_3d(int elem_type)const
    {
        if ((elem_type == 2)||(elem_type == 3)) return false;
        return true;
    }
    //TODO for now seems that it has too many if's clauses so it's better not use it frequently; for now it's used only in preface
    //unfortunatly i did not figure out how to make all this static (there is no template methods specialization inside template classes)
    //NOTE for plananr elements 3rd component in reference coordinates is simply ignored
    __DEVICE_TAG__ T                shape_func(int elem_type, int vert_i,const vec &ref)const
    {
        if (elem_type == 2) {
            if (vert_i == 0) return (T(1.f) - ref[0] - ref[1]); else
            if (vert_i == 1) return ref[0]; else
            if (vert_i == 2) return ref[1]; //TODO else ERROR
        } else if (elem_type == 3) {
            //from course http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch11.d/AFEM.Ch11.pdf
            return  (T(1.f) + verts[elem_type][vert_i][0]*ref[0])*
                (T(1.f) + verts[elem_type][vert_i][1]*ref[1])/T(4.f);
        } else if (elem_type == 4) {
            if (vert_i == 0) return (T(1.f) - ref[0] - ref[1] - ref[2]); else
            if (vert_i == 1) return ref[0]; else
            if (vert_i == 2) return ref[1]; else
            if (vert_i == 3) return ref[2]; //TODO else ERROR
        } else if (elem_type == 5) {
            //from course http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch11.d/AFEM.Ch11.pdf
            return  (T(1.f) + verts[elem_type][vert_i][0]*ref[0])*
                (T(1.f) + verts[elem_type][vert_i][1]*ref[1])*
                (T(1.f) + verts[elem_type][vert_i][2]*ref[2])/T(8.f);
        } else if (elem_type == 6) {
            T       tri_res, lin_res;
            if ((vert_i == 0)||(vert_i == 3)) tri_res = (T(1.f) - ref[0] - ref[1]); else
            if ((vert_i == 1)||(vert_i == 4)) tri_res = ref[0]; else
            if ((vert_i == 2)||(vert_i == 5)) tri_res = ref[1];
            //TODO error else
            lin_res = (T(1.f) + verts[elem_type][vert_i][2]*ref[2])/T(2.f);
            return lin_res*tri_res;
        } else {
            //TODO others
        }
    }
    __DEVICE_TAG__ T                shape_func_der(int elem_type, int vert_i,const vec &ref, int j)const
    {
        if (elem_type == 2) {
            if (j == 2) return T(0.f);
            if (vert_i == 0) return T(-1.f); else
            return (vert_i-1 == j?T(1.f):T(0.f));
            //TODO check whether vert_i < 3, otherwise ERROR (use assert, because it's logic_error)
        } else if (elem_type == 3) {
            if (j == 2) return T(0.f);
            //from course http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch11.d/AFEM.Ch11.pdf
            T       res( verts[elem_type][vert_i][j] );
            for (int jj = 0;jj < 2;++jj) if (jj != j)
                res *= (T(1.f) + verts[elem_type][vert_i][jj]*ref[jj]);
            return  res/T(4.f);
        } else if (elem_type == 4) {
            if (vert_i == 0) return T(-1.f); else
            return (vert_i-1 == j?T(1.f):T(0.f));
            //TODO check whether vert_i < 4, otherwise ERROR (use assert, because it's logic_error)
        } else if (elem_type == 5) {
            //from course http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch11.d/AFEM.Ch11.pdf
            T       res( verts[elem_type][vert_i][j] );
            for (int jj = 0;jj < 3;++jj) if (jj != j)
                res *= (T(1.f) + verts[elem_type][vert_i][jj]*ref[jj]);
            return  res/T(8.f);
        } else if (elem_type == 6) {
            if (j == 2) {
                T       tri_res;
                if ((vert_i == 0)||(vert_i == 3)) tri_res = (T(1.f) - ref[0] - ref[1]); else
                if ((vert_i == 1)||(vert_i == 4)) tri_res = ref[0]; else
                if ((vert_i == 2)||(vert_i == 5)) tri_res = ref[1];
                //TODO error else
                return tri_res*verts[elem_type][vert_i][2]/T(2.f);
            }
            T       lin_res;
            lin_res = (T(1.f) + verts[elem_type][vert_i][2]*ref[2])/T(2.f);
            if (vert_i%3 == 0) return lin_res*T(-1.f); else
            return lin_res*(vert_i%3-1 == j?T(1.f):T(0.f));
        } else {
            //TODO others
        }
    }
    //TODO we use pointer to vec   here which is unaccaptable for __device__ code
    //for 2d elements eembedded in 3d this is not actualy Jacobian but rather metric multiplier that appears in surface integral
    __DEVICE_TAG__ T                ref_to_phys_jacob_det(int elem_type, const vec *vertexes, const vec   &ref)const
    {
        if (is_3d(elem_type)) {
            T       J[3][3];
            for (int i = 0;i < 3;++i)
            for (int j = 0;j < 3;++j) {
                J[i][j] = T(0.f);
                for (int vert_i = 0;vert_i < get_verts_n(elem_type);++vert_i) {
                    //TODO check order of i and j
                    J[i][j] += vertexes[vert_i][i] * shape_func_der(elem_type, vert_i, ref, j);
                }
            }
            return mat33_det(J);
        } else {
            vec     r_u, r_v;
            for (int i = 0;i < 3;++i) {
                r_u[i] = T(0.f);
                r_v[i] = T(0.f);
                for (int vert_i = 0;vert_i < get_verts_n(elem_type);++vert_i) {
                    //TODO check order of i and j
                    r_u[i] += vertexes[vert_i][i] * shape_func_der(elem_type, vert_i, ref, 0);
                    r_v[i] += vertexes[vert_i][i] * shape_func_der(elem_type, vert_i, ref, 1);
                }
            }
            T       E = scalar_prod( r_u, r_u ),
                G = scalar_prod( r_v, r_v ),
                F = scalar_prod( r_u, r_v );
            return sqrt( E*G - F*F );
        }
    }
    //TODO see shape_func comment and ref_to_phys_jacob_det comment
    //NOTE for plananr elements 3rd component in reference coordinates is simply ignored
    __DEVICE_TAG__ void             ref_to_phys(int elem_type, const vec *vertexes, const vec &ref, vec &res)const
    {
        if (elem_type == 2) {
            for (int j = 0;j < 3;++j) {
                res[j] =        vertexes[0][j] * shape_func(elem_type,0,ref) +
                        vertexes[1][j] * shape_func(elem_type,1,ref) +
                        vertexes[2][j] * shape_func(elem_type,2,ref);
            }
        } else if (elem_type == 3) {
            for (int j = 0;j < 3;++j) {
                res[j] =        vertexes[0][j] * shape_func(elem_type,0,ref) +
                        vertexes[1][j] * shape_func(elem_type,1,ref) +
                        vertexes[2][j] * shape_func(elem_type,2,ref) +
                        vertexes[3][j] * shape_func(elem_type,3,ref);
            }
        } else if (elem_type == 4) {
            for (int j = 0;j < 3;++j) {
                res[j] =        vertexes[0][j] * shape_func(elem_type,0,ref) +
                        vertexes[1][j] * shape_func(elem_type,1,ref) +
                        vertexes[2][j] * shape_func(elem_type,2,ref) +
                        vertexes[3][j] * shape_func(elem_type,3,ref);
            }
        } else if (elem_type == 5) {
            for (int j = 0;j < 3;++j) {
                res[j] =        vertexes[0][j] * shape_func(elem_type,0,ref) +
                        vertexes[1][j] * shape_func(elem_type,1,ref) +
                        vertexes[2][j] * shape_func(elem_type,2,ref) +
                        vertexes[3][j] * shape_func(elem_type,3,ref) +
                        vertexes[4][j] * shape_func(elem_type,4,ref) +
                        vertexes[5][j] * shape_func(elem_type,5,ref) +
                        vertexes[6][j] * shape_func(elem_type,6,ref) +
                        vertexes[7][j] * shape_func(elem_type,7,ref);
            }
        } else if (elem_type == 6) {
            for (int j = 0;j < 3;++j) {
                res[j] =        vertexes[0][j] * shape_func(elem_type,0,ref) +
                        vertexes[1][j] * shape_func(elem_type,1,ref) +
                        vertexes[2][j] * shape_func(elem_type,2,ref) +
                        vertexes[3][j] * shape_func(elem_type,3,ref) +
                        vertexes[4][j] * shape_func(elem_type,4,ref) +
                        vertexes[5][j] * shape_func(elem_type,5,ref);
            }
        } else {
            //TODO
        }
    }

    gmsh_mesh_elem_reference()
    {
        faces_n[2] = 3;
        faces_n[3] = 4;
        faces_n[4] = 4;
        faces_n[5] = 6;
        faces_n[6] = 5;
        faces_n[7] = 5;

        prim_verts_n[2] = verts_n[2] = 3;
        prim_verts_n[3] = verts_n[3] = 4;
        prim_verts_n[4] = verts_n[4] = 4;
        prim_verts_n[5] = verts_n[5] = 8;
        prim_verts_n[6] = verts_n[6] = 6;
        prim_verts_n[7] = verts_n[7] = 5;

        //NOTE reference vertexes coordinates are taken
        //from gmsh source files, i.e. Geo/MTetrahedron.h, Geo/MHexahedron.h, etc (getNode method)

        //triangle vertexes
        prim_verts[2][0] = verts[2][0] = vec(0., 0., 0.);
        prim_verts[2][1] = verts[2][1] = vec(1., 0., 0.);
        prim_verts[2][2] = verts[2][2] = vec(0., 1., 0.);
        //quadrangle vertexes
        prim_verts[3][0] = verts[3][0] = vec(-1., -1., 0.);
        prim_verts[3][1] = verts[3][1] = vec( 1., -1., 0.);
        prim_verts[3][2] = verts[3][2] = vec( 1.,  1., 0.);
        prim_verts[3][3] = verts[3][3] = vec(-1.,  1., 0.);
        //tetrahedron vertexes
        prim_verts[4][0] = verts[4][0] = vec(0., 0., 0.);
        prim_verts[4][1] = verts[4][1] = vec(1., 0., 0.);
        prim_verts[4][2] = verts[4][2] = vec(0., 1., 0.);
        prim_verts[4][3] = verts[4][3] = vec(0., 0., 1.);
        //hexahedron vertexes
        prim_verts[5][0] = verts[5][0] = vec(-1., -1., -1.);
        prim_verts[5][1] = verts[5][1] = vec( 1., -1., -1.);
        prim_verts[5][2] = verts[5][2] = vec( 1.,  1., -1.);
        prim_verts[5][3] = verts[5][3] = vec(-1.,  1., -1.);
        prim_verts[5][4] = verts[5][4] = vec(-1., -1.,  1.);
        prim_verts[5][5] = verts[5][5] = vec( 1., -1.,  1.);
        prim_verts[5][6] = verts[5][6] = vec( 1.,  1.,  1.);
        prim_verts[5][7] = verts[5][7] = vec(-1.,  1.,  1.);
        //prism vertexes
        prim_verts[6][0] = verts[6][0] = vec( 0.,  0., -1.);
        prim_verts[6][1] = verts[6][1] = vec( 1.,  0., -1.);
        prim_verts[6][2] = verts[6][2] = vec( 0.,  1., -1.);
        prim_verts[6][3] = verts[6][3] = vec( 0.,  0.,  1.);
        prim_verts[6][4] = verts[6][4] = vec( 1.,  0.,  1.);
        prim_verts[6][5] = verts[6][5] = vec( 0.,  1.,  1.);
        //TODO other types

        //faces types
        face_elem_type[4][0] = 2;
        face_elem_type[4][1] = 2;
        face_elem_type[4][2] = 2;
        face_elem_type[4][3] = 2;
        face_elem_type[5][0] = 3;
        face_elem_type[5][1] = 3;
        face_elem_type[5][2] = 3;
        face_elem_type[5][3] = 3;
        face_elem_type[5][4] = 3;
        face_elem_type[5][5] = 3;
        face_elem_type[6][0] = 2;
        face_elem_type[6][1] = 2;
        face_elem_type[6][2] = 3;
        face_elem_type[6][3] = 3;
        face_elem_type[6][4] = 3;
        face_elem_type[7][0] = 2;
        face_elem_type[7][1] = 2;
        face_elem_type[7][2] = 2;
        face_elem_type[7][3] = 2;
        face_elem_type[7][4] = 3;
        //TODO other types

        //NOTE faces enumeration and vertexes enumeration inside faces is taken
        //from gmsh source files, i.e. Geo/MTetrahedron.h (static int faces_tetra),
        //Geo/MHexahedron.h (static int faces_hexa), etc.
        //TODO 6 (prism)
        //faces verts numbers
        face_prim_verts_n[4][0] = face_verts_n[4][0] = 3;
        face_prim_verts_n[4][1] = face_verts_n[4][1] = 3;
        face_prim_verts_n[4][2] = face_verts_n[4][2] = 3;
        face_prim_verts_n[4][3] = face_verts_n[4][3] = 3;
        face_prim_verts_n[5][0] = face_verts_n[5][0] = 4;
        face_prim_verts_n[5][1] = face_verts_n[5][1] = 4;
        face_prim_verts_n[5][2] = face_verts_n[5][2] = 4;
        face_prim_verts_n[5][3] = face_verts_n[5][3] = 4;
        face_prim_verts_n[5][4] = face_verts_n[5][4] = 4;
        face_prim_verts_n[5][5] = face_verts_n[5][5] = 4;
        face_prim_verts_n[6][0] = face_verts_n[6][0] = 3;
        face_prim_verts_n[6][1] = face_verts_n[6][1] = 3;
        face_prim_verts_n[6][2] = face_verts_n[6][2] = 4;
        face_prim_verts_n[6][3] = face_verts_n[6][3] = 4;
        face_prim_verts_n[6][4] = face_verts_n[6][4] = 4;
        face_prim_verts_n[7][0] = face_verts_n[7][0] = 3;
        face_prim_verts_n[7][1] = face_verts_n[7][1] = 3;
        face_prim_verts_n[7][2] = face_verts_n[7][2] = 3;
        face_prim_verts_n[7][3] = face_verts_n[7][3] = 3;
        face_prim_verts_n[7][4] = face_verts_n[7][4] = 4;
        /*face_verts_n[6][0] = 4;
        face_verts_n[6][1] = 4;
        face_verts_n[6][2] = 4;
        face_verts_n[6][3] = 4;
        face_verts_n[6][4] = 4;*/

        /*face_verts[4][0][0] = 0;
        face_verts[4][0][1] = 2;
        face_verts[4][0][2] = 3;
        face_verts[4][1][0] = 0;
        face_verts[4][1][1] = 1;
        face_verts[4][1][2] = 3;
        face_verts[4][2][0] = 0;
        face_verts[4][2][1] = 3;
        face_verts[4][2][2] = 2;
        face_verts[4][3][0] = 3;
        face_verts[4][3][1] = 1;
        face_verts[4][3][2] = 2;*/
        //tetrahedron faces verts
        //TODO first tetrahedron face in gmsh source is 0,2,1 (change for it?)
        face_prim_verts[4][0][0] = face_verts[4][0][0] = 0;
        face_prim_verts[4][0][1] = face_verts[4][0][1] = 1;
        face_prim_verts[4][0][2] = face_verts[4][0][2] = 2;
        face_prim_verts[4][1][0] = face_verts[4][1][0] = 0;
        face_prim_verts[4][1][1] = face_verts[4][1][1] = 1;
        face_prim_verts[4][1][2] = face_verts[4][1][2] = 3;
        face_prim_verts[4][2][0] = face_verts[4][2][0] = 0;
        face_prim_verts[4][2][1] = face_verts[4][2][1] = 3;
        face_prim_verts[4][2][2] = face_verts[4][2][2] = 2;
        face_prim_verts[4][3][0] = face_verts[4][3][0] = 3;
        face_prim_verts[4][3][1] = face_verts[4][3][1] = 1;
        face_prim_verts[4][3][2] = face_verts[4][3][2] = 2;
        //hexahedron faces verts
        face_prim_verts[5][0][0] = face_verts[5][0][0] = 0;
        face_prim_verts[5][0][1] = face_verts[5][0][1] = 3;
        face_prim_verts[5][0][2] = face_verts[5][0][2] = 2;
        face_prim_verts[5][0][3] = face_verts[5][0][3] = 1;
        face_prim_verts[5][1][0] = face_verts[5][1][0] = 0;
        face_prim_verts[5][1][1] = face_verts[5][1][1] = 1;
        face_prim_verts[5][1][2] = face_verts[5][1][2] = 5;
        face_prim_verts[5][1][3] = face_verts[5][1][3] = 4;
        face_prim_verts[5][2][0] = face_verts[5][2][0] = 0;
        face_prim_verts[5][2][1] = face_verts[5][2][1] = 4;
        face_prim_verts[5][2][2] = face_verts[5][2][2] = 7;
        face_prim_verts[5][2][3] = face_verts[5][2][3] = 3;
        face_prim_verts[5][3][0] = face_verts[5][3][0] = 1;
        face_prim_verts[5][3][1] = face_verts[5][3][1] = 2;
        face_prim_verts[5][3][2] = face_verts[5][3][2] = 6;
        face_prim_verts[5][3][3] = face_verts[5][3][3] = 5;
        face_prim_verts[5][4][0] = face_verts[5][4][0] = 2;
        face_prim_verts[5][4][1] = face_verts[5][4][1] = 3;
        face_prim_verts[5][4][2] = face_verts[5][4][2] = 7;
        face_prim_verts[5][4][3] = face_verts[5][4][3] = 6;
        face_prim_verts[5][5][0] = face_verts[5][5][0] = 4;
        face_prim_verts[5][5][1] = face_verts[5][5][1] = 5;
        face_prim_verts[5][5][2] = face_verts[5][5][2] = 6;
        face_prim_verts[5][5][3] = face_verts[5][5][3] = 7;
        //prism faces verts
        face_prim_verts[6][0][0] = face_verts[6][0][0] = 0;
        face_prim_verts[6][0][1] = face_verts[6][0][1] = 2;
        face_prim_verts[6][0][2] = face_verts[6][0][2] = 1;
        face_prim_verts[6][1][0] = face_verts[6][1][0] = 3;
        face_prim_verts[6][1][1] = face_verts[6][1][1] = 4;
        face_prim_verts[6][1][2] = face_verts[6][1][2] = 5;
        face_prim_verts[6][2][0] = face_verts[6][2][0] = 0;
        face_prim_verts[6][2][1] = face_verts[6][2][1] = 1;
        face_prim_verts[6][2][2] = face_verts[6][2][2] = 4;
        face_prim_verts[6][2][3] = face_verts[6][2][3] = 3;
        face_prim_verts[6][3][0] = face_verts[6][3][0] = 0;
        face_prim_verts[6][3][1] = face_verts[6][3][1] = 3;
        face_prim_verts[6][3][2] = face_verts[6][3][2] = 5;
        face_prim_verts[6][3][3] = face_verts[6][3][3] = 2;
        face_prim_verts[6][4][0] = face_verts[6][4][0] = 1;
        face_prim_verts[6][4][1] = face_verts[6][4][1] = 2;
        face_prim_verts[6][4][2] = face_verts[6][4][2] = 5;
        face_prim_verts[6][4][3] = face_verts[6][4][3] = 4;
        //pyramid faces verts
        face_prim_verts[7][0][0] = face_verts[7][0][0] = 0;
        face_prim_verts[7][0][1] = face_verts[7][0][1] = 1; 
        face_prim_verts[7][0][2] = face_verts[7][0][2] = 4;
        face_prim_verts[7][1][0] = face_verts[7][1][0] = 3; 
        face_prim_verts[7][1][1] = face_verts[7][1][1] = 0; 
        face_prim_verts[7][1][2] = face_verts[7][1][2] = 4;
        face_prim_verts[7][2][0] = face_verts[7][2][0] = 1; 
        face_prim_verts[7][2][1] = face_verts[7][2][1] = 2; 
        face_prim_verts[7][2][2] = face_verts[7][2][2] = 4;
        face_prim_verts[7][3][0] = face_verts[7][3][0] = 2; 
        face_prim_verts[7][3][1] = face_verts[7][3][1] = 3; 
        face_prim_verts[7][3][2] = face_verts[7][3][2] = 4;
        face_prim_verts[7][4][0] = face_verts[7][4][0] = 0; 
        face_prim_verts[7][4][1] = face_verts[7][4][1] = 3; 
        face_prim_verts[7][4][2] = face_verts[7][4][2] = 2; 
        face_prim_verts[7][4][3] = face_verts[7][4][3] = 1;
        //TODO other types
    }
};

}  /// namespace mesh
}  /// namespace scfd

#endif
