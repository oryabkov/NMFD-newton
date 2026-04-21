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

#ifndef __SCFD_CUBATURE_REFERENCE_H__
#define __SCFD_CUBATURE_REFERENCE_H__

//TODO mesh/device_tag.h is temporal

#include <cmath>
#include <stdexcept>
#include <scfd/utils/device_tag.h>
#include <scfd/static_vec/vec.h>

//NOTE element types and local coordinate systems are synced with t_mesh_elem_reference_tml.h (and, accordingly, with gmsh)

//TODO we must use some macros for definition of the 'most complex' element type, that is supported (i.e., tetrahedra, 3prism or hexahedron)
//TODO decide what fp type we use to initialize quadrature points and weights; i prefer to use double initially and then cast them to actuall type T (like float)
//TODO it would be useful also to have cubature structure for given element and given order (see for example some TODO's in t_DG_tml constructor)

//TODO it must be 8 if we need pyramids! for now we just don't need em.
#define CUBATURE_REFERENCE_ELEM_TYPES_N 7

namespace scfd
{
namespace mesh
{

template<int max_order>
struct cubature_reference_max_pnts_n
{
};

template<int max_order>
struct face_cubature_reference_max_pnts_n
{
};

//ISSUE am i sure that product of nth order rules gives nth order rule in corresponding dimension?

template<>
struct cubature_reference_max_pnts_n<0>
{
    //static const int value = 1;   //for tetrahedron
    //static const int value = 1;     //for prism
    static const int value = 1;     //for hexahedron
};

template<>
struct cubature_reference_max_pnts_n<1>
{
    //static const int value = 1;   //for tetrahedron
    //static const int value = 1;     //for prism
    static const int value = 1;     //for hexahedron
};

template<>
struct cubature_reference_max_pnts_n<2>
{
    //static const int value = 4;   //for tetrahedron
    //static const int value = 6;   //for prism
    static const int value = 8;     //for hexahedron (product of 1d 3dr order rules)
};

template<>
struct cubature_reference_max_pnts_n<3>
{
    //static const int value = 5;   //for tetrahedron
    //static const int value = 8;   //for prism
    static const int value = 8;     //for hexahedron (product of 1d 3dr order rules)
};

/*template<>
struct cubature_reference_max_pnts_n<4>
{
        static const int value = 10;
};*/
template<>
struct cubature_reference_max_pnts_n<4>
{
        //static const int value = 14;  //for tetrahedron
        static const int value = 18;    //for prism
    //static const int value = 14;  //for hexahedron (HEXAHEDRON(5) from getfem++)
    //static const int value = 27;
};

template<>
struct cubature_reference_max_pnts_n<5>
{
    //static const int value = 14;  //for tetrahedron
    static const int value = 21;    //for prism
    //static const int value = 14;  //for hexahedron (HEXAHEDRON(5) from getfem++)
    //static const int value = 27;
};

template<>
struct cubature_reference_max_pnts_n<6>
{
    //static const int value = 15;  //for tetrahedron
    static const int value = 58;    //for hexahedron (HEXAHEDRON(9) from getfem++)
};

/// runtime version of cubature_reference_max_pnts_n structure
__DEVICE_TAG__ int get_cubature_reference_max_pnts_n(int max_order)
{
    switch (max_order)
    {
        case 0: return cubature_reference_max_pnts_n<0>::value;
        case 1: return cubature_reference_max_pnts_n<1>::value;
        case 2: return cubature_reference_max_pnts_n<2>::value;
        case 3: return cubature_reference_max_pnts_n<3>::value;
        case 4: return cubature_reference_max_pnts_n<4>::value;
        case 5: return cubature_reference_max_pnts_n<5>::value;
        case 6: return cubature_reference_max_pnts_n<6>::value;
        default: return -1;
        /// TODO error
    }
}

template<>
struct face_cubature_reference_max_pnts_n<0>
{
    //static const int value = 1;   //for triangle
    static const int value = 1;     //for quadrangle
};

template<>
struct face_cubature_reference_max_pnts_n<1>
{
    //static const int value = 1;   //for triangle
    static const int value = 1;     //for quadrangle
};

template<>
struct face_cubature_reference_max_pnts_n<2>
{
    //static const int value = 3;   //for triangle
    static const int value = 4;     //for quadrangle
};

template<>
struct face_cubature_reference_max_pnts_n<3>
{
    //static const int value = 4;   //for triangle
    static const int value = 4;     //for quadrangle
};

template<>
struct face_cubature_reference_max_pnts_n<4>
{
    //static const int value = 6;   //for triangle
    static const int value = 9;     //for quadrangle
};

template<>
struct face_cubature_reference_max_pnts_n<5>
{
    //static const int value = 7;   //for triangle
    static const int value = 9;     //for quadrangle
};

/// runtime version of face_cubature_reference_max_pnts_n structure
__DEVICE_TAG__ int get_face_cubature_reference_max_pnts_n(int max_order)
{
    switch (max_order)
    {
        case 0: return face_cubature_reference_max_pnts_n<0>::value;
        case 1: return face_cubature_reference_max_pnts_n<1>::value;
        case 2: return face_cubature_reference_max_pnts_n<2>::value;
        case 3: return face_cubature_reference_max_pnts_n<3>::value;
        case 4: return face_cubature_reference_max_pnts_n<4>::value;
        case 5: return face_cubature_reference_max_pnts_n<5>::value;
        default: return -1;
        /// TODO error
    }
}

template<int max_order, class T>
struct elem_cubature_reference
{
    typedef static_vec::vec<T,3>  vec_t;

    static const int max_pnts_n = cubature_reference_max_pnts_n<max_order>::value;

    //ISSUE add elem_type field?

    unsigned char   pnts_n[max_order+1];
    T               weights[max_order+1][max_pnts_n];
    vec_t           pnts[max_order+1][max_pnts_n];

    __DEVICE_TAG__ unsigned char    get_pnts_n(unsigned char order)const
    {
        return pnts_n[order];
    }
    __DEVICE_TAG__ T                get_weight(unsigned char order, unsigned char pnt_i)const
    {
        return weights[order][pnt_i];
    }
    //several interfaces to get cubature points coordinates
    __DEVICE_TAG__ const vec_t      &get_pnt(unsigned char order, unsigned char pnt_i)const
    {
        return pnts[order][pnt_i];
    }
    __DEVICE_TAG__ void             get_pnt(unsigned char order, unsigned char pnt_i, vec_t &res)const
    {
        res = pnts[order][pnt_i];
    }
    __DEVICE_TAG__ T                get_pnt(unsigned char order, unsigned char pnt_i, unsigned char j)const
    {
        return pnts[order][pnt_i][j];
    }

    void    __init_triangle()
    {
        //all W_XX are normalized to one but area of reference triangle is 0.5, so i simply multiply them on 0.5
        T   ref_area(0.5f);

        T W_0[1] = {1.0};
        T X_0[1] = {1.0/3.0};
        T Y_0[1] = {1.0/3.0};

                pnts_n[0] = 1;
        for (int pnt_i = 0; pnt_i < 1; pnt_i++) 
        {
            weights[0][pnt_i] = W_0[pnt_i]*ref_area;
            pnts[0][pnt_i][0] = X_0[pnt_i];
            pnts[0][pnt_i][1] = Y_0[pnt_i];
            pnts[0][pnt_i][2] = T(0.f);
        };

        pnts_n[1] = 1;
        for (int pnt_i = 0; pnt_i < 1; pnt_i++) 
        {
            weights[1][pnt_i] = W_0[pnt_i]*ref_area;
            pnts[1][pnt_i][0] = X_0[pnt_i];
            pnts[1][pnt_i][1] = Y_0[pnt_i];
            pnts[1][pnt_i][2] = T(0.f);
        };
        if (max_order == 1) return;

        T W_1[3] = {1.0/3.0, 1.0/3.0, 1.0/3.0};
        T X_1[3] = {1.0/6.0, 2.0/3.0, 1.0/6.0};
        T Y_1[3] = {1.0/6.0, 1.0/6.0, 2.0/3.0};
        pnts_n[2] = 3;
        for (int pnt_i = 0; pnt_i < 3; pnt_i++) 
        {
            weights[2][pnt_i] = W_1[pnt_i]*ref_area;
            pnts[2][pnt_i][0] = X_1[pnt_i];
            pnts[2][pnt_i][1] = Y_1[pnt_i];
            pnts[2][pnt_i][2] = T(0.f);
        };
        if (max_order == 2) return;

        T W_2[4] = {-27./48., 25./48., 25./48., 25./48.};
        T X_2[4] = {1.0/3.0, 1.0/5.0, 1.0/5.0, 3.0/5.0};
        T Y_2[4] = {1.0/3.0, 3.0/5.0, 1.0/5.0, 1.0/5.0};
        pnts_n[3] = 4;
        for (int pnt_i = 0; pnt_i < 4; pnt_i++) 
        {
            weights[3][pnt_i] = W_2[pnt_i]*ref_area;
            pnts[3][pnt_i][0] = X_2[pnt_i];
            pnts[3][pnt_i][1] = Y_2[pnt_i];
            pnts[3][pnt_i][2] = T(0.f);
        };
        if (max_order == 3) return;

        T W_3[6] = {0.22338158967801, 0.22338158967801, 0.22338158967801, 0.10995174365532, 0.10995174365532, 0.10995174365532};
        T X_3[6] = {0.44594849091597, 0.44594849091597, 0.10810301816807, 0.09157621350977, 0.09157621350977, 0.81684757298046};
        T Y_3[6] = {0.44594849091597, 0.10810301816807, 0.44594849091597, 0.09157621350977, 0.81684757298046, 0.09157621350977};
        pnts_n[4] = 6;
        for (int pnt_i = 0; pnt_i < 6; pnt_i++) 
        {
            weights[4][pnt_i] = W_3[pnt_i]*ref_area;
            pnts[4][pnt_i][0] = X_3[pnt_i];
            pnts[4][pnt_i][1] = Y_3[pnt_i];
            pnts[4][pnt_i][2] = T(0.f);
        };
        if (max_order == 4) return;

        T W_4[7] = {0.22500000000000, 0.13239415278851, 0.13239415278851, 0.13239415278851, 0.12593918054483, 0.12593918054483, 0.12593918054483};
        T X_4[7] = {0.33333333333333, 0.47014206410511, 0.47014206410511, 0.05971587178977, 0.10128650732346, 0.10128650732346, 0.79742698535309};
        T Y_4[7] = {0.33333333333333, 0.47014206410511, 0.05971587178977, 0.47014206410511, 0.10128650732346, 0.79742698535309, 0.10128650732346};
        pnts_n[5] = 7;
        for (int pnt_i = 0; pnt_i < 7; pnt_i++) 
        {
            weights[5][pnt_i] = W_4[pnt_i]*ref_area;
            pnts[5][pnt_i][0] = X_4[pnt_i];
            pnts[5][pnt_i][1] = Y_4[pnt_i];
            pnts[5][pnt_i][2] = T(0.f);
        };
        if (max_order == 5) return;
        
        throw std::runtime_error("elem_cubature_reference::__init_triangle : not supported order");

        //TODO
    }
    void    __init_quad()
    {
        //all W_XX are normalized to one but area of reference quad is 4., so i simply multiply them on 4.
        T   ref_area(4.f);
        
        T W_0[1] = {1.};
        T X_0[1] = {0.};
        T Y_0[1] = {0.};
        T Z_0[1] = {0.};

        pnts_n[0] = 1;
        for (int pnt_i=0; pnt_i < 1; pnt_i++) 
        {
            weights[0][pnt_i] = W_0[pnt_i]*ref_area;
            pnts[0][pnt_i][0] = X_0[pnt_i];
            pnts[0][pnt_i][1] = Y_0[pnt_i];
            pnts[0][pnt_i][2] = Z_0[pnt_i];
        };

        pnts_n[1] = 1;
        for (int pnt_i=0; pnt_i < 1; pnt_i++) 
        {
            weights[1][pnt_i] = W_0[pnt_i]*ref_area;
            pnts[1][pnt_i][0] = X_0[pnt_i];
            pnts[1][pnt_i][1] = Y_0[pnt_i];
            pnts[1][pnt_i][2] = Z_0[pnt_i];
        };
        if (max_order == 1) return;

        //this is direct product of 2 3rd order Gauss quadratures
        T W_1[4] = {    0.25,0.25,0.25,0.25 };
        T X_1[4] = {    -0.5773502691896257,  0.5773502691896257, -0.5773502691896257,  0.5773502691896257 };
        T Y_1[4] = {    -0.5773502691896257, -0.5773502691896257,  0.5773502691896257,  0.5773502691896257 };
        T Z_1[4] = {     0.0000000000000000,  0.0000000000000000,  0.0000000000000000,  0.0000000000000000 };

        pnts_n[2] = 4;
        for (int pnt_i=0; pnt_i < 4; pnt_i++) 
        {
            weights[2][pnt_i] = W_1[pnt_i]*ref_area;
            pnts[2][pnt_i][0] = X_1[pnt_i];
            pnts[2][pnt_i][1] = Y_1[pnt_i];
            pnts[2][pnt_i][2] = Z_1[pnt_i];
        };
        if (max_order == 2) return;

        pnts_n[3] = 4;
        for (int pnt_i=0; pnt_i < 4; pnt_i++) 
        {
            weights[3][pnt_i] = W_1[pnt_i]*ref_area;
            pnts[3][pnt_i][0] = X_1[pnt_i];
            pnts[3][pnt_i][1] = Y_1[pnt_i];
            pnts[3][pnt_i][2] = Z_1[pnt_i];
        };
        if (max_order == 3) return;

        //5/18 8/18 5/18 are coefficients for 1d quadrature
        //this is direct product of 2 5th order Gauss quadratures
        double W_2[9] = {   0.07716049382716049382716049382716,     0.12345679012345679012345679012346,     0.07716049382716049382716049382716,
                            0.12345679012345679012345679012346,     0.19753086419753086419753086419753,     0.12345679012345679012345679012346,
                            0.07716049382716049382716049382716,     0.12345679012345679012345679012346,     0.07716049382716049382716049382716 };
        double X_2[9] = {  -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648,
                           -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648,
                           -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648 };
        double Y_2[9] = {  -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,
                            0.00000000000000000000000000000000,     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                            0.77459666924148337703585307995648,     0.77459666924148337703585307995648,     0.77459666924148337703585307995648 };
        double Z_2[9] = {   0.00000000000000000000000000000000,     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                            0.00000000000000000000000000000000,     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                            0.00000000000000000000000000000000,     0.00000000000000000000000000000000,     0.00000000000000000000000000000000 };

        pnts_n[4] = 9;
        for (int pnt_i=0; pnt_i < 9; pnt_i++) 
        {
            weights[4][pnt_i] = W_2[pnt_i]*ref_area;
            pnts[4][pnt_i][0] = X_2[pnt_i];
            pnts[4][pnt_i][1] = Y_2[pnt_i];
            pnts[4][pnt_i][2] = Z_2[pnt_i];
        };
        if (max_order == 4) return;

        pnts_n[5] = 9;
        for (int pnt_i=0; pnt_i < 9; pnt_i++) 
        {
            weights[5][pnt_i] = W_2[pnt_i]*ref_area;
            pnts[5][pnt_i][0] = X_2[pnt_i];
            pnts[5][pnt_i][1] = Y_2[pnt_i];
            pnts[5][pnt_i][2] = Z_2[pnt_i];
        };
        if (max_order == 5) return;

        throw std::runtime_error("elem_cubature_reference::__init_quad : not supported order");

        //TODO
    }
    void    __init_tetrahedron()
    {
        //all W_XX are normalized to one but area of reference tetrahedron is 1/6, so i simply multiply them on 1/6
        T   ref_vol(1./6.);
        T   a,b,c,d,e,f,h,i;

        /// https://getfem.org/userdoc/appendixB.html# IM_TETRAHEDRON(1)
        /// Note that ref_vol is already accounted in getfem, no need to multiply on it second time
        T W_0[1] = {1./6.};
        T X_0[1] = {1./4.};
        T Y_0[1] = {1./4.};
        T Z_0[1] = {1./4.};

        pnts_n[0] = 1;
        for (int pnt_i=0; pnt_i < 1; pnt_i++) 
        {
            weights[0][pnt_i] = W_0[pnt_i];
            pnts[0][pnt_i][0] = X_0[pnt_i];
            pnts[0][pnt_i][1] = Y_0[pnt_i];
            pnts[0][pnt_i][2] = Z_0[pnt_i];
        };

        pnts_n[1] = 1;
        for (int pnt_i=0; pnt_i < 1; pnt_i++) 
        {
            weights[1][pnt_i] = W_0[pnt_i];
            pnts[1][pnt_i][0] = X_0[pnt_i];
            pnts[1][pnt_i][1] = Y_0[pnt_i];
            pnts[1][pnt_i][2] = Z_0[pnt_i];
        };
        if (max_order == 1) return;

        /// https://getfem.org/userdoc/appendixB.html# IM_TETRAHEDRON(2)
        /// Note that ref_vol is already accounted in getfem, no need to multiply on it second time
        a = (5.-std::sqrt(5.))/20.;
        b = (5.+3.*std::sqrt(5.))/20.;
        T W_1[4] = {1./24.,1./24.,1./24.,1./24.};
        T X_1[4] = {a,a,a,b};
        T Y_1[4] = {a,b,a,a};
        T Z_1[4] = {a,a,b,a};
        pnts_n[2] = 4;
        for (int pnt_i=0; pnt_i<4; pnt_i++) 
        {
            weights[2][pnt_i] = W_1[pnt_i];
            pnts[2][pnt_i][0] = X_1[pnt_i];
            pnts[2][pnt_i][1] = Y_1[pnt_i];
            pnts[2][pnt_i][2] = Z_1[pnt_i];
        };
        if (max_order == 2) return;

        /// https://getfem.org/userdoc/appendixB.html# IM_TETRAHEDRON(3)
        /// Note that ref_vol is already accounted in getfem, no need to multiply on it second time
        T W_2[5] = {-2./15.,3./40.,3./40.,3./40.,3./40.};
        T X_2[5] = {1./4.,1./6.,1./6.,1./6.,1./2.};
        T Y_2[5] = {1./4.,1./6.,1./2.,1./6.,1./6.};
        T Z_2[5] = {1./4.,1./6.,1./6.,1./2.,1./6.};
        pnts_n[3] = 5;
        for (int pnt_i=0; pnt_i<5; pnt_i++) 
        {
            weights[3][pnt_i] = W_2[pnt_i];
            pnts[3][pnt_i][0] = X_2[pnt_i];
            pnts[3][pnt_i][1] = Y_2[pnt_i];
            pnts[3][pnt_i][2] = Z_2[pnt_i];
        };
        if (max_order == 3) return;

        //TODO TEST this 10 point qubature.. is it wrong??
        /*T W_3[10] = {0.217765,0.217765,0.217765,0.217765,0.02149,0.02149,0.02149,0.02149,0.02149,0.02149};
        T X_3[10] = {0.568431,0.143856,0.143856,0.143856,0,0.5,0.5,0.5,0,0};
        T Y_3[10] = {0.143856,0.143856,0.143856,0.568431,0.5,0,0.5,0,0.5,0};
        T Z_3[10] = {0.143856,0.143856,0.568431,0.143856,0.5,0.5,0,0,0,0.5};
        pnts_n[4] = 10;
        for (int pnt_i=0; pnt_i<10; pnt_i++) {
            weights[4][4][pnt_i] = W_3[pnt_i];
            pnts[4][4][pnt_i][0] = X_3[pnt_i];
            pnts[4][4][pnt_i][1] = Y_3[pnt_i];
            pnts[4][4][pnt_i][2] = Z_3[pnt_i];
        };
        if (max_order == 4) return;*/

        T X_3[11] = {   0.2500000000000000, 0.7857142857142857, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714,
                        0.1005964238332008, 0.3994035761667992, 0.3994035761667992, 0.3994035761667992, 0.1005964238332008, 0.1005964238332008};
        T Y_3[11] = {   0.2500000000000000, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.7857142857142857,
                        0.3994035761667992, 0.1005964238332008, 0.3994035761667992, 0.1005964238332008, 0.3994035761667992, 0.1005964238332008};
        T Z_3[11] = {   0.2500000000000000, 0.0714285714285714, 0.0714285714285714, 0.7857142857142857, 0.0714285714285714,
                        0.3994035761667992, 0.3994035761667992, 0.1005964238332008, 0.1005964238332008, 0.1005964238332008, 0.3994035761667992};
        T W_3[11] = {  -0.0789333333333333, 0.0457333333333333, 0.0457333333333333, 0.0457333333333333, 0.0457333333333333,
                        0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333};
        pnts_n[4] = 11;
        for (int pnt_i=0; pnt_i<11; pnt_i++) 
        {
            weights[4][pnt_i] = W_3[pnt_i]*ref_vol;
            pnts[4][pnt_i][0] = X_3[pnt_i];
            pnts[4][pnt_i][1] = Y_3[pnt_i];
            pnts[4][pnt_i][2] = Z_3[pnt_i];
        };
        /*T W_3[14] = {0.0190476,0.0190476,0.0190476,0.0190476,0.0190476,0.0190476,0.0885898,0.0885898,0.0885898,0.0885898,0.132839,0.132839,0.132839,0.132839};
        T X_3[14] = {0  ,0.5,0.5,0.5,0  ,0  ,0.69842 ,0.100527,0.100527,0.100527,0.0568814,0.314373 ,0.314373 ,0.314373};
        T Y_3[14] = {0.5,0  ,0.5,0  ,0.5,0  ,0.100527,0.100527,0.100527,0.69842 ,0.314373 ,0.314373 ,0.314373 ,0.0568814};
        T Z_3[14] = {0.5,0.5,0  ,0  ,0  ,0.5,0.100527,0.100527,0.69842 ,0.100527,0.314373 ,0.314373 ,0.0568814,0.314373};*/
        /*pnts_n[4] = 14;
        for (int pnt_i=0; pnt_i<14; pnt_i++) {
            weights[4][4][pnt_i] = W_3[pnt_i];
            pnts[4][4][pnt_i][0] = X_3[pnt_i];
            pnts[4][4][pnt_i][1] = Y_3[pnt_i];
            pnts[4][4][pnt_i][2] = Z_3[pnt_i];
        };*/
        if (max_order == 4) return;

        /// https://getfem.org/userdoc/appendixB.html# IM_TETRAHEDRON(5)
        /// Note that ref_vol is already accounted in getfem, no need to multiply on it second time
        a = (7.+std::sqrt(15.))/34.;
        b = (7.-std::sqrt(15.))/34.;
        c = (13.+3.*std::sqrt(15.))/34.;
        d = (13.-3.*std::sqrt(15.))/34.;
        e = (5.-std::sqrt(15.))/20.;
        f = (5.+std::sqrt(15.))/20.;
        h = (2665.-14.*std::sqrt(15.))/226800.;
        i = (2665.+14.*std::sqrt(15.))/226800.;
        T W_4[15] = {8./405.,h,h,h,h,i,i,i,i,5./567.,5./567.,5./567.,5./567.,5./567.,5./567.};
        T X_4[15] = {1./4.,a,a,a,c,b,b,b,d,e,e,f,e,f,f};
        T Y_4[15] = {1./4.,a,a,c,a,b,b,d,b,e,f,e,f,e,f};
        T Z_4[15] = {1./4.,a,c,a,a,b,d,b,b,f,e,e,f,f,e};
        pnts_n[5] = 15;
        for (int pnt_i=0; pnt_i<15; pnt_i++) 
        {
            weights[5][pnt_i] = W_4[pnt_i];
            pnts[5][pnt_i][0] = X_4[pnt_i];
            pnts[5][pnt_i][1] = Y_4[pnt_i];
            pnts[5][pnt_i][2] = Z_4[pnt_i];
        };
        if (max_order == 5) return;

        /*T W_5[15] = {0.181702,0.0361607,0.0361607,0.0361607,0.0361607,0.0698715,0.0698715,0.0698715,0.0698715,0.0656948,0.0656948,0.0656948,0.0656948,0.0656948,0.0656948};
        T X_5[15] = {0.25,0,0.333333,0.333333,0.333333,0.727273,0.0909091,0.0909091,0.0909091,0.43345,0.0665502,0.0665502,0.0665502,0.43345,0.43345};
        T Y_5[15] = {0.25,0.333333,0.333333,0.333333,0,0.0909091,0.0909091,0.0909091,0.727273,0.0665502,0.43345,0.0665502,0.43345,0.0665502,0.43345};
        T Z_5[15] = {0.25,0.333333,0.333333,0,0.333333,0.0909091,0.0909091,0.727273,0.0909091,0.0665502,0.0665502,0.43345,0.43345,0.43345,0.0665502};
        pnts_n[6] = 15;
        for (int pnt_i=0; pnt_i<15; pnt_i++) 
        {
            weights[6][pnt_i] = W_5[pnt_i]*ref_vol;
            pnts[6][pnt_i][0] = X_5[pnt_i];
            pnts[6][pnt_i][1] = Y_5[pnt_i];
            pnts[6][pnt_i][2] = Z_5[pnt_i];
        };
        if (max_order == 6) return;*/

        throw std::runtime_error("elem_cubature_reference::__init_tetrahedron : not supported order");

        //TODO 8th order

        /*T W_8[45] = {-0.235962,0.0244879,0.0244879,0.0244879,0.0244879,0.00394852,0.00394852,0.00394852,0.00394852,0.0263056,0.0263056,0.0263056,0.0263056,0.0263056,0.0263056,0.0829804,0.0829804,0.0829804,0.0829804,0.0829804,0.0829804,0.0254426,0.0254426,0.0254426,0.0254426,0.0254426,0.0254426,0.0254426,0.0254426,0.0254426,0.0254426,0.0254426,0.0254426,0.0134324,0.0134324,0.0134324,0.0134324,0.0134324,0.0134324,0.0134324,0.0134324,0.0134324,0.0134324,0.0134324,0.0134324};
        T X_8[45] = {0.25,0.617587,0.127471,0.127471,0.127471,0.903764,0.0320788,0.0320788,0.0320788,0.450223,0.0497771,0.0497771,0.0497771,0.450223,0.450223,0.31627,0.18373,0.18373,0.18373,0.31627,0.31627,0.0229178,0.231901,0.231901,0.51328,0.231901,0.231901,0.231901,0.0229178,0.51328,0.231901,0.0229178,0.51328,0.730313,0.03797,0.03797,0.193746,0.03797,0.03797,0.03797,0.730313,0.193746,0.03797,0.730313,0.193746};
        T Y_8[45] = {0.25,0.127471,0.127471,0.127471,0.617587,0.0320788,0.0320788,0.0320788,0.903764,0.0497771,0.450223,0.0497771,0.450223,0.0497771,0.450223,0.18373,0.31627,0.18373,0.31627,0.18373,0.31627,0.231901,0.0229178,0.231901,0.231901,0.51328,0.231901,0.0229178,0.51328,0.231901,0.51328,0.231901,0.0229178,0.03797,0.730313,0.03797,0.03797,0.193746,0.03797,0.730313,0.193746,0.03797,0.193746,0.03797,0.730313};
        T Z_8[45] = {0.25,0.127471,0.127471,0.617587,0.127471,0.0320788,0.0320788,0.903764,0.0320788,0.0497771,0.0497771,0.450223,0.450223,0.450223,0.0497771,0.18373,0.18373,0.31627,0.31627,0.31627,0.18373,0.231901,0.231901,0.0229178,0.231901,0.231901,0.51328,0.51328,0.231901,0.0229178,0.0229178,0.51328,0.231901,0.03797,0.03797,0.730313,0.03797,0.03797,0.193746,0.193746,0.03797,0.730313,0.730313,0.193746,0.03797};
        for (int pnt_i=0; pnt_i < 45; pnt_i++) {
            weights[4][6][pnt_i] = W_6[pnt_i];
            pnts[4][6][pnt_i][0] = X_6[pnt_i];
            pnts[4][6][pnt_i][1] = Y_6[pnt_i];
            pnts[4][6][pnt_i][2] = Z_6[pnt_i];
        }*/
    }
    void    __init_hexahedron()
    {
        //all W_XX are normalized to one but area of reference hexahedron is 8, so i simply multiply them on 8
        T   ref_vol(8.f);

        T W_0[1] = {1.};
        T X_0[1] = {0.};
        T Y_0[1] = {0.};
        T Z_0[1] = {0.};

        pnts_n[0] = 1;
        for (int pnt_i=0; pnt_i < 1; pnt_i++) 
        {
            weights[0][pnt_i] = W_0[pnt_i]*ref_vol;
            pnts[0][pnt_i][0] = X_0[pnt_i];
            pnts[0][pnt_i][1] = Y_0[pnt_i];
            pnts[0][pnt_i][2] = Z_0[pnt_i];
        };

        pnts_n[1] = 1;
        for (int pnt_i=0; pnt_i < 1; pnt_i++) 
        {
            weights[1][pnt_i] = W_0[pnt_i]*ref_vol;
            pnts[1][pnt_i][0] = X_0[pnt_i];
            pnts[1][pnt_i][1] = Y_0[pnt_i];
            pnts[1][pnt_i][2] = Z_0[pnt_i];
        };
        if (max_order == 1) return;

        //this is direct product of 3 3rd order Gauss quadratures
        T W_1[8] = {     0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125};
        T X_1[8] = {    -0.5773502691896257, -0.5773502691896257, -0.5773502691896257, -0.5773502691896257,
                         0.5773502691896257,  0.5773502691896257,  0.5773502691896257,  0.5773502691896257};
        T Y_1[8] = {    -0.5773502691896257, -0.5773502691896257,  0.5773502691896257,  0.5773502691896257,
                        -0.5773502691896257, -0.5773502691896257,  0.5773502691896257,  0.5773502691896257};
        T Z_1[8] = {    -0.5773502691896257,  0.5773502691896257, -0.5773502691896257,  0.5773502691896257,
                        -0.5773502691896257,  0.5773502691896257, -0.5773502691896257,  0.5773502691896257};

        pnts_n[2] = 8;
        for (int pnt_i=0; pnt_i < 8; pnt_i++) 
        {
            weights[2][pnt_i] = W_1[pnt_i]*ref_vol;
            pnts[2][pnt_i][0] = X_1[pnt_i];
            pnts[2][pnt_i][1] = Y_1[pnt_i];
            pnts[2][pnt_i][2] = Z_1[pnt_i];
        };
        if (max_order == 2) return;

        pnts_n[3] = 8;
        for (int pnt_i=0; pnt_i < 8; pnt_i++) 
        {
            weights[3][pnt_i] = W_1[pnt_i]*ref_vol;
            pnts[3][pnt_i][0] = X_1[pnt_i];
            pnts[3][pnt_i][1] = Y_1[pnt_i];
            pnts[3][pnt_i][2] = Z_1[pnt_i];
        };
        if (max_order == 3) return;

        //HEXAHEDRON(5) from getfem++ see cubature/HEXAHEDRON_5.IM (originally was on 1x1x1 cube and had 2 points with full hexahedra symmetry)
        T W_2[14] = {    0.11080332409972299168975069252077,     0.11080332409972299168975069252077,
                         0.11080332409972299168975069252077,     0.11080332409972299168975069252077,
                         0.11080332409972299168975069252077,     0.11080332409972299168975069252077,
                         0.04189750692520775623268698060942,     0.04189750692520775623268698060942,
                         0.04189750692520775623268698060942,     0.04189750692520775623268698060942,
                         0.04189750692520775623268698060942,     0.04189750692520775623268698060942,
                         0.04189750692520775623268698060942,     0.04189750692520775623268698060942 };
        T X_2[14] = {   -0.79582242575422146326454882047612,     0.79582242575422146326454882047612,
                         0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                         0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                        -0.75878691063932814626903427811226,     0.75878691063932814626903427811226,
                        -0.75878691063932814626903427811226,     0.75878691063932814626903427811226,
                        -0.75878691063932814626903427811226,     0.75878691063932814626903427811226,
                        -0.75878691063932814626903427811226,     0.75878691063932814626903427811226 };
        T Y_2[14] = {    0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                        -0.79582242575422146326454882047612,     0.79582242575422146326454882047612,
                         0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                        -0.75878691063932814626903427811226,    -0.75878691063932814626903427811226,
                         0.75878691063932814626903427811226,     0.75878691063932814626903427811226,
                        -0.75878691063932814626903427811226,    -0.75878691063932814626903427811226,
                         0.75878691063932814626903427811226,     0.75878691063932814626903427811226 };
        T Z_2[14] = {    0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                         0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                        -0.79582242575422146326454882047612,     0.79582242575422146326454882047612,
                        -0.75878691063932814626903427811226,    -0.75878691063932814626903427811226,
                        -0.75878691063932814626903427811226,    -0.75878691063932814626903427811226,
                         0.75878691063932814626903427811226,     0.75878691063932814626903427811226,
                         0.75878691063932814626903427811226,     0.75878691063932814626903427811226 };

        pnts_n[4] = 14;
        for (int pnt_i=0; pnt_i < 14; pnt_i++) 
        {
            weights[4][pnt_i] = W_2[pnt_i]*ref_vol;
            pnts[4][pnt_i][0] = X_2[pnt_i];
            pnts[4][pnt_i][1] = Y_2[pnt_i];
            pnts[4][pnt_i][2] = Z_2[pnt_i];
        };
        if (max_order == 4) return;

        pnts_n[5] = 14;
        for (int pnt_i=0; pnt_i < 14; pnt_i++) 
        {
            weights[5][pnt_i] = W_2[pnt_i]*ref_vol;
            pnts[5][pnt_i][0] = X_2[pnt_i];
            pnts[5][pnt_i][1] = Y_2[pnt_i];
            pnts[5][pnt_i][2] = Z_2[pnt_i];
        };
        if (max_order == 5) return;

        /*double W_2[27] = {     0.07716049382716049382716049382716*5./18.,  0.12345679012345679012345679012346*5./18.,  0.07716049382716049382716049382716*5./18.,
                     0.12345679012345679012345679012346*5./18.,  0.19753086419753086419753086419753*5./18.,  0.12345679012345679012345679012346*5./18.,
                     0.07716049382716049382716049382716*5./18.,  0.12345679012345679012345679012346*5./18.,  0.07716049382716049382716049382716*5./18.,
                     0.07716049382716049382716049382716*8./18.,  0.12345679012345679012345679012346*8./18.,  0.07716049382716049382716049382716*8./18.,
                     0.12345679012345679012345679012346*8./18.,  0.19753086419753086419753086419753*8./18.,  0.12345679012345679012345679012346*8./18.,
                     0.07716049382716049382716049382716*8./18.,  0.12345679012345679012345679012346*8./18.,  0.07716049382716049382716049382716*8./18.,
                     0.07716049382716049382716049382716*5./18.,  0.12345679012345679012345679012346*5./18.,  0.07716049382716049382716049382716*5./18.,
                     0.12345679012345679012345679012346*5./18.,  0.19753086419753086419753086419753*5./18.,  0.12345679012345679012345679012346*5./18.,
                     0.07716049382716049382716049382716*5./18.,  0.12345679012345679012345679012346*5./18.,  0.07716049382716049382716049382716*5./18. };
        double X_2[27] = {  -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648,
                    -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648,
                    -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648,
                    -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648,
                    -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648,
                    -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648,
                    -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648,
                    -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648,
                    -0.77459666924148337703585307995648,     0.00000000000000000000000000000000,     0.77459666924148337703585307995648 };
        double Y_2[27] = {  -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,
                     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                     0.77459666924148337703585307995648,     0.77459666924148337703585307995648,     0.77459666924148337703585307995648,
                     -0.77459666924148337703585307995648,   -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,
                     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                     0.77459666924148337703585307995648,     0.77459666924148337703585307995648,     0.77459666924148337703585307995648,
                     -0.77459666924148337703585307995648,   -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,
                     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                     0.77459666924148337703585307995648,     0.77459666924148337703585307995648,     0.77459666924148337703585307995648 };
        double Z_2[27] = {  -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,
                    -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,
                    -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,    -0.77459666924148337703585307995648,
                     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,     0.00000000000000000000000000000000,
                     0.77459666924148337703585307995648,     0.77459666924148337703585307995648,     0.77459666924148337703585307995648,
                     0.77459666924148337703585307995648,     0.77459666924148337703585307995648,     0.77459666924148337703585307995648,
                     0.77459666924148337703585307995648,     0.77459666924148337703585307995648,     0.77459666924148337703585307995648 };*/

        /*pnts_n[4] = 27;
                for (int pnt_i=0; pnt_i < 27; pnt_i++) {
            weights[4][pnt_i] = W_2[pnt_i]*ref_vol;
            pnts[4][pnt_i][0] = X_2[pnt_i];
            pnts[4][pnt_i][1] = Y_2[pnt_i];
            pnts[4][pnt_i][2] = Z_2[pnt_i];
        };
        if (max_order == 4) return;

        pnts_n[5] = 27;
                for (int pnt_i=0; pnt_i < 27; pnt_i++) {
            weights[5][pnt_i] = W_2[pnt_i]*ref_vol;
            pnts[5][pnt_i][0] = X_2[pnt_i];
            pnts[5][pnt_i][1] = Y_2[pnt_i];
            pnts[5][pnt_i][2] = Z_2[pnt_i];
        };
        if (max_order == 5) return;*/

        throw std::runtime_error("elem_cubature_reference::__init_hexahedron : not supported order");
    }
    void    __init_prism()
    {
        //all W_XX are normalized to one
        T   ref_vol(1.f);

        T W_0[1] = {1.0};
        T X_0[1] = {1.0/3.0};
        T Y_0[1] = {1.0/3.0};
        T Z_0[1] = {0.};

        pnts_n[0] = 1;
        for (int pnt_i = 0; pnt_i < 1; pnt_i++) 
        {
            weights[0][pnt_i] = W_0[pnt_i]*ref_vol;
            pnts[0][pnt_i][0] = X_0[pnt_i];
            pnts[0][pnt_i][1] = Y_0[pnt_i];
            pnts[0][pnt_i][2] = Z_0[pnt_i];
        };

        pnts_n[1] = 1;
        for (int pnt_i = 0; pnt_i < 1; pnt_i++) 
        {
            weights[1][pnt_i] = W_0[pnt_i]*ref_vol;
            pnts[1][pnt_i][0] = X_0[pnt_i];
            pnts[1][pnt_i][1] = Y_0[pnt_i];
            pnts[1][pnt_i][2] = Z_0[pnt_i];
        };
        if (max_order == 1) return;

        T W_1[6] = {1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0};
        T X_1[6] = {1.0/6.0, 2.0/3.0, 1.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/6.0};
        T Y_1[6] = {1.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/6.0, 1.0/6.0, 2.0/3.0};
        T Z_1[6] = {-0.5773502691896257,  -0.5773502691896257, -0.5773502691896257,  0.5773502691896257,  0.5773502691896257,  0.5773502691896257};
        pnts_n[2] = 6;
        for (int pnt_i = 0; pnt_i < 6; pnt_i++) 
        {
            weights[2][pnt_i] = W_1[pnt_i]*ref_vol;
            pnts[2][pnt_i][0] = X_1[pnt_i];
            pnts[2][pnt_i][1] = Y_1[pnt_i];
            pnts[2][pnt_i][2] = Z_1[pnt_i];
        };
        if (max_order == 2) return;

        T W_2[8] = {-27./96., 25./96., 25./96., 25./96., -27./96., 25./96., 25./96., 25./96.};
        T X_2[8] = {1.0/3.0, 1.0/5.0, 1.0/5.0, 3.0/5.0, 1.0/3.0, 1.0/5.0, 1.0/5.0, 3.0/5.0};
        T Y_2[8] = {1.0/3.0, 3.0/5.0, 1.0/5.0, 1.0/5.0, 1.0/3.0, 3.0/5.0, 1.0/5.0, 1.0/5.0};
        T Z_2[8] = {-0.5773502691896257,  -0.5773502691896257, -0.5773502691896257, -0.5773502691896257,
                     0.5773502691896257,   0.5773502691896257,  0.5773502691896257,  0.5773502691896257};
        pnts_n[3] = 8;
        for (int pnt_i = 0; pnt_i < 8; pnt_i++) 
        {
            weights[3][pnt_i] = W_2[pnt_i]*ref_vol;
            pnts[3][pnt_i][0] = X_2[pnt_i];
            pnts[3][pnt_i][1] = Y_2[pnt_i];
            pnts[3][pnt_i][2] = Z_2[pnt_i];
        };
        if (max_order == 3) return;

        T W_3[18] = {   0.22338158967801*5./18., 0.22338158967801*5./18., 0.22338158967801*5./18., 0.10995174365532*5./18., 0.10995174365532*5./18., 0.10995174365532*5./18.,
                        0.22338158967801*8./18., 0.22338158967801*8./18., 0.22338158967801*8./18., 0.10995174365532*8./18., 0.10995174365532*8./18., 0.10995174365532*8./18.,
                        0.22338158967801*5./18., 0.22338158967801*5./18., 0.22338158967801*5./18., 0.10995174365532*5./18., 0.10995174365532*5./18., 0.10995174365532*5./18.};
        T X_3[18] = {   0.44594849091597, 0.44594849091597, 0.10810301816807, 0.09157621350977, 0.09157621350977, 0.81684757298046,
                        0.44594849091597, 0.44594849091597, 0.10810301816807, 0.09157621350977, 0.09157621350977, 0.81684757298046,
                        0.44594849091597, 0.44594849091597, 0.10810301816807, 0.09157621350977, 0.09157621350977, 0.81684757298046};
        T Y_3[18] = {   0.44594849091597, 0.10810301816807, 0.44594849091597, 0.09157621350977, 0.81684757298046, 0.09157621350977,
                        0.44594849091597, 0.10810301816807, 0.44594849091597, 0.09157621350977, 0.81684757298046, 0.09157621350977,
                        0.44594849091597, 0.10810301816807, 0.44594849091597, 0.09157621350977, 0.81684757298046, 0.09157621350977};
        T Z_3[18] = {  -0.77459666924148337703585307995648, -0.77459666924148337703585307995648, -0.77459666924148337703585307995648, -0.77459666924148337703585307995648, -0.77459666924148337703585307995648, -0.77459666924148337703585307995648,
                        0.00000000000000000000000000000000,  0.00000000000000000000000000000000,  0.00000000000000000000000000000000,  0.00000000000000000000000000000000,  0.00000000000000000000000000000000,  0.00000000000000000000000000000000,
                        0.77459666924148337703585307995648,  0.77459666924148337703585307995648,  0.77459666924148337703585307995648,  0.77459666924148337703585307995648,  0.77459666924148337703585307995648,  0.77459666924148337703585307995648};
        pnts_n[4] = 18;
        for (int pnt_i = 0; pnt_i < 18; pnt_i++) 
        {
            weights[4][pnt_i] = W_3[pnt_i]*ref_vol;
            pnts[4][pnt_i][0] = X_3[pnt_i];
            pnts[4][pnt_i][1] = Y_3[pnt_i];
            pnts[4][pnt_i][2] = Z_3[pnt_i];
        };
        if (max_order == 4) return;

        T W_4[21] = {   0.22500000000000*5./18, 0.13239415278851*5./18, 0.13239415278851*5./18, 0.13239415278851*5./18, 0.12593918054483*5./18, 0.12593918054483*5./18, 0.12593918054483*5./18,
                        0.22500000000000*8./18, 0.13239415278851*8./18, 0.13239415278851*8./18, 0.13239415278851*8./18, 0.12593918054483*8./18, 0.12593918054483*8./18, 0.12593918054483*8./18,
                        0.22500000000000*5./18, 0.13239415278851*5./18, 0.13239415278851*5./18, 0.13239415278851*5./18, 0.12593918054483*5./18, 0.12593918054483*5./18, 0.12593918054483*5./18};
        T X_4[21] = {   0.33333333333333, 0.47014206410511, 0.47014206410511, 0.05971587178977, 0.10128650732346, 0.10128650732346, 0.79742698535309,
                        0.33333333333333, 0.47014206410511, 0.47014206410511, 0.05971587178977, 0.10128650732346, 0.10128650732346, 0.79742698535309,
                        0.33333333333333, 0.47014206410511, 0.47014206410511, 0.05971587178977, 0.10128650732346, 0.10128650732346, 0.79742698535309};
        T Y_4[21] = {   0.33333333333333, 0.47014206410511, 0.05971587178977, 0.47014206410511, 0.10128650732346, 0.79742698535309, 0.10128650732346,
                        0.33333333333333, 0.47014206410511, 0.05971587178977, 0.47014206410511, 0.10128650732346, 0.79742698535309, 0.10128650732346,
                        0.33333333333333, 0.47014206410511, 0.05971587178977, 0.47014206410511, 0.10128650732346, 0.79742698535309, 0.10128650732346};
        T Z_4[21] = {-0.77459666924148337703585307995648, -0.77459666924148337703585307995648, -0.77459666924148337703585307995648, -0.77459666924148337703585307995648, -0.77459666924148337703585307995648, -0.77459666924148337703585307995648, -0.77459666924148337703585307995648,
                      0.00000000000000000000000000000000,  0.00000000000000000000000000000000,  0.00000000000000000000000000000000,  0.00000000000000000000000000000000,  0.00000000000000000000000000000000,  0.00000000000000000000000000000000,  0.00000000000000000000000000000000,
                      0.77459666924148337703585307995648,  0.77459666924148337703585307995648,  0.77459666924148337703585307995648,  0.77459666924148337703585307995648,  0.77459666924148337703585307995648,  0.77459666924148337703585307995648,  0.77459666924148337703585307995648};
        pnts_n[5] = 21;
        for (int pnt_i = 0; pnt_i < 21; pnt_i++) 
        {
            weights[5][pnt_i] = W_4[pnt_i]*ref_vol;
            pnts[5][pnt_i][0] = X_4[pnt_i];
            pnts[5][pnt_i][1] = Y_4[pnt_i];
            pnts[5][pnt_i][2] = Z_4[pnt_i];
        };
        if (max_order == 5) return;

        throw std::runtime_error("elem_cubature_reference::__init_prism : not supported order");

        //TODO
    }

    /*template<class ElemRef>
    void    init_for_coords
    (
        const elem_cubature_reference &q_ref, int elem_type,
        const ElemRef &e, const vec_t *vertexes
    )
    {
        for (unsigned char ord = 0;ord <= max_order;++ord) 
        {
            pnts_n[ord] = q_ref.pnts_n[ord];
            for (unsigned char pnt_i = 0;pnt_i < pnts_n[ord];++pnt_i) 
            {
                //abs here is just in case of wrong vertexes orientation
                weights[ord][pnt_i] = std::abs( e.ref_to_phys_jacob_det(elem_type, vertexes, q_ref.pnts[ord][pnt_i]) )*q_ref.weights[ord][pnt_i];
                e.ref_to_phys(elem_type, vertexes, q_ref.pnts[ord][pnt_i], pnts[ord][pnt_i]);
            }
        }
    }*/
};

//ISSUE what about dimension? divide it or make it 'all in one'?
//NOTE it's recommended to use methods accessors instead of directly accessing arrays
template<int max_order, class T>
struct cubature_reference
{
    typedef static_vec::vec<T,3>                  vec_t;
    typedef elem_cubature_reference<max_order,T>  elem_cubature_t;

    static const int                    max_pnts_n = cubature_reference_max_pnts_n<max_order>::value;

    /*int       pnts_n[CUBATURE_REFERENCE_ELEM_TYPES_N][max_order+1];
    T       weights[CUBATURE_REFERENCE_ELEM_TYPES_N][max_order+1][max_pnts_n];
    vec_t       pnts[CUBATURE_REFERENCE_ELEM_TYPES_N][max_order+1][max_pnts_n];*/
    /// TODO dirty hack!! won't work for 2d solver case
    elem_cubature_t cubatures[CUBATURE_REFERENCE_ELEM_TYPES_N-2];

    __DEVICE_TAG__ const elem_cubature_t    &get_elem_cubature(int elem_type)const
    {
        return cubatures[elem_type-2];
    }
    __DEVICE_TAG__ unsigned char            get_pnts_n(int elem_type, unsigned char order)const
    {
        return cubatures[elem_type-2].pnts_n[order];
    }
    __DEVICE_TAG__ T                        get_weight(int elem_type, unsigned char order, unsigned char pnt_i)const
    {
        return cubatures[elem_type-2].weights[order][pnt_i];
    }
    //several interfaces to get cubature points coordinates
    __DEVICE_TAG__ const vec_t              &get_pnt(int elem_type, unsigned char order, unsigned char pnt_i)const
    {
        return cubatures[elem_type-2].pnts[order][pnt_i];
    }
    __DEVICE_TAG__ void                     get_pnt(int elem_type, unsigned char order, unsigned char pnt_i, vec_t &res)const
    {
        res = cubatures[elem_type-2].pnts[order][pnt_i];
    }
        __DEVICE_TAG__ T                    get_pnt(int elem_type, unsigned char order, unsigned char pnt_i, unsigned char j)const
    {
        return cubatures[elem_type-2].pnts[order][pnt_i][j];
    }



    cubature_reference()
    {
        //fill data
        /*cubatures[2].__init_triangle();
        cubatures[3].__init_quad();
        cubatures[4].__init_tetrahedron();
        cubatures[5].__init_hexahedron();
        cubatures[6].__init_prism();*/
        /// TODO dirty hack!! won't work for 2d solver case
        cubatures[0].__init_triangle();
        cubatures[1].__init_quad();
        cubatures[2].__init_tetrahedron();
        cubatures[3].__init_hexahedron();
        cubatures[4].__init_prism();
        //TODO other types
    }

};

}  /// namespace mesh
}  /// namespace scfd

#endif
