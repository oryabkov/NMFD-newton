#ifndef __SCFD_DG_DEVICE_POLYNOM_REFERENCE_H__
#define __SCFD_DG_DEVICE_POLYNOM_REFERENCE_H__

#include "polylen_static.h"
#include "polynom_indexing.h"

namespace scfd
{
namespace dg
{
namespace detail 
{

template<class T, int Dim, int MaxPolyDeg>
struct device_polynom_reference
{
    static const int poly_max_len = polylen_static<Dim,MaxPolyDeg>::len;

    T       coeffs[poly_max_len];

    /// Special function to access specific array componont without dynamic indexing
    __DEVICE_TAG__ T  access_comp(const T p[Dim],int j)const
    {
        switch (j)
        {
            case 0: return p[0];
            case 1: return p[1];
            case 2: return p[2];
            //default:
            /// TODO some error
            default: return T(-1e+6);
        }
    }
    __DEVICE_TAG__ T  eval(const T p[Dim])const
    {
        T res(0);

        int deg;
        int mon[MaxPolyDeg];
        for (int mon_i = 0;mon_i < poly_max_len;++mon_i)
        {
            polynom_index_to_monom<int,MaxPolyDeg>(Dim, mon_i, deg, mon);
            T mon_val(1);
            #pragma unroll
            for (int mon_comp_i = 0;mon_comp_i < MaxPolyDeg;++mon_comp_i)
            {
                if (!(mon_comp_i < deg)) break;
                mon_val *= access_comp(p,mon[mon_comp_i]);
            }
            res += mon_val*coeffs[mon_i];
        }

        return res;
    }
};

}  /// namespace detail
}  /// namespace dg
}  /// namespace scfd

#endif


