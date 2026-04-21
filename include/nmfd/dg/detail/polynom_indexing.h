#ifndef __SCFD_POLYNOM_INDEXING_H__
#define __SCFD_POLYNOM_INDEXING_H__

#include <scfd/utils/device_tag.h>

namespace scfd
{
namespace dg
{
namespace detail 
{

// = binomial(n,k)
template<class Ord>
__DEVICE_TAG__ Ord binomial(Ord n, Ord k)
{
    Ord res = 1;
    for (Ord i=1;i<=k;i++)
    {
        res *= n - k + i;
        res /= i;
    }
    return res;
}

// Length of a vars_n,neg polynomial
// = binomial(vars_n+deg,deg)
template<class Ord>
__DEVICE_TAG__ Ord polylen(Ord vars_n, Ord deg)
{
    return binomial(vars_n+deg,deg);
}

template<class Ord, Ord MaxDeg>
__DEVICE_TAG__ void polynom_index_to_monom(Ord vars_n, Ord index, Ord &deg, Ord mon[MaxDeg])
{
    deg = 0;
    while (index >= polylen(vars_n,deg)) ++deg;
    if (deg > 0) index -= polylen(vars_n,deg-1);
    Ord curr_vars_n = vars_n,
        curr_base_i = 0;
    //TODO deg-d problem
    //#pragma unroll
    //for (int d = MaxDeg;d >= 1;--d) {
    //    if (d > deg) continue;
    #pragma unroll
    for (int curr_mon_i = 0;curr_mon_i < MaxDeg;++curr_mon_i) {
        if (!(curr_mon_i < deg)) break;

        //Ord     curr_mon_i = deg-d;
        Ord     d = deg-curr_mon_i;
        Ord     i_curr_0 = 0,
                i_curr_1 = curr_vars_n;
        while (i_curr_1 - i_curr_0 > 1) {
            Ord i_curr_mid = (i_curr_1 + i_curr_0)/2;
            Ord i_curr_mid_sum = binomial(curr_vars_n+d-1,d) - binomial(curr_vars_n+d-1-i_curr_mid,d);
            if (index >= i_curr_mid_sum)
                i_curr_0 = i_curr_mid;
            else
                i_curr_1 = i_curr_mid;
        }
        //std::cout << "index = " << index << " d = " << d << " i_curr_0 = " << i_curr_0 << " i_curr_1 = " << i_curr_1 << std::endl;
        //assert(i_curr_1 == i_curr_0+1);
        mon[curr_mon_i] = i_curr_0 + curr_base_i;
        //std::cout << mon[deg-d] << " " << i_curr_0 << std::endl;
        index -= binomial(curr_vars_n+d-1,d) - binomial(curr_vars_n+d-1-i_curr_0,d);
        curr_vars_n = vars_n - mon[curr_mon_i];
        curr_base_i = mon[curr_mon_i];
    }
}

//NOTE works only for static variables number
//because we need to know array size compile-time (CUDA problems)
template<class Ord, Ord MaxDeg, Ord VarsN>
__DEVICE_TAG__ void monom_to_exponents(Ord deg, Ord mon[MaxDeg], Ord exp[VarsN])
{
    //TODO static upper index in loop
    #pragma unroll
    for (int j = 0;j < VarsN;++j) exp[j] = 0;
    #pragma unroll
    for (int j = 0;j < deg;++j) {
        exp[mon[j]]++;
    }
}

//NOTE works for run-time variables number but for CUDA suppose some global memory array (slower then static version)
template<class Ord, Ord MaxDeg>
__DEVICE_TAG__ void monom_to_exponents(Ord deg, Ord vars_n, Ord mon[MaxDeg], Ord *exp)
{
    //TODO static upper index in loop
    #pragma unroll
    for (int j = 0;j < vars_n;++j) exp[j] = 0;
    #pragma unroll
    for (int j = 0;j < deg;++j) {
        exp[mon[j]]++;
    }
}

//NOTE works only for static variables number
//because we need to know array size compile-time (CUDA problems)
template<class Ord, Ord MaxDeg, Ord VarsN>
__DEVICE_TAG__ void polynom_index_to_exponents(Ord index, Ord exp[VarsN])
{
    Ord deg, mon[MaxDeg];
    polynom_index_to_monom<Ord,MaxDeg>(VarsN, index, deg, mon);
    monom_to_exponents<Ord,MaxDeg,VarsN>(deg, mon, exp);
}

template<class Ord, Ord MaxDeg>
__DEVICE_TAG__ void polynom_index_to_exponents(Ord vars_n, Ord index, Ord *exp)
{
    Ord deg, mon[MaxDeg];
    polynom_index_to_monom<Ord,MaxDeg>(vars_n, index, deg, mon);
    monom_to_exponents<Ord,MaxDeg>(deg, vars_n, mon, exp);
}

template<class Ord>
__DEVICE_TAG__ Ord pow(Ord base, Ord deg)
{
    Ord     res = 1;
    for (Ord i=1;i<=deg;i++) res *= base;
    return res;
}

template<class Ord>
__DEVICE_TAG__ Ord bitwise_get(Ord x, Ord bit)
{
    return (x & (1 << bit)) >> bit;
}

template<class Ord, Ord MaxDeg>
__DEVICE_TAG__ bool monom_sub_term_enum(Ord deg, Ord mon[MaxDeg], Ord sub_term_i)
{
    #pragma unroll
    for (int j = 1;j < MaxDeg;++j) {
        if (!(j < deg)) break;
        if ((mon[j]==mon[j-1])&&(bitwise_get(sub_term_i,j) > bitwise_get(sub_term_i,j-1))) return false;
    }
    return true;
}

template<class Ord, Ord MaxDeg>
__DEVICE_TAG__ Ord monom_to_polynom_index(Ord vars_n, Ord deg, Ord mon[MaxDeg], Ord sub_term_mask)
{
    Ord     masked_deg = 0;
    #pragma unroll
    for (int j = 0;j < MaxDeg;++j) {
        if (!(j < deg)) break;
        if (bitwise_get(sub_term_mask,j)) ++masked_deg;
    }
    Ord     res = 0;
    if (masked_deg > 0) res += polylen(vars_n,masked_deg-1);
    //std::cout << std::endl << "masked_deg = " << masked_deg << std::endl;
    //std::cout << std::endl << "res = " << res << std::endl;

    Ord curr_vars_n = vars_n,
        curr_base_i = 0;
    Ord masked_d = masked_deg;
    //#pragma unroll
    //for (int d = MaxDeg;d >= 1;--d) {
    #pragma unroll
    for (int curr_mon_i = 0;curr_mon_i < MaxDeg;++curr_mon_i) {
        if (!(curr_mon_i < deg)) break;

        //if (d > deg) continue;
        if (!bitwise_get(sub_term_mask,curr_mon_i)) continue;

        res += binomial(curr_vars_n+masked_d-1,masked_d) - 
               binomial(curr_vars_n+masked_d-1-(mon[curr_mon_i]-curr_base_i),masked_d);

        curr_vars_n = vars_n - mon[curr_mon_i];
        curr_base_i = mon[curr_mon_i];

        --masked_d;
    }

    return res;
}

}  /// namespace detail
}  /// namespace dg
}  /// namespace scfd

#endif
