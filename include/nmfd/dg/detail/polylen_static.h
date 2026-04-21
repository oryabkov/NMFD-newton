#ifndef __SCFD_DG_POLYLEN_STATIC_H__
#define __SCFD_DG_POLYLEN_STATIC_H__

namespace scfd
{
namespace dg
{
namespace detail 
{
  
template <int n>
struct factorial_static
{
  enum { fac = n*factorial_static<n-1>::fac };
};
  
template <>
struct factorial_static<0>
{
  enum { fac = 1 };
};

template <>
struct factorial_static<-1>
{
  enum { fac = 1 };
};
  
// Template to evaluate binomial_static coefficients at compile time.
template <int n, int k>
struct binomial_static
{
    enum { value = binomial_static<n-1,k-1>::value + binomial_static<n-1,k>::value };
};

template <int n>
struct binomial_static<n,n>
{
    enum { value = 1 };
};

template<>
struct binomial_static<0,0>
{
  enum { value = 1 };
};

template <int n>
struct binomial_static<n,0>
{
  enum { value = n > 0 ? 1 : 0 };
};

template <int k>
struct binomial_static<0,k>
{
  enum { value = k > 0 ? 1 : 0 };
};

template <int Nvar, int Ndeg>
struct polylen_static
{
  enum { len = binomial_static<Nvar+Ndeg,Ndeg>::value };
};

}  /// namespace detail
}  /// namespace dg
}  /// namespace scfd

#endif
