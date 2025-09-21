
#include <type_traits>
#include <iostream>
#include <nmfd/detail/algo_hierarchy_macro.h>

struct subalg1
{
    struct utils_hierarchy
    {
        void *some_util;
    };
};

struct subalg2
{
    struct params_hierarchy
    {
        int p1,p2;
    };
};

struct alg
{
    struct params
    {
    };
    struct utils
    {
    };

    NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL(subalg1,a1,subalg2,a2)
    struct params_hierarchy : public params
    {
        NMFD_ALGO_HIERARCHY_PARAMS_DEF(subalg1,a1,subalg2,a2)
        params_hierarchy(
            const params &prm_, 
            NMFD_ALGO_HIERARCHY_PARAMS_PASS(subalg1,a1,subalg2,a2)
        ) : 
          params(prm_), 
          NMFD_ALGO_HIERARCHY_INIT_LIST(subalg1,a1,subalg2,a2)
        {
        }
    };
    NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL(subalg1,a1,subalg2,a2)
    struct utils_hierarchy : public utils
    {
        NMFD_ALGO_HIERARCHY_UTILS_DEF(subalg1,a1,subalg2,a2)
        utils_hierarchy() = default;
        template<class ...Args>
        utils_hierarchy(
            NMFD_ALGO_HIERARCHY_UTILS_PASS(subalg1,a1,subalg2,a2),
            Args... args
        ) : 
          utils(args...), 
          NMFD_ALGO_HIERARCHY_INIT_LIST(subalg1,a1,subalg2,a2)
        {
        }
    };
};

int main(int argc, char const *args[])
{
    int errors = 0;

    alg a;

    if (!std::is_same<alg::a1_params_hierarchy_type,nmfd::detail::params_hierarchy_dummy>::value) errors++;
    if (!std::is_same<alg::a2_params_hierarchy_type,subalg2::params_hierarchy>::value) errors++;

    if (!std::is_same<alg::a1_utils_hierarchy_type,subalg2::utils_hierarchy>::value>::value) errors++;
    if (!std::is_same<alg::a2_utils_hierarchy_type,nmfd::detail::utils_hierarchy_dummy) errors++;
    
    if (errors != 0)
    {
        std::cout << "tests FAILED, errors = " << errors << std::endl;
    }
    else
    {
        std::cout << "tests PASSED" << std::endl;
    }

    return 0;
}