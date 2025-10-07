
#include <type_traits>
#include <iostream>
#include <string>
#include <nmfd/detail/algo_hierarchy_macro.h>

struct subalg1
{
    using vector_space_type = int;

    struct utils_hierarchy
    {
        void *some_util;
        utils_hierarchy() = default;
        template<class Backend>
        utils_hierarchy(Backend &backend, std::shared_ptr<vector_space_type> vec_space)
        {
        }
    };
};

struct subalg2
{
    struct params_hierarchy
    {
        int p1 = 1,p2 = 2;

        params_hierarchy(const std::string &log_prefix = "", const std::string &log_name = "subalg2::")
        {
        }
        #ifdef NMFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
            p1 = j.value("p1", p1);
            p2 = j.value("p2", p2);
        }
        nlohmann::json to_json() const
        {
            return
                nlohmann::json{{"type", "subalg2"},{"p1",p1},{"p2",p2}};
        }
        #endif
    };
};

/// First test internal macros

struct alg
{
    using vector_space_type = int;

    struct params
    {
        std::string log_msg_prefix;

        params(const std::string &log_prefix = "", const std::string &log_name = "alg::")
        {
        }
        #ifdef NMFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
        }
        nlohmann::json to_json() const
        {
            return
                nlohmann::json{{"type", "alg"}};
        }
        #endif
    };
    struct utils
    {
        template<class Backend>
        utils(Backend &backend, std::shared_ptr<vector_space_type> vec_space_)
        {
        }
    };

    NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL(subalg1,a1,subalg2,a2)
    struct params_hierarchy : public params
    {
        NMFD_ALGO_HIERARCHY_PARAMS_DEF(subalg1,a1,subalg2,a2)
        params_hierarchy(const std::string &log_prefix = "", const std::string &log_name = "alg::") : 
          params(log_prefix, log_name)
          NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST(subalg1,a1,subalg2,a2)
        {
        }
        params_hierarchy(
            const params &prm_
            NMFD_ALGO_HIERARCHY_PARAMS_PASS(subalg1,a1,subalg2,a2)
        ) : 
          params(prm_)
          NMFD_ALGO_HIERARCHY_INIT_LIST(subalg1,a1,subalg2,a2)
        {
        }
        #ifdef NMFD_ENABLE_NLOHMANN
        void from_json(const nlohmann::json& j)
        {
            params::from_json(j);
            NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON(subalg1,a1,subalg2,a2)
        }
        nlohmann::json to_json() const
        {
            nlohmann::json  j = params::to_json();
            NMFD_ALGO_HIERARCHY_PARAMS_TOJSON(subalg1,a1,subalg2,a2)
            return j;
        }
        #endif
    };
    NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL(subalg1,a1,subalg2,a2)
    struct utils_hierarchy : public utils
    {
        NMFD_ALGO_HIERARCHY_UTILS_DEF(subalg1,a1,subalg2,a2)
        utils_hierarchy() = default;
        template<class ...Args>
        utils_hierarchy(
            NMFD_ALGO_HIERARCHY_UTILS_PASS(subalg1,a1,subalg2,a2)
            Args... args
        ) : 
          utils(args...)
          NMFD_ALGO_HIERARCHY_INIT_LIST(subalg1,a1,subalg2,a2)
        {
        }
        template<class Backend>
        utils_hierarchy(Backend &backend, std::shared_ptr<vector_space_type> vec_space) : 
          utils(backend, vec_space)
          NMFD_ALGO_HIERARCHY_UTILS_INIT_LIST(subalg1,a1,subalg2,a2)
        {
        }
    };
};

/// Then test macros for user-usage (should do the same as above)

struct alg1
{
    using vector_space_type = int;

    NMFD_ALGO_EMPTY_PARAMS_TYPE_DEFINE(alg1)
    NMFD_ALGO_EMPTY_UTILS_TYPE_DEFINE(alg1)

    NMFD_ALGO_HIERARCHY_TYPES_DEFINE(alg1,subalg1,a1,subalg2,a2)
};



int main(int argc, char const *args[])
{
    int errors = 0;

    alg a;
    alg1 a1;

    if (!std::is_same<alg::a1_params_hierarchy_type,nmfd::detail::params_hierarchy_dummy>::value) errors++;
    if (!std::is_same<alg::a2_params_hierarchy_type,subalg2::params_hierarchy>::value) errors++;

    if (!std::is_same<alg::a1_utils_hierarchy_type,subalg1::utils_hierarchy>::value) errors++;
    if (!std::is_same<alg::a2_utils_hierarchy_type,nmfd::detail::utils_hierarchy_dummy>::value) errors++;
    
    if (!std::is_same<alg1::a1_params_hierarchy_type,nmfd::detail::params_hierarchy_dummy>::value) errors++;
    if (!std::is_same<alg1::a2_params_hierarchy_type,subalg2::params_hierarchy>::value) errors++;

    if (!std::is_same<alg1::a1_utils_hierarchy_type,subalg1::utils_hierarchy>::value) errors++;
    if (!std::is_same<alg1::a2_utils_hierarchy_type,nmfd::detail::utils_hierarchy_dummy>::value) errors++;

    #ifdef NMFD_ENABLE_NLOHMANN
    alg1::params_hierarchy alg1_params;
    /// Check default values
    if (alg1_params.a2.p1 != 1) errors++;
    if (alg1_params.a2.p2 != 2) errors++;
    auto j = alg1_params.to_json();
    j["a2"]["p1"] = 2;
    j["a2"]["p2"] = 3;
    alg1_params.from_json(j);
    /// Check values from json
    if (alg1_params.a2.p1 != 2) errors++;
    if (alg1_params.a2.p2 != 3) errors++;
    #endif

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