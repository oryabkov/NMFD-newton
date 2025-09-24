// Copyright © 2016-2020 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is subalgot of SCFD.

// SCFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// SCFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with SCFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __NMFD_ALGO_HIERARCHY_MACRO_H__
#define __NMFD_ALGO_HIERARCHY_MACRO_H__

//#include "for_each_config.h"
#ifdef NMFD_ENABLE_NLOHMANN
#include <nlohmann/json.hpp>
#endif
#include <nmfd/detail/algo_utils_hierarchy.h>
#include <nmfd/detail/algo_params_hierarchy.h>
#include <nmfd/detail/str_source_helper.h>


//TODO not working for MSVC 2008
//i suppose we can use boost instead?
//NOTE however this variant is totally standart-compliant

/************************************ General macros ************************************/

#define NMFD_ALGO_HIERARCHY_CONCATENATE(arg1, arg2)   NMFD_ALGO_HIERARCHY_CONCATENATE1(arg1, arg2)
#define NMFD_ALGO_HIERARCHY_CONCATENATE1(arg1, arg2)  NMFD_ALGO_HIERARCHY_CONCATENATE2(arg1, arg2)
#define NMFD_ALGO_HIERARCHY_CONCATENATE2(arg1, arg2)  arg1##arg2

#define NMFD_ALGO_HIERARCHY_NARG(...) NMFD_ALGO_HIERARCHY_NARG_(__VA_ARGS__, NMFD_ALGO_HIERARCHY_RSEQ_N())
#define NMFD_ALGO_HIERARCHY_NARG_(...) NMFD_ALGO_HIERARCHY_ARG_N(__VA_ARGS__) 
#define NMFD_ALGO_HIERARCHY_ARG_N(_11, _12, _21, _22, _31, _32, _41, _42, N, ...) N 
#define NMFD_ALGO_HIERARCHY_RSEQ_N() 4, 4, 3, 3, 2, 2, 1, 1, 0, 0

/************************************ Common utils/params macros ************************/

#define NMFD_ALGO_HIERARCHY_INIT_LIST_0(...)
#define NMFD_ALGO_HIERARCHY_INIT_LIST_1(subalgo_type, subalgo_name, ...) \
  subalgo_name(subalgo_name)
#define NMFD_ALGO_HIERARCHY_INIT_LIST_2(subalgo_type, subalgo_name, ...) \
  subalgo_name(subalgo_name),                                            \
  NMFD_ALGO_HIERARCHY_INIT_LIST_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_INIT_LIST_3(subalgo_type, subalgo_name, ...) \
  subalgo_name(subalgo_name),                                            \
  NMFD_ALGO_HIERARCHY_INIT_LIST_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_INIT_LIST_4(subalgo_type, subalgo_name, ...) \
  subalgo_name(subalgo_name),                                            \
  NMFD_ALGO_HIERARCHY_INIT_LIST_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_INIT_LIST_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_INIT_LIST_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_INIT_LIST(...) NMFD_ALGO_HIERARCHY_INIT_LIST_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

/************************************ Params-specific macros ****************************/

#define NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_0(...)
#define NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_1(subalgo_type, subalgo_name, ...)                                                                  \
  using NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) = typename nmfd::detail::algo_params_hierarchy<subalgo_type>::type;
#define NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_2(subalgo_type, subalgo_name, ...)                                                                  \
  using NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) = typename nmfd::detail::algo_params_hierarchy<subalgo_type>::type;  \
  NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_3(subalgo_type, subalgo_name, ...)                                                                  \
  using NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) = typename nmfd::detail::algo_params_hierarchy<subalgo_type>::type;  \
  NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_4(subalgo_type, subalgo_name, ...)                                                                  \
  using NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) = typename nmfd::detail::algo_params_hierarchy<subalgo_type>::type;  \
  NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL(...) NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_DEF_0(...)
#define NMFD_ALGO_HIERARCHY_PARAMS_DEF_1(subalgo_type, subalgo_name, ...)             \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) subalgo_name;
#define NMFD_ALGO_HIERARCHY_PARAMS_DEF_2(subalgo_type, subalgo_name, ...)             \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) subalgo_name;  \
  NMFD_ALGO_HIERARCHY_PARAMS_DEF_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_DEF_3(subalgo_type, subalgo_name, ...)             \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) subalgo_name;  \
  NMFD_ALGO_HIERARCHY_PARAMS_DEF_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_DEF_4(subalgo_type, subalgo_name, ...)             \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) subalgo_name;  \
  NMFD_ALGO_HIERARCHY_PARAMS_DEF_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_DEF_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_PARAMS_DEF_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_DEF(...) NMFD_ALGO_HIERARCHY_PARAMS_DEF_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_PASS_0(...)
#define NMFD_ALGO_HIERARCHY_PARAMS_PASS_1(subalgo_type, subalgo_name, ...)                   \
  const NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) &subalgo_name
#define NMFD_ALGO_HIERARCHY_PARAMS_PASS_2(subalgo_type, subalgo_name, ...)                   \
  const NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) &subalgo_name,  \
  NMFD_ALGO_HIERARCHY_PARAMS_PASS_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_PASS_3(subalgo_type, subalgo_name, ...)                   \
  const NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) &subalgo_name,  \
  NMFD_ALGO_HIERARCHY_PARAMS_PASS_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_PASS_4(subalgo_type, subalgo_name, ...)                   \
  const NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_params_hierarchy_type) &subalgo_name,  \
  NMFD_ALGO_HIERARCHY_PARAMS_PASS_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_PASS_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_PARAMS_PASS_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_PASS(...) NMFD_ALGO_HIERARCHY_PARAMS_PASS_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_0(...)
#define NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_1(subalgo_type, subalgo_name, ...) \
  subalgo_name(this->log_msg_prefix)
#define NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_2(subalgo_type, subalgo_name, ...) \
  subalgo_name(this->log_msg_prefix),                                           \
  NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_3(subalgo_type, subalgo_name, ...) \
  subalgo_name(this->log_msg_prefix),                                           \
  NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_4(subalgo_type, subalgo_name, ...) \
  subalgo_name(this->log_msg_prefix),                                           \
  NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST(...) NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

/************************************ Params-json-specific macros ***********************/

#define NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_0(...)
#define NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_1(subalgo_type, subalgo_name, ...)                   \
  j[__STR(subalgo_name)] = subalgo_name.to_json();
#define NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_2(subalgo_type, subalgo_name, ...)                   \
  j[__STR(subalgo_name)] = subalgo_name.to_json();                                             \
  NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_3(subalgo_type, subalgo_name, ...)                   \
  j[__STR(subalgo_name)] = subalgo_name.to_json();                                             \
  NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_4(subalgo_type, subalgo_name, ...)                   \
  j[__STR(subalgo_name)] = subalgo_name.to_json();                                             \
  NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_TOJSON(...) NMFD_ALGO_HIERARCHY_PARAMS_TOJSON_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_0(...)
#define NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_1(subalgo_type, subalgo_name, ...)                   \
  subalgo_name.from_json(j.at(__STR(subalgo_name)));
#define NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_2(subalgo_type, subalgo_name, ...)                   \
  subalgo_name.from_json(j.at(__STR(subalgo_name)));                                             \
  NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_3(subalgo_type, subalgo_name, ...)                   \
  subalgo_name.from_json(j.at(__STR(subalgo_name)));                                             \
  NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_4(subalgo_type, subalgo_name, ...)                   \
  subalgo_name.from_json(j.at(__STR(subalgo_name)));                                             \
  NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON(...) NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

#ifdef NMFD_ENABLE_NLOHMANN
#define NMFD_ALGO_HIERARCHY_PARAMS_JSON_METHODS(...)    \
  void from_json(const nlohmann::json& j)               \
  {                                                     \
      params::from_json(j);                             \
      NMFD_ALGO_HIERARCHY_PARAMS_FROMJSON(__VA_ARGS__)  \
  }                                                     \
  nlohmann::json to_json() const                        \
  {                                                     \
      nlohmann::json  j = params::to_json();            \
      NMFD_ALGO_HIERARCHY_PARAMS_TOJSON(__VA_ARGS__)    \
      return j;                                         \
  }
#else
#define NMFD_ALGO_HIERARCHY_PARAMS_JSON_METHODS(...)
#endif

/************************************ Utils-specific macros *****************************/

#define NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_0(...)
#define NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_1(subalgo_type, subalgo_name, ...)                                                                  \
  using NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) = typename nmfd::detail::algo_utils_hierarchy<subalgo_type>::type;
#define NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_2(subalgo_type, subalgo_name, ...)                                                                  \
  using NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) = typename nmfd::detail::algo_utils_hierarchy<subalgo_type>::type;  \
  NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_3(subalgo_type, subalgo_name, ...)                                                                  \
  using NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) = typename nmfd::detail::algo_utils_hierarchy<subalgo_type>::type;  \
  NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_4(subalgo_type, subalgo_name, ...)                                                                  \
  using NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) = typename nmfd::detail::algo_utils_hierarchy<subalgo_type>::type;  \
  NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL(...) NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_UTILS_DEF_0(...)
#define NMFD_ALGO_HIERARCHY_UTILS_DEF_1(subalgo_type, subalgo_name, ...)             \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) subalgo_name;
#define NMFD_ALGO_HIERARCHY_UTILS_DEF_2(subalgo_type, subalgo_name, ...)             \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) subalgo_name;  \
  NMFD_ALGO_HIERARCHY_UTILS_DEF_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_UTILS_DEF_3(subalgo_type, subalgo_name, ...)             \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) subalgo_name;  \
  NMFD_ALGO_HIERARCHY_UTILS_DEF_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_UTILS_DEF_4(subalgo_type, subalgo_name, ...)             \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) subalgo_name;  \
  NMFD_ALGO_HIERARCHY_UTILS_DEF_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_UTILS_DEF_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_UTILS_DEF_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_UTILS_DEF(...) NMFD_ALGO_HIERARCHY_UTILS_DEF_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_UTILS_PASS_0(...)
#define NMFD_ALGO_HIERARCHY_UTILS_PASS_1(subalgo_type, subalgo_name, ...)            \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) subalgo_name
#define NMFD_ALGO_HIERARCHY_UTILS_PASS_2(subalgo_type, subalgo_name, ...)            \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) subalgo_name,  \
  NMFD_ALGO_HIERARCHY_UTILS_PASS_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_UTILS_PASS_3(subalgo_type, subalgo_name, ...)            \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) subalgo_name,  \
  NMFD_ALGO_HIERARCHY_UTILS_PASS_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_UTILS_PASS_4(subalgo_type, subalgo_name, ...)            \
  NMFD_ALGO_HIERARCHY_CONCATENATE(subalgo_name,_utils_hierarchy_type) subalgo_name,  \
  NMFD_ALGO_HIERARCHY_UTILS_PASS_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_UTILS_PASS_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_UTILS_PASS_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_UTILS_PASS(...) NMFD_ALGO_HIERARCHY_UTILS_PASS_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

/************************************ General macros ************************************/

#define NMFD_ALGO_HIERARCHY_TYPES_DEFINE(algo_type, ...)    \
  NMFD_ALGO_HIERARCHY_PARAMS_TYPES_DECL(__VA_ARGS__)        \
  struct params_hierarchy : public params                   \
  {                                                         \
      NMFD_ALGO_HIERARCHY_PARAMS_DEF(__VA_ARGS__)           \
      params_hierarchy(                                     \
          const std::string &log_prefix = "",               \
          const std::string &log_name = __STR(algo_type)"::"\
      ) :                                                   \
        params(log_prefix, log_name),                       \
        NMFD_ALGO_HIERARCHY_PARAMS_INIT_LIST(__VA_ARGS__)   \
      {                                                     \
      }                                                     \
      params_hierarchy(                                     \
          const params &prm_,                               \
          NMFD_ALGO_HIERARCHY_PARAMS_PASS(__VA_ARGS__)      \
      ) :                                                   \
        params(prm_),                                       \
        NMFD_ALGO_HIERARCHY_INIT_LIST(__VA_ARGS__)          \
      {                                                     \
      }                                                     \
      NMFD_ALGO_HIERARCHY_PARAMS_JSON_METHODS(__VA_ARGS__)  \
  };                                                        \
  NMFD_ALGO_HIERARCHY_UTILS_TYPES_DECL(__VA_ARGS__)         \
  struct utils_hierarchy : public utils                     \
  {                                                         \
      NMFD_ALGO_HIERARCHY_UTILS_DEF(__VA_ARGS__)            \
      utils_hierarchy() = default;                          \
      template<class ...Args>                               \
      utils_hierarchy(                                      \
          NMFD_ALGO_HIERARCHY_UTILS_PASS(__VA_ARGS__),      \
          Args... args                                      \
      ) :                                                   \
        utils(args...),                                     \
        NMFD_ALGO_HIERARCHY_INIT_LIST(__VA_ARGS__)          \
      {                                                     \
      }                                                     \
  };

#ifdef NMFD_ENABLE_NLOHMANN
#define NMFD_ALGO_EMPTY_PARAMS_JSON_METHODS(algo_type)  \
  void from_json(const nlohmann::json& j)               \
  {                                                     \
  }                                                     \
  nlohmann::json to_json() const                        \
  {                                                     \
      return                                            \
          nlohmann::json{{"type", __STR(algo_type)}};   \
  }
#else
#define NMFD_ALGO_EMPTY_PARAMS_JSON_METHODS(algo_type)
#endif

#define NMFD_ALGO_EMPTY_PARAMS_TYPE_DEFINE(algo_type)        \
  struct params                                              \
  {                                                          \
      std::string log_msg_prefix;                            \
                                                             \
      params(                                                \
          const std::string &log_prefix = "",                \
          const std::string &log_name = __STR(algo_type)"::" \
      )                                                      \
      {                                                      \
      }                                                      \
      NMFD_ALGO_EMPTY_PARAMS_JSON_METHODS(algo_type)         \
  };
#define NMFD_ALGO_EMPTY_UTILS_TYPE_DEFINE(algo_type)         \
  struct utils                                               \
  {                                                          \
  };

/*#define NMFD_ALGO_HIERARCHY_PARAMS_LIST_1(subalgo_type, subalgo_name, ...) subalgo_type NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)
#define NMFD_ALGO_HIERARCHY_PARAMS_LIST_2(subalgo_type, subalgo_name, ...)                                                            \
  subalgo_type NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name),                                                                       \
  NMFD_ALGO_HIERARCHY_PARAMS_LIST_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_LIST_3(subalgo_type, subalgo_name, ...)                                                            \
  subalgo_type NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name),                                                                       \
  NMFD_ALGO_HIERARCHY_PARAMS_LIST_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_LIST_4(subalgo_type, subalgo_name, ...)                                                            \
  subalgo_type NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name),                                                                       \
  NMFD_ALGO_HIERARCHY_PARAMS_LIST_3(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_LIST_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_PARAMS_LIST_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_LIST(...) NMFD_ALGO_HIERARCHY_PARAMS_LIST_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_CC_1(subalgo_type, subalgo_name, ...) subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name))
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_2(subalgo_type, subalgo_name, ...)                                                              \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_1(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_3(subalgo_type, subalgo_name, ...)                                                              \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_2(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_4(subalgo_type, subalgo_name, ...)                                                              \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_3(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_5(subalgo_type, subalgo_name, ...)                                                              \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_4(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_6(subalgo_type, subalgo_name, ...)                                                              \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_5(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_7(subalgo_type, subalgo_name, ...)                                                              \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_6(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_8(subalgo_type, subalgo_name, ...)                                                              \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_7(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_9(subalgo_type, subalgo_name, ...)                                                              \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_8(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_10(subalgo_type, subalgo_name, ...)                                                             \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_9(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_11(subalgo_type, subalgo_name, ...)                                                             \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_10(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_12(subalgo_type, subalgo_name, ...)                                                             \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_11(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_13(subalgo_type, subalgo_name, ...)                                                             \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_12(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_14(subalgo_type, subalgo_name, ...)                                                             \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_13(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_15(subalgo_type, subalgo_name, ...)                                                             \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_14(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_16(subalgo_type, subalgo_name, ...)                                                             \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_15(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_17(subalgo_type, subalgo_name, ...)                                                             \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_16(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC_18(subalgo_type, subalgo_name, ...)                                                             \
  subalgo_name(NMFD_ALGO_HIERARCHY_CONCATENATE(_,subalgo_name)),                                                                      \
  NMFD_ALGO_HIERARCHY_PARAMS_CC_17(__VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS_CC_(N, ...) NMFD_ALGO_HIERARCHY_CONCATENATE(NMFD_ALGO_HIERARCHY_PARAMS_CC_, N)(__VA_ARGS__)
#define NMFD_ALGO_HIERARCHY_PARAMS_CC(...) NMFD_ALGO_HIERARCHY_PARAMS_CC_(NMFD_ALGO_HIERARCHY_NARG(__VA_ARGS__), __VA_ARGS__)

#define NMFD_ALGO_HIERARCHY_PARAMS(func_name, ...)                                                                       \
        NMFD_ALGO_HIERARCHY_PARAMS_DEF(__VA_ARGS__)                                                                           \
        func_name(NMFD_ALGO_HIERARCHY_PARAMS_LIST(__VA_ARGS__)) : NMFD_ALGO_HIERARCHY_PARAMS_CC(__VA_ARGS__) {}
*/
#endif
