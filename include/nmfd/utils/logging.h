#ifndef __NMFD_UTILS_LOGGING_H__
#define __NMFD_UTILS_LOGGING_H__

#ifdef SCFD_BACKEND_ENABLE_MPI
#include <scfd/utils/log_mpi.h>
using current_log = scfd::utils::log_mpi;
#else
#include <scfd/utils/log_std.h>
using current_log = scfd::utils::log_std;
#endif

#endif // __NMFD_UTILS_LOGGING_H__
