#include <iostream>
#include <cmath>

#include "nmfd/operations/rect_vector_space.h"
#include <scfd/backend/serial_cpu.h>
#include <scfd/communication/mpi_wrap.h>

constexpr int dim        = 3;
constexpr int tensor_dim = 2;
using scalar             = double;

using comm_info_t  = scfd::communication::mpi_comm_info;
using vector_space = nmfd::rect_vector_space<scalar, dim, tensor_dim, scfd::backend::serial_cpu, comm_info_t>;
using vector_t      = typename vector_space::vector_type;
using vector_view_t = typename vector_t::view_type;

int main(int argc, char* argv[])
{
    scfd::communication::mpi_wrap comm(argc, argv);
    comm_info_t comm_world = comm.comm_world();

    const int np   = comm_world.num_procs;
    const int rank = comm_world.myid;

    vector_space vec_space({10, 10, 10}, comm_world);

    vector_t x;
    vec_space.init_vector(x);
    vec_space.start_use_vector(x);

    vec_space.assign_scalar(2.0, x);   // constant field == 2 everywhere

    const long   local_elems  = 10L * 10 * 10 * tensor_dim;
    const long   global_elems = local_elems * np;

    const scalar got_norm_l2   = vec_space.norm_l2(x);            // expect 2.0 (size-independent)
    const scalar got_sum       = vec_space.sum(x);               // expect 2 * global_elems
    const scalar got_dot       = vec_space.scalar_prod(x, x);    // expect 4 * global_elems
    const scalar got_norm_sq   = vec_space.norm_sq(x);           // expect 4 * global_elems
    const scalar got_norm      = vec_space.norm(x);              // expect sqrt(4*global_elems)
    const scalar got_norm2_sq  = vec_space.norm2_sq(x);          // expect 4.0 (== 2^2)

    const scalar exp_norm_l2   = 2.0;
    const scalar exp_sum       = 2.0 * global_elems;
    const scalar exp_dot       = 4.0 * global_elems;
    const scalar exp_norm_sq   = 4.0 * global_elems;
    const scalar exp_norm      = std::sqrt(4.0 * global_elems);
    const scalar exp_norm2_sq  = 4.0;

    auto close = [](scalar a, scalar b) {
        scalar denom = std::max(scalar(1), std::abs(b));
        return std::abs(a - b) / denom < scalar(1e-12);
    };

    bool ok = close(got_norm_l2, exp_norm_l2) && close(got_sum, exp_sum) &&
              close(got_dot, exp_dot) && close(got_norm_sq, exp_norm_sq) &&
              close(got_norm, exp_norm) && close(got_norm2_sq, exp_norm2_sq);

    if (rank == 0)
    {
        std::cout << "num_procs   = " << np << std::endl;
        std::cout << "norm_l2(x)  = " << got_norm_l2  << " (expect " << exp_norm_l2  << ")" << std::endl;
        std::cout << "sum(x)      = " << got_sum      << " (expect " << exp_sum      << ")" << std::endl;
        std::cout << "dot(x,x)    = " << got_dot      << " (expect " << exp_dot      << ")" << std::endl;
        std::cout << "norm_sq(x)  = " << got_norm_sq  << " (expect " << exp_norm_sq  << ")" << std::endl;
        std::cout << "norm(x)     = " << got_norm     << " (expect " << exp_norm     << ")" << std::endl;
        std::cout << "norm2_sq(x) = " << got_norm2_sq << " (expect " << exp_norm2_sq << ")" << std::endl;
        std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
    }

    vec_space.free_vector(x);
    return ok ? 0 : 1;
}
