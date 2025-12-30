#include <utility>
#include <iostream>

#include "nmfd/operations/rect_vector_space.h"
#include <scfd/backend/serial_cpu.h>

constexpr int dim        = 3;
constexpr int tensor_dim = 2;
using scalar             = double;

using vector_space = nmfd::rect_vector_space<scalar,/*dim=*/3,/*tensor_dim=*/2, scfd::backend::serial_cpu>;

using vector_t      = typename vector_space::vector_type;
using multivector_t = typename vector_space::multivector_type;
using vector_view_t = typename vector_t::view_type;
using ordinal_type = typename vector_space::ordinal_type;

int main()
{
    vector_t x, y;

    vector_space vec_space({10,10,10});

    vec_space.init_vector(x);
    vec_space.init_vector(y);

    vec_space.start_use_vector(x);
    vec_space.start_use_vector(y);


    // ----- Vector operations -----
    std::cout << "----- Vector operations -----" << std::endl;

    vec_space.assign_scalar(1.f, x);               //x=1
    vec_space.scale(2.f, x);                       //x=2
    vec_space.assign(x, y);                        //y=x
    vec_space.add_mul_scalar(3.f, -1.f, y);        //y=-2*1+3=1
    vec_space.add_lin_comb(4.f, x, -3.f, y);       //y=2*4-3*1=5
    vec_space.assign_lin_comb(2.f, y, x);          //x=5*2=10
    vector_view_t x_view(x), y_view(y);

    std::cout << "size        = (" << vec_space.get_size()[0] << ", " << vec_space.get_size()[1] << ", " << vec_space.get_size()[2] << ")" << std::endl;

    std::cout << "x(0,0,0)[0] = " << x(0, 0, 0, 0) << std::endl;
    std::cout << "x(0,0,0)[1] = " << x(0, 0, 0, 1) << std::endl;

    std::cout << "y(0,0,0)[0] = " << y(0, 0, 0, 0) << std::endl;
    std::cout << "y(0,0,0)[1] = " << y(0, 0, 0, 1) << std::endl;

    std::cout << "|x|         = " << vec_space.norm2(x) << std::endl;
    std::cout << "|y|         = " << vec_space.norm2(y) << std::endl;

    x_view.release();
    y_view.release();

    std::cout << "dot(x, y)   = " << vec_space.scalar_prod(x,y) << std::endl;
    std::cout << "sum(x)      = " << vec_space.sum(x) << std::endl;
    std::cout << "norm(x)     = " << vec_space.norm(x) << std::endl;
    std::cout << "norm_sq(x)  = " << vec_space.norm_sq(x) << std::endl;
    std::cout << "norm2(x)    = " << vec_space.norm2(x) << std::endl;
    std::cout << "norm2_sq(x) = " << vec_space.norm2_sq(x) << std::endl;

    // ----- Multivector operations -----
    std::cout << std::endl << "----- Multivector operations -----" << std::endl;

    multivector_t mx, my;
    ordinal_type m = 2;

    vec_space.init_multivector(mx, m);
    vec_space.init_multivector(my, m);

    vec_space.start_use_multivector(mx, m);
    vec_space.start_use_multivector(my, m);

    vec_space.assign_scalar(1.f, mx[0]);
    vec_space.assign_scalar(2.f, mx[1]);
    std::cout << "mx[0](0, 0, 0)[0] = " << mx[0](0, 0, 0, 0) << std::endl;
    std::cout << "mx[1](0, 0, 0)[1] = " << mx[1](0, 0, 0, 1) << std::endl;

    vec_space.assign_scalar(3.f, my[0]);
    vec_space.assign_scalar(4.f, my[1]);
    std::cout << "my[0](0, 0, 0)[0] = " << my[0](0, 0, 0, 0) << std::endl;
    std::cout << "my[1](0, 0, 0)[1] = " << my[1](0, 0, 0, 1) << std::endl;

    std::cout << "scalar_prod(mx[1], y)    = " << vec_space.scalar_prod(mx, m, 1, y) << std::endl;
    std::cout << "scalar_prod_l2(mx[1], y) = " << vec_space.scalar_prod_l2(mx, m, 1, y) << std::endl;

    vec_space.free_vector(x);
    vec_space.free_vector(y);
    vec_space.free_multivector(mx, m);
    vec_space.free_multivector(my, m);

    return 0;
}
