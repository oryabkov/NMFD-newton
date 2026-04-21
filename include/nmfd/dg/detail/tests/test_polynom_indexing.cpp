
#include <iostream>
#include <scfd/dg/detail/polynom_indexing.h>

using namespace scfd::dg::detail;

static const int                     deg = 2;
static const int                     vars_n = 5;

int main(int argc, char const *argv[])
{
    

    for (int i = 0;i < polylen<int>(vars_n, deg);++i) {
        std::cout << "polynom_index = " << i << "; ";
    //for (int i = 11;i < 12;++i) {
        int     curr_deg;
        int     mon[deg];
        polynom_index_to_monom<int,deg>(vars_n, i, curr_deg, mon);

        for (int j = 0;j < curr_deg;++j) std::cout << mon[j] << " ";
        std::cout << "; ";
        //std::cout << mon[0] << std::endl;

        int     exp[vars_n];
        monom_to_exponents<int,deg,vars_n>(curr_deg, mon, exp);

        for (int j = 0;j < vars_n;++j) std::cout << exp[j] << " ";

        std::cout << "; monom_to_polynom_index = " << monom_to_polynom_index<int,deg>(vars_n, curr_deg, mon, (1 << deg)-1) << std::endl;
    }

    return 0;
}