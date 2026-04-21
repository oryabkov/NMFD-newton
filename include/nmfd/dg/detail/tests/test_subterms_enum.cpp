
#include <iostream>
#include <scfd/dg/detail/polynom_indexing.h>

using namespace scfd::dg::detail;

static const int                     poly_deg = 2;
static const int                     vars_n = 5;

template<int Deg>
void enum_subterms(int deg, int mon[Deg])
{
    for (int sub_term_mask = 1;sub_term_mask < (1 << (Deg-1));++sub_term_mask) {
        bool    pass_sub_term = false;

        if ( (deg == 0) || (!(sub_term_mask < (1 << (deg-1)))) ) pass_sub_term = true;
        if (!monom_sub_term_enum<int,Deg>(deg, mon, sub_term_mask)) pass_sub_term = true;
        
        int i_x_term, i_y_term;

        if (!pass_sub_term) {
            i_x_term = monom_to_polynom_index<int,Deg>(vars_n, deg, mon,  sub_term_mask);
            i_y_term = monom_to_polynom_index<int,Deg>(vars_n, deg, mon, ~sub_term_mask);
        } else {
            i_x_term = -1;
            i_y_term = -1;
        }

        if (i_x_term != -1) {
            int     exp[vars_n];
            polynom_index_to_exponents<int,Deg,vars_n>(i_x_term, exp);
            for (int j = 0;j < vars_n;++j) std::cout << exp[j] << " ";
            std::cout << " : ";
            polynom_index_to_exponents<int,Deg,vars_n>(i_y_term, exp);
            for (int j = 0;j < vars_n;++j) std::cout << exp[j] << " ";
            std::cout << std::endl;
            if (i_x_term != i_y_term) {
                polynom_index_to_exponents<int,Deg,vars_n>(i_y_term, exp);
                for (int j = 0;j < vars_n;++j) std::cout << exp[j] << " ";
                std::cout << " : ";
                polynom_index_to_exponents<int,Deg,vars_n>(i_x_term, exp);
                for (int j = 0;j < vars_n;++j) std::cout << exp[j] << " ";
                std::cout << std::endl;
            }
        }
    }
}

int main(int argc, char const *argv[])
{
    for (int i = 0;i < polylen<int>(vars_n, poly_deg);++i) {
        std::cout << "polynom_index = " << i << "; ";
    //for (int i = 11;i < 12;++i) {
        int     deg;
        int     mon[poly_deg];
        polynom_index_to_monom<int,poly_deg>(vars_n, i, deg, mon);
        //std::cout << mon[0] << std::endl;

        int     exp[vars_n];
        monom_to_exponents<int,poly_deg,vars_n>(deg, mon, exp);

        for (int j = 0;j < vars_n;++j) std::cout << exp[j] << " ";

        std::cout << "subterms:" << std::endl;
        enum_subterms<poly_deg>(deg, mon);
    }

    return 0;
}