
#include <iostream>
#include <string>
#include <scfd/dg/detail/polynom_indexing.h>

using namespace scfd::dg::detail;

#ifndef DEGREE
#define DEGREE 3
#endif

static const int deg = DEGREE;

std::string var_name(int j)
{
    switch (j)
    {
        case 0: 
            return "x";
        case 1: 
            return "y";
        case 2: 
            return "z";
        default:
            throw std::logic_error("var_name: j > 3");
    }
}

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        std::cout << "USAGE: " << argv[0] << " VARS_N" << std::endl;
        std::cout << "static polynom degree: " << deg << std::endl;
        return 0;
    }

    int     vars_n = std::stoi(argv[1]);

    std::cout << "static polynom degree: " << deg << std::endl;
    std::cout << "variables number: " << vars_n << std::endl;

    for (int i = 0;i < polylen<int>(vars_n, deg);++i) 
    {
        std::cout << "monom_index = " << i << "; ";
        int     curr_deg;
        int     mon[deg];
        polynom_index_to_monom<int,deg>(vars_n, i, curr_deg, mon);

        std::cout << "monom variables: ";
        if (curr_deg == 0)
            std::cout << "<empty>";
        else
            for (int j = 0;j < curr_deg;++j) std::cout << mon[j] << " ";
        std::cout << std::endl;
    }

    if (vars_n <= 3)
    {
        for (int i = 0;i < polylen<int>(vars_n, deg);++i) 
        {
            std::cout << "monom_index = " << i << "; ";
            int     exp[vars_n];            
            polynom_index_to_exponents<int,deg>(vars_n, i, exp);

            std::cout << "monom: ";
            bool has_something = false;
            for (int j = 0;j < vars_n;++j) 
            {
                if (exp[j] > 0)
                {
                    has_something = true;
                    std::cout << var_name(j);
                    if (exp[j] > 1)
                        std::cout << "^" << exp[j];
                    std::cout << " ";
                }
            }
            if (!has_something)
            {
                std::cout << "1.";
            }
            std::cout << std::endl;
        }

    }

    return 0;
}