#!/usr/bin/env python3

import sys

def csv_to_code(in_fn,matrix_name):
    # Using readlines()
    file1 = open(in_fn, 'r')
    Lines = file1.readlines()
     
    count = 0
    res_lines = []
    # Strips the newline character
    for line in Lines:
        l = line.strip()
        if count == 0:
            monom_names = l.split(',')
            monom_names.pop(0)
            #print(monom_names)
            count += 1
            continue
        monom_coeffs = l.split(',')
        monom_coeffs.pop(0)

        for n in range(len(monom_coeffs)):
            if monom_coeffs[n] == "":
                monom_coeffs[n] = "0"
            monom_coeffs[n] = monom_coeffs[n].replace(monom_names[n], '')
            monom_coeffs[n] = monom_coeffs[n].replace('sqrt', 'std::sqrt')
            monom_coeffs[n] = 'T(' + monom_coeffs[n] + ')'

        res_str = '        {'+','.join(monom_coeffs)+'}'
        #print(res_str)

        res_lines.append(res_str)

        #print("Line{}: {}".format(count, line.strip()))
        count += 1

    res_code = 'T '+matrix_name + "[{}][{}] = ".format(count-1,count-1) + '\n    {\n'+',\n'.join(res_lines)+'\n    };\n'

    return res_code
