#!/usr/bin/env python3

import os
import sys
import glob

from csv_to_code import csv_to_code

if (len(sys.argv) < 2):
    print("Usage: {} DIR_NAME".format(sys.argv[0]))
    sys.exit()

dir_fn = sys.argv[1]

os.chdir('./'+dir_fn)

csv_files_list = glob.glob('./*.csv')

res_code = ''

for fn in csv_files_list:
    matrix_name = fn[2:-4]
    #print(fn)
    #print(matrix_name)
    res_code_1 = csv_to_code(fn,matrix_name)

    res_code += res_code_1

print(res_code)