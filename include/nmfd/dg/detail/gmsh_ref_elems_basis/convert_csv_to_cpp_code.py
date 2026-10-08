#!/usr/bin/env python3

import sys
from csv_to_code import csv_to_code

if (len(sys.argv) < 3):
    print("Usage: {} INPUT_CSV MATRIX_NAME".format(sys.argv[0]))
    sys.exit()

in_fn = sys.argv[1]
matrix_name = sys.argv[2]

res_code = csv_to_code(in_fn,matrix_name)

print(res_code)
