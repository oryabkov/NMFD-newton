import sys
import re
import pandas as pd

def main(argv):
    print("Argument List: " + str(sys.argv))
    basis2d = {
        "*x^2*y":[],
        "*x*y^2":[],
        "*x*y":[],
        "*y^2":[],
        "*y^3":[],
        "*x^3":[],      
        "*x^2":[],
        "*y":[],
        "*x":[],
        "*1":[]
        }
    basis2d_sorted = {
        "*1":[],
        "*x":[],
        "*y":[],
        "*x^2":[],
        "*x*y":[],
        "*y^2":[],
        "*x^3":[],
        "*x^2*y":[],
        "*x*y^2":[],
        "*y^3":[] 
        }    
    basis3d = {
        "*x^3":[],
        "*y^3":[],
        "*z^3":[],
        "*x^2*y":[],
        "*x^2*z":[],
        "*x*y^2":[],
        "*x*y*z":[],
        "*x*z^2":[],
        "*y^2*z":[],
        "*y*z^2":[],
        "*x*y":[],
        "*x*z":[],
        "*y*z":[],
        "*x^2":[],
        "*y^2":[],
        "*z^2":[],
        "*x":[],
        "*y":[],
        "*z":[],
        "*1":[]
        }    
    basis3d_sorted = {
        "*1":[],
        "*x":[],
        "*y":[],
        "*z":[],
        "*x^2":[],
        "*x*y":[],
        "*x*z":[],
        "*y^2":[],
        "*y*z":[],
        "*z^2":[],
        "*x^3":[],
        "*x^2*y":[],
        "*x^2*z":[],
        "*x*y^2":[],
        "*x*y*z":[],
        "*x*z^2":[],
        "*y^3":[],
        "*y^2*z":[],
        "*y*z^2":[],
        "*z^3":[]
        }        
    try:
        dim = int(sys.argv[1])
        file_name = sys.argv[2]
    except IndexError:
        print("\n usage: "+sys.argv[0] + " dim file_name; dim - problem dimension, file_name - file from the Wolfram Mathematica\n")
        sys.exit()
    
    print(dim)
    if dim == 2:
        basis = basis2d
        basis_sorted =basis2d_sorted
    elif dim == 3:
        basis = basis3d
        basis_sorted = basis3d_sorted
    else:
        print("\n incorrect problem dimension. Only 2 or 3 are supported")
        sys.exit()
    
    print("dim = " + str(dim) + "; file_name = " + file_name + "\n")
    with open(file_name) as file:
        main_str = file.read()
    
    # prepare the string
    main_str = main_str.replace('\n','')
    main_str = main_str.replace('\r','')
    main_str = main_str.replace('{{','{')
    main_str = main_str.replace('}}','}')
    main_str = main_str.replace('Sqrt[','sqrt(')
    main_str = main_str.replace(']',')')
    split_str = re.findall("\{(.*?)\}", main_str) 
    data = []
    for spl in split_str:
        split_str_inside = spl.split(',')
        data.append(split_str_inside)
        
    # append all monomials, all but constants
    for j in range(len(data)):
        for key in basis.keys():
            for k in range(len(data[j])):
                val = data[j][k]
                if key in val:
                    basis[key].append({j: val})
                    data[j][k]=''
                
    # append the remaining constants
    for j in range(len(data)):
        for k in range(len(data[j])):
            if data[j][k] != '':
                basis['*1'].append({j: data[j][k]})


    # create a table of monomials
    table = pd.DataFrame( columns=basis_sorted.keys(), index = range(len(basis)) )
    
    for key in basis:
        pairs = basis[key]
        for pair in pairs:
            for key_l in pair:
                table[key][key_l] = pair[key_l]
            
    # save            
    name = file_name.split('.')
    filename_tmpl = name[0].split('/')
    file_name_csv = filename_tmpl[-1]+".csv"
    table.to_csv(file_name_csv)


if __name__ == '__main__':
    main(sys.argv[1:])