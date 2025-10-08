#!/bin/bash

cd solvers
make all -B
#make test
cd ..

cd detail
make all -B
#make test
cd ..