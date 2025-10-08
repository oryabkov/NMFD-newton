#!/bin/bash

cd solvers
make test
cd ..

cd detail
make test
cd ..