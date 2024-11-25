#!/bin/bash

mkdir -p build
cd build
cmake ..
make
cd ..
export PYTHONPATH=$(pwd)/build:$PYTHONPATH

echo "Finished build"