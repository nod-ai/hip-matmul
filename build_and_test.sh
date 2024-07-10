#!/bin/bash

set -eux

if [[ ! -d build ]]
then
  mkdir build
fi

executable=build/hip-matmul

hipcc matmul.hip -std=c++17 -Wall -Wextra -O3 -o "${executable}" -save-temps=obj

"${executable}"
