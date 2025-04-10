#!/bin/bash

set -euxo pipefail

if [[ ! -d build ]]
then
  mkdir build
fi

declare -a KERNELS_TO_RUN=("matmul" "matvec")

if [ ! -z ${KERNEL+x} ]; then
  KERNELS_TO_RUN=("${KERNEL}")
fi

for kernel in "${KERNELS_TO_RUN[@]}" ; do
  echo "Build and test: ${kernel}"
  executable="build/hip-${kernel}"
  hipcc "${kernel}.hip" -std=c++20 -Wall -Wextra -O3 -o "${executable}" -save-temps=obj
  "${executable}"
done
