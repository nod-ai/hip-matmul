# Toying with matmuls in HIP

Trying to put together a self-contained testbed for data-tiled matmul kernels
using MFMA intrinsics.

To build and test, just run this script,

```
./build_and_test.sh
```

but it's really just a single `hipcc` command line to build, and then runs the
resulting executable.
