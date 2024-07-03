# Toying with matmuls in HIP

Trying to put together a self-contained testbed for data-tiled matmul kernels
using MFMA intrinsics.

```
hipcc matmul.hip -o /tmp/matmul && /tmp/matmul
```
