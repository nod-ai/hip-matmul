// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Self-contained program proving that __builtin_amdgcn_mfma_f32_16x16x16f16
// performs internal reassociation, by computing a 16x16 matmul and checking
// the element [0, 0] in the result matrix against a reference implementation
// not reassociating.
//
// Two reference results are computed: using std::fmaf, and using plain add and
// mul. The results are much closer with std::fmaf, but still not exactly the
// same. This suggests that the GPU instruction internally really is doing a
// fused multiply-add, but reassociating.

/* Compile and run:

hipcc mfma_reassociating.hip -Wall -Wextra -O3 -o ~/mfma_reassociating &&
~/mfma_reassociating

*/

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

void hip_check_impl(hipError_t hip_error_code, const char *condstr,
                    const char *file, int line) {
  if (hip_error_code != hipSuccess) {
    fprintf(stderr, "HIP Error \"%s\" produced by `%s` at %s:%d\n",
            hipGetErrorString(hip_error_code), condstr, file, line);
    exit(EXIT_FAILURE);
  }
}

#define HIP_CHECK(expr) hip_check_impl(expr, #expr, __FILE__, __LINE__)

using float16x4_t =
    __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;

__global__ void device_kernel(const float16x4_t *a, const float16x4_t *b,
                              floatx4_t *c) {
  int tid = threadIdx.x;
  c[tid] =
      __builtin_amdgcn_mfma_f32_16x16x16f16(a[tid], b[tid], c[tid], 0, 0, 0);
}

int main() {
  std::vector<_Float16> A_host_data(16 * 16, static_cast<_Float16>(0.f));
  std::vector<_Float16> B_host_data(16 * 16, static_cast<_Float16>(0.f));
  // This lambda encodes just what we need to know about this MFMA instruction's
  // layout - we just use row 0, for both A and B thanks to MFMA instructions
  // having transposed B.
  auto row0_layout = [](int i) { return 64 * (i / 4) + (i % 4); };
  for (int i = 0; i < 16; ++i) {
    A_host_data[row0_layout(i)] =
        static_cast<_Float16>(static_cast<float>(0 + i) / 7);
    B_host_data[row0_layout(i)] =
        static_cast<_Float16>(static_cast<float>(0 + i) / 7);
  }
  std::vector<float> C_host_data(16 * 16, 0.f);
  float16x4_t *A_device_buffer{};
  float16x4_t *B_device_buffer{};
  floatx4_t *C_device_buffer{};
  int A_bytes = A_host_data.size() * sizeof A_host_data[0];
  int B_bytes = B_host_data.size() * sizeof B_host_data[0];
  int C_bytes = C_host_data.size() * sizeof C_host_data[0];

  HIP_CHECK(hipMalloc(&A_device_buffer, A_bytes));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_bytes));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_bytes));

  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_bytes,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_bytes,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_bytes,
                      hipMemcpyHostToDevice));

  const dim3 grid_dim(1, 1, 1);
  const dim3 block_dim(64, 1, 1);

  device_kernel<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
      A_device_buffer, B_device_buffer, C_device_buffer);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_host_data.data(), C_device_buffer, C_host_data.size(),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));

  float reference_C00_fma = 0.f;
  float reference_C00_nonfma = 0.f;
  for (int i = 0; i < 16; ++i) {
    reference_C00_fma =
        std::fmaf(A_host_data[row0_layout(i)], B_host_data[row0_layout(i)],
                  reference_C00_fma);
    reference_C00_nonfma +=
        A_host_data[row0_layout(i)] * B_host_data[row0_layout(i)];
  }
  std::printf("GPU result:                            %.8g\n", C_host_data[0]);
  std::printf("Non-reassociating reference (using FMA):    %.8g\n",
              reference_C00_fma);
  std::printf("Difference:                         %g\n",
              C_host_data[0] - reference_C00_fma);
  std::printf("Non-reassociating reference (not using FMA):    %.8g\n",
              reference_C00_nonfma);
  std::printf("Difference:                         %g\n",
              C_host_data[0] - reference_C00_nonfma);
}
