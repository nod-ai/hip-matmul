// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>

#include <random>
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

struct Layout {
  int rows0, cols0;
  int rows1, cols1;
  int rows2, cols2;
  int rows3, cols3;
};

__device__ __host__ int flatsize(const Layout &layout) {
  return layout.rows0 * layout.cols0 * layout.rows1 * layout.cols1 *
         layout.rows2 * layout.cols2 * layout.rows3 * layout.cols3;
}

__device__ __host__ int offset(const Layout &layout, int r0, int c0, int r1,
                               int c1, int r2, int c2, int r3, int c3) {
  return c3 +
         layout.cols3 *
             (r3 +
              layout.rows3 *
                  (c2 + layout.cols2 *
                            (r2 + layout.rows2 *
                                      (c1 + layout.cols1 *
                                                (r1 + layout.rows1 *
                                                          (c0 + layout.cols0 *
                                                                    r0))))));
}

__global__ void mmt8d(const float *A_data, const float *B_data, float *C_data,
                      int M0, int M1, int M2, int M3, int N0, int N1, int N2,
                      int N3, int K0, int K1, int K2, int K3) {
  int m0 = blockIdx.x;
  int n0 = blockIdx.y;
  int m1m2m3 = threadIdx.x;
  int n1n2n3 = threadIdx.y;
  int m1 = m1m2m3 / (M2 * M3);
  int n1 = n1n2n3 / (N2 * N3);
  int m2 = (m1m2m3 - m1 * M2 * M3) / M3;
  int n2 = (n1n2n3 - n1 * N2 * N3) / N3;
  int m3 = m1m2m3 % M3;
  int n3 = n1n2n3 % N3;

  Layout A_layout{M0, K0, M1, K1, M2, K2, M3, K3};
  Layout B_layout{N0, K0, N1, K1, N2, K2, N3, K3};
  Layout C_layout{M0, N0, M1, N1, M2, N2, M3, N3};

  float acc = 0.f;
  for (int k0 = 0; k0 < K0; ++k0) {
    for (int k1 = 0; k1 < K1; ++k1) {
      for (int k2 = 0; k2 < K2; ++k2) {
        for (int k3 = 0; k3 < K3; ++k3) {
          float a = A_data[offset(A_layout, m0, k0, m1, k1, m2, k2, m3, k3)];
          float b = B_data[offset(B_layout, n0, k0, n1, k1, n2, k2, n3, k3)];
          acc += a * b;
        }
      }
    }
  }

  C_data[offset(C_layout, m0, n0, m1, n1, m2, n2, m3, n3)] = acc;
}

std::vector<float> makeRandomFloatVector(int size, std::minstd_rand &r) {
  std::vector<float> v(size);
  for (float &x : v) {
    // Generate small integers in [-2, +2] so products are in [-4, +4] so
    // accumulators are in [-4K, +4K] for accumulation depth K so they're
    // exactly representable, float rounding is exact and we don't need
    // fuzzy compares.
    x = static_cast<float>(static_cast<int>((r() % 5)) - 2);
  }
  return v;
}

void checkMmt8dResults(const float *A_data, const float *B_data,
                       const float *C_data, int M0, int M1, int M2, int M3,
                       int N0, int N1, int N2, int N3, int K0, int K1, int K2,
                       int K3) {
  Layout A_layout{M0, K0, M1, K1, M2, K2, M3, K3};
  Layout B_layout{N0, K0, N1, K1, N2, K2, N3, K3};
  Layout C_layout{M0, N0, M1, N1, M2, N2, M3, N3};
  for (int m0 = 0; m0 < M0; ++m0) {
    for (int n0 = 0; n0 < N0; ++n0) {
      for (int m1 = 0; m1 < M1; ++m1) {
        for (int n1 = 0; n1 < N1; ++n1) {
          for (int m2 = 0; m2 < M2; ++m2) {
            for (int n2 = 0; n2 < N2; ++n2) {
              for (int m3 = 0; m3 < M3; ++m3) {
                for (int n3 = 0; n3 < N3; ++n3) {
                  float acc = 0.f;
                  for (int k0 = 0; k0 < K0; ++k0) {
                    for (int k1 = 0; k1 < K1; ++k1) {
                      for (int k2 = 0; k2 < K2; ++k2) {
                        for (int k3 = 0; k3 < K3; ++k3) {
                          float a = A_data[offset(A_layout, m0, k0, m1, k1, m2,
                                                  k2, m3, k3)];
                          float b = B_data[offset(B_layout, n0, k0, n1, k1, n2,
                                                  k2, n3, k3)];
                          acc += a * b;
                        }
                      }
                    }
                  }
                  float expected = acc;
                  float actual =
                      C_data[offset(C_layout, m0, n0, m1, n1, m2, n2, m3, n3)];
                  if (actual != expected) {
                    fprintf(stderr,
                            "matmul numerical error: actual(%g) != "
                            "expected(%g), at m0=%d n0=%d m1=%d n1=%d m2=%d "
                            "n2=%d, m3=%d n3=%d, at %s:%d",
                            actual, expected, m0, n0, m1, n1, m2, n2, m3, n3,
                            __FILE__, __LINE__);
                    abort();
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

int main() {
  auto getIntEnvVar = [](const char *name, int default_val) {
    const char *env = std::getenv(name);
    return env ? std::stoi(env) : default_val;
  };

  const int M0 = getIntEnvVar("M0", 5);
  const int M1 = getIntEnvVar("M1", 16);
  const int M2 = getIntEnvVar("M2", 1);
  const int M3 = getIntEnvVar("M3", 1);

  const int N0 = getIntEnvVar("N0", 5);
  const int N1 = getIntEnvVar("N1", 16);
  const int N2 = getIntEnvVar("N2", 1);
  const int N3 = getIntEnvVar("N3", 1);

  const int K0 = getIntEnvVar("K0", 100);
  const int K1 = getIntEnvVar("K1", 1);
  const int K2 = getIntEnvVar("K2", 1);
  const int K3 = getIntEnvVar("K3", 1);

  Layout A_layout{M0, K0, M1, K1, M2, K2, M3, K3};
  Layout B_layout{N0, K0, N1, K1, N2, K2, N3, K3};
  Layout C_layout{M0, N0, M1, N1, M2, N2, M3, N3};

  std::minstd_rand random_engine;
  std::vector<float> A_host_data =
      makeRandomFloatVector(flatsize(A_layout), random_engine);
  std::vector<float> B_host_data =
      makeRandomFloatVector(flatsize(B_layout), random_engine);
  std::vector<float> C_host_data(flatsize(C_layout));

  const size_t A_bytes = sizeof(float) * A_host_data.size();
  const size_t B_bytes = sizeof(float) * B_host_data.size();
  const size_t C_bytes = sizeof(float) * C_host_data.size();
  float *A_device_buffer{};
  float *B_device_buffer{};
  float *C_device_buffer{};
  HIP_CHECK(hipMalloc(&A_device_buffer, A_bytes));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_bytes));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_bytes));

  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_bytes,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_bytes,
                      hipMemcpyHostToDevice));

  const dim3 grid_dim(M0, N0);
  const dim3 block_dim(M1 * M2 * M3, M1 * M2 * M3);

  // Launch matrix multiplication kernel.
  printf("mmt8d: M0=%d, M1=%d, M2=%d, M3=%d, N0=%d, N1=%d, "
         "N2=%d, N3=%d, K0=%d, K1=%d, K2=%d, K3=%d\n",
         M0, M1, M2, M3, N0, N1, N2, N3, K0, K1, K2, K3);
  mmt8d<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
      A_device_buffer, B_device_buffer, C_device_buffer, M0, M1, M2, M3, N0, N1,
      N2, N3, K0, K1, K2, K3);
  // Check if the kernel launch was successful.
  HIP_CHECK(hipGetLastError());

  // Copy the resulting matrix to the host. This call synchronizes with the
  // host.
  HIP_CHECK(hipMemcpy(C_host_data.data(), C_device_buffer, C_bytes,
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));

  checkMmt8dResults(A_host_data.data(), B_host_data.data(), C_host_data.data(),
                    M0, M1, M2, M3, N0, N1, N2, N3, K0, K1, K2, K3);
}

