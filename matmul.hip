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

int getIntEnvVar(const char *name, int default_val) {
  const char *env = std::getenv(name);
  return env ? std::stoi(env) : default_val;
};

struct Matrix8dShape {
  int rows0, cols0;
  int rows1, cols1;
  int rows2, cols2;
  int rows3, cols3;
};

struct Mmt8dShape {
  int M0, N0, K0;
  int M1, N1, K1;
  int M2, N2, K2;
  int M3, N3, K3;
};

__device__ __host__ Matrix8dShape lhs(const Mmt8dShape &s) {
  return {s.M0, s.K0, s.M1, s.K1, s.M2, s.K2, s.M3, s.K3};
}

__device__ __host__ Matrix8dShape rhs(const Mmt8dShape &s) {
  return {s.N0, s.K0, s.N1, s.K1, s.N2, s.K2, s.N3, s.K3};
}

__device__ __host__ Matrix8dShape acc(const Mmt8dShape &s) {
  return {s.M0, s.N0, s.M1, s.N1, s.M2, s.N2, s.M3, s.N3};
}

__device__ __host__ int flatsize(const Matrix8dShape &s) {
  return s.rows0 * s.cols0 * s.rows1 * s.cols1 * s.rows2 * s.cols2 * s.rows3 *
         s.cols3;
}

__device__ __host__ int offset(const Matrix8dShape &s, int r0, int c0, int r1,
                               int c1, int r2, int c2, int r3, int c3) {
  return c3 +
         s.cols3 *
             (r3 +
              s.rows3 *
                  (c2 +
                   s.cols2 *
                       (r2 +
                        s.rows2 *
                            (c1 +
                             s.cols1 * (r1 + s.rows1 * (c0 + s.cols0 * r0))))));
}

typedef void (*mmt8d_func_t)(const float *, const float *, float *,
                             const Mmt8dShape &);

__global__ void mmt8d_generic(const float *A_data, const float *B_data,
                              float *C_data, const Mmt8dShape &s) {
  int m0 = blockIdx.x;
  int n0 = blockIdx.y;
  int m1m2m3 = threadIdx.x;
  int n1n2n3 = threadIdx.y;
  int m1 = m1m2m3 / (s.M2 * s.M3);
  int n1 = n1n2n3 / (s.N2 * s.N3);
  int m2 = (m1m2m3 - m1 * s.M2 * s.M3) / s.M3;
  int n2 = (n1n2n3 - n1 * s.N2 * s.N3) / s.N3;
  int m3 = m1m2m3 % s.M3;
  int n3 = n1n2n3 % s.N3;

  float c = 0.f;
  for (int k0 = 0; k0 < s.K0; ++k0) {
    for (int k1 = 0; k1 < s.K1; ++k1) {
      for (int k2 = 0; k2 < s.K2; ++k2) {
        for (int k3 = 0; k3 < s.K3; ++k3) {
          float a = A_data[offset(lhs(s), m0, k0, m1, k1, m2, k2, m3, k3)];
          float b = B_data[offset(rhs(s), n0, k0, n1, k1, n2, k2, n3, k3)];
          c += a * b;
        }
      }
    }
  }

  C_data[offset(acc(s), m0, n0, m1, n1, m2, n2, m3, n3)] = c;
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
                       const float *C_data, const Mmt8dShape &s) {
  // This reference code is slow. To make the checks not too slow on
  // large matmuls, we only check at most 16 steps along the major M0, N0 dims.
  int m0_step = std::max(1, s.M0 / 16);
  int n0_step = std::max(1, s.N0 / 16);
  for (int m0 = 0; m0 < s.M0; m0 += m0_step) {
    for (int n0 = 0; n0 < s.N0; n0 += n0_step) {
      for (int m1 = 0; m1 < s.M1; ++m1) {
        for (int n1 = 0; n1 < s.N1; ++n1) {
          for (int m2 = 0; m2 < s.M2; ++m2) {
            for (int n2 = 0; n2 < s.N2; ++n2) {
              for (int m3 = 0; m3 < s.M3; ++m3) {
                for (int n3 = 0; n3 < s.N3; ++n3) {
                  float c = 0.f;
                  for (int k0 = 0; k0 < s.K0; ++k0) {
                    for (int k1 = 0; k1 < s.K1; ++k1) {
                      for (int k2 = 0; k2 < s.K2; ++k2) {
                        for (int k3 = 0; k3 < s.K3; ++k3) {
                          float a = A_data[offset(lhs(s), m0, k0, m1, k1, m2,
                                                  k2, m3, k3)];
                          float b = B_data[offset(rhs(s), n0, k0, n1, k1, n2,
                                                  k2, n3, k3)];
                          c += a * b;
                        }
                      }
                    }
                  }
                  float expected = c;
                  float actual =
                      C_data[offset(acc(s), m0, n0, m1, n1, m2, n2, m3, n3)];
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

void test(const char *name, mmt8d_func_t mmt8d_func, const Mmt8dShape &s) {
  std::minstd_rand random_engine;
  std::vector<float> A_host_data =
      makeRandomFloatVector(flatsize(lhs(s)), random_engine);
  std::vector<float> B_host_data =
      makeRandomFloatVector(flatsize(rhs(s)), random_engine);
  std::vector<float> C_host_data(flatsize(acc(s)));

  const size_t A_bytes = sizeof(float) * A_host_data.size();
  const size_t B_bytes = sizeof(float) * B_host_data.size();
  const size_t C_bytes = sizeof(float) * C_host_data.size();
  float *A_device_buffer{};
  float *B_device_buffer{};
  float *C_device_buffer{};
  Mmt8dShape *shape_device_buffer{};
  HIP_CHECK(hipMalloc(&A_device_buffer, A_bytes));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_bytes));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_bytes));
  HIP_CHECK(hipMalloc(&shape_device_buffer, sizeof s));

  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_bytes,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_bytes,
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(shape_device_buffer, &s, sizeof s, hipMemcpyHostToDevice));

  const dim3 grid_dim(s.M0, s.N0);
  const dim3 block_dim(s.M1 * s.M2 * s.M3, s.M1 * s.M2 * s.M3);

  printf("%s: M0=%d, M1=%d, M2=%d, M3=%d, N0=%d, N1=%d, "
         "N2=%d, N3=%d, K0=%d, K1=%d, K2=%d, K3=%d\n",
         name, s.M0, s.M1, s.M2, s.M3, s.N0, s.N1, s.N2, s.N3, s.K0, s.K1, s.K2,
         s.K3);

  printf("  Checking correctness...\n");
  mmt8d_func<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
      A_device_buffer, B_device_buffer, C_device_buffer, *shape_device_buffer);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_host_data.data(), C_device_buffer, C_bytes,
                      hipMemcpyDeviceToHost));
  checkMmt8dResults(A_host_data.data(), B_host_data.data(), C_host_data.data(),
                    s);

  printf("  Benchmarking... ");
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  float kernel_ms{};
  float min_batch_ms = getIntEnvVar("BENCHMARK_MIN_MS", 100);
  int batch_size = 1;
  while (true) {
    HIP_CHECK(hipEventRecord(start, hipStreamDefault));
    for (int b = 0; b < batch_size; ++b) {
      mmt8d_func<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
          A_device_buffer, B_device_buffer, C_device_buffer,
          *shape_device_buffer);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));
    float batch_ms{};
    HIP_CHECK(hipEventElapsedTime(&batch_ms, start, stop));
    if (batch_ms >= min_batch_ms) {
      kernel_ms = batch_ms / batch_size;
      break;
    }
    if (batch_size > (1 << 20)) {
      fprintf(stderr, "Vacuous kernel? Only taking %g ms at batch_size=%d.\n",
              batch_ms, batch_size);
      abort();
    }
    batch_size *= 2;
  }
  float kernel_ops = 2.f * s.M0 * s.M1 * s.M2 * s.M3 * s.N0 * s.N1 * s.N2 *
                     s.N3 * s.K0 * s.K1 * s.K2 * s.K3;
  float kernel_ops_per_s = 1000.f * kernel_ops / kernel_ms;
  printf("    %.3g Tflop/s\n", 1.e-12f * kernel_ops_per_s);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
  HIP_CHECK(hipFree(shape_device_buffer));
}

void test(const char *name, mmt8d_func_t mmt8d_func, int M1, int N1, int K1,
          int M2, int N2, int K2, int M3, int N3, int K3) {
  int M = getIntEnvVar("M", 1024);
  int N = getIntEnvVar("N", 1024);
  int K = getIntEnvVar("K", 1024);

  Mmt8dShape s;
  s.M0 = M / (M1 * M2 * M3);
  s.N0 = N / (N1 * N2 * N3);
  s.K0 = K / (K1 * K2 * K3);
  s.M1 = M1;
  s.N1 = N1;
  s.K1 = K1;
  s.M2 = M2;
  s.N2 = N2;
  s.K2 = K2;
  s.M3 = M3;
  s.N3 = N3;
  s.K3 = K3;

  test(name, mmt8d_func, s);
}

#define TEST(kernel, ...) test(#kernel, kernel, __VA_ARGS__)

int main() {
  TEST(mmt8d_generic, /*M1=*/16, /*N1=*/16, /*K1=*/1, 1, 1, 1, 1, 1, 1);
}
