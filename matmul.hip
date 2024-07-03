// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>

#include <functional>
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

typedef int (*tile_layout_func_t)(int, int);

struct TiledMatrixShape {
  int rows_outer, cols_outer;
  int rows_tile0, cols_tile0;
  tile_layout_func_t tile_layout;
};

struct TiledMmtShape {
  int M_outer, N_outer, K_outer;
  int M_tile0, N_tile0, K_tile0;
  tile_layout_func_t A_tile_layout, B_tile_layout, C_tile_layout;
};

__device__ __host__ TiledMatrixShape A_shape(const TiledMmtShape &s) {
  return {s.M_outer, s.K_outer, s.M_tile0, s.K_tile0, s.A_tile_layout};
}

__device__ __host__ TiledMatrixShape B_shape(const TiledMmtShape &s) {
  return {s.N_outer, s.K_outer, s.N_tile0, s.K_tile0, s.B_tile_layout};
}

__device__ __host__ TiledMatrixShape C_shape(const TiledMmtShape &s) {
  return {s.M_outer, s.N_outer, s.M_tile0, s.N_tile0, s.C_tile_layout};
}

__device__ __host__ int flatsize(const TiledMatrixShape &s) {
  return s.rows_outer * s.cols_outer * s.rows_tile0 * s.cols_tile0;
}

__device__ __host__ int offset(const TiledMatrixShape &s, int r_outer,
                               int c_outer, int r_tile0, int c_tile0) {
  return s.tile_layout(r_tile0, c_tile0) +
         s.rows_tile0 * s.cols_tile0 * (c_outer + s.cols_outer * r_outer);
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
                       const float *C_data, const TiledMmtShape &s) {
  // This reference code is slow. To make the checks not too slow on
  // large matmuls, we only check at most 16 steps along the major M_outer,
  // N_outer dims.
  int m_outer_step = std::max(1, s.M_outer / 16);
  int n_outer_step = std::max(1, s.N_outer / 16);
  for (int m_outer = 0; m_outer < s.M_outer; m_outer += m_outer_step) {
    for (int n_outer = 0; n_outer < s.N_outer; n_outer += n_outer_step) {
      for (int m_tile0 = 0; m_tile0 < s.M_tile0; ++m_tile0) {
        for (int n_tile0 = 0; n_tile0 < s.N_tile0; ++n_tile0) {
          float c = 0.f;
          for (int k_outer = 0; k_outer < s.K_outer; ++k_outer) {
            for (int k_tile0 = 0; k_tile0 < s.K_tile0; ++k_tile0) {
              float a = A_data[offset(A_shape(s), m_outer, k_outer, m_tile0,
                                      k_tile0)];
              float b = B_data[offset(B_shape(s), n_outer, k_outer, n_tile0,
                                      k_tile0)];
              c += a * b;
            }
          }
          float expected = c;
          float actual =
              C_data[offset(C_shape(s), m_outer, n_outer, m_tile0, n_tile0)];
          if (actual != expected) {
            fprintf(stderr,
                    "matmul numerical error: actual(%g) != "
                    "expected(%g), at m_outer=%d n_outer=%d m_tile0=%d "
                    "n_tile0=%d, at %s:%d\n",
                    actual, expected, m_outer, n_outer, m_tile0, n_tile0,
                    __FILE__, __LINE__);
            abort();
          }
        }
      }
    }
  }
}

typedef void (*mmt_func_t)(const float *, const float *, float *, int, int,
                           int);

class MmtKernel {
public:
  virtual ~MmtKernel(){};
  virtual const char *name() const = 0;
  virtual int M_tile0() const = 0;
  virtual int N_tile0() const = 0;
  virtual int K_tile0() const = 0;
  virtual tile_layout_func_t A_tile_layout() const = 0;
  virtual tile_layout_func_t B_tile_layout() const = 0;
  virtual tile_layout_func_t C_tile_layout() const = 0;
  virtual int num_threads() const = 0;
  virtual mmt_func_t mmt_func() const = 0;
};

TiledMmtShape getTestShape(const MmtKernel &kernel) {
  int M = getIntEnvVar("M", 1024);
  int N = getIntEnvVar("N", 1024);
  int K = getIntEnvVar("K", 1024);
  TiledMmtShape s;
  s.M_outer = std::max(1, M / kernel.M_tile0());
  s.N_outer = std::max(1, N / kernel.N_tile0());
  s.K_outer = std::max(1, K / kernel.K_tile0());
  s.M_tile0 = kernel.M_tile0();
  s.N_tile0 = kernel.N_tile0();
  s.K_tile0 = kernel.K_tile0();
  s.A_tile_layout = kernel.A_tile_layout();
  s.B_tile_layout = kernel.B_tile_layout();
  s.C_tile_layout = kernel.C_tile_layout();
  return s;
}

void test(const MmtKernel &kernel) {
  TiledMmtShape s = getTestShape(kernel);
  std::minstd_rand random_engine;
  std::vector<float> A_host_data =
      makeRandomFloatVector(flatsize(A_shape(s)), random_engine);
  std::vector<float> B_host_data =
      makeRandomFloatVector(flatsize(B_shape(s)), random_engine);
  std::vector<float> C_host_data(flatsize(C_shape(s)));

  const size_t A_bytes = sizeof(float) * A_host_data.size();
  const size_t B_bytes = sizeof(float) * B_host_data.size();
  const size_t C_bytes = sizeof(float) * C_host_data.size();
  float *A_device_buffer{};
  float *B_device_buffer{};
  float *C_device_buffer{};
  TiledMmtShape *shape_device_buffer{};
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

  const dim3 grid_dim(s.M_outer, s.N_outer);
  const dim3 block_dim(kernel.num_threads());

  printf("%s: MxNxK outer=(%dx%dx%d) tile0=(%dx%dx%d) num_threads=%d\n",
         kernel.name(), s.M_outer, s.N_outer, s.K_outer, s.M_tile0, s.N_tile0,
         s.K_tile0, kernel.num_threads());

  printf("  Checking correctness... ");
  kernel.mmt_func()<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
      A_device_buffer, B_device_buffer, C_device_buffer, s.M_outer, s.N_outer,
      s.K_outer);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_host_data.data(), C_device_buffer, C_bytes,
                      hipMemcpyDeviceToHost));
  checkMmt8dResults(A_host_data.data(), B_host_data.data(), C_host_data.data(),
                    s);
  printf("OK\n");

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
      kernel.mmt_func()<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
          A_device_buffer, B_device_buffer, C_device_buffer, s.M_outer,
          s.N_outer, s.K_outer);
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
  float kernel_ops = 2.f * s.M_outer * s.N_outer * s.K_outer * s.M_tile0 *
                     s.N_tile0 * s.K_tile0;
  float kernel_ops_per_s = 1000.f * kernel_ops / kernel_ms;
  printf("%.3g Tflop/s\n", 1.e-12f * kernel_ops_per_s);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
  HIP_CHECK(hipFree(shape_device_buffer));
}

template <int T_M_tile0, int T_N_tile0, int T_K_tile0>
class MmtKernelGeneric : public MmtKernel {
  virtual const char *name() const override { return "MmtKernelGeneric"; }
  virtual int M_tile0() const override { return T_M_tile0; }
  virtual int N_tile0() const override { return T_N_tile0; }
  virtual int K_tile0() const override { return T_K_tile0; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return k + T_K_tile0 * m; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return k + T_K_tile0 * n; };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return n + T_N_tile0 * m; };
  }
  virtual int num_threads() const override { return T_M_tile0 * T_N_tile0; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ static void run(const float *A_data, const float *B_data,
                             float *C_data, int M_outer, int N_outer,
                             int K_outer) {
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int m_tile0 = threadIdx.x / T_N_tile0;
    int n_tile0 = threadIdx.x % T_N_tile0;
    float c = 0.f;
    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int k_tile0 = 0; k_tile0 < T_K_tile0; ++k_tile0) {
        float a =
            A_data[k_tile0 +
                   T_K_tile0 *
                       (m_tile0 + T_M_tile0 * (k_outer + K_outer * m_outer))];
        float b =
            B_data[k_tile0 +
                   T_K_tile0 *
                       (n_tile0 + T_N_tile0 * (k_outer + K_outer * n_outer))];
        c += a * b;
      }
    }
    C_data[n_tile0 +
           T_N_tile0 * (m_tile0 + T_M_tile0 * (n_outer + N_outer * m_outer))] =
        c;
  }
};

#if 0
struct mmt_mfma_f32_16x16x4f32 {
  static constexpr int Mblock = 64;
  static constexpr int Nblock = 1;
  static constexpr int M_tile0 = 1;
  static constexpr int N_tile0 = 1;
  static constexpr int K_tile0 = 4;
  static constexpr int M2 = 4;
  static constexpr int N2 = 16;
  static constexpr int K2 = 1;
  static constexpr int M3 = 4;
  static constexpr int N3 = 1;
  static constexpr int K3 = 1;
  __global__ static void run(const float *A_data, const float *B_data,
                             float *C_data, int M_outer, int N_outer,
                             int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t c = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int t = threadIdx.x;

    int k_tile0 = t / 16;
    int m2 = (t % 16) / 4;
    int m3 = t % 4;
    int n2 = t % 16;

    const float *A_ptr = A_data + 16 * k_tile0 + 4 * m2;
    const float *B_ptr = B_data + t;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      c = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, c, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 64;
    }

    for (int m3 = 0; m3 < 4; ++m3) {
      C_data[64 * m2 + 4 * n2 + m3] = c[m3];
    }
    (void)M_outer;
  }
};
#endif

int main() { test(MmtKernelGeneric<3, 5, 2>()); }
