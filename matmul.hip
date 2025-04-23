// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hip"

#include <cstdio>
#include <cxxabi.h>
#include <optional>
#include <random>
#include <typeinfo>
#include <vector>

typedef void (*mmt_func_t)(const void *, const void *, void *, void *, int, int,
                           int);

typedef int (*tile_layout_func_t)(int, int);

struct TiledMatrixShape {
  int rows_outer, cols_outer;
  int rows_tile, cols_tile;
  tile_layout_func_t tile_layout;
};

struct TiledMmtShape {
  MNKShape outer, tile;
  tile_layout_func_t A_tile_layout, B_tile_layout, C_tile_layout;
};

__device__ __host__ TiledMatrixShape A_shape(const TiledMmtShape &s) {
  return {s.outer.M, s.outer.K, s.tile.M, s.tile.K, s.A_tile_layout};
}

__device__ __host__ TiledMatrixShape B_shape(const TiledMmtShape &s) {
  return {s.outer.N, s.outer.K, s.tile.N, s.tile.K, s.B_tile_layout};
}

__device__ __host__ TiledMatrixShape C_shape(const TiledMmtShape &s) {
  return {s.outer.M, s.outer.N, s.tile.M, s.tile.N, s.C_tile_layout};
}

__device__ __host__ int flatsize(const TiledMatrixShape &s) {
  return s.rows_outer * s.cols_outer * s.rows_tile * s.cols_tile;
}

__device__ __host__ int offset(const TiledMatrixShape &s, int r_outer,
                               int c_outer, int r_tile, int c_tile) {
  return s.tile_layout(r_tile, c_tile) +
         s.rows_tile * s.cols_tile * (c_outer + s.cols_outer * r_outer);
}

// Base class for matrix-times-matrix-transposed ("mmt") kernels.
// As the RHS is transposed, the dimensions are:
// LHS = "A-matrix" : MxK
// RHS = "B-matrix" : NxK
// Accumulator = "C-matrix": MxN
//
// The data layout is tiled with tile sizes given by the {M,N,K}_tile methods
// and tile layouts given by the {A,B,C}_layout methods.
class MmtKernel {
public:
  virtual ~MmtKernel() {};
  // Returns the element type of the A-matrix (LHS)
  virtual Type A_type() const = 0;
  // Returns the element type of the B-matrix (RHS)
  virtual Type B_type() const = 0;
  // Returns the element type of the C-matrix (accumulator/result)
  virtual Type C_type() const = 0;
  // Returns the M-dimension tile size (rows of accumulator)
  virtual int M_tile() const = 0;
  // Returns the N-dimension tile size (columns of accumulator)
  virtual int N_tile() const = 0;
  // Returns the K-dimension tile size (reduction dimension)
  virtual int K_tile() const = 0;
  // Returns the offset-computation function describing the A-matrix layout.
  virtual tile_layout_func_t A_tile_layout() const = 0;
  // Returns the offset-computation function describing the B-matrix layout.
  virtual tile_layout_func_t B_tile_layout() const = 0;
  // Returns the offset-computation function describing the C-matrix layout.
  virtual tile_layout_func_t C_tile_layout() const = 0;
  // Returns the number of threads that the kernel requires running on.
  virtual int num_threads() const = 0;
  // Returns a pointer to the device kernel.
  virtual mmt_func_t mmt_func() const = 0;
  // Optional: kernels may override this method to override the default grid.
  virtual std::optional<dim3>
  get_work_centric_grid(const MNKShape & /*outer*/) const {
    return {};
  }
  // Optional: kernels may override this to get an auxiliary device buffer of
  // the given size in bytes.
  virtual int aux_buffer_size(const MNKShape & /*outer*/) const { return 0; }
};

MNKShape getBenchmarkMNKShape(const MmtKernel &kernel) {
  int M = getIntEnvVar("M", 4096);
  int N = getIntEnvVar("N", 4096);
  int K = getIntEnvVar("K", 4096);
  MNKShape o;
  o.M = std::max(1, M / kernel.M_tile());
  o.N = std::max(1, N / kernel.N_tile());
  o.K = std::max(1, K / kernel.K_tile());
  return o;
}

TiledMmtShape getTestShape(const MmtKernel &kernel, const MNKShape &o) {
  TiledMmtShape s;
  s.outer = o;
  s.tile.M = kernel.M_tile();
  s.tile.N = kernel.N_tile();
  s.tile.K = kernel.K_tile();
  s.A_tile_layout = kernel.A_tile_layout();
  s.B_tile_layout = kernel.B_tile_layout();
  s.C_tile_layout = kernel.C_tile_layout();
  return s;
}

dim3 getLaunchGrid(const MmtKernel &kernel, const TiledMmtShape &s) {
  return kernel.get_work_centric_grid(s.outer).value_or(
      dim3(s.outer.M, s.outer.N));
}

template <Type A_type, Type B_type, Type C_type>
void checkMmtResults(const void *A_data_void, const void *B_data_void,
                     const void *C_data_void, const TiledMmtShape &s) {
  using TA = CType<A_type>;
  using TB = CType<B_type>;
  using TC = CType<C_type>;
  const TA *A_data = static_cast<const TA *>(A_data_void);
  const TB *B_data = static_cast<const TB *>(B_data_void);
  const TC *C_data = static_cast<const TC *>(C_data_void);
  // This reference code is slow. To make the checks not too slow on
  // large matmuls, we only check the 4 corner tiles.
  for (int m_outer : {0, s.outer.M - 1}) {
    for (int n_outer : {0, s.outer.N - 1}) {
      for (int m_tile = 0; m_tile < s.tile.M; ++m_tile) {
        for (int n_tile = 0; n_tile < s.tile.N; ++n_tile) {
          float c = 0.f;
          for (int k_outer = 0; k_outer < s.outer.K; ++k_outer) {
            for (int k_tile = 0; k_tile < s.tile.K; ++k_tile) {
              TA a =
                  A_data[offset(A_shape(s), m_outer, k_outer, m_tile, k_tile)];
              TB b =
                  B_data[offset(B_shape(s), n_outer, k_outer, n_tile, k_tile)];
              c += static_cast<TC>(a) * static_cast<TC>(b);
            }
          }
          TC expected = c;
          TC actual =
              C_data[offset(C_shape(s), m_outer, n_outer, m_tile, n_tile)];
          if (actual != expected) {
            fprintf(stderr,
                    "matmul numerical error: actual(%g) != "
                    "expected(%g), at m_outer=%d n_outer=%d m_tile=%d "
                    "n_tile=%d, at %s:%d. Note: outer MxNxK = %dx%dx%d\n",
                    static_cast<float>(actual), static_cast<float>(expected),
                    m_outer, n_outer, m_tile, n_tile, __FILE__, __LINE__,
                    s.outer.M, s.outer.N, s.outer.K);
            abort();
          }
        }
      }
    }
  }
}

void checkMmtResults(Type A_type, Type B_type, Type C_type,
                     const void *A_data_void, const void *B_data_void,
                     const void *C_data_void, const TiledMmtShape &s) {
#define HANDLE_CASE(A, B, C)                                                   \
  if (A_type == Type::A && B_type == Type::B && C_type == Type::C) {           \
    checkMmtResults<Type::A, Type::B, Type::C>(A_data_void, B_data_void,       \
                                               C_data_void, s);                \
    return;                                                                    \
  }
  HANDLE_CASE(FP32, FP32, FP32)
  HANDLE_CASE(FP16, FP16, FP32)
  HANDLE_CASE(SI8, SI8, SI32)
#undef HANDLE_CASE

  fprintf(stderr, "%s:%d: unhandled types\n", __FILE__, __LINE__);
  abort();
}

void check(const MmtKernel &kernel, const MNKShape &o) {
  TiledMmtShape s = getTestShape(kernel, o);
  std::minstd_rand random_engine;
  std::vector<std::byte> A_host_data =
      makeRandomBuffer(kernel.A_type(), flatsize(A_shape(s)), random_engine);
  std::vector<std::byte> B_host_data =
      makeRandomBuffer(kernel.B_type(), flatsize(B_shape(s)), random_engine);
  std::vector<std::byte> C_host_data =
      makeRandomBuffer(kernel.C_type(), flatsize(C_shape(s)), random_engine);

  void *A_device_buffer{};
  void *B_device_buffer{};
  void *C_device_buffer{};
  void *aux_device_buffer{};
  TiledMmtShape *shape_device_buffer{};
  HIP_CHECK(hipMalloc(&A_device_buffer, A_host_data.size()));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_host_data.size()));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_host_data.size()));
  HIP_CHECK(hipGetLastError());
  int aux_buffer_size = kernel.aux_buffer_size(o);
  HIP_CHECK(hipMalloc(&aux_device_buffer, aux_buffer_size));
  HIP_CHECK(hipMemset(aux_device_buffer, 0, aux_buffer_size));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&shape_device_buffer, sizeof s));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_host_data.size(),
                      hipMemcpyHostToDevice));

  HIP_CHECK(
      hipMemcpy(shape_device_buffer, &s, sizeof s, hipMemcpyHostToDevice));
  HIP_CHECK(hipGetLastError());
  const dim3 grid_dim = getLaunchGrid(kernel, s);
  const dim3 block_dim(kernel.num_threads());
  HIP_CHECK(hipGetLastError());
  kernel.mmt_func()<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
      A_device_buffer, B_device_buffer, C_device_buffer, aux_device_buffer,
      s.outer.M, s.outer.N, s.outer.K);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_host_data.data(), C_device_buffer, C_host_data.size(),
                      hipMemcpyDeviceToHost));
  checkMmtResults(kernel.A_type(), kernel.B_type(), kernel.C_type(),
                  A_host_data.data(), B_host_data.data(), C_host_data.data(),
                  s);

  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
  HIP_CHECK(hipFree(aux_device_buffer));
  HIP_CHECK(hipFree(shape_device_buffer));
}

void check(const MmtKernel &kernel) {
  std::printf("  Checking correctness... ");
  // Test with more generic shapes than just M==N==K==2^x.
  for (MNKShape o : {MNKShape{1, 1, 1}, MNKShape{2, 1, 1}, MNKShape{1, 2, 1},
                     MNKShape{1, 1, 2}, MNKShape{1, 1, 3}, MNKShape{1, 1, 4},
                     MNKShape{1, 1, 5}, MNKShape{2, 2, 2}, MNKShape{2, 3, 4},
                     MNKShape{5, 2, 3}, MNKShape{1, 1, 10}, MNKShape{4, 4, 8},
                     MNKShape{305, 1, 1}, MNKShape{20, 20, 20}}) {
    check(kernel, o);
  }
  std::printf("OK\n");
}

void benchmark(const MmtKernel &kernel, const MNKShape &o) {
  TiledMmtShape s = getTestShape(kernel, o);
  std::printf("  Benchmarking: total MxNxK=%dx%dx%d, outer MxNxK=%dx%dx%d ... ",
              s.outer.M * s.tile.M, s.outer.N * s.tile.N, s.outer.K * s.tile.K,
              s.outer.M, s.outer.N, s.outer.K);

  std::minstd_rand random_engine;
  std::vector<std::byte> A_host_data =
      makeRandomBuffer(kernel.A_type(), flatsize(A_shape(s)), random_engine);
  std::vector<std::byte> B_host_data =
      makeRandomBuffer(kernel.B_type(), flatsize(B_shape(s)), random_engine);
  std::vector<std::byte> C_host_data =
      makeRandomBuffer(kernel.C_type(), flatsize(C_shape(s)), random_engine);

  void *A_device_buffer{};
  void *B_device_buffer{};
  void *C_device_buffer{};
  void *aux_device_buffer{};
  TiledMmtShape *shape_device_buffer{};
  HIP_CHECK(hipMalloc(&A_device_buffer, A_host_data.size()));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_host_data.size()));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_host_data.size()));
  int aux_buffer_size = kernel.aux_buffer_size(o);
  HIP_CHECK(hipMalloc(&aux_device_buffer, aux_buffer_size));
  HIP_CHECK(hipMemset(aux_device_buffer, 0, aux_buffer_size));
  HIP_CHECK(hipMalloc(&shape_device_buffer, sizeof s));

  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(shape_device_buffer, &s, sizeof s, hipMemcpyHostToDevice));

  const dim3 grid_dim = getLaunchGrid(kernel, s);
  const dim3 block_dim(kernel.num_threads());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  float elapsed_ms{};
  float min_elapsed_ms = getIntEnvVar("BENCHMARK_MIN_MS", 100);
  int fixed_iterations = getIntEnvVar("FIXED_ITERATIONS", 0);
  int iterations = fixed_iterations ? fixed_iterations : 1;
  while (true) {
    HIP_CHECK(hipEventRecord(start, hipStreamDefault));
    for (int b = 0; b < iterations; ++b) {
      kernel.mmt_func()<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
          A_device_buffer, B_device_buffer, C_device_buffer, aux_device_buffer,
          s.outer.M, s.outer.N, s.outer.K);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    if (elapsed_ms >= min_elapsed_ms || fixed_iterations) {
      break;
    }
    if (iterations > (1 << 20)) {
      fprintf(stderr, "Vacuous kernel? Only taking %g ms at iterations=%d.\n",
              elapsed_ms, iterations);
      abort();
    }
    iterations *= 2;
  }

  // Calculate the actual amount of memory read during the calculation, taking
  // into account the tile sizes.
  float A_element_bytes = type_size(kernel.A_type());
  float B_element_bytes = type_size(kernel.B_type());
  float MNK = static_cast<float>(s.outer.M) * s.outer.N * s.outer.K;
  float kernel_bytes_read = MNK * ((A_element_bytes * s.tile.M * s.tile.K) +
                                   (B_element_bytes * s.tile.N * s.tile.K));

  float kernel_ms = elapsed_ms / iterations;
  float kernel_ops =
      2.f * s.outer.M * s.outer.N * s.outer.K * s.tile.M * s.tile.N * s.tile.K;
  float kernel_ops_per_s = 1000.f * kernel_ops / kernel_ms;
  float kernel_bytes_read_per_s = 1000.f * kernel_bytes_read / kernel_ms;
  std::printf("%.4g Tflop/s, read %.4g TB/s, latency %.2g ms, iterations=%d\n",
              1.e-12f * kernel_ops_per_s, 1e-12f * kernel_bytes_read_per_s,
              kernel_ms, iterations);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
  HIP_CHECK(hipFree(aux_device_buffer));
  HIP_CHECK(hipFree(shape_device_buffer));
}

void test(const MmtKernel &kernel) {
  char *name =
      abi::__cxa_demangle(typeid(kernel).name(), nullptr, nullptr, nullptr);
  const char *filter = getenv("FILTER");
  if (filter && !strstr(name, filter)) {
    return;
  }
  std::printf("%s: A:%s, B:%s, C:%s, tile MxNxK=%dx%dx%d, num_threads=%d\n",
              name, str(kernel.A_type()), str(kernel.B_type()),
              str(kernel.C_type()), kernel.M_tile(), kernel.N_tile(),
              kernel.K_tile(), kernel.num_threads());
  free(name);

  if (!getenv("SKIP_CHECK")) {
    check(kernel);
  }

  MNKShape o = getBenchmarkMNKShape(kernel);
  benchmark(kernel, o);
}

template <Type T_A_type, Type T_B_type, Type T_C_type, int T_M_tile,
          int T_N_tile, int T_K_tile>
class MmtKernel_generic : public MmtKernel {
  virtual Type A_type() const override { return T_A_type; }
  virtual Type B_type() const override { return T_B_type; }
  virtual Type C_type() const override { return T_C_type; }
  virtual int M_tile() const override { return T_M_tile; }
  virtual int N_tile() const override { return T_N_tile; }
  virtual int K_tile() const override { return T_K_tile; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return k + T_K_tile * m; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return k + T_K_tile * n; };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return n + T_N_tile * m; };
  }
  virtual int num_threads() const override { return T_M_tile * T_N_tile; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, void * /*aux_data*/, int /*M_outer*/,
                             int N_outer, int K_outer) {
    using TA = CType<T_A_type>;
    using TB = CType<T_B_type>;
    using TC = CType<T_C_type>;
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int m_tile = threadIdx.x / T_N_tile;
    int n_tile = threadIdx.x % T_N_tile;
    TC c = {0};
    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int k_tile = 0; k_tile < T_K_tile; ++k_tile) {
        TA a = static_cast<const TA *>(
            A_data)[k_tile +
                    T_K_tile *
                        (m_tile + T_M_tile * (k_outer + K_outer * m_outer))];
        TB b = static_cast<const TB *>(
            B_data)[k_tile +
                    T_K_tile *
                        (n_tile + T_N_tile * (k_outer + K_outer * n_outer))];
        c += static_cast<TC>(a) * static_cast<TC>(b);
      }
    }
    static_cast<TC *>(
        C_data)[n_tile + T_N_tile * (m_tile + T_M_tile * (n_outer +
                                                          N_outer * m_outer))] =
        c;
  }
};

class MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_rowmajor : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 16; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return 4 * m + k; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return 16 * k + n; };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return 16 * m + n; };
  }
  virtual int num_threads() const override { return 64; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(64) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;
    int ai = tid % 16;
    int ak = tid / 16;
    int bj = tid % 16;
    int bk = tid / 16;

    const float *A_ptr = static_cast<const float *>(A_data) +
                         m_outer * K_outer * 64 + ai * 4 + ak;
    const float *B_ptr = static_cast<const float *>(B_data) +
                         n_outer * K_outer * 64 + bk * 16 + bj;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, acc, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 64;
    }

    for (int gpr = 0; gpr < 4; ++gpr) {
      int ci = 4 * (tid / 16) + gpr;
      int cj = tid % 16;
      static_cast<float *>(
          C_data)[m_outer * N_outer * 256 + n_outer * 256 + ci * 16 + cj] =
          acc[gpr];
    }
  }
};

class MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_directAB_rowmajorC
    : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 16; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m + 16 * k; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return n + 16 * k; };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return 16 * m + n; };
  }
  virtual int num_threads() const override { return 64; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(64) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const float *A_ptr =
        static_cast<const float *>(A_data) + m_outer * K_outer * 64 + tid;
    const float *B_ptr =
        static_cast<const float *>(B_data) + n_outer * K_outer * 64 + tid;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, acc, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 64;
    }

    for (int gpr = 0; gpr < 4; ++gpr) {
      int ci = 4 * (tid / 16) + gpr;
      int cj = tid % 16;
      static_cast<float *>(
          C_data)[m_outer * N_outer * 256 + n_outer * 256 + ci * 16 + cj] =
          acc[gpr];
    }
  }
};

class MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 16; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return m + 16 * k; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return n + 16 * k; };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return 64 * (m / 4) + 4 * n + (m % 4); };
  }
  virtual int num_threads() const override { return 64; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(64) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const float *A_ptr =
        static_cast<const float *>(A_data) + m_outer * K_outer * 64 + tid;
    const float *B_ptr =
        static_cast<const float *>(B_data) + n_outer * K_outer * 64 + tid;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, acc, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 64;
    }

    static_cast<floatx4_t *>(C_data)[64 * (N_outer * m_outer + n_outer) + tid] =
        acc;
  }
};

class MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct_Kx4 : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return (k / 4) + 4 * (m + 16 * (k % 4)); };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return (k / 4) + 4 * (n + 16 * (k % 4)); };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return 64 * (m / 4) + 4 * n + (m % 4); };
  }
  virtual int num_threads() const override { return 64; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(64) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const float *A_ptr =
        static_cast<const float *>(A_data) + m_outer * K_outer * 256 + 4 * tid;
    const float *B_ptr =
        static_cast<const float *>(B_data) + n_outer * K_outer * 256 + 4 * tid;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(A_ptr[0], B_ptr[0], acc, 0, 0,
                                                 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(A_ptr[1], B_ptr[1], acc, 0, 0,
                                                 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(A_ptr[2], B_ptr[2], acc, 0, 0,
                                                 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(A_ptr[3], B_ptr[3], acc, 0, 0,
                                                 0);
      A_ptr += 256;
      B_ptr += 256;
    }

    static_cast<floatx4_t *>(C_data)[64 * (N_outer * m_outer + n_outer) + tid] =
        acc;
  }
};

class MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct_Kx4_unrollx4
    : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return (k / 4) + 4 * (m + 16 * (k % 4)); };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) { return (k / 4) + 4 * (n + 16 * (k % 4)); };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) { return 64 * (m / 4) + 4 * n + (m % 4); };
  }
  virtual int num_threads() const override { return 64; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(64) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const floatx4_t *A_ptr =
        static_cast<const floatx4_t *>(A_data) + m_outer * K_outer * 64 + tid;
    const floatx4_t *B_ptr =
        static_cast<const floatx4_t *>(B_data) + n_outer * K_outer * 64 + tid;

    int k_outer = 0;
    for (; k_outer <= K_outer - 4; k_outer += 4) {
      floatx4_t a0 = A_ptr[0];
      floatx4_t b0 = B_ptr[0];
      floatx4_t a1 = A_ptr[64];
      floatx4_t b1 = B_ptr[64];
      floatx4_t a2 = A_ptr[128];
      floatx4_t b2 = B_ptr[128];
      floatx4_t a3 = A_ptr[192];
      floatx4_t b3 = B_ptr[192];
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[0], b0[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[1], b0[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[2], b0[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[3], b0[3], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a1[0], b1[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a1[1], b1[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a1[2], b1[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a1[3], b1[3], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a2[0], b2[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a2[1], b2[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a2[2], b2[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a2[3], b2[3], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a3[0], b3[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a3[1], b3[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a3[2], b3[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a3[3], b3[3], acc, 0, 0, 0);
      A_ptr += 256;
      B_ptr += 256;
    }
    for (; k_outer < K_outer; ++k_outer) {
      floatx4_t a0 = A_ptr[0];
      floatx4_t b0 = B_ptr[0];
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[0], b0[0], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[1], b0[1], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[2], b0[2], acc, 0, 0, 0);
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a0[3], b0[3], acc, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 64;
    }

    static_cast<floatx4_t *>(C_data)[64 * (N_outer * m_outer + n_outer) + tid] =
        acc;
  }
};

class MmtKernel_128t_1x2_amdgcn_mfma_f32_16x16x4f32_direct : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 16; }
  virtual int N_tile() const override { return 32; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) { return 16 * k + m; };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 64 * no + 16 * k + ni;
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int ni = n % 16;
      int no = n / 16;
      return 256 * no + 64 * (m / 4) + 4 * ni + (m % 4);
    };
  }
  virtual int num_threads() const override { return 128; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(128) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const float *A_ptr = static_cast<const float *>(A_data) +
                         m_outer * K_outer * 64 + (tid % 64);
    const float *B_ptr =
        static_cast<const float *>(B_data) + n_outer * K_outer * 128 + tid;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, acc, 0, 0, 0);
      A_ptr += 64;
      B_ptr += 128;
    }

    static_cast<floatx4_t *>(
        C_data)[128 * (N_outer * m_outer + n_outer) + tid] = acc;
  }
};

class MmtKernel_256t_2x2_amdgcn_mfma_f32_16x16x4f32_direct : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 32; }
  virtual int N_tile() const override { return 32; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 64 * mo + 16 * k + mi;
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 64 * no + 16 * k + ni;
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return 512 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    const float *A_ptr = static_cast<const float *>(A_data) +
                         m_outer * K_outer * 128 + (tid % 64) +
                         64 * (tid / 128);
    const float *B_ptr = static_cast<const float *>(B_data) +
                         n_outer * K_outer * 128 + (tid % 128);

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      acc = __builtin_amdgcn_mfma_f32_16x16x4f32(*A_ptr, *B_ptr, acc, 0, 0, 0);
      A_ptr += 128;
      B_ptr += 128;
    }

    static_cast<floatx4_t *>(
        C_data)[256 * (N_outer * m_outer + n_outer) + tid] = acc;
  }
};

class MmtKernel_256t_2x2_amdgcn_mfma_f32_16x16x4f32_shared : public MmtKernel {
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return 32; }
  virtual int N_tile() const override { return 32; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 64 * mo + 16 * k + mi;
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 64 * no + 16 * k + ni;
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return 512 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc = {0};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    int A_thread_offset = (tid % 64) + 64 * (tid / 128);
    int B_thread_offset = tid % 128;

    constexpr int A_tile_size = 32 * 4;
    constexpr int B_tile_size = 32 * 4;

    const float *A_global =
        static_cast<const float *>(A_data) + m_outer * K_outer * A_tile_size;
    const float *B_global =
        static_cast<const float *>(B_data) + n_outer * K_outer * B_tile_size;

    constexpr int K_outer_shared_size = 4; // Tuned.

    __shared__ float A_shared[K_outer_shared_size * A_tile_size];
    __shared__ float B_shared[K_outer_shared_size * B_tile_size];

    const float *A_global_ptr = A_global + A_thread_offset;
    const float *B_global_ptr = B_global + B_thread_offset;
    float *A_shared_base_ptr = A_shared + A_thread_offset;
    float *B_shared_base_ptr = B_shared + B_thread_offset;

    // Main loop: handle full-size shared tiles.
    int k_outer_global = 0;
    for (; k_outer_global <= K_outer - K_outer_shared_size;
         k_outer_global += K_outer_shared_size) {
      {
        float *A_shared_ptr = A_shared_base_ptr;
        float *B_shared_ptr = B_shared_base_ptr;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          *A_shared_ptr = *A_global_ptr;
          *B_shared_ptr = *B_global_ptr;
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
          A_global_ptr += A_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *A_shared_ptr = A_shared_base_ptr;
        const float *B_shared_ptr = B_shared_base_ptr;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          acc = __builtin_amdgcn_mfma_f32_16x16x4f32(
              *A_shared_ptr, *B_shared_ptr, acc, 0, 0, 0);
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    // Handle remainder: the last shared tile has a smaller K-size.
    if (k_outer_global < K_outer) {
      int K_remaining_outer_size = K_outer - k_outer_global;
      {
        float *A_shared_ptr = A_shared_base_ptr;
        float *B_shared_ptr = B_shared_base_ptr;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          *A_shared_ptr = *A_global_ptr;
          *B_shared_ptr = *B_global_ptr;
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
          A_global_ptr += A_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *A_shared_ptr = A_shared_base_ptr;
        const float *B_shared_ptr = B_shared_base_ptr;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          acc = __builtin_amdgcn_mfma_f32_16x16x4f32(
              *A_shared_ptr, *B_shared_ptr, acc, 0, 0, 0);
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    static_cast<floatx4_t *>(
        C_data)[256 * (N_outer * m_outer + n_outer) + tid] = acc;
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 64 * mo + 16 * k + mi;
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 64 * no + 16 * k + ni;
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size = MS * 16 * 4;
    constexpr int B_tile_size = NS * 16 * 4;

    const float *A_global =
        static_cast<const float *>(A_data) + m_outer * K_outer * A_tile_size;
    const float *B_global =
        static_cast<const float *>(B_data) + n_outer * K_outer * B_tile_size;

    constexpr int K_outer_shared_size = 4; // Tuned.

    __shared__ float B_shared[K_outer_shared_size * B_tile_size];

    const float *A_global_ptr = A_global + (tid % 64);
    const float *B_global_ptr = B_global + tid;

    // Main loop: handle full-size shared tiles.
    int k_outer_global = 0;
    for (; k_outer_global <= K_outer - K_outer_shared_size;
         k_outer_global += K_outer_shared_size) {
      {
        float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          for (int j = 0; j < B_tile_size; j += 256) {
            B_shared_ptr[j] = B_global_ptr[j];
          }
          B_shared_ptr += B_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          for (int i = 0; i < MS; ++i) {
            for (int j = 0; j < NS / 4; ++j) {
              acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                  A_global_ptr[64 * i], B_shared_ptr[256 * j], acc[i][j], 0, 0,
                  0);
            }
          }
          A_global_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    // Handle remainder: the last shared tile has a smaller K-size.
    if (k_outer_global < K_outer) {
      int K_remaining_outer_size = K_outer - k_outer_global;
      {
        float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          for (int j = 0; j < B_tile_size; j += 256) {
            B_shared_ptr[j] = B_global_ptr[j];
          }
          B_shared_ptr += B_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          for (int i = 0; i < MS; ++i) {
            for (int j = 0; j < NS / 4; ++j) {
              acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                  A_global_ptr[64 * i], B_shared_ptr[256 * j], acc[i][j], 0, 0,
                  0);
            }
          }
          A_global_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[16 * 16 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 4; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 64 * mo + 16 * k + mi;
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 64 * no + 16 * k + ni;
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size = MS * 16 * 4;
    constexpr int B_tile_size = NS * 16 * 4;

    const float *A_global =
        static_cast<const float *>(A_data) + m_outer * K_outer * A_tile_size;
    const float *B_global =
        static_cast<const float *>(B_data) + n_outer * K_outer * B_tile_size;

    constexpr int K_outer_shared_size = 4; // Tuned.

    __shared__ float A_shared[K_outer_shared_size * A_tile_size];
    __shared__ float B_shared[K_outer_shared_size * B_tile_size];

    const float *A_global_ptr = A_global + tid;
    const float *B_global_ptr = B_global + tid;

    // Main loop: handle full-size shared tiles.
    int k_outer_global = 0;
    for (; k_outer_global <= K_outer - K_outer_shared_size;
         k_outer_global += K_outer_shared_size) {
      {
        float *A_shared_ptr = A_shared + tid;
        float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          for (int i = 0; i < A_tile_size; i += 256) {
            A_shared_ptr[i] = A_global_ptr[i];
          }
          for (int j = 0; j < B_tile_size; j += 256) {
            B_shared_ptr[j] = B_global_ptr[j];
          }
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
          A_global_ptr += A_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *A_shared_ptr = A_shared + (tid % 64);
        const float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_outer_shared_size;
             ++k_outer_shared) {
          for (int i = 0; i < MS; ++i) {
            for (int j = 0; j < NS / 4; ++j) {
              acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                  A_shared_ptr[64 * i], B_shared_ptr[256 * j], acc[i][j], 0, 0,
                  0);
            }
          }
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    // Handle remainder: the last shared tile has a smaller K-size.
    if (k_outer_global < K_outer) {
      int K_remaining_outer_size = K_outer - k_outer_global;
      {
        float *A_shared_ptr = A_shared + tid;
        float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          for (int i = 0; i < A_tile_size; i += 256) {
            A_shared_ptr[i] = A_global_ptr[i];
          }
          for (int j = 0; j < B_tile_size; j += 256) {
            B_shared_ptr[j] = B_global_ptr[j];
          }
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
          A_global_ptr += A_tile_size;
          B_global_ptr += B_tile_size;
        }
      }
      __syncthreads();

      {
        const float *A_shared_ptr = A_shared + (tid % 64);
        const float *B_shared_ptr = B_shared + tid;
        for (int k_outer_shared = 0; k_outer_shared < K_remaining_outer_size;
             ++k_outer_shared) {
          for (int i = 0; i < MS; ++i) {
            for (int j = 0; j < NS / 4; ++j) {
              acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                  A_shared_ptr[64 * i], B_shared_ptr[256 * j], acc[i][j], 0, 0,
                  0);
            }
          }
          A_shared_ptr += A_tile_size;
          B_shared_ptr += B_tile_size;
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[16 * 16 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const floatx4_t *A_global = static_cast<const floatx4_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec4;
    const floatx4_t *B_global = static_cast<const floatx4_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec4;
    const floatx4_t *A_global_ptr = A_global + tid;
    const floatx4_t *B_global_ptr = B_global + tid;

    __shared__ floatx4_t A_shared[A_tile_size_in_vec4];
    __shared__ floatx4_t B_shared[B_tile_size_in_vec4];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
      __syncthreads();

      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < MS; ++i) {
          for (int j = 0; j < NS / 4; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[64 * i + (tid % 64)][k], B_shared[256 * j + tid][k],
                acc[i][j], 0, 0, 0);
          }
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_subgroup2x2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return (NS / 2) * 1024 * (mo / 2) + 1024 * (no / 2) + 512 * (mo % 2) +
             256 * (no % 2) + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS / 2][NS / 2] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const floatx4_t *A_global = static_cast<const floatx4_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec4;
    const floatx4_t *B_global = static_cast<const floatx4_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec4;
    const floatx4_t *A_global_ptr = A_global + tid;
    const floatx4_t *B_global_ptr = B_global + tid;

    __shared__ floatx4_t A_shared[A_tile_size_in_vec4];
    __shared__ floatx4_t B_shared[B_tile_size_in_vec4];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
      __syncthreads();

      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < MS / 2; ++i) {
          for (int j = 0; j < NS / 2; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[128 * i + 64 * (tid / 128) + (tid % 64)][k],
                B_shared[128 * j + (tid % 128)][k], acc[i][j], 0, 0, 0);
          }
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS / 2; ++i) {
      for (int j = 0; j < NS / 2; ++j) {
        C_ptr[256 * (NS / 2 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class
    MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_doublebuffer_naive
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const floatx4_t *A_global = static_cast<const floatx4_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec4;
    const floatx4_t *B_global = static_cast<const floatx4_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec4;
    const floatx4_t *A_global_ptr = A_global + tid;
    const floatx4_t *B_global_ptr = B_global + tid;

    __shared__ floatx4_t A_shared[2][A_tile_size_in_vec4];
    __shared__ floatx4_t B_shared[2][B_tile_size_in_vec4];

    for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
      A_shared[0][i + tid] = A_global_ptr[i];
    }
    for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
      B_shared[0][j + tid] = B_global_ptr[j];
    }
    A_global_ptr += A_tile_size_in_vec4;
    B_global_ptr += B_tile_size_in_vec4;
    // No __syncthreads here, the loop starts by loading separate parts of
    // A_shared and B_shared and then does a __syncthreads.

    for (int k_outer = 1; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
        A_shared[k_outer & 1][i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
        B_shared[k_outer & 1][j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
      __syncthreads();

      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < MS; ++i) {
          for (int j = 0; j < NS / 4; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[(k_outer ^ 1) & 1][64 * i + (tid % 64)][k],
                B_shared[(k_outer ^ 1) & 1][256 * j + tid][k], acc[i][j], 0, 0,
                0);
          }
        }
      }
      // We thought that there should be no __syncthreads here, as the next
      // iteration will load into separate parts of A_shared and B_shared.
      // But we observed a data race giving intermittent failures in practice.
      __syncthreads();
    }

    for (int k = 0; k < 4; ++k) {
      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
              A_shared[(K_outer ^ 1) & 1][64 * i + (tid % 64)][k],
              B_shared[(K_outer ^ 1) & 1][256 * j + tid][k], acc[i][j], 0, 0,
              0);
        }
      }
    }
    __syncthreads();

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class
    MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_doublebuffer_take2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const floatx4_t *A_global = static_cast<const floatx4_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec4;
    const floatx4_t *B_global = static_cast<const floatx4_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec4;
    const floatx4_t *A_global_ptr = A_global + tid;
    const floatx4_t *B_global_ptr = B_global + tid;

    __shared__ floatx4_t A_shared[2][A_tile_size_in_vec4];
    __shared__ floatx4_t B_shared[2][B_tile_size_in_vec4];

    auto load_shared_from_global = [&](int buffer_id) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
        A_shared[buffer_id][i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
        B_shared[buffer_id][j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
    };

    auto mfma = [&](int buffer_id) {
      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < MS; ++i) {
          for (int j = 0; j < NS / 4; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[buffer_id][64 * i + (tid % 64)][k],
                B_shared[buffer_id][256 * j + tid][k], acc[i][j], 0, 0, 0);
          }
        }
      }
    };

    load_shared_from_global(0);

    while (K_outer >= 3) {
      load_shared_from_global(1);
      __syncthreads();
      mfma(0);
      load_shared_from_global(0);
      mfma(1);
      K_outer -= 2;
    }

    if (K_outer == 2) {
      load_shared_from_global(1);
      __syncthreads();
      mfma(0);
      mfma(1);
    } else {
      __syncthreads();
      mfma(0);
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = (m % 64) / 16;
      int mx = m / 64;
      int ni = n % 16;
      int no = (n % 64) / 16;
      int nx = n / 64;
      return (NS / 4) * 4096 * mx + 4096 * nx + 1024 * mo + 256 * no +
             64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 1024; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(1024) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS / 4][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const floatx4_t *A_global = static_cast<const floatx4_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec4;
    const floatx4_t *B_global = static_cast<const floatx4_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec4;
    const floatx4_t *A_global_ptr = A_global + tid;
    const floatx4_t *B_global_ptr = B_global + tid;

    __shared__ floatx4_t A_shared[A_tile_size_in_vec4];
    __shared__ floatx4_t B_shared[B_tile_size_in_vec4];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 1024) {
        if (tid < MS * 64)
          A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 1024) {
        if (tid < NS * 64)
          B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
      __syncthreads();

      for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < MS / 4; ++i) {
          for (int j = 0; j < NS / 4; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[256 * i + 64 * (tid / 256) + (tid % 64)][k],
                B_shared[256 * j + (tid % 256)][k], acc[i][j], 0, 0, 0);
          }
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS / 4; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[1024 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class
    MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_misguidednobankconflicts
    : public MmtKernel {
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP32; }
  virtual Type B_type() const override { return Type::FP32; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return k / 4 + 4 * (64 * mo + 16 * (k % 4) + mi);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return k / 4 + 4 * (64 * no + 16 * (k % 4) + ni);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using floatx2_t = __attribute__((__vector_size__(2 * sizeof(float)))) float;
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec2 = MS * 16 * 4 * 2;
    constexpr int B_tile_size_in_vec2 = NS * 16 * 4 * 2;

    const floatx2_t *A_global = static_cast<const floatx2_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec2;
    const floatx2_t *B_global = static_cast<const floatx2_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec2;
    const floatx2_t *A_global_ptr = A_global + 2 * tid;
    const floatx2_t *B_global_ptr = B_global + 2 * tid;

    __shared__ floatx2_t A_shared[A_tile_size_in_vec2];
    __shared__ floatx2_t B_shared[B_tile_size_in_vec2];

    int k0 = (tid / 8) % 2;
    int k1 = 1 ^ k0;

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec2; i += 2 * 256) {
        A_shared[i + 2 * tid + k0] = A_global_ptr[i];
        A_shared[i + 2 * tid + k1] = A_global_ptr[i + 1];
      }
      for (int j = 0; j < B_tile_size_in_vec2; j += 2 * 256) {
        B_shared[j + 2 * tid + k0] = B_global_ptr[j];
        B_shared[j + 2 * tid + k1] = B_global_ptr[j + 1];
      }
      A_global_ptr += A_tile_size_in_vec2;
      B_global_ptr += B_tile_size_in_vec2;
      __syncthreads();

      for (int p = 0; p < 2; ++p) {
        for (int i = 0; i < MS; ++i) {
          for (int j = 0; j < NS / 4; ++j) {
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[2 * 64 * i + 2 * (tid % 64) + k0][p],
                B_shared[2 * 256 * j + 2 * tid + k0][p], acc[i][j], 0, 0, 0);
            acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                A_shared[2 * 64 * i + 2 * (tid % 64) + k1][p],
                B_shared[2 * 256 * j + 2 * tid + k1][p], acc[i][j], 0, 0, 0);
          }
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[16 * 16 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP16; }
  virtual Type B_type() const override { return Type::FP16; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 256 * mo + 64 * (k / 4) + 4 * mi + (k % 4);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 256 * no + 64 * (k / 4) + 4 * ni + (k % 4);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using float16x4_t =
        __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec4 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec4 = NS * 16 * 4;

    const float16x4_t *A_global = static_cast<const float16x4_t *>(A_data) +
                                  m_outer * K_outer * A_tile_size_in_vec4;
    const float16x4_t *B_global = static_cast<const float16x4_t *>(B_data) +
                                  n_outer * K_outer * B_tile_size_in_vec4;
    const float16x4_t *A_global_ptr = A_global + tid;
    const float16x4_t *B_global_ptr = B_global + tid;

    __shared__ float16x4_t A_shared[A_tile_size_in_vec4];
    __shared__ float16x4_t B_shared[B_tile_size_in_vec4];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec4; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec4; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec4;
      B_global_ptr += B_tile_size_in_vec4;
      __syncthreads();

      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x16f16(
              A_shared[64 * i + (tid % 64)], B_shared[256 * j + tid], acc[i][j],
              0, 0, 0);
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::FP16; }
  virtual Type B_type() const override { return Type::FP16; }
  virtual Type C_type() const override { return Type::FP32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 32; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 512 * mo + 128 * ((k % 16) / 4) + 8 * mi + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 512 * no + 128 * ((k % 16) / 4) + 8 * ni + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using float16x4_t =
        __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
    using float16x8_t =
        __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
    using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    floatx4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec8 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec8 = NS * 16 * 4;

    const float16x8_t *A_global = static_cast<const float16x8_t *>(A_data) +
                                  m_outer * K_outer * A_tile_size_in_vec8;
    const float16x8_t *B_global = static_cast<const float16x8_t *>(B_data) +
                                  n_outer * K_outer * B_tile_size_in_vec8;
    const float16x8_t *A_global_ptr = A_global + tid;
    const float16x8_t *B_global_ptr = B_global + tid;

    __shared__ float16x8_t A_shared[A_tile_size_in_vec8];
    __shared__ float16x8_t B_shared[B_tile_size_in_vec8];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();

      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          float16x8_t a = A_shared[64 * i + (tid % 64)];
          float16x8_t b = B_shared[256 * j + tid];
          float16x4_t a0 = (float16x4_t){a[0], a[1], a[2], a[3]};
          float16x4_t a1 = (float16x4_t){a[4], a[5], a[6], a[7]};
          float16x4_t b0 = (float16x4_t){b[0], b[1], b[2], b[3]};
          float16x4_t b1 = (float16x4_t){b[4], b[5], b[6], b[7]};
          acc[i][j] =
              __builtin_amdgcn_mfma_f32_16x16x16f16(a0, b0, acc[i][j], 0, 0, 0);
          acc[i][j] =
              __builtin_amdgcn_mfma_f32_16x16x16f16(a1, b1, acc[i][j], 0, 0, 0);
        }
      }
      __syncthreads();
    }

    floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

// A simple Stream-K kernel variant of
// MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2.
//
// The synchronization between workgroups cooperating on a tile is wait-free
// and produces deterministic results. This is achieved using a per-tile atomic
// counter to determine which workgroup is the last to complete its share of the
// tile's computation. That workgroup will then go on to perform the final
// reduction of the multiple workgroups' local accumulators into the final
// accumulator. The results are still deterministic thanks to performing that
// final reduction in order of increasing K-dimension indices, regardless of
// which workgroup happens to be doing it.
//
// The inner arithmetic loops isn't particularly optimized, being the same as in
// MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2.
// Like in other kernels, the template parameters MS and NS control the number
// of MFMA intrinsics in the kernel along the M and N dimensions respectively:
// MS x (NS / 4), while 4 subgroups are stacked along the NS dimension,
// achieving a total width of MS x NS MFMA intrinsics across all 4 subgroups.
//
template <int MS, int NS>
class MmtKernel_StreamK_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  // Defining some values as constants as they will be needed in static methods
  // below.
  static constexpr Type static_A_type = Type::FP16;
  static constexpr Type static_B_type = Type::FP16;
  static constexpr Type static_C_type = Type::FP32;
  static constexpr int static_M_tile = MS * 16;
  static constexpr int static_N_tile = NS * 16;
  static constexpr int static_K_tile = 32;
  // Standard MmtKernel interface methods override describing element types.
  virtual Type A_type() const override { return static_A_type; }
  virtual Type B_type() const override { return static_B_type; }
  virtual Type C_type() const override { return static_C_type; }
  // Standard MmtKernel interface methods override describing tile shapes.
  virtual int M_tile() const override { return static_M_tile; }
  virtual int N_tile() const override { return static_N_tile; }
  virtual int K_tile() const override { return static_K_tile; }
  // Standard MmtKernel interface methods override describing tile layouts.
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 512 * mo + 128 * ((k % 16) / 4) + 8 * mi + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 512 * no + 128 * ((k % 16) / 4) + 8 * ni + 4 * (k / 16) + (k % 4);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 64 * (mi / 4) + 4 * ni + (mi % 4);
    };
  }
  // Standard MmtKernel interface method override describing number of threads.
  virtual int num_threads() const override { return 256; }
  // Additional constants and local helpers (static methods) for work-centric
  // grid and auxiliary buffer allocation.
  //
  // Maximum number of Compute Units that we aim to use on sufficiently large
  // problems. This value assumes a MI300X in SPX mode.
  static constexpr int MaxWorkgroups = 304;
  // Alignment to use when coalescing device buffers to avoid alignment-related
  // performance caveats.
  static constexpr int device_alignment = 128;
  // Alignment to use for atomics to avoid false-sharing effects.
  static constexpr int atomics_alignment = 128;
  // Returns the number of Compute Units that we aim to use for the problem
  // shape with given `outer` shape (expressed in units of this kernel's tiles).
  static __device__ __host__ int get_workgroup_count(const MNKShape &outer) {
    int64_t total_iters = static_cast<int64_t>(outer.M) * outer.N * outer.K;
    return std::min(static_cast<int64_t>(MaxWorkgroups), total_iters);
  }
  static __device__ __host__ int get_tile_count(const MNKShape &outer) {
    return outer.M * outer.N;
  }
  // Returns the buffer bytes to allocate in the auxiliary buffer for the atomic
  // counters (one per tile).
  using atomic_t = int;
  static __device__ __host__ int atomics_buffer_size(const MNKShape &outer) {
    return round_up_to_po2<device_alignment>(
        get_tile_count(outer) *
        round_up_to_po2<atomics_alignment>(sizeof(atomic_t)));
  }
  // Returns the number of local accumulator buffers to allocate in the
  // auxiliary buffer, per tile. This value is a coarse upper bound, could be
  // refined to trim memory usage.
  static __device__ __host__ int
  local_accumulators_count_per_tile(const MNKShape &outer) {
    return 1 + ceil_div(get_workgroup_count(outer), get_tile_count(outer));
  }
  // Return the overall number of local accumulator buffers to allocate in the
  // auxiliary buffer. This value is a coarse upper bound, could be
  // refined to trim memory usage.
  static __device__ __host__ int
  local_accumulators_count(const MNKShape &outer) {
    return local_accumulators_count_per_tile(outer) * get_tile_count(outer);
  }
  // Return the overall auxiliary buffer bytes to allocate.
  static __device__ __host__ int
  local_accumulators_buffer_size(const MNKShape &outer) {
    return round_up_to_po2<device_alignment>(local_accumulators_count(outer) *
                                             static_M_tile * static_N_tile *
                                             type_size(static_C_type));
  }
  // Standard MmtKernel interface method override describing the work-centric
  // grid.
  virtual std::optional<dim3>
  get_work_centric_grid(const MNKShape &outer) const override {
    return {get_workgroup_count(outer)};
  }
  // Standard MmtKernel interface method override describing the auxiliary
  // buffer size to allocate.
  virtual int aux_buffer_size(const MNKShape &outer) const override {
    return atomics_buffer_size(outer) + local_accumulators_buffer_size(outer);
  }
  // Standard MmtKernel interface method override describing the actual device
  // kernel.
  virtual mmt_func_t mmt_func() const override { return run; };
  // Some typedefs used in the kernel implementation.
  using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  using float16x4_t =
      __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
  using float16x8_t =
      __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
  static constexpr int num_acc_vec4s = MS * NS / 4;
  using acc_array_t = std::array<floatx4_t, num_acc_vec4s>;
  // Device kernel implementation.
  __global__ __launch_bounds__(256) static void run(const void *A_data,
                                                    const void *B_data,
                                                    void *C_data,
                                                    void *aux_data, int M_outer,
                                                    int N_outer, int K_outer) {
    MNKShape outer{M_outer, N_outer, K_outer};
    // Get local accumulators buffer.
    char *aux_bytes = static_cast<char *>(aux_data);
    float *local_accumulators_buffer =
        reinterpret_cast<float *>(aux_bytes + atomics_buffer_size(outer));
    // General arithmetic just like in Algorithm 5 in
    // https://arxiv.org/pdf/2301.03598, except we use the term "workgroup"
    // instead of "CTA".
    int iters_per_tile = K_outer;
    int total_tiles = M_outer * N_outer;
    int total_iters = total_tiles * iters_per_tile;
    int workgroup_count = get_workgroup_count(outer);
    int workgroup = blockIdx.x;
    int iter_start = workgroup * total_iters / workgroup_count;
    int iter_end = (workgroup + 1) * total_iters / workgroup_count;
    int iter = iter_start;
    // Work-centric loop on iterations to be handled by this compute unit.
    while (iter < iter_end) {
      // Still same variable names as in Algorithm 5 in
      // https://arxiv.org/pdf/2301.03598.
      int tile_idx = iter / iters_per_tile;
      int tile_iter_start = tile_idx * iters_per_tile;
      int tile_iter_end = tile_iter_start + iters_per_tile;
      int local_iter = iter - tile_iter_start;
      int local_iter_end = std::min(tile_iter_end, iter_end) - tile_iter_start;
      if (local_iter == local_iter_end) {
        break;
      }
      // Map the 1D tile_idx to 2D (m_outer, n_outer) position in the C-matrix
      // tile space, using MxN-lexicographic order.
      int m_outer = tile_idx / N_outer;
      int n_outer = tile_idx - m_outer * N_outer;

      acc_array_t acc = mac_loop(A_data, B_data, K_outer, m_outer, n_outer,
                                 local_iter, local_iter_end);

      // The next step is specific to our wait-free atomic approach: we need to
      // know the interval of other compute units cooperating with the current
      // compute unit on the current tile.
      int cooperating_workgroup_start =
          ceil_div((tile_iter_start + 1) * workgroup_count, total_iters) - 1;
      int cooperating_workgroup_end =
          ceil_div(tile_iter_end * workgroup_count, total_iters);
      assert(workgroup >= cooperating_workgroup_start);
      assert(workgroup < cooperating_workgroup_end);

      int cooperating_workgroup_count =
          cooperating_workgroup_end - cooperating_workgroup_start;

      // Track whether we will ultimately perform the store of the final
      // accumulator. This is always true, except when cooperating with other
      // workgroups on the tile AND determining that another workgroup will be
      // the one doing the final reduction.
      bool should_store_final_accumulator = true;

      // Hard case: cooperating with other workgroups on this tile.
      if (cooperating_workgroup_count != 1) {
        int workgroup_idx_in_cooperating_group =
            workgroup - cooperating_workgroup_start;
        int local_accumulator_idx =
            workgroup_idx_in_cooperating_group +
            tile_idx * local_accumulators_count_per_tile(
                           MNKShape{M_outer, N_outer, K_outer});
        floatx4_t *local_accumulator = reinterpret_cast<floatx4_t *>(
            local_accumulators_buffer +
            local_accumulator_idx * static_M_tile * static_N_tile);

        // Store our own accumulators to memory. We don't know yet which
        // workgroup will perform the final reduction. If it is not us, then it
        // will need our local accumulators to have been stored to global
        // memory. If it is us, then this store is potentially redundant,
        // however:
        // 1. Something needs to provide memory ordering on the local
        // accumulator
        //    accesses made by different workgroups. At the moment, that thing
        //    is the same barrier that also determines which workgroup does the
        //    final reduction. If we try to potentially save the global store
        //    here, we will need to add a second fence.
        // 2. Trying to keep this in registers would mean more register pressure
        //    and potentially a new constraint on kernel tile size choice.
        // 3. Likewise if trying to keep that in shared memory.
        acc_store(acc, local_accumulator);

        atomic_t *atomic_counter = reinterpret_cast<atomic_t *>(
            aux_bytes + tile_idx * atomics_alignment);

        // Increment the atomic counter and agent-scope acquire-release fence.
        // This has the effect of ordering the above accumulator store before
        // the below accumulator loads across workgroups ("agent scope").
        int counter_value = workgroup_atomic_increment(atomic_counter);

        // Determine if we are the workgroup that should perform the final
        // reduction.
        if (counter_value == cooperating_workgroup_count) {
          // Perform the final reduction.
          acc = final_reduction(
              M_outer, N_outer, K_outer, local_accumulators_buffer, tile_idx,
              cooperating_workgroup_start, cooperating_workgroup_end);

          // Reset the atomic counter to 0 so that the auxiliary buffer is ready
          // to reuse in another kernel launch.
          if (threadIdx.x == 0) {
            *atomic_counter = 0;
          }
        } else {
          // Not performing the final reduction, so also not doing the store.
          should_store_final_accumulator = false;
        }
      }

      floatx4_t *C_ptr = static_cast<floatx4_t *>(C_data) +
                         MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);

      if (should_store_final_accumulator) {
        // Store the final accumulator to the destination.
        // Note: consumer fusions go here.
        acc_store(acc, C_ptr);
      }

      iter = tile_iter_end;
    }
  }
  // Helper to load accumulators from memory.
  __device__ static acc_array_t acc_load(const floatx4_t *ptr) {
    acc_array_t acc;
    for (int i = 0; i < num_acc_vec4s; ++i) {
      acc[i] = ptr[256 * i + threadIdx.x];
    }
    return acc;
  }
  // Helper to store accumulators to memory.
  __device__ static void acc_store(acc_array_t acc, floatx4_t *ptr) {
    for (int i = 0; i < num_acc_vec4s; ++i) {
      ptr[256 * i + threadIdx.x] = acc[i];
    }
  }

  // Helper: inner MFMA loop. Similar to the MacLoop function given in
  // Algorithm 3 in https://arxiv.org/pdf/2301.03598.
  __device__ static acc_array_t mac_loop(const void *A_data, const void *B_data,
                                         int K_outer, int m_outer, int n_outer,
                                         int k_iter_start, int k_iter_end) {
    // Accumulator VGPRs.
    acc_array_t acc = {{0}};
    int tid = threadIdx.x;
    constexpr int A_tile_size_in_vec8 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec8 = NS * 16 * 4;
    const float16x8_t *A_global =
        static_cast<const float16x8_t *>(A_data) +
        (m_outer * K_outer + k_iter_start) * A_tile_size_in_vec8;
    const float16x8_t *B_global =
        static_cast<const float16x8_t *>(B_data) +
        (n_outer * K_outer + k_iter_start) * B_tile_size_in_vec8;
    const float16x8_t *A_global_ptr = A_global + tid;
    const float16x8_t *B_global_ptr = B_global + tid;

    // Shared memory buffers for tiles of A and B matrices.
    __shared__ float16x8_t A_shared[A_tile_size_in_vec8];
    __shared__ float16x8_t B_shared[B_tile_size_in_vec8];

    // Inner loop on K dimenion.
    for (int k_iter = k_iter_start; k_iter < k_iter_end; ++k_iter) {
      // Load data from global to shared memory.
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();
      // Perform MFMA arithmetic on data in shared memory.
      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          float16x8_t a = A_shared[64 * i + (tid % 64)];
          float16x8_t b = B_shared[256 * j + tid];
          float16x4_t a0 = (float16x4_t){a[0], a[1], a[2], a[3]};
          float16x4_t a1 = (float16x4_t){a[4], a[5], a[6], a[7]};
          float16x4_t b0 = (float16x4_t){b[0], b[1], b[2], b[3]};
          float16x4_t b1 = (float16x4_t){b[4], b[5], b[6], b[7]};
          floatx4_t acc0 = acc[i * (NS / 4) + j];
          acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a0, b0, acc0, 0, 0, 0);
          acc0 = __builtin_amdgcn_mfma_f32_16x16x16f16(a1, b1, acc0, 0, 0, 0);
          acc[i * (NS / 4) + j] = acc0;
        }
      }
      __syncthreads();
    }
    return acc;
  }
  __device__ static int workgroup_atomic_increment(atomic_t *atomic) {
    __shared__ int workgroup_value;
    int tid = threadIdx.x;
    if (tid == 0) {
      // Thread0 fetches and increments the device-wide atomic.
      // This is a relaxed atomic, which is fine, as we provide the memory
      // ordering separately below with __builtin_amdgcn_fence, which usefully
      // allows us to specify the memory scope.
      int thread0_value = 1 + atomicAdd(atomic, 1);
      // Thread0 stores the value to the workgroup-wide shared location.
      workgroup_value = thread0_value;
      // Memory scope is "agent" because we need other workgroups to see this.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    }
    // Now all threads in the workgroup load the workgroup-wide shared value.
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    return workgroup_value;
  }
  // Helper: maybe perform the final reduction of partial accumulators to the
  // destination C-matrix. The condition depends on atomic counter increments
  // telling us if we are the workgroup that is responsible for this among other
  // workgroups cooperating on a tile. It is thus non-deterministic which
  // workgroup will perform this final reduction, but the final C-matrix
  // elements must be deterministic. This is achieved by deterministically
  // looping over the partial accumulators in order of increasing K-dimension
  // indices, regardless of which partial accumulators were computed earlier
  // than others.
  __device__ static acc_array_t
  final_reduction(int M_outer, int N_outer, int K_outer,
                  float *local_accumulators_buffer, int tile_idx,
                  int cooperating_workgroup_start,
                  int cooperating_workgroup_end) {
    acc_array_t acc = {0};
    // Sum the partial accumulators in deterministic order of increasing
    // compute unit index, which makes to determinisic order of increasing K
    // dimension index.
    for (int other_workgroup = cooperating_workgroup_start;
         other_workgroup < cooperating_workgroup_end; ++other_workgroup) {
      int other_workgroup_idx_in_cooperating_group =
          other_workgroup - cooperating_workgroup_start;
      int other_local_accumulator_idx =
          other_workgroup_idx_in_cooperating_group +
          tile_idx * local_accumulators_count_per_tile(
                         MNKShape{M_outer, N_outer, K_outer});
      floatx4_t *other_accum_ptr = reinterpret_cast<floatx4_t *>(
          local_accumulators_buffer +
          other_local_accumulator_idx * static_M_tile * static_N_tile);
      for (int i = 0; i < num_acc_vec4s; ++i) {
        acc[i] += other_accum_ptr[256 * i + threadIdx.x];
      }
    }
    return acc;
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_i32_32x32x16i8_shared
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return MS * 32; }
  virtual int N_tile() const override { return NS * 32; }
  virtual int K_tile() const override { return 16; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 32;
      int mo = m / 32;
      return 512 * mo + 256 * (k / 8) + 8 * mi + (k % 8);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 32;
      int no = n / 32;
      return 512 * no + 256 * (k / 8) + 8 * ni + (k % 8);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 32;
      int mo = m / 32;
      int ni = n % 32;
      int no = n / 32;
      return NS * 1024 * mo + 1024 * no + 16 * (((32 * (mi / 4)) % 64) + ni) +
             4 * (mi / 8) + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int8x8_t = int64_t;
    using int32x16_t = __attribute__((__vector_size__(4 * 16))) int32_t;
    int32x16_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec8 = MS * 32 * 2;
    constexpr int B_tile_size_in_vec8 = NS * 32 * 2;

    const int8x8_t *A_global = static_cast<const int8x8_t *>(A_data) +
                               m_outer * K_outer * A_tile_size_in_vec8;
    const int8x8_t *B_global = static_cast<const int8x8_t *>(B_data) +
                               n_outer * K_outer * B_tile_size_in_vec8;
    const int8x8_t *A_global_ptr = A_global + tid;
    const int8x8_t *B_global_ptr = B_global + tid;

    __shared__ int8x8_t A_shared[A_tile_size_in_vec8];
    __shared__ int8x8_t B_shared[B_tile_size_in_vec8];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();

      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          acc[i][j] = __builtin_amdgcn_mfma_i32_32x32x16_i8(
              A_shared[64 * i + (tid % 64)], B_shared[256 * j + tid], acc[i][j],
              0, 0, 0);
        }
      }
      __syncthreads();
    }

    int32x16_t *C_ptr = static_cast<int32x16_t *>(C_data) +
                        MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 32; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 512 * mo + 128 * (k / 8) + 8 * mi + (k % 8);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 512 * no + 128 * (k / 8) + 8 * ni + (k % 8);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 4 * (16 * (mi / 4) + ni) + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int8x8_t = int64_t;
    using int32x4_t = __attribute__((__vector_size__(4 * 4))) int32_t;
    int32x4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec8 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec8 = NS * 16 * 4;

    const int8x8_t *A_global = static_cast<const int8x8_t *>(A_data) +
                               m_outer * K_outer * A_tile_size_in_vec8;
    const int8x8_t *B_global = static_cast<const int8x8_t *>(B_data) +
                               n_outer * K_outer * B_tile_size_in_vec8;
    const int8x8_t *A_global_ptr = A_global + tid;
    const int8x8_t *B_global_ptr = B_global + tid;

    __shared__ int8x8_t A_shared[A_tile_size_in_vec8];
    __shared__ int8x8_t B_shared[B_tile_size_in_vec8];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec8; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec8; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec8;
      B_global_ptr += B_tile_size_in_vec8;
      __syncthreads();

      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
              A_shared[64 * i + (tid % 64)], B_shared[256 * j + tid], acc[i][j],
              0, 0, 0);
        }
      }
      __syncthreads();
    }

    int32x4_t *C_ptr = static_cast<int32x4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 64; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 1024 * mo + 256 * ((k % 32) / 8) + 16 * mi + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 1024 * no + 256 * ((k % 32) / 8) + 16 * ni + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 4 * (16 * (mi / 4) + ni) + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int64x2_t = __attribute__((__vector_size__(8 * 2))) int64_t;
    using int32x4_t = __attribute__((__vector_size__(4 * 4))) int32_t;
    int32x4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec16 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec16 = NS * 16 * 4;

    const int64x2_t *A_global = static_cast<const int64x2_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec16;
    const int64x2_t *B_global = static_cast<const int64x2_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec16;
    const int64x2_t *A_global_ptr = A_global + tid;
    const int64x2_t *B_global_ptr = B_global + tid;

    __shared__ int64x2_t A_shared[A_tile_size_in_vec16];
    __shared__ int64x2_t B_shared[B_tile_size_in_vec16];

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int i = 0; i < A_tile_size_in_vec16; i += 256) {
        A_shared[i + tid] = A_global_ptr[i];
      }
      for (int j = 0; j < B_tile_size_in_vec16; j += 256) {
        B_shared[j + tid] = B_global_ptr[j];
      }
      A_global_ptr += A_tile_size_in_vec16;
      B_global_ptr += B_tile_size_in_vec16;
      __syncthreads();

      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          for (int k = 0; k < 2; ++k) {
            acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
                A_shared[64 * i + (tid % 64)][k], B_shared[256 * j + tid][k],
                acc[i][j], 0, 0, 0);
          }
        }
      }
      __syncthreads();
    }

    int32x4_t *C_ptr = static_cast<int32x4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipelineload
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 64; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 1024 * mo + 256 * ((k % 32) / 8) + 16 * mi + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 1024 * no + 256 * ((k % 32) / 8) + 16 * ni + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 4 * (16 * (mi / 4) + ni) + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int64x2_t = __attribute__((__vector_size__(8 * 2))) int64_t;
    using int32x4_t = __attribute__((__vector_size__(4 * 4))) int32_t;
    int32x4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec16 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec16 = NS * 16 * 4;
    constexpr int num_threads = 256;

    const int64x2_t *A_global = static_cast<const int64x2_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec16;
    const int64x2_t *B_global = static_cast<const int64x2_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec16;

    __shared__ int64x2_t A_shared[A_tile_size_in_vec16];
    __shared__ int64x2_t B_shared[B_tile_size_in_vec16];

    int64x2_t A_vgpr0[A_tile_size_in_vec16 / num_threads];
    int64x2_t B_vgpr0[B_tile_size_in_vec16 / num_threads];

    auto global_to_vgpr0 = [&]() {
      for (int i = 0; i < A_tile_size_in_vec16 / num_threads; ++i) {
        A_vgpr0[i] = A_global[i * num_threads + tid];
      }
      for (int j = 0; j < B_tile_size_in_vec16 / num_threads; ++j) {
        B_vgpr0[j] = B_global[j * num_threads + tid];
      }
      A_global += A_tile_size_in_vec16;
      B_global += B_tile_size_in_vec16;
    };

    auto vpgr0_to_shared = [&]() {
      for (int i = 0; i < A_tile_size_in_vec16 / num_threads; ++i) {
        A_shared[i * num_threads + tid] = A_vgpr0[i];
      }
      for (int j = 0; j < B_tile_size_in_vec16 / num_threads; ++j) {
        B_shared[j * num_threads + tid] = B_vgpr0[j];
      }
    };

    auto mfma = [&]() {
      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          for (int k = 0; k < 2; ++k) {
            acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
                A_shared[64 * i + (tid % 64)][k], B_shared[256 * j + tid][k],
                acc[i][j], 0, 0, 0);
          }
        }
      }
    };

    global_to_vgpr0();
    vpgr0_to_shared();
    if (K_outer >= 2) {
      global_to_vgpr0();
      for (int k_outer = 0; k_outer < K_outer - 2; ++k_outer) {
        // Wait for all shared memory to be written by all subgroups.
        __syncthreads();
        mfma();
        // Wait for all subgroups to finish reading from shared memory, so that
        // we can overwrite it.
        __syncthreads();
        vpgr0_to_shared();
        global_to_vgpr0();
      }
      // Wait for all shared memory to be written by all subgroups.
      __syncthreads();
      mfma();
      // Wait for all subgroups to finish reading from shared memory, so that
      // we can overwrite it.
      __syncthreads();
      vpgr0_to_shared();
    }
    // Wait for all shared memory to be written by all subgroups.
    __syncthreads();
    mfma();

    int32x4_t *C_ptr = static_cast<int32x4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipeline_v3
    : public MmtKernel {
  static_assert(MS >= 4 && !(MS % 4));
  static_assert(NS >= 4 && !(NS % 4));
  virtual Type A_type() const override { return Type::SI8; }
  virtual Type B_type() const override { return Type::SI8; }
  virtual Type C_type() const override { return Type::SI32; }
  virtual int M_tile() const override { return MS * 16; }
  virtual int N_tile() const override { return NS * 16; }
  virtual int K_tile() const override { return 64; }
  virtual tile_layout_func_t A_tile_layout() const override {
    return [](int m, int k) {
      int mi = m % 16;
      int mo = m / 16;
      return 1024 * mo + 256 * ((k % 32) / 8) + 16 * mi + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t B_tile_layout() const override {
    return [](int n, int k) {
      int ni = n % 16;
      int no = n / 16;
      return 1024 * no + 256 * ((k % 32) / 8) + 16 * ni + 8 * (k / 32) +
             (k % 8);
    };
  }
  virtual tile_layout_func_t C_tile_layout() const override {
    return [](int m, int n) {
      int mi = m % 16;
      int mo = m / 16;
      int ni = n % 16;
      int no = n / 16;
      return NS * 256 * mo + 256 * no + 4 * (16 * (mi / 4) + ni) + (mi % 4);
    };
  }
  virtual int num_threads() const override { return 256; }
  virtual mmt_func_t mmt_func() const override { return run; };
  __global__ __launch_bounds__(256) static void run(
      const void *A_data, const void *B_data, void *C_data, void * /*aux_data*/,
      int /*M_outer*/, int N_outer, int K_outer) {
    using int64x2_t = __attribute__((__vector_size__(8 * 2))) int64_t;
    using int32x4_t = __attribute__((__vector_size__(4 * 4))) int32_t;
    int32x4_t acc[MS][NS / 4] = {{0}};

    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int tid = threadIdx.x;

    constexpr int A_tile_size_in_vec16 = MS * 16 * 4;
    constexpr int B_tile_size_in_vec16 = NS * 16 * 4;
    constexpr int num_threads = 256;

    const int64x2_t *A_global = static_cast<const int64x2_t *>(A_data) +
                                m_outer * K_outer * A_tile_size_in_vec16;
    const int64x2_t *B_global = static_cast<const int64x2_t *>(B_data) +
                                n_outer * K_outer * B_tile_size_in_vec16;

    __shared__ int64x2_t A_shared[A_tile_size_in_vec16];
    __shared__ int64x2_t B_shared[B_tile_size_in_vec16];

    int64x2_t A_vgpr0[A_tile_size_in_vec16 / num_threads];
    int64x2_t B_vgpr0[B_tile_size_in_vec16 / num_threads];

    int64x2_t A_block_vgpr1[MS];
    int64x2_t B_block_vgpr1[NS / 2];

    auto global_to_vgpr0 = [&]() {
      for (int i = 0; i < A_tile_size_in_vec16 / num_threads; ++i) {
        A_vgpr0[i] = A_global[i * num_threads + tid];
      }
      for (int j = 0; j < B_tile_size_in_vec16 / num_threads; ++j) {
        B_vgpr0[j] = B_global[j * num_threads + tid];
      }
      A_global += A_tile_size_in_vec16;
      B_global += B_tile_size_in_vec16;
    };

    auto vpgr0_to_shared = [&]() {
      for (int i = 0; i < A_tile_size_in_vec16 / num_threads; ++i) {
        A_shared[i * num_threads + tid] = A_vgpr0[i];
      }
      for (int j = 0; j < B_tile_size_in_vec16 / num_threads; ++j) {
        B_shared[j * num_threads + tid] = B_vgpr0[j];
      }
    };

    auto shared_to_vgpr = [&]() {
      for (int i = 0; i < MS; ++i)
        A_block_vgpr1[i] = A_shared[64 * i + (tid % 64)];
      for (int j = 0; j < NS / 4; ++j)
        B_block_vgpr1[j] = B_shared[256 * j + tid];
    };

    auto mfma = [&]() {
      for (int i = 0; i < MS; ++i) {
        for (int j = 0; j < NS / 4; ++j) {
          for (int k = 0; k < 2; ++k) {
            acc[i][j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
                A_block_vgpr1[i][k], B_block_vgpr1[j][k], acc[i][j], 0, 0, 0);
          }
        }
      }
    };

    auto sync = [] { __syncthreads(); };

    global_to_vgpr0();
    vpgr0_to_shared();
    if (K_outer >= 2) {
      global_to_vgpr0();
      sync();
      shared_to_vgpr();
      for (int k_outer = 0; k_outer < K_outer - 2; ++k_outer) {
        sync();
        vpgr0_to_shared();
        global_to_vgpr0();
        mfma();
        sync();
        shared_to_vgpr();
      }
      sync();
      vpgr0_to_shared();
      mfma();
    }
    sync();
    shared_to_vgpr();
    mfma();

    int32x4_t *C_ptr = static_cast<int32x4_t *>(C_data) +
                       MS * NS * 16 * 4 * (N_outer * m_outer + n_outer);
    for (int i = 0; i < MS; ++i) {
      for (int j = 0; j < NS / 4; ++j) {
        C_ptr[256 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

int main() {
  std::printf("Best-performing kernels for each element types:\n\n");
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipeline_v3<
       8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipelineload<
       12, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<8, 8>());
  test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 8>());

  std::printf("\n\n\nOther kernels:\n\n");
  test(MmtKernel_generic<Type::SI8, Type::SI8, Type::SI32, 3, 5, 2>());
  test(MmtKernel_generic<Type::FP16, Type::FP16, Type::FP32, 3, 5, 2>());
  test(MmtKernel_generic<Type::FP32, Type::FP32, Type::FP32, 3, 5, 2>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_32x32x16i8_shared<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_32x32x16i8_shared<4, 8>());
  test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_rowmajor());
  test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_directAB_rowmajorC());
  test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct());
  test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct_Kx4());
  test(MmtKernel_64t_amdgcn_mfma_f32_16x16x4f32_direct_Kx4_unrollx4());
  test(MmtKernel_128t_1x2_amdgcn_mfma_f32_16x16x4f32_direct());
  test(MmtKernel_256t_2x2_amdgcn_mfma_f32_16x16x4f32_direct());
  test(MmtKernel_256t_2x2_amdgcn_mfma_f32_16x16x4f32_shared());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<4, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<4, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<8, 8>());
  test(
      MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<8, 12>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<4, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<4, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<8, 12>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<4, 4>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<4, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 12>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 16>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<16, 16>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_subgroup2x2<
       8, 8>());
  test(
      MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_misguidednobankconflicts<
          8, 8>());
  test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 12>());
  test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 16>());
  test(MmtKernel_1024t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<16, 16>());
  test(
      MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_doublebuffer_naive<
          8, 8>());
  test(
      MmtKernel_256t_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_doublebuffer_take2<
          8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2<8, 8>());
  test(MmtKernel_256t_MSxNS_amdgcn_mfma_i32_16x16x32i8_shared_Kx2_pipelineload<
       8, 8>());

  std::printf("\n\n\nStream-K experiments:\n\n");
  test(
      MmtKernel_StreamK_256t_MSxNS_amdgcn_mfma_f32_16x16x16f16_shared_Kx2<8,
                                                                          8>());
}
