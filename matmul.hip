// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>

#include <cxxabi.h>
#include <functional>
#include <random>
#include <typeinfo>
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
  int rows_tile, cols_tile;
  tile_layout_func_t tile_layout;
};

struct TiledMmtShape {
  int M_outer, N_outer, K_outer;
  int M_tile, N_tile, K_tile;
  tile_layout_func_t A_tile_layout, B_tile_layout, C_tile_layout;
};

__device__ __host__ TiledMatrixShape A_shape(const TiledMmtShape &s) {
  return {s.M_outer, s.K_outer, s.M_tile, s.K_tile, s.A_tile_layout};
}

__device__ __host__ TiledMatrixShape B_shape(const TiledMmtShape &s) {
  return {s.N_outer, s.K_outer, s.N_tile, s.K_tile, s.B_tile_layout};
}

__device__ __host__ TiledMatrixShape C_shape(const TiledMmtShape &s) {
  return {s.M_outer, s.N_outer, s.M_tile, s.N_tile, s.C_tile_layout};
}

__device__ __host__ int flatsize(const TiledMatrixShape &s) {
  return s.rows_outer * s.cols_outer * s.rows_tile * s.cols_tile;
}

__device__ __host__ int offset(const TiledMatrixShape &s, int r_outer,
                               int c_outer, int r_tile, int c_tile) {
  return s.tile_layout(r_tile, c_tile) +
         s.rows_tile * s.cols_tile * (c_outer + s.cols_outer * r_outer);
}

enum class Type { SI8, SI16, SI32, FP16, FP32 };

const char *str(Type t) {
  switch (t) {
  case Type::SI8:
    return "si8";
  case Type::SI16:
    return "si16";
  case Type::SI32:
    return "si32";
  case Type::FP16:
    return "fp16";
  case Type::FP32:
    return "fp32";
  }
}

int type_size(Type t) {
  switch (t) {
  case Type::SI8:
    return 1;
  case Type::SI16:
    return 2;
  case Type::SI32:
    return 4;
  case Type::FP16:
    return 2;
  case Type::FP32:
    return 4;
  }
}

template <Type t> struct CTypeImpl {};
template <> struct CTypeImpl<Type::SI8> {
  using type = int8_t;
};
template <> struct CTypeImpl<Type::SI16> {
  using type = int16_t;
};
template <> struct CTypeImpl<Type::SI32> {
  using type = int32_t;
};
template <> struct CTypeImpl<Type::FP16> {
  using type = __fp16;
};
template <> struct CTypeImpl<Type::FP32> {
  using type = float;
};
template <Type t> using CType = CTypeImpl<t>::type;

template <Type type>
void fillRandomBuffer(int size, std::minstd_rand &r, void *out_buffer) {
  using T = CType<type>;
  T *out_buffer_typed = static_cast<T *>(out_buffer);
  for (int i = 0; i < size; ++i) {
    // Generate small integers in [-2, +2] so products are in [-4, +4] so
    // accumulators are in [-4K, +4K] for accumulation depth K so they're
    // exactly representable, float rounding is exact and we don't need
    // fuzzy compares.
    out_buffer_typed[i] = static_cast<T>(static_cast<int>((r() % 5)) - 2);
  }
}

std::vector<std::byte> makeRandomBuffer(Type type, int size,
                                        std::minstd_rand &r) {
  int bytes = size * type_size(type);
  std::vector<std::byte> result(bytes);
  if (type == Type::SI8) {
    fillRandomBuffer<Type::SI8>(size, r, result.data());
  } else if (type == Type::SI16) {
    fillRandomBuffer<Type::SI16>(size, r, result.data());
  } else if (type == Type::SI32) {
    fillRandomBuffer<Type::SI32>(size, r, result.data());
  } else if (type == Type::FP16) {
    fillRandomBuffer<Type::FP16>(size, r, result.data());
  } else if (type == Type::FP32) {
    fillRandomBuffer<Type::FP32>(size, r, result.data());
  }
  return result;
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
  for (int m_outer : {0, s.M_outer - 1}) {
    for (int n_outer : {0, s.N_outer - 1}) {
      for (int m_tile = 0; m_tile < s.M_tile; ++m_tile) {
        for (int n_tile = 0; n_tile < s.N_tile; ++n_tile) {
          float c = 0.f;
          for (int k_outer = 0; k_outer < s.K_outer; ++k_outer) {
            for (int k_tile = 0; k_tile < s.K_tile; ++k_tile) {
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
                    "n_tile=%d, at %s:%d\n",
                    static_cast<float>(actual), static_cast<float>(expected),
                    m_outer, n_outer, m_tile, n_tile, __FILE__, __LINE__);
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

typedef void (*mmt_func_t)(const void *, const void *, void *, int, int);

class MmtKernel {
public:
  virtual ~MmtKernel(){};
  virtual Type A_type() const = 0;
  virtual Type B_type() const = 0;
  virtual Type C_type() const = 0;
  virtual int M_tile() const = 0;
  virtual int N_tile() const = 0;
  virtual int K_tile() const = 0;
  virtual tile_layout_func_t A_tile_layout() const = 0;
  virtual tile_layout_func_t B_tile_layout() const = 0;
  virtual tile_layout_func_t C_tile_layout() const = 0;
  virtual int num_threads() const = 0;
  virtual mmt_func_t mmt_func() const = 0;
};

TiledMmtShape getTestShape(const MmtKernel &kernel) {
  int M = getIntEnvVar("M", 4096);
  int N = getIntEnvVar("N", 4096);
  int K = getIntEnvVar("K", 4096);
  TiledMmtShape s;
  s.M_outer = std::max(1, M / kernel.M_tile());
  s.N_outer = std::max(1, N / kernel.N_tile());
  s.K_outer = std::max(1, K / kernel.K_tile());
  s.M_tile = kernel.M_tile();
  s.N_tile = kernel.N_tile();
  s.K_tile = kernel.K_tile();
  s.A_tile_layout = kernel.A_tile_layout();
  s.B_tile_layout = kernel.B_tile_layout();
  s.C_tile_layout = kernel.C_tile_layout();
  return s;
}

void check(const MmtKernel &kernel, const TiledMmtShape &s) {
  std::minstd_rand random_engine;
  std::vector<std::byte> A_host_data =
      makeRandomBuffer(kernel.A_type(), flatsize(A_shape(s)), random_engine);
  std::vector<std::byte> B_host_data =
      makeRandomBuffer(kernel.B_type(), flatsize(B_shape(s)), random_engine);
  std::vector<std::byte> C_host_data =
      makeRandomBuffer(kernel.C_type(), flatsize(C_shape(s)), random_engine);

  float *A_device_buffer{};
  float *B_device_buffer{};
  float *C_device_buffer{};
  TiledMmtShape *shape_device_buffer{};
  HIP_CHECK(hipMalloc(&A_device_buffer, A_host_data.size()));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_host_data.size()));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_host_data.size()));
  HIP_CHECK(hipMalloc(&shape_device_buffer, sizeof s));

  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(shape_device_buffer, &s, sizeof s, hipMemcpyHostToDevice));

  const dim3 grid_dim(s.M_outer, s.N_outer);
  const dim3 block_dim(kernel.num_threads());

  kernel.mmt_func()<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
      A_device_buffer, B_device_buffer, C_device_buffer, s.N_outer, s.K_outer);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_host_data.data(), C_device_buffer, C_host_data.size(),
                      hipMemcpyDeviceToHost));
  checkMmtResults(kernel.A_type(), kernel.B_type(), kernel.C_type(),
                  A_host_data.data(), B_host_data.data(), C_host_data.data(),
                  s);

  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
  HIP_CHECK(hipFree(shape_device_buffer));
}

void benchmark(const MmtKernel &kernel, const TiledMmtShape &s) {
  std::minstd_rand random_engine;
  std::vector<std::byte> A_host_data =
      makeRandomBuffer(kernel.A_type(), flatsize(A_shape(s)), random_engine);
  std::vector<std::byte> B_host_data =
      makeRandomBuffer(kernel.B_type(), flatsize(B_shape(s)), random_engine);
  std::vector<std::byte> C_host_data =
      makeRandomBuffer(kernel.C_type(), flatsize(C_shape(s)), random_engine);

  float *A_device_buffer{};
  float *B_device_buffer{};
  float *C_device_buffer{};
  TiledMmtShape *shape_device_buffer{};
  HIP_CHECK(hipMalloc(&A_device_buffer, A_host_data.size()));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_host_data.size()));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_host_data.size()));
  HIP_CHECK(hipMalloc(&shape_device_buffer, sizeof s));

  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(shape_device_buffer, &s, sizeof s, hipMemcpyHostToDevice));

  const dim3 grid_dim(s.M_outer, s.N_outer);
  const dim3 block_dim(kernel.num_threads());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  float batch_ms{};
  float min_batch_ms = getIntEnvVar("BENCHMARK_MIN_MS", 100);
  int fixed_batch_size = getIntEnvVar("FIXED_BATCH_SIZE", 0);
  int batch_size = fixed_batch_size ? fixed_batch_size : 1;
  while (true) {
    HIP_CHECK(hipEventRecord(start, hipStreamDefault));
    for (int b = 0; b < batch_size; ++b) {
      kernel.mmt_func()<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
          A_device_buffer, B_device_buffer, C_device_buffer, s.N_outer,
          s.K_outer);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));
    HIP_CHECK(hipEventElapsedTime(&batch_ms, start, stop));
    if (batch_ms >= min_batch_ms || fixed_batch_size) {
      break;
    }
    if (batch_size > (1 << 20)) {
      fprintf(stderr, "Vacuous kernel? Only taking %g ms at batch_size=%d.\n",
              batch_ms, batch_size);
      abort();
    }
    batch_size *= 2;
  }
  float kernel_ms = batch_ms / batch_size;
  float kernel_ops =
      2.f * s.M_outer * s.N_outer * s.K_outer * s.M_tile * s.N_tile * s.K_tile;
  float kernel_bytes_read = static_cast<float>(sizeof(float)) * s.M_outer *
                            s.N_outer * s.K_outer * (s.M_tile + s.N_tile) *
                            s.K_tile;
  float kernel_ops_per_s = 1000.f * kernel_ops / kernel_ms;
  float kernel_bytes_read_per_s = 1000.f * kernel_bytes_read / kernel_ms;
  printf("%.3g Tflop/s, read %.3g TB/s, batch_size=%d\n", 1.e-12f * kernel_ops_per_s,
         1e-12f * kernel_bytes_read_per_s, batch_size);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
  HIP_CHECK(hipFree(shape_device_buffer));
}

void test(const MmtKernel &kernel) {
  char *name =
      abi::__cxa_demangle(typeid(kernel).name(), nullptr, nullptr, nullptr);
  const char *filter = getenv("FILTER");
  if (filter && !strstr(name, filter)) {
    return;
  }
  TiledMmtShape s = getTestShape(kernel);
  printf("%s: A:%s, B:%s, C:%s, MxNxK: outer=(%dx%dx%d), tile=(%dx%dx%d), "
         "num_threads=%d\n",
         name, str(kernel.A_type()), str(kernel.B_type()), str(kernel.C_type()),
         s.M_outer, s.N_outer, s.K_outer, s.M_tile, s.N_tile, s.K_tile,
         kernel.num_threads());
  free(name);

  if (!getenv("SKIP_CHECK")) {
    printf("  Checking correctness... ");
    // Test with more generic shapes than just M==N==K==2^x.
    TiledMmtShape s_tweaked{s};
    if (!getenv("SKIP_SHAPE_TWEAK")) {
      s_tweaked.M_outer += 1;
      s_tweaked.N_outer += 3;
      s_tweaked.K_outer += 5;
    }
    check(kernel, s_tweaked);
    printf("OK\n");
  }

  printf("  Benchmarking... ");
  benchmark(kernel, s);
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
                             void *C_data, int N_outer, int K_outer) {
    using TA = CType<T_A_type>;
    using TB = CType<T_B_type>;
    using TC = CType<T_C_type>;
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int m_tile = threadIdx.x / T_N_tile;
    int n_tile = threadIdx.x % T_N_tile;
    float c = 0.f;
    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int k_tile = 0; k_tile < T_K_tile; ++k_tile) {
        float a = static_cast<const TA *>(
            A_data)[k_tile +
                    T_K_tile *
                        (m_tile + T_M_tile * (k_outer + K_outer * m_outer))];
        float b = static_cast<const TB *>(
            B_data)[k_tile +
                    T_K_tile *
                        (n_tile + T_N_tile * (k_outer + K_outer * n_outer))];
        c += a * b;
      }
    }
    static_cast<TC *>(
        C_data)[n_tile + T_N_tile * (m_tile + T_M_tile * (n_outer +
                                                          N_outer * m_outer))] =
        c;
  }
};

class MmtKernel_amdgcn_mfma_f32_16x16x4f32_rowmajor : public MmtKernel {
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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

class MmtKernel_amdgcn_mfma_f32_16x16x4f32_directAB_rowmajorC
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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

class MmtKernel_amdgcn_mfma_f32_16x16x4f32_direct : public MmtKernel {
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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

class MmtKernel_amdgcn_mfma_f32_16x16x4f32_direct_Kx4 : public MmtKernel {
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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

class MmtKernel_amdgcn_mfma_f32_16x16x4f32_direct_Kx4_unrollx4
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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

class MmtKernel_1x2_amdgcn_mfma_f32_16x16x4f32_direct : public MmtKernel {
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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

class MmtKernel_2x2_amdgcn_mfma_f32_16x16x4f32_direct : public MmtKernel {
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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

class MmtKernel_2x2_amdgcn_mfma_f32_16x16x4f32_shared : public MmtKernel {
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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
class MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB
    : public MmtKernel {
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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
class MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared : public MmtKernel {
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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
class MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4 : public MmtKernel {
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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
        C_ptr[16 * 16 * (NS / 4 * i + j) + tid] = acc[i][j];
      }
    }
  }
};

template <int MS, int NS>
class MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_nobankconflicts
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
  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int N_outer, int K_outer) {
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

int main() {
  test(MmtKernel_generic<Type::FP32, Type::FP32, Type::FP32, 3, 5, 2>());
  test(MmtKernel_generic<Type::FP16, Type::FP16, Type::FP32, 3, 5, 2>());
  test(MmtKernel_generic<Type::SI8, Type::SI8, Type::SI32, 3, 5, 2>());
  test(MmtKernel_amdgcn_mfma_f32_16x16x4f32_rowmajor());
  test(MmtKernel_amdgcn_mfma_f32_16x16x4f32_directAB_rowmajorC());
  test(MmtKernel_amdgcn_mfma_f32_16x16x4f32_direct());
  test(MmtKernel_amdgcn_mfma_f32_16x16x4f32_direct_Kx4());
  test(MmtKernel_amdgcn_mfma_f32_16x16x4f32_direct_Kx4_unrollx4());
  test(MmtKernel_1x2_amdgcn_mfma_f32_16x16x4f32_direct());
  test(MmtKernel_2x2_amdgcn_mfma_f32_16x16x4f32_direct());
  test(MmtKernel_2x2_amdgcn_mfma_f32_16x16x4f32_shared());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<4, 4>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<4, 4>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<4, 4>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<4, 8>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<4, 8>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<4, 8>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<8, 8>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<8, 8>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4_nobankconflicts<
       8, 8>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 8>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_directA_sharedB<8, 12>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared<8, 12>());
  test(MmtKernel_MSxNS_amdgcn_mfma_f32_16x16x4f32_shared_Kx4<8, 12>());
}
