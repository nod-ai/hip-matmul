// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hip"

#include <cassert>
#include <cstdio>
#include <random>
#include <type_traits>
#include <vector>

// Device function that implements a matrix vector multiplication over data in
// the 'global' layout. This is not intended to work with data tiling.
using matvec_func_t = void (*)(const void * /*A*/, const void * /*B*/,
                               void * /*C*/, int /*M*/, int /*N*/, int /*K*/);

struct ProblemProperties {
  Type A_type;
  Type B_type;
  Type C_type;
  MNKShape total;
  MNKShape outer;
  MNKShape tile;
};

using num_threads_func_t = int (*)(const ProblemProperties & /*problem*/);
using shared_mem_bytes_func_t = size_t (*)(const ProblemProperties & /*problem*/);

size_t no_shared_mem(const ProblemProperties &) { return 0; }

void print(const ProblemProperties &problem, FILE *file = stderr) {
  fprintf(file, "A:%s, B:%s, C:%s, Total MxNxK=%dx%dx%d\n", str(problem.A_type),
          str(problem.B_type), str(problem.C_type), problem.total.M,
          problem.total.N, problem.total.K);
  fprintf(file, "\tOuter MxNxK=%dx%dx%d, Tile MxNxK=%dx%dx%d\n",
          problem.outer.M, problem.outer.N, problem.outer.K, problem.tile.M,
          problem.tile.N, problem.tile.K);
}

int flat_a_size(const ProblemProperties &problem) {
  return type_size(problem.A_type) * problem.total.M * problem.total.K;
}

int flat_b_size(const ProblemProperties &problem) {
  return type_size(problem.B_type) * problem.total.N * problem.total.K;
}

int flat_c_size(const ProblemProperties &problem) {
  return type_size(problem.C_type) * problem.total.M * problem.total.N;
}

struct MatvecKernel {
  const char *name; // Kernel name.
  Type A_type;      // Element type of the LHS matrix.
  Type B_type;      // Element type of the RHS matrix.
  Type C_type;      // Element type of the result matrix.
  int M_tile;       // M-dimension tile size (rows of accumulator).
  int N_tile;       // N-dimension tile size (columns of accumulator).
  int K_tile;       // K-dimension tile size (reduction dimension).
  num_threads_func_t num_threads_func; // Number of threads that the kernel
                                       // requires running on.
  shared_mem_bytes_func_t
      shared_mem_bytes_func;      // Amount of shared memory per workgroup.
  matvec_func_t matvec_func; // Device kernel pointer.
};

ProblemProperties getBenchmarkProblemSize(const MatvecKernel &kernel,
                                    MNKShape total) {
  ProblemProperties problem_size = {};
  problem_size.A_type = kernel.A_type;
  problem_size.B_type = kernel.B_type;
  problem_size.C_type = kernel.C_type;
  problem_size.tile = {kernel.M_tile, kernel.N_tile, kernel.K_tile};

  problem_size.total = total;
  problem_size.outer.M = ceil_div(total.M, kernel.M_tile);
  problem_size.outer.N = ceil_div(total.N, kernel.N_tile);
  problem_size.outer.K = ceil_div(total.K, kernel.K_tile);
  return problem_size;
}

ProblemProperties getCheckProblemSize(const MatvecKernel &kernel) {
  int M = getIntEnvVar("M", 4096);
  int N = getIntEnvVar("N", 1);
  int K = getIntEnvVar("K", 4096);
  MNKShape total = {M, N, K};
  return getBenchmarkProblemSize(kernel, total);
}

template <Type A_type, Type B_type, Type C_type>
void checkMatvecResults(const void *A_data_void, const void *B_data_void,
                        const void *C_data_void, const ProblemProperties &problem) {
  if (getIntEnvVar("DEBUG", 0)) {
    fprintf(stderr, "Checking matvec result\n");
    print(problem);
  }
  using TA = CType<A_type>;
  using TB = CType<B_type>;
  using TC = CType<C_type>;
  const TA *A_data = static_cast<const TA *>(A_data_void);
  const TB *B_data = static_cast<const TB *>(B_data_void);
  const TC *C_data = static_cast<const TC *>(C_data_void);
  // This reference code is slow. To make the checks not too slow on
  // large matmuls, we only check the 9 corner/middle tiles.
  for (int m_outer : {0, problem.outer.M / 2, problem.outer.M - 1}) {
    for (int n_outer : {0, problem.outer.N / 2, problem.outer.N - 1}) {
      for (int m_tile = 0; m_tile < problem.tile.M; ++m_tile) {
        for (int n_tile = 0; n_tile < problem.tile.N; ++n_tile) {
          int global_m = m_outer * problem.tile.M + m_tile;
          int global_n = n_outer * problem.tile.N + n_tile;
          TC c = {0};
          for (int k_outer = 0; k_outer < problem.outer.K; ++k_outer) {
            for (int k_tile = 0; k_tile < problem.tile.K; ++k_tile) {
              int global_k = k_outer * problem.tile.K + k_tile;
              TA a = A_data[global_m * problem.total.K + global_k];
              TB b = B_data[global_n * problem.total.K + global_k];
              c += static_cast<TC>(a) * static_cast<TC>(b);
            }
          }
          TC expected = c;
          TC actual = C_data[global_m * problem.total.N + global_n];
          if (actual != expected) {
            fprintf(stderr,
                    "matmul numerical error: actual(%g) != "
                    "expected(%g), at m_outer=%d n_outer=%d m_tile=%d "
                    "n_tile=%d, at %s:%d. Note: outer MxNxK = %dx%dx%d\n",
                    static_cast<float>(actual), static_cast<float>(expected),
                    m_outer, n_outer, m_tile, n_tile, __FILE__, __LINE__,
                    problem.outer.M, problem.outer.N, problem.outer.K);
            abort();
          }
        }
      }
    }
  }
}

void checkMatvecResults(Type A_type, Type B_type, Type C_type,
                        const void *A_data_void, const void *B_data_void,
                        const void *C_data_void, const ProblemProperties &problem) {
#define HANDLE_CASE(A, B, C)                                                   \
  if (A_type == Type::A && B_type == Type::B && C_type == Type::C) {           \
    checkMatvecResults<Type::A, Type::B, Type::C>(A_data_void, B_data_void,    \
                                                  C_data_void, problem);       \
    return;                                                                    \
  }
  HANDLE_CASE(FP32, FP32, FP32)
  HANDLE_CASE(FP16, FP16, FP32)
  HANDLE_CASE(SI8, SI8, SI32)
#undef HANDLE_CASE

  fprintf(stderr, "%s:%d: unhandled types\n", __FILE__, __LINE__);
  abort();
}

void check(const MatvecKernel &kernel, const ProblemProperties &problem) {
  std::minstd_rand random_engine;
  std::vector<std::byte> A_host_data =
      makeRandomBuffer(kernel.A_type, flat_a_size(problem), random_engine);
  std::vector<std::byte> B_host_data =
      makeRandomBuffer(kernel.B_type, flat_b_size(problem), random_engine);
  std::vector<std::byte> C_host_data =
      makeRandomBuffer(kernel.C_type, flat_c_size(problem), random_engine);

  void *A_device_buffer = nullptr;
  void *B_device_buffer = nullptr;
  void *C_device_buffer = nullptr;
  HIP_CHECK(hipMalloc(&A_device_buffer, A_host_data.size()));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&B_device_buffer, B_host_data.size()));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&C_device_buffer, C_host_data.size()));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipGetLastError());

  const dim3 grid_dim(problem.outer.M, problem.outer.N);
  const dim3 block_dim(kernel.num_threads_func(problem));
  const size_t shared_mem_bytes = kernel.shared_mem_bytes_func(problem);
  fprintf(stderr, "Num threads: %d\n", block_dim.x);
  fprintf(stderr, "Shared mem bytes: %zu\n", shared_mem_bytes);

  HIP_CHECK(hipGetLastError());
  kernel
      .matvec_func<<<grid_dim, block_dim, shared_mem_bytes, hipStreamDefault>>>(
          A_device_buffer, B_device_buffer, C_device_buffer, problem.total.M,
          problem.total.N, problem.total.K);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(C_host_data.data(), C_device_buffer, C_host_data.size(),
                      hipMemcpyDeviceToHost));
  checkMatvecResults(kernel.A_type, kernel.B_type, kernel.C_type,
                     A_host_data.data(), B_host_data.data(), C_host_data.data(),
                     problem);

  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));
}

void check(const MatvecKernel &kernel) {
  check(kernel, getCheckProblemSize(kernel));
}

void benchmark(const MatvecKernel &kernel, MNKShape total) {
  ProblemProperties problem = getBenchmarkProblemSize(kernel, total);
  std::printf("  Benchmarking: ");
  print(problem, stdout);

  std::minstd_rand random_engine;
  std::vector<std::byte> A_host_data =
      makeRandomBuffer(kernel.A_type, flat_a_size(problem), random_engine);
  std::vector<std::byte> B_host_data =
      makeRandomBuffer(kernel.B_type, flat_b_size(problem), random_engine);
  std::vector<std::byte> C_host_data =
      makeRandomBuffer(kernel.C_type, flat_c_size(problem), random_engine);

  void *A_device_buffer{};
  void *B_device_buffer{};
  void *C_device_buffer{};
  HIP_CHECK(hipMalloc(&A_device_buffer, A_host_data.size()));
  HIP_CHECK(hipMalloc(&B_device_buffer, B_host_data.size()));
  HIP_CHECK(hipMalloc(&C_device_buffer, C_host_data.size()));

  HIP_CHECK(hipMemcpy(A_device_buffer, A_host_data.data(), A_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_device_buffer, B_host_data.data(), B_host_data.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(C_device_buffer, C_host_data.data(), C_host_data.size(),
                      hipMemcpyHostToDevice));

  const dim3 grid_dim(problem.outer.M, problem.outer.N);
  const dim3 block_dim(kernel.num_threads_func(problem));
  const size_t shared_mem_bytes = kernel.shared_mem_bytes_func(problem);

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
      kernel.matvec_func<<<grid_dim, block_dim, shared_mem_bytes,
                           hipStreamDefault>>>(
          A_device_buffer, B_device_buffer, C_device_buffer, problem.total.M,
          problem.total.N, problem.total.K);
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

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(A_device_buffer));
  HIP_CHECK(hipFree(B_device_buffer));
  HIP_CHECK(hipFree(C_device_buffer));

  // Calculate the actual amount of memory read during the calculation, taking
  // into account the tile sizes.
  float A_element_bytes = type_size(kernel.A_type);
  float B_element_bytes = type_size(kernel.B_type);
  float A_bytes = A_element_bytes * problem.total.M * problem.total.K;
  float B_bytes = B_element_bytes * problem.total.N * problem.total.K;
  float kernel_bytes_read = A_bytes + B_bytes;

  float kernel_ms = elapsed_ms / iterations;
  float kernel_ops = 2.f * problem.total.M * problem.total.N * problem.total.K;
  float kernel_ops_per_s = 1000.f * kernel_ops / kernel_ms;
  float kernel_bytes_read_per_s = 1000.f * kernel_bytes_read / kernel_ms;
  std::printf(
      "\tRead %.4g TB/s, %.4g Tflop/s, latency %.2g ms, iterations=%d\n",
      1.e-12f * kernel_ops_per_s, 1e-12f * kernel_bytes_read_per_s, kernel_ms,
      iterations);
}

struct NaiveMatmulKernel : MatvecKernel {
  using TA = float;
  using TB = float;
  using TC = float;
  static constexpr int T_M_tile = 64; // One subgroup.
  static constexpr int T_N_tile = 1;
  static constexpr int T_K_tile = 4; // dword_x4.

  NaiveMatmulKernel() {
    name = __FUNCTION__;
    A_type = Type::FP32;
    B_type = Type::FP32;
    C_type = Type::FP32;
    M_tile = T_M_tile;
    N_tile = T_N_tile;
    K_tile = T_K_tile;
    num_threads_func = [](const ProblemProperties &) {
      return T_M_tile * T_N_tile;
    };
    shared_mem_bytes_func = no_shared_mem;
    matvec_func = run;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int m_tile = threadIdx.x / T_N_tile;
    int n_tile = threadIdx.x % T_N_tile;

    int K_outer = ceil_div(K, T_K_tile);

    int global_m = m_outer * T_M_tile + m_tile;
    int global_n = n_outer * T_N_tile + n_tile;
    if (global_m >= M || global_n >= N)
      return;

    TC c = {0};
    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int k_tile = 0; k_tile < T_K_tile; ++k_tile) {
        int global_k = k_outer * T_K_tile + k_tile;
        if (global_k >= K) {
          break;
        }

        TA a = static_cast<const TA *>(A_data)[global_m * K + global_k];
        TB b = static_cast<const TB *>(B_data)[global_n * K + global_k];
        c += static_cast<TC>(a) * static_cast<TC>(b);
      }
    }

    static_cast<TC *>(C_data)[global_m * N + global_n] = c;
  }
};

template <typename T> __device__ static T fma(T a, T b, T c) {
  if constexpr (std::is_integral_v<T>) {
    return a * b + c;
  } else {
    return std::fma(a, b, c);
  }
}

template <typename T> __device__ static T subgroup_reduce(T value) {
  for (int offset = warpSize >> 1; offset > 0; offset = offset >> 1) {
    value += __shfl_down(value, offset, warpSize);
  }
  return value;
};

template <int T_M_tile, int T_K_tile>
struct SubgroupReduceKernel : MatvecKernel {
  using TA = float;
  using TB = float;
  using TC = float;
  static constexpr int T_N_tile = 1;

  // In this matvec kernel, the whole subgroup processes a row and cooperatively
  // reduces the result using subgroup operations. At the end, only one thread
  // within the subgroups stores the row value to the main memory.
  SubgroupReduceKernel() {
    name = __FUNCTION__;
    A_type = Type::FP32;
    B_type = Type::FP32;
    C_type = Type::FP32;
    M_tile = T_M_tile;
    N_tile = T_N_tile;
    K_tile = T_K_tile;
    num_threads_func = [](const ProblemProperties &) { return warpSize; };
    shared_mem_bytes_func = no_shared_mem;
    matvec_func = run;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int lane = threadIdx.x % warpSize;

    // The number of adjacent elements to be accessed by a thread. This is used
    // to guarantee vectorized memory accesses, e.g., `global_load_dwordx4`.
    static constexpr int K_VEC = T_K_tile / warpSize;
    int K_outer = ceil_div(K, T_K_tile);

    int global_m = m_outer * T_M_tile;
    int global_n = n_outer * T_N_tile;
    if (global_m >= M || global_n >= N)
      return;

    int tile_k = lane * K_VEC;
    // Partial accumulator for the thread,
    // storing `K_VEC` values per row.
    TC c[T_M_tile][K_VEC] = {{0}};

    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      // Load the vector operand first to hide the latency. We will need these
      // values for every row slice of the matrix operand.
      TA b[T_N_tile][K_VEC] = {{0}};
      for (int k = 0; k < K_VEC; ++k) {
        int global_k = k_outer * T_K_tile + tile_k + k;
        b[0][k] = static_cast<const TB *>(B_data)[global_n * K + global_k];
      }

      TA a[T_M_tile][K_VEC] = {{0}};
      for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
        for (int k = 0; k < K_VEC; ++k) {
          int global_k = k_outer * T_K_tile + tile_k + k;
          a[m_tile][k] = static_cast<const TA *>(
              A_data)[(global_m + m_tile) * K + global_k];
        }
      }

      // Partial reduciton. Ideally, we'd like this to generate FMAs.
      for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
        for (int k = 0; k < K_VEC; ++k) {
          c[m_tile][k] = fma(static_cast<TC>(a[m_tile][k]),
                             static_cast<TC>(b[0][k]), c[m_tile][k]);
        }
      }
    }

    // Perform thread-level reduction first, producing a single partial result
    // per thread.
    TC res[T_M_tile] = {{0}};
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      for (int k = 0; k < K_VEC; ++k) {
        res[m_tile] += c[m_tile][k];
      }
    }

    // Perform subgroup-level reduction, such that thread 0 contains the final
    // result for the whole subgroup.
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      res[m_tile] = subgroup_reduce(res[m_tile]);
    }
    if (lane == 0) {
      for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
        static_cast<TC *>(C_data)[(global_m + m_tile) * N + global_n] =
            res[m_tile];
      }
    }
  }
};

template <int T_M_tile, int T_K_tile>
struct WorkgroupReduceKernel : MatvecKernel {
  using TA = float;
  using TB = float;
  using TC = float;
  static constexpr int T_N_tile = 1;
  static constexpr int K_VEC =
      std::min(T_K_tile, 16 / static_cast<int>(sizeof(TA)));
  static constexpr int K_outer = T_K_tile / K_VEC;
  static_assert(T_K_tile == K_VEC * K_outer, "Invalid configuration");

  // In this matvec kernel, the whole workgroups processes a row and cooperatively
  // reduces the result using subgroup operations. At the end, only one thread
  // within the subgroups stores the row value to the main memory.
  WorkgroupReduceKernel() {
    name = __FUNCTION__;
    A_type = Type::FP32;
    B_type = Type::FP32;
    C_type = Type::FP32;
    M_tile = T_M_tile;
    N_tile = T_N_tile;
    K_tile = T_K_tile;
    num_threads_func = [](const ProblemProperties &problem) {
      assert(problem.total.K % T_K_tile == 0 && "Unsupported input");
      return problem.total.K / T_K_tile;
    };
    shared_mem_bytes_func = [](const ProblemProperties &problem) {
      int num_subgroups = ceil_div(problem.total.K / T_K_tile, warpSize);
      assert(num_subgroups <= 32 && "Too many subgroups per workgroup");
      return sizeof(TC) * T_M_tile * num_subgroups;
    };
    matvec_func = run;
  }

  __global__ static void run(const void *A_data, const void *B_data,
                             void *C_data, int M, int N, int K) {
    extern __shared__ TC shared_mem[];

    const int num_subgroups = ceil_div(K / T_K_tile, warpSize);
    int m_outer = blockIdx.x;
    int n_outer = blockIdx.y;
    int lane = threadIdx.x % warpSize;
    int subgroup = threadIdx.x / warpSize;

    int global_m = m_outer * T_M_tile;
    int global_n = n_outer * T_N_tile;
    if (global_m >= M || global_n >= N)
      return;

    static constexpr int K_per_subgroup = T_K_tile * warpSize;
    static constexpr int K_per_outer = K_VEC * warpSize;
    int tile_k = subgroup * K_per_subgroup + lane * K_VEC;

    // Load the vector operand first to hide the latency. We will need these
    // values for every row slice of the matrix operand.
    TA b[T_N_tile][K_outer][K_VEC] = {{{0}}};
    for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
      for (int k = 0; k < K_VEC; ++k) {
        int global_k = tile_k + k_outer * K_per_outer + k;
        b[0][k_outer][k] =
            static_cast<const TB *>(B_data)[global_n * K + global_k];
      }
    }

    TA a[T_M_tile][K_outer][K_VEC] = {{{0}}};
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
        for (int k = 0; k < K_VEC; ++k) {
          int global_k = tile_k + k_outer * K_per_outer + k;
          a[m_tile][k_outer][k] = static_cast<const TA *>(
              A_data)[(global_m + m_tile) * K + global_k];
        }
      }
    }

    // Partial accumulator for the thread, storing `T_K_tile` values per row.
    TC c[T_M_tile][K_outer][K_VEC] = {{{0}}};
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
        for (int k = 0; k < K_VEC; ++k) {
          c[m_tile][k_outer][k] =
              fma(static_cast<TC>(a[m_tile][k_outer][k]),
                  static_cast<TC>(b[0][k_outer][k]), c[m_tile][k_outer][k]);
        }
      }
    }

    // Perform thread-level reduction first, producing a single partial
    // result per thread.
    TC res[T_M_tile] = {0};
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      for (int k_outer = 0; k_outer < K_outer; ++k_outer) {
        for (int k = 0; k < K_VEC; ++k) {
          res[m_tile] += c[m_tile][k_outer][k];
        }
      }
    }

    // Perform subgroup-level reduction, such that thread 0 contains the
    // final result for the whole subgroup.
    for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
      TC thread_res = subgroup_reduce(res[m_tile]);
      res[m_tile] = thread_res;
    }

    // Store the subgroup results in shared memory.
    if (lane == 0) {
      for (int m_tile = 0; m_tile < T_M_tile; ++m_tile) {
        shared_mem[subgroup * T_M_tile + m_tile] = res[m_tile];
      }
    }

    __syncthreads();

    // Each subgroup will reduce one or more rows.
    for (int m_tile = subgroup; m_tile < T_M_tile; m_tile += num_subgroups) {
      TC thread_res = -TC{0};
      if (lane < num_subgroups) {
        thread_res = shared_mem[lane * T_M_tile + m_tile];
      }

      TC final = subgroup_reduce(thread_res);
      if (lane == 0) {
        static_cast<TC *>(C_data)[(global_m + m_tile) * N + global_n] = final;
      }
    }
  }
};

void test(const MatvecKernel &kernel) {
  const char *filter = getenv("FILTER");
  if (filter && !strstr(kernel.name, filter)) {
    return;
  }
  std::printf("%s: A:%s, B:%s, C:%s, tile MxNxK=%dx%dx%d\n", kernel.name,
              str(kernel.A_type), str(kernel.B_type), str(kernel.C_type),
              kernel.M_tile, kernel.N_tile, kernel.K_tile);

  if (!getenv("SKIP_CHECK")) {
    check(kernel);
  }

  MNKShape test_shapes[] = {
      {14336, 1, 4096}, {4096, 1, 14336}, {4096, 1, 4096}, {1024, 1, 4096}};
  for (MNKShape shape : test_shapes) {
    benchmark(kernel, shape);
  }
}

int main() {
  test(NaiveMatmulKernel{});
  test(SubgroupReduceKernel<1, warpSize>{});
  test(SubgroupReduceKernel<1, warpSize * 4>{});
  test(SubgroupReduceKernel<4, warpSize>{});
  test(SubgroupReduceKernel<4, warpSize * 4>{});
  test(WorkgroupReduceKernel<1, 4 * 4>{});
  test(WorkgroupReduceKernel<2, 4 * 4>{});
  test(WorkgroupReduceKernel<3, 4 * 4>{});
  test(WorkgroupReduceKernel<4, 4 * 4>{});
  return 0;
}
