// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demonstrate modelling AMDGPU MFMA builtins on CPU, with normal CPU threads
// playing the role of GPU threads.

// Compile and run:
//
//   clang++ --std=c++20 mfma_on_cpu_threads.cc -o /tmp/m && /tmp/m
//

#include <barrier>
#include <cassert>
#include <functional>
#include <thread>

using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;

constexpr int threads_per_subgroup = 64;

// Device kernels have a `threadIdx` implicitly passed to them. We mimic that
// here with a thread-local thread-id in [0, number_of_threads - 1].
thread_local int tid;

// Functionally equivalent to AMDGPU __builtin_amdgcn_mfma_f32_16x16x4f32,
// just with normal CPU threads instead of GPU threads!
floatx4_t cpu_mfma_f32_16x16x4f32(float a, float b, floatx4_t c) {
  assert(tid < threads_per_subgroup && "current limitation: only one subgroup");
  // Each thread is going to need `a` and `b` matrix elements passed to other
  // threads, not just the single ones that were directly passed to it.
  static float a_tile[threads_per_subgroup];
  static float b_tile[threads_per_subgroup];
  // Record the single `a` and `b` elements directly passed to us, so other
  // threads can see them.
  a_tile[tid] = a;
  b_tile[tid] = b;
  // Wait for all threads to have made their contributions to `{a,b}_tile`.
  static std::barrier barrier(threads_per_subgroup, []() {});
  barrier.arrive_and_wait();
  // Now perform the computation, now that we can see the whole `{a,b}_tile`.
  int m = 4 * (tid / 16);
  int n = tid % 16;
  for (int k = 0; k < 4; ++k) {
    for (int p = 0; p < 4; ++p) {
      c[p] += a_tile[16 * k + m + p] * b_tile[16 * k + n];
    }
  }
  return c;
}

// A "device kernel" using cpu_mfma_f32_16x16x4f32. Think of it as the
// __global__ kernel that you would use in a kernel launch.
void matmul_kernel_f32_16x16x4f32(const float *a, const float *b,
                                  floatx4_t *c) {
  c[tid] = cpu_mfma_f32_16x16x4f32(a[tid], b[tid], c[tid]);
}

// Test helper.
void init_test_matrices(float *a, float *b, floatx4_t *c) {
  for (int i = 0; i < 64; ++i) {
    a[i] = ((i / 16) == (i % 4));
    b[i] = ((i / 16) == (i % 4));
    c[i] = floatx4_t{static_cast<float>(i), 0, 0, 0};
  }
}

// Test helper.
void print_a_matrix(const char *label, const float *a) {
  printf("%s:\n", label);
  for (int m = 0; m < 16; ++m) {
    for (int k = 0; k < 4; ++k) {
      printf("%4g ", a[16 * k + m]);
    }
    printf("\n");
  }
  printf("\n");
}

// Test helper.
void print_b_matrix(const char *label, const float *b) {
  printf("%s:\n", label);
  for (int k = 0; k < 4; ++k) {
    for (int n = 0; n < 16; ++n) {
      printf("%4g ", b[16 * k + n]);
    }
    printf("\n");
  }
  printf("\n");
}

// Test helper.
void print_c_matrix(const char *label, const floatx4_t *c) {
  printf("%s:\n", label);
  for (int m = 0; m < 16; m += 4) {
    for (int p = 0; p < 4; ++p) {
      for (int n = 0; n < 16; ++n) {
        printf("%4g ", c[4 * m + n][p]);
      }
      printf("\n");
    }
  }
  printf("\n");
}

int main() {
  float a[64];
  float b[64];
  floatx4_t c[64];
  init_test_matrices(a, b, c);
  print_a_matrix("A matrix", a);
  print_b_matrix("B matrix", b);
  print_c_matrix("C matrix", c);

  // Think of the following block as a device kernel<<<...>>> launch.
  {
    std::vector<std::jthread> threads;
    for (int i = 0; i < threads_per_subgroup; ++i) {
      auto thread_func = [&](int tid_arg) {
        tid = tid_arg;
        matmul_kernel_f32_16x16x4f32(a, b, c);
      };
      threads.emplace_back(thread_func, /*tid_arg=*/i);
    }
  }

  print_c_matrix("Result matrix", c);
}
