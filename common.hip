// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HIP_MATMUL_COMMON
#define HIP_MATMUL_COMMON

#include <hip/hip_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

inline void hip_check_impl(hipError_t hip_error_code, const char *condstr,
                    const char *file, int line) {
  if (hip_error_code != hipSuccess) {
    fprintf(stderr, "HIP Error \"%s\" produced by `%s` at %s:%d\n",
            hipGetErrorString(hip_error_code), condstr, file, line);
    exit(EXIT_FAILURE);
  }
}

#define HIP_CHECK(expr) hip_check_impl(expr, #expr, __FILE__, __LINE__)

inline int getIntEnvVar(const char *name, int default_val) {
  const char *env = std::getenv(name);
  return env ? std::stoi(env) : default_val;
};

struct MNKShape {
  int M, N, K;
};

enum class Type { SI8, SI16, SI32, FP16, FP32 };

inline const char *str(Type t) {
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

inline __device__ __host__ int type_size(Type t) {
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
  using type = _Float16;
};
template <> struct CTypeImpl<Type::FP32> {
  using type = float;
};
template <Type t> using CType = typename CTypeImpl<t>::type;

template <typename A, typename B>
__device__ __host__ std::common_type_t<A, B> ceil_div(A a, B b) {
  return (a + b - 1) / b;
}

template <int Po2, typename A> __device__ __host__ A round_up_to_po2(A a) {
  static_assert(Po2 > 0 && (Po2 & (Po2 - 1)) == 0);
  return ((a - 1) | (Po2 - 1)) + 1;
}

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

inline std::vector<std::byte> makeRandomBuffer(Type type, int size,
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

#endif
