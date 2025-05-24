#pragma once

// #include <inttypes.h>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

// Define custom is_same and is_same_v for cutlass
namespace std {

template<typename T, typename U>
struct is_same {
    static constexpr bool value = false;
};

template<typename T>
struct is_same<T, T> {
    static constexpr bool value = true;
};

template<typename T, typename U>
inline constexpr bool is_same_v = is_same<T, U>::value;

}

using i32 = int32_t;
using i64 = int64_t;

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

template <typename T> constexpr T c_one = T(1.0);
template <typename T> constexpr T c_zero = T(0.0);
template <typename T> constexpr T c_two = T(2.0);
template <typename T> constexpr T c_negative_one = T(-1.0);
template <typename T> constexpr T c_negative_two = T(-2.0);

enum class NormType {
    L1,
    L2,
    P
};

enum class Majorness {
    COL_MAJOR,
    ROW_MAJOR
};

enum class Alignment {
    ALIGN_1,
    ALIGN_4
};

enum class InnerOperator {
    DIFF,
    DOT
};

enum class PowerType {
    NOOP,
    ABS,
    SQUARE,
    SQRT,
    POW
};

enum class KernelType {
    NONE,
    LAPLACE,
    GAUSSIAN
};

template <class T>
static
__forceinline__
__device__
T
_abs(
    T v
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);
    if constexpr (std::is_same_v<T, f32>) {
        return fabsf(v);
    } else if constexpr (std::is_same_v<T, f64>) {
        return fabs(v);
    }
}

// Forces sqrt.approx for f32. Skips having to use --use_fast_math
template <class T>
static
__forceinline__
__device__
T
_sqrt(
    T v
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);
    if constexpr (std::is_same_v<T, f32>) {
        float result;
        asm(
            "sqrt.approx.ftz.f32 %0, %1;\n\t" 
            : "=f"(result) : "f"(v)
        );
        return result;
    } else if constexpr (std::is_same_v<T, f64>) {
        return sqrt(v);
    }
}

// Forces mul and ex2.approx for f32. Skips having to use --use_fast_math
template <class T>
static
__forceinline__
__device__
T 
_exp(
	T v
) {
	static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);
	if constexpr (std::is_same_v<T,f32>) {
        float result;
        asm(
            "mul.ftz.f32 %0, %1, 0f3FB8AA3B;\n\t"
            "ex2.approx.ftz.f32 %0, %0;\n\t" 
            : "=f"(result) : "f"(v)
        );
        return result;
    } else if constexpr (std::is_same_v<T,f64>) {
        return exp(v);
    }
}


// Forces lg2.approx, mul and ex2.approx for f32. Skips having to use --use_fast_math
template <class T>
static
__forceinline__
__device__
T 
_pow(
	T v,
    T p
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);
    if constexpr (std::is_same_v<T,f32>) {
        float result;
        asm(
            "lg2.approx.ftz.f32 %0, %1;\n\t"
            "mul.ftz.f32 %0, %0, %2;\n\t"
            "ex2.approx.ftz.f32 %0, %0;\n\t"
            : "=f"(result) : "f"(v), "f"(p)
        );
        return result;
    } else if constexpr (std::is_same_v<T,f64>) {
        return pow(v,p);
    }
}

template <class T>
static
__forceinline__
__device__
T
_nan() {
	static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);

	if constexpr (std::is_same_v<T,f32>) {
		return nanf("");
	} else if constexpr (std::is_same_v<T,f64>) {
		return nan("");
	}
}

template <class T>
static
__forceinline__
__device__
T 
_signum(
    T v
) {
    static_assert(std::is_same_v<T, f32> || std::is_same_v<T, f64>);
    if constexpr (std::is_same_v<T,f32>) {
		return copysign(1.0f, v);
	} else if constexpr (std::is_same_v<T,f64>) {
		return copysign(1.0, v);
	}
}
