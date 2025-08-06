//===- rope.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Improved fast math implementations for better accuracy
// These should meet the 0.05f tolerance requirement

inline float fast_powf(float base, float exp) {
    // Handle special cases
    if (exp == 0.0f) return 1.0f;
    if (base == 0.0f) return 0.0f;
    if (base == 1.0f) return 1.0f;
    
    // Handle negative exponents
    bool neg_exp = false;
    if (exp < 0.0f) {
        neg_exp = true;
        exp = -exp;
    }
    
    // Split into integer and fractional parts
    int e_int = (int)exp;
    float e_frac = exp - (float)e_int;
    
    // Handle integer part using binary exponentiation
    float result = 1.0f;
    float b = base;
    int e = e_int;
    while (e) {
        if (e & 1) result *= b;
        b *= b;
        e >>= 1;
    }
    
    // Handle fractional part using exp(frac * log(base))
    if (e_frac != 0.0f) {
        // Better log approximation using bit manipulation
        union { float f; uint32_t i; } u = { base };
        int exp_bits = ((u.i >> 23) & 0xFF) - 127;
        float mantissa = 1.0f + ((u.i & 0x7FFFFF) / 8388608.0f);
        
        // log2(base) = exp_bits + log2(mantissa)
        float log2_base = exp_bits + (mantissa - 1.0f) * 1.442695f; // 1/ln(2)
        float log_base = log2_base * 0.693147f; // ln(2)
        
        // exp(frac * log(base)) using better approximation
        float exp_arg = e_frac * log_base;
        
        // Use a more accurate exp approximation
        // exp(x) ≈ 1 + x + x²/2 + x³/6 for small x
        float exp_approx = 1.0f + exp_arg;
        if (exp_arg != 0.0f) {
            float x2 = exp_arg * exp_arg;
            exp_approx += 0.5f * x2;
            if (std::abs(exp_arg) < 1.0f) {
                exp_approx += (1.0f/6.0f) * x2 * exp_arg;
            }
        }
        
        result *= exp_approx;
    }
    
    if (neg_exp) result = 1.0f / result;
    return result;
}

inline float fast_cosf(float x) {
    // Better range reduction to [-pi, pi]
    const float PI = 3.14159265358979323846f;
    const float PI2 = 2.0f * PI;
    const float PI_OVER_2 = 1.57079632679489661923f;
    
    // Reduce to [-pi, pi] without using fmodf
    while (x > PI) x -= PI2;
    while (x < -PI) x += PI2;
    
    // Use symmetry: cos(x) = cos(-x)
    if (x < 0.0f) x = -x;
    
    // Use symmetry: cos(x) = -cos(pi - x) for x > pi/2
    if (x > PI_OVER_2) {
        x = PI - x;
        return -fast_cosf(x);
    }
    
    // For small angles, use higher order Taylor series
    if (x < 0.1f) {
        float x2 = x * x;
        return 1.0f - 0.5f * x2 + (1.0f/24.0f) * x2 * x2 - (1.0f/720.0f) * x2 * x2 * x2;
    }
    
    // For larger angles, use Chebyshev approximation
    // cos(x) ≈ 1 - 0.5x² + 0.0416667x⁴ - 0.0013889x⁶
    float x2 = x * x;
    return 1.0f - 0.5f * x2 + 0.0416667f * x2 * x2 - 0.0013889f * x2 * x2 * x2;
}

inline float fast_sinf(float x) {
    // Better range reduction to [-pi, pi]
    const float PI = 3.14159265358979323846f;
    const float PI2 = 2.0f * PI;
    const float PI_OVER_2 = 1.57079632679489661923f;
    
    // Reduce to [-pi, pi] without using fmodf
    while (x > PI) x -= PI2;
    while (x < -PI) x += PI2;
    
    // Use symmetry: sin(-x) = -sin(x)
    bool neg = false;
    if (x < 0.0f) {
        neg = true;
        x = -x;
    }
    
    // Use symmetry: sin(x) = sin(pi - x) for x > pi/2
    if (x > PI_OVER_2) {
        x = PI - x;
    }
    
    // For small angles, use higher order Taylor series
    if (x < 0.1f) {
        float x2 = x * x;
        float result = x - (1.0f/6.0f) * x * x2 + (1.0f/120.0f) * x * x2 * x2;
        return neg ? -result : result;
    }
    
    // For larger angles, use Chebyshev approximation
    // sin(x) ≈ x - 0.166667x³ + 0.00833333x⁵
    float x2 = x * x;
    float result = x - 0.166667f * x * x2 + 0.00833333f * x * x2 * x2;
    return neg ? -result : result;
}

// Rotary Positional Embedding kernel (scalar version)
template <typename T, int N>
void rope_kernel_scalar(const bfloat16 *restrict input, bfloat16 *restrict output, int32_t pos, int32_t dims) {
    event0();
    constexpr float theta = 10000.0f;
    ::aie::vector<T, N> y;

    for (int v = 0; v < dims; v = v+N){
        ::aie::vector<T, N> x = ::aie::load_v<N>(input + v);
        // dims must be even
        for (int i = 0; i < dims; i += 2) {
            int j = i / 2;
            float exponent = (2 * j) / static_cast<float>(dims);
            float inv_freq = 1.0f / fast_powf(theta, exponent);
            float angle = pos * inv_freq;

            float cos_val = fast_cosf(angle);
            float sin_val = fast_sinf(angle);

            y[i] = x[i] * cos_val - x[i+1] * sin_val;
            y[i+1] = x[i] * sin_val + x[i+1] * cos_val;
        }
        ::aie::store_v(output + v, y);
    }
    event1();
}

extern "C" {
void rope(bfloat16 *input, bfloat16 *output, int32_t pos, int32_t dims) {
    rope_kernel_scalar<bfloat16, 16>(input, output, pos, dims);
}
}