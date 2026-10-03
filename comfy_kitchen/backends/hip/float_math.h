// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hip/hip_runtime.h>

namespace comfy::hip_backend {

// Preserve IEEE division in kernels compiled with -ffast-math.
__forceinline__ __device__ float ieee_div_f32(float numerator, float denominator) {
    bool denominator_scale = false;
    bool scale = false;
    const float scaled_denominator = __builtin_amdgcn_div_scalef(
        numerator, denominator, false, &denominator_scale);
    const float scaled_numerator = __builtin_amdgcn_div_scalef(
        numerator, denominator, true, &scale);
    float reciprocal = __builtin_amdgcn_rcpf(scaled_denominator);
    const float reciprocal_error = fmaf(-scaled_denominator, reciprocal, 1.0f);
    reciprocal = fmaf(reciprocal_error, reciprocal, reciprocal);
    float quotient = scaled_numerator * reciprocal;
    float remainder = fmaf(-scaled_denominator, quotient, scaled_numerator);
    quotient = fmaf(remainder, reciprocal, quotient);
    remainder = fmaf(-scaled_denominator, quotient, scaled_numerator);
    quotient = __builtin_amdgcn_div_fmasf(remainder, reciprocal, quotient, scale);
    return __builtin_amdgcn_div_fixupf(quotient, denominator, numerator);
}

}  // namespace comfy::hip_backend
