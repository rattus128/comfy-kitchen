# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as functional

import comfy_kitchen as ck


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.cuda,
    reason="requires an NVIDIA CUDA runtime",
)


def _cuda_backend():
    from comfy_kitchen.backends import cuda

    if not cuda._EXT_AVAILABLE:
        pytest.skip("CUDA extension is not built")
    return cuda


@pytest.mark.parametrize("width", [256, 3840])
def test_rms_gated_residual_matches_explicit_stages(width):
    _cuda_backend()

    values = torch.arange(2 * width, device="cuda", dtype=torch.float32)
    activation = (((values % 37) - 18) / 11).to(torch.bfloat16).reshape(1, 2, width)
    residual = (((values % 29) - 7) / 13).to(torch.bfloat16).reshape_as(activation)
    weight = (((torch.arange(width, device="cuda") % 17) + 3) / 19).to(torch.bfloat16)
    gate = (((torch.arange(width, device="cuda") % 23) - 9) / 12).to(torch.bfloat16)

    normalized = functional.rms_norm(activation, (width,), weight, 3.0e-4)
    product = gate * normalized
    expected = residual + product
    with ck.use_backend("cuda"):
        actual = ck.rms_gated_residual(activation, weight, residual, gate, 3.0e-4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_int8_rms_scale_shift_keeps_fp32_intermediates(input_dtype, weight_dtype):
    cuda = _cuda_backend()

    k = 256
    x = (((torch.arange(2 * k, device="cuda") % 41) - 20) / 9).to(input_dtype).reshape(2, k)
    # The FP32-only increment catches wrappers that silently cast the weight to BF16.
    norm_weight = ((((torch.arange(k, device="cuda") % 13) + 2) / 11) + 0.0013).to(weight_dtype)
    scale = (((torch.arange(k, device="cuda") % 19) - 8) / 31).to(input_dtype)
    shift = (((torch.arange(k, device="cuda") % 11) - 3) / 17).to(input_dtype)
    weight = ((torch.arange(64 * k, device="cuda") % 15) - 7).to(torch.int8).reshape(64, k)
    weight_scale = torch.tensor(0.02, device="cuda")

    x_fp32 = x.float()
    rstd = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + 3.0e-4)
    normalized = x_fp32 * rstd * norm_weight.float()
    modulated = normalized * (1 + scale.float()) + shift.float()
    expected = cuda.int8_linear(
        modulated, weight, weight_scale, convrot=True, convrot_groupsize=256)
    with ck.use_backend("cuda"):
        actual = ck.int8_linear(
            x, weight, weight_scale, convrot=True, convrot_groupsize=256,
            input_act="rms_norm", input_act_weight=norm_weight, input_act_eps=3.0e-4,
            input_act_scale=scale, input_act_shift=shift)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_int8_swiglu_keeps_fp32_intermediates(input_dtype):
    cuda = _cuda_backend()

    k = 256
    raw = (((torch.arange(4 * k, device="cuda") % 53) - 26) / 9).to(input_dtype).reshape(2, 2 * k)
    weight = ((torch.arange(32 * k, device="cuda") % 15) - 7).to(torch.int8).reshape(32, k)
    weight_scale = torch.tensor(0.02, device="cuda")
    gate, up = raw.chunk(2, dim=-1)
    activated = functional.silu(gate.float()) * up.float()
    expected = cuda.int8_linear(
        activated, weight, weight_scale, convrot=True, convrot_groupsize=256)

    with ck.use_backend("cuda"):
        actual = ck.int8_linear(
            raw, weight, weight_scale, convrot=True, convrot_groupsize=256,
            input_act="swiglu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
