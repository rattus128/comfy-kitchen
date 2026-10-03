# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Activations that a quantizer can absorb on the way in.

An MLP's ``linear(act(proj(x)))`` otherwise writes act's output to HBM and reads
it straight back to quantize it. Backends that can fold the activation into the
quantizer do so; the rest apply it here first with ordinary tensor-storage
rounding, then quantize. Fused FP32 intermediates can give different results.

``swiglu`` computes ``silu(gate) * up`` from two operands, or from
``[gate | up]`` halves on the last dim when the second operand is omitted.

``rms_norm`` is the row-wise ``x * rsqrt(mean(x^2) + eps) * w``; it carries a
K-element weight and an eps alongside the code.
"""

import torch

# Must match the enum in backends/cuda/input_act_codes.h; the CUDA backend
# passes these codes straight to the kernel.
INPUT_ACT_TO_CODE: dict[str | None, int] = {
    None: 0,
    "none": 0,
    "gelu_tanh": 1,
    "swiglu": 2,
    "rms_norm": 3,
}


def input_act_code(input_act: str | None) -> int:
    """Kernel code for `input_act`, rejecting anything unsupported."""
    try:
        return INPUT_ACT_TO_CODE[input_act]
    except KeyError:
        raise ValueError(_unsupported(input_act)) from None


def input_act_width(input_act: str | None) -> int:
    """How many input columns produce one activated column (1, or 2 for swiglu)."""
    return 2 if input_act == "swiglu" else 1


def validate_input_act_up(x, input_act, act_up):
    if act_up is not None:
        if input_act != "swiglu":
            raise ValueError("input_act_up requires input_act='swiglu'")
        if act_up.shape != x.shape or act_up.dtype != x.dtype or act_up.device != x.device:
            raise ValueError("input_act_up must match x's shape, dtype and device")


def apply_input_act(
    x: torch.Tensor,
    input_act: str | None,
    act_weight: torch.Tensor | None = None,
    act_eps: float = 0.0,
    act_scale: torch.Tensor | None = None,
    act_shift: torch.Tensor | None = None,
    act_up: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the pre-quantization activation eagerly.

    Used by backends and shapes the fused quantizer does not cover. These
    tensor operations retain their storage rounding, unlike the fused FP32
    activation intermediates.
    """
    validate_input_act_up(x, input_act, act_up)
    if (act_scale is not None or act_shift is not None) and input_act not in ("rms_norm", "adaln"):
        raise ValueError("input modulation requires input_act 'rms_norm' or 'adaln'")
    if input_act in (None, "none"):
        return x
    if input_act == "gelu_tanh":
        activated = torch.nn.functional.gelu(x, approximate="tanh")
    elif input_act == "swiglu":
        gate, up = x.chunk(2, dim=-1) if act_up is None else (x, act_up)
        activated = torch.nn.functional.silu(gate).mul_(up)
    elif input_act == "rms_norm":
        if act_weight is None:
            raise ValueError("input_act 'rms_norm' requires act_weight")
        activated = torch.nn.functional.rms_norm(
            x.float(), (x.shape[-1],), weight=act_weight.float(), eps=act_eps).to(x.dtype)
    elif input_act == "adaln":
        zero = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
        return torch.ops.comfy_kitchen.adaln(
            x, zero if act_scale is None else act_scale,
            zero if act_shift is None else act_shift, act_eps)
    else:
        raise ValueError(_unsupported(input_act))

    if act_scale is not None:
        activated = activated * (1 + act_scale)
    return activated if act_shift is None else activated + act_shift


def apply_residual(
    out: torch.Tensor,
    residual: torch.Tensor | None,
    residual_scale: torch.Tensor | None,
) -> torch.Tensor:
    """``residual + residual_scale * out`` in out's dtype (addcmul's type
    promotion would otherwise widen it)."""
    if residual is None:
        return out
    if residual_scale is None:
        raise ValueError("residual requires residual_scale")
    return torch.addcmul(residual.to(out.dtype), out, residual_scale.to(out.dtype))


def _unsupported(input_act) -> str:
    known = sorted([*(k for k in INPUT_ACT_TO_CODE if k is not None), "adaln"])
    return f"unsupported input_act: {input_act!r} (expected one of {known})"
