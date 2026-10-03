# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""int8_linear(input_act=...) — folding an MLP's activation into the quantizer."""

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from comfy_kitchen.tensor.int8_utils import _build_hadamard
from tests.conftest import cuda_backend_available, get_capable_backends, rel_err

_GROUP = 256


def _gelu(x):
    return functional.gelu(x, approximate="tanh")


def _exact_fp64(h, group=_GROUP):
    """gelu -> ConvRot -> row-wise int8, with no intermediate rounding."""
    hd = h.double()
    g = 0.5 * hd * (1 + torch.tanh(0.7978845608028654 * (hd + 0.044715 * hd**3)))
    k = h.shape[-1]
    mat = _build_hadamard(group, device=h.device, dtype=torch.float64)
    rot = (g.reshape(-1, k // group, group) @ mat).reshape(-1, k)
    scale = (rot.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
    return (rot / scale).round().clamp(-128, 127), scale


class TestInputActQuantizer:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(512, 256), (1024, 4096), (256, 16384), (37, 2048)])
    def test_at_least_as_accurate_as_eager_chain(self, dtype, shape, seed, cuda_available):
        """Folding gelu in must not be worse than gelu-then-quantize.

        It is normally better: the eager chain rounds gelu's output to the input
        dtype before quantizing, while the fused path keeps float32 all the way
        to the int8 conversion.
        """
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        h = torch.randn(shape, dtype=dtype, device="cuda") * 2.0
        exact_q, _ = _exact_fp64(h)

        chain_q, _ = cuda_backend.quantize_int8_rowwise_convrot64(
            _gelu(h).contiguous(), _GROUP
        )
        fused_q, fused_s = cuda_backend.quantize_int8_rowwise_convrot64(
            h, _GROUP, input_act="gelu_tanh"
        )

        err_chain = (chain_q.double() - exact_q).abs().mean().item()
        err_fused = (fused_q.double() - exact_q).abs().mean().item()
        assert err_fused <= max(err_chain * 1.05, 1e-4), (
            f"fused ({err_fused:.6f}) less accurate than chain ({err_chain:.6f})"
        )
        assert fused_q.shape == h.shape
        assert fused_q.dtype == torch.int8
        assert fused_s.shape == (h.shape[0], 1)

    @pytest.mark.parametrize(
        "kwargs", [{"convrot": True, "convrot_groupsize": _GROUP}, {"convrot": False}]
    )
    def test_rejects_unknown_activation(self, kwargs, cuda_available):
        """Every route must reject the same way - the fused path used to raise a
        bare KeyError while the fallbacks raised a descriptive ValueError."""
        if not cuda_available:
            pytest.skip("CUDA required")

        h = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, 4096), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="unsupported input_act"):
            ck.int8_linear(
                h, weight, wscale, None, torch.bfloat16, input_act="silu", **kwargs
            )


class TestInt8LinearInputAct:
    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    def test_matches_eager_activation(self, backend, seed, cuda_available):
        """int8_linear(x, input_act=a) == int8_linear(a(x)) on every backend."""
        device = "cuda" if cuda_available else "cpu"
        if backend not in get_capable_backends("int8_linear", device):
            pytest.skip(f"backend '{backend}' not capable")

        m, k, n = 1024, 4096, 512
        h = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        weight = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=device)
        wscale = torch.tensor(0.01, dtype=torch.float32, device=device)

        with ck.use_backend(backend):
            ref = ck.int8_linear(
                _gelu(h), weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP,
            )
            got = ck.int8_linear(
                h, weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP, input_act="gelu_tanh",
            )

        rel = rel_err(got, ref)
        # Both quantize to int8; they differ only by the intermediate rounding
        # the fused path avoids.
        assert rel < 0.05, f"{backend}: rel={rel:.3e}"

    @pytest.mark.parametrize(
        "tag,shape,kwargs",
        [
            ("no convrot", (1024, 4096), {"convrot": False}),
            ("K not %256", (1024, 300), {"convrot": False}),
            ("K over smem cap", (64, 256 * 72), {"convrot": True, "convrot_groupsize": 256}),
            ("m == 1", (1, 4096), {"convrot": True, "convrot_groupsize": 256}),
        ],
    )
    def test_fallback_paths_agree(self, tag, shape, kwargs, seed, cuda_available):
        """Paths the fused kernel cannot serve must still apply the activation."""
        if not cuda_available:
            pytest.skip("CUDA required")

        m, k = shape
        h = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, k), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")

        ref = ck.int8_linear(_gelu(h), weight, wscale, None, torch.bfloat16, **kwargs)
        got = ck.int8_linear(
            h, weight, wscale, None, torch.bfloat16, input_act="gelu_tanh", **kwargs
        )
        rel = rel_err(got, ref)
        assert rel < 0.05, f"{tag}: rel={rel:.3e}"

    def test_none_is_identity(self, seed, cuda_available):
        """input_act=None and omitting it must be bit-identical."""
        if not cuda_available:
            pytest.skip("CUDA required")

        h = torch.randn(512, 4096, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, 4096), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")

        a = ck.int8_linear(h, weight, wscale, None, torch.bfloat16,
                           convrot=True, convrot_groupsize=_GROUP)
        b = ck.int8_linear(h, weight, wscale, None, torch.bfloat16,
                           convrot=True, convrot_groupsize=_GROUP, input_act=None)
        c = ck.int8_linear(h, weight, wscale, None, torch.bfloat16,
                           convrot=True, convrot_groupsize=_GROUP, input_act="none")
        assert torch.equal(a, b)
        assert torch.equal(a, c)

    def test_3d_input(self, seed, cuda_available):
        """Batched (B, T, K) input keeps its shape and applies the activation."""
        if not cuda_available:
            pytest.skip("CUDA required")

        h = torch.randn(2, 512, 4096, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, 4096), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")

        ref = ck.int8_linear(_gelu(h), weight, wscale, None, torch.bfloat16,
                             convrot=True, convrot_groupsize=_GROUP)
        got = ck.int8_linear(h, weight, wscale, None, torch.bfloat16,
                             convrot=True, convrot_groupsize=_GROUP,
                             input_act="gelu_tanh")
        assert got.shape == (2, 512, 256)
        # Same metric as the 2D cases: an elementwise tolerance is meaningless
        # for int8 GEMM outputs, where a single-LSB difference in the quantized
        # activation shifts the whole accumulated sum for that output element.
        rel = rel_err(got, ref)
        assert rel < 0.05, f"3d: rel={rel:.3e}"


def _swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return functional.silu(gate) * up


class TestSwiGLUInputAct:
    """swiglu is the gated pair: input rows are [gate | up], the activated row
    silu(gate) * up is half as wide, and the weight matches the halved width."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(512, 256), (1024, 4096), (37, 14336)])
    def test_at_least_as_accurate_as_eager_chain(self, dtype, shape, seed, cuda_available):
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        m, k = shape
        h = torch.randn((m, 2 * k), dtype=dtype, device="cuda") * 2.0

        hd = h.double()
        g = functional.silu(hd[:, :k]) * hd[:, k:]
        mat = _build_hadamard(_GROUP, device=h.device, dtype=torch.float64)
        rot = (g.reshape(-1, k // _GROUP, _GROUP) @ mat).reshape(-1, k)
        scale = (rot.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
        exact_q = (rot / scale).round().clamp(-128, 127)

        chain_q, _ = cuda_backend.quantize_int8_rowwise_convrot64(
            _swiglu(h).contiguous(), _GROUP
        )
        fused_q, fused_s = cuda_backend.quantize_int8_rowwise_convrot64(
            h, _GROUP, input_act="swiglu"
        )

        err_chain = (chain_q.double() - exact_q).abs().mean().item()
        err_fused = (fused_q.double() - exact_q).abs().mean().item()
        assert err_fused <= max(err_chain * 1.05, 1e-4), (
            f"fused ({err_fused:.6f}) less accurate than chain ({err_chain:.6f})"
        )
        assert fused_q.shape == (m, k)
        assert fused_q.dtype == torch.int8
        assert fused_s.shape == (m, 1)

    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    def test_matches_eager_activation(self, backend, seed, cuda_available):
        """int8_linear(x, input_act="swiglu") == int8_linear(swiglu(x)) everywhere."""
        device = "cuda" if cuda_available else "cpu"
        if backend not in get_capable_backends("int8_linear", device):
            pytest.skip(f"backend '{backend}' not capable")

        m, k, n = 1024, 4096, 512
        h = torch.randn(m, 2 * k, dtype=torch.bfloat16, device=device)
        weight = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=device)
        wscale = torch.tensor(0.01, dtype=torch.float32, device=device)

        with ck.use_backend(backend):
            ref = ck.int8_linear(
                _swiglu(h), weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP,
            )
            got = ck.int8_linear(
                h, weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP, input_act="swiglu",
            )

        assert got.shape == (m, n)
        rel = rel_err(got, ref)
        assert rel < 0.05, f"{backend}: rel={rel:.3e}"

    @pytest.mark.parametrize(
        "tag,shape,kwargs",
        [
            ("no convrot", (1024, 2 * 4096), {"convrot": False}),
            ("K over smem cap", (64, 2 * 256 * 72), {"convrot": True, "convrot_groupsize": 256}),
            ("m == 1", (1, 2 * 4096), {"convrot": True, "convrot_groupsize": 256}),
        ],
    )
    def test_fallback_paths_agree(self, tag, shape, kwargs, seed, cuda_available):
        """Paths the fused kernel cannot serve must still apply the activation."""
        if not cuda_available:
            pytest.skip("CUDA required")

        m, k2 = shape
        k = k2 // 2
        h = torch.randn(m, k2, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, k), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")

        ref = ck.int8_linear(_swiglu(h), weight, wscale, None, torch.bfloat16, **kwargs)
        got = ck.int8_linear(
            h, weight, wscale, None, torch.bfloat16, input_act="swiglu", **kwargs
        )
        rel = rel_err(got, ref)
        assert rel < 0.05, f"{tag}: rel={rel:.3e}"


_EPS = 1e-5


def _rms_norm(x, w, eps=_EPS):
    return functional.rms_norm(x, (x.shape[-1],), weight=w.to(x.dtype), eps=eps)


class TestRmsNormInputAct:
    """rms_norm folds a pre-norm block's ``linear(rms_norm(x))`` into the
    quantizer; it carries a K-element weight and an eps alongside the code."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("shape", [(512, 256), (1024, 4096), (37, 2048), (256, 8192)])
    def test_at_least_as_accurate_as_eager_chain(self, dtype, shape, seed, cuda_available):
        if not cuda_backend_available():
            pytest.skip("compiled CUDA backend required")
        from comfy_kitchen.backends import cuda as cuda_backend

        m, k = shape
        h = torch.randn((m, k), dtype=dtype, device="cuda") * 2.0
        w = torch.randn((k,), dtype=dtype, device="cuda")

        hd, wd = h.double(), w.double()
        g = hd * torch.rsqrt(hd.pow(2).mean(-1, keepdim=True) + _EPS) * wd
        mat = _build_hadamard(_GROUP, device=h.device, dtype=torch.float64)
        rot = (g.reshape(-1, k // _GROUP, _GROUP) @ mat).reshape(-1, k)
        scale = (rot.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
        exact_q = (rot / scale).round().clamp(-128, 127)

        chain_q, _ = cuda_backend.quantize_int8_rowwise_convrot64(
            _rms_norm(h, w).contiguous(), _GROUP
        )
        fused_q, fused_s = cuda_backend.quantize_int8_rowwise_convrot64(
            h, _GROUP, input_act="rms_norm", input_act_weight=w, input_act_eps=_EPS
        )

        err_chain = (chain_q.double() - exact_q).abs().mean().item()
        err_fused = (fused_q.double() - exact_q).abs().mean().item()
        assert err_fused <= max(err_chain * 1.05, 1e-4), (
            f"fused ({err_fused:.6f}) less accurate than chain ({err_chain:.6f})"
        )
        assert fused_q.shape == (m, k)
        assert fused_q.dtype == torch.int8
        assert fused_s.shape == (m, 1)

    @pytest.mark.parametrize("backend", ["cuda", "hip", "triton", "eager"])
    def test_matches_eager_activation(self, backend, seed, cuda_available):
        """int8_linear(x, input_act="rms_norm") == int8_linear(rms_norm(x))."""
        device = "cuda" if cuda_available else "cpu"
        if backend not in get_capable_backends("int8_linear", device):
            pytest.skip(f"backend '{backend}' not capable")

        m, k, n = 1024, 4096, 512
        h = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        w = torch.randn(k, dtype=torch.bfloat16, device=device)
        weight = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=device)
        wscale = torch.tensor(0.01, dtype=torch.float32, device=device)

        with ck.use_backend(backend):
            ref = ck.int8_linear(
                _rms_norm(h, w), weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP,
            )
            got = ck.int8_linear(
                h, weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP,
                input_act="rms_norm", input_act_weight=w, input_act_eps=_EPS,
            )

        assert got.shape == (m, n)
        rel = rel_err(got, ref)
        assert rel < 0.05, f"{backend}: rel={rel:.3e}"

    @pytest.mark.parametrize(
        "tag,shape,kwargs",
        [
            ("no convrot", (1024, 4096), {"convrot": False}),
            ("K over smem cap", (64, 256 * 72), {"convrot": True, "convrot_groupsize": 256}),
            ("m == 1", (1, 4096), {"convrot": True, "convrot_groupsize": 256}),
        ],
    )
    def test_fallback_paths_agree(self, tag, shape, kwargs, seed, cuda_available):
        """Paths the fused kernel cannot serve must still apply the norm."""
        if not cuda_available:
            pytest.skip("CUDA required")

        m, k = shape
        h = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(k, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, k), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")

        ref = ck.int8_linear(_rms_norm(h, w), weight, wscale, None, torch.bfloat16, **kwargs)
        got = ck.int8_linear(
            h, weight, wscale, None, torch.bfloat16,
            input_act="rms_norm", input_act_weight=w, input_act_eps=_EPS, **kwargs
        )
        rel = rel_err(got, ref)
        assert rel < 0.05, f"{tag}: rel={rel:.3e}"

    def test_requires_weight(self, cuda_available):
        """rms_norm without a weight must raise, not silently skip the norm."""
        if not cuda_available:
            pytest.skip("CUDA required")

        h = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
        weight = torch.randint(-127, 127, (256, 4096), dtype=torch.int8, device="cuda")
        wscale = torch.tensor(0.01, dtype=torch.float32, device="cuda")
        with pytest.raises((ValueError, RuntimeError)):
            ck.int8_linear(
                h, weight, wscale, None, torch.bfloat16,
                convrot=True, convrot_groupsize=_GROUP, input_act="rms_norm",
            )


@pytest.mark.parametrize("backend", ["eager", "hip", "cuda", "triton"])
@pytest.mark.parametrize("with_scale", [False, True])
def test_rms_modulation_broadcast_and_rounding(backend, cuda_available, with_scale):
    device = "cpu" if backend == "eager" else "cuda"
    if device == "cuda" and not cuda_available:
        pytest.skip("GPU required")
    if backend not in get_capable_backends("int8_linear", device):
        pytest.skip(f"backend '{backend}' not capable")
    torch.manual_seed(224)
    x = torch.randn(2, 17, 256, device=device, dtype=torch.bfloat16)
    gamma = torch.randn(256, device=device, dtype=torch.bfloat16) + 0.25
    scale = torch.randn(2, 1, 512, device=device, dtype=torch.bfloat16)[..., ::2] * 0.3
    shift = torch.randn(2, 1, 512, device=device, dtype=torch.bfloat16)[..., ::2] * 0.2
    weight = torch.randint(-127, 128, (128, 256), device=device, dtype=torch.int8)
    ws = torch.rand(128, device=device) * 0.01
    normalized = functional.rms_norm(x, (256,), gamma, 3e-4)
    if with_scale:
        normalized = normalized * (1 + scale)
    else:
        scale = None
    modulated = normalized + shift
    with ck.use_backend(backend):
        expected = ck.int8_linear(modulated, weight, ws)
        actual = ck.int8_linear(x, weight, ws, input_act="rms_norm",
                                input_act_weight=gamma, input_act_eps=3e-4,
                                input_act_scale=scale, input_act_shift=shift)
        assert torch.equal(actual, expected)
        with pytest.raises(ValueError, match=r"requires.*rms_norm"):
            ck.int8_linear(x, weight, ws, input_act_shift=shift)


def test_swiglu_rounds_before_multiply():
    from comfy_kitchen.backends._activations import apply_input_act

    torch.manual_seed(31)
    x = torch.randn(5, 32, dtype=torch.bfloat16)
    gate, up = x.chunk(2, dim=-1)
    expected = functional.silu(gate) * up
    actual = apply_input_act(x, "swiglu")
    wrong = (functional.silu(gate.float()) * up.float()).to(x.dtype)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, wrong)


def test_rms_norm_preserves_fp32_weight():
    from comfy_kitchen.backends._activations import apply_input_act

    torch.manual_seed(32)
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    weight = torch.randn(32, dtype=torch.float32)
    actual = apply_input_act(x, "rms_norm", weight, 1e-5)
    expected = functional.rms_norm(x.float(), (32,), weight, 1e-5).to(torch.bfloat16)
    truncated = functional.rms_norm(x.float(), (32,), weight.bfloat16().float(), 1e-5).to(torch.bfloat16)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, truncated)


def test_adaln_shift_only():
    from comfy_kitchen.backends._activations import apply_input_act

    torch.manual_seed(33)
    x = torch.randn(3, 32, dtype=torch.bfloat16)
    shift = torch.randn(32, dtype=torch.bfloat16)
    actual = apply_input_act(x, "adaln", act_eps=1e-5, act_shift=shift)
    expected = ck.adaln(x, torch.zeros_like(shift), shift, 1e-5)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("backend", ["eager", "hip", "cuda", "triton"])
@pytest.mark.parametrize("layout", ["views", "independent", "strided", "transposed_batch"])
@pytest.mark.parametrize("convrot", [False, True])
def test_two_input_swiglu_linear(backend, layout, convrot, cuda_available, monkeypatch):
    device = "cpu" if backend == "eager" else "cuda"
    if device == "cuda" and not cuda_available:
        pytest.skip("GPU required")
    if backend not in get_capable_backends("int8_linear", device):
        pytest.skip(f"backend '{backend}' not capable")
    torch.manual_seed(83)
    packed = torch.randn(2, 17, 512, dtype=torch.bfloat16, device=device)
    gate, up = packed.chunk(2, -1)
    if layout == "independent":
        gate, up = gate.contiguous(), up.contiguous()
    elif layout == "strided":
        gate, up = packed[..., ::2], packed[..., 1::2]
    elif layout == "transposed_batch":
        gate, up = gate.transpose(0, 1), up.transpose(0, 1)
    weight = torch.randint(-80, 81, (128, 256), dtype=torch.int8, device=device)
    scale = torch.rand(128, device=device) * 0.01
    kwargs = {"convrot": convrot, "input_act": "swiglu"}
    with ck.use_backend(backend):
        expected = ck.int8_linear(torch.cat((gate, up), -1), weight, scale, **kwargs)
        if backend == "hip" and convrot and layout == "views":
            from comfy_kitchen.backends import hip

            quantize = hip._rotate_quant_int8

            def check_views(x, *args, **kwargs):
                assert x.data_ptr() == gate.data_ptr()
                assert kwargs["act_up"].data_ptr() == up.data_ptr()
                return quantize(x, *args, **kwargs)

            monkeypatch.setattr(hip, "_rotate_quant_int8", check_views)
        actual = ck.int8_linear(gate, weight, scale, input_act_up=up, **kwargs)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("m,k,group", [
    (7, 64, 16), (9, 256, 64), (37, 2048, 256),
    (512, 10240, 256), (128, 32768, 256),
    (512, 12288, 256), (512, 16384, 256),
])
@pytest.mark.parametrize("independent", [False, True])
def test_hip_two_input_swiglu_quantizer(m, k, group, independent, monkeypatch, cuda_available):
    if not cuda_available or "hip" not in get_capable_backends("int8_linear", "cuda"):
        pytest.skip("HIP backend required")
    from comfy_kitchen.backends import hip

    torch.manual_seed(84)
    packed = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)
    gate, up = packed.chunk(2, -1)
    if independent:
        # Different positive row strides and storage offsets catch a kernel
        # that still assumes up = gate + K or uses the gate's stride for both.
        storage = torch.empty(m, k + 8, device="cuda", dtype=packed.dtype)
        storage[:, 4:4+k].copy_(up)
        up = storage[:, 4:4+k]
    expected = hip._rotate_quant_int8(packed, group, "swiglu")
    native = hip._C.quantize_int8_convrot
    seen = []

    def capture(*args):
        # DLPack must receive the original two allocations, not copies.
        gate_arg, up_arg = torch.from_dlpack(args[0]), torch.from_dlpack(args[-1])
        seen.append((gate_arg.data_ptr(), up_arg.data_ptr()))
        return native(hip._dl(gate_arg), *args[1:-1], hip._dl(up_arg))

    monkeypatch.setattr(hip._C, "quantize_int8_convrot", capture)
    actual = hip._rotate_quant_int8(gate, group, "swiglu", act_up=up)
    assert seen[0] == (gate.data_ptr(), up.data_ptr())
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=False))


def test_two_input_swiglu_precision_and_validation():
    from comfy_kitchen.backends._activations import apply_input_act

    gate = torch.tensor([[-2.13, 0.77, 1.29, 4.51]], dtype=torch.float32)
    up = torch.tensor([[3.17, -1.37, 0.17, 2.51]], dtype=torch.float32)
    expected = functional.silu(gate) * up
    actual = apply_input_act(gate, "swiglu", act_up=up)
    assert torch.equal(actual, expected)
    for act, bad_up in [(None, up), ("swiglu", up[:, :2]), ("swiglu", up.bfloat16())]:
        with pytest.raises(ValueError, match="input_act_up"):
            apply_input_act(gate, act, act_up=bad_up)
