"""SiLU-and-mul fused with dynamic per-tensor fp8 quantization.

For a dense MLP whose down projection takes per-tensor fp8 activations, this
replaces ``SiluAndMul`` followed by a dynamic per-tensor quant. The first launch
takes every tile's absmax of ``silu(gate) * up``, rounded as ``SiluAndMul``
rounds it; the second reduces those maxima to the scale and quantizes its tile.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max

_BLOCK = 2048
# Every quant program reads all tile maxima, which only pays off for decode.
MAX_TOKENS = 128


@triton.jit
def _silu_and_mul_tile(x_ptr, row, cols, mask, inter, stride):
    gate = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * stride + inter + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    act = (gate / (1.0 + tl.exp(-gate))).to(x_ptr.dtype.element_ty).to(tl.float32)
    return (act * up).to(x_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _tile_absmax_kernel(
    x_ptr, max_ptr, inter, stride, TILES: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid // TILES
    cols = (pid % TILES) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < inter
    y = _silu_and_mul_tile(x_ptr, row, cols, mask, inter, stride)
    tl.store(max_ptr + pid, tl.max(tl.abs(y), axis=0))


@triton.jit
def _quant_kernel(
    x_ptr,
    max_ptr,
    out_ptr,
    scale_ptr,
    inter,
    stride,
    n_tiles,
    FP8_MAX: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    N_TILES_POW2: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, N_TILES_POW2)
    amax = tl.max(tl.load(max_ptr + offs, mask=offs < n_tiles, other=0.0), axis=0)
    scale = tl.maximum(amax / FP8_MAX, 1e-12)
    row = pid // TILES
    cols = (pid % TILES) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < inter
    y = _silu_and_mul_tile(x_ptr, row, cols, mask, inter, stride)
    q = tl.clamp(y / scale, -FP8_MAX, FP8_MAX)
    tl.store(out_ptr + row * inter + cols, q.to(out_ptr.dtype.element_ty), mask=mask)
    if pid == 0:
        tl.store(scale_ptr, scale)


def silu_and_mul_dynamic_per_tensor_quant_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` is ``(tokens, 2 * inter)``, gate then up, with a unit last stride.

    Returns the fp8 ``(tokens, inter)`` activation and its fp32 ``(1,)`` scale.
    """
    tokens, width = x.shape
    inter = width // 2
    tiles = triton.cdiv(inter, _BLOCK)
    n_tiles = tokens * tiles
    tile_max = torch.empty(n_tiles, device=x.device, dtype=torch.float32)
    out = torch.empty((tokens, inter), device=x.device, dtype=fp8_dtype)
    scale = torch.empty(1, device=x.device, dtype=torch.float32)
    _tile_absmax_kernel[(n_tiles,)](
        x, tile_max, inter, x.stride(0), TILES=tiles, BLOCK=_BLOCK
    )
    _quant_kernel[(n_tiles,)](
        x,
        tile_max,
        out,
        scale,
        inter,
        x.stride(0),
        n_tiles,
        FP8_MAX=fp8_max,
        TILES=tiles,
        BLOCK=_BLOCK,
        N_TILES_POW2=triton.next_power_of_2(n_tiles),
    )
    return out, scale
