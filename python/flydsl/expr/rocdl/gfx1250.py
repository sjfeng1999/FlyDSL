# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""gfx1250-specific copy atoms and ops (TDM)."""

from ..._mlir._mlir_libs._mlirDialectsFlyROCDL import (
    CopyOpGFX1250TDMGatherType,
    CopyOpGFX1250TDMLoad2DType,
    CopyOpGFX1250TDMStore2DType,
)


def TDMLoad2D(
    elem_bits,
    tile_dim0,
    tile_dim1,
    *,
    pad_interval=0,
    pad_amount=0,
    num_warps=1,
    atomic_barrier_enable=False,
):
    """Create a gfx1250 TDM 2D tensor load atom (global -> LDS).

    Static descriptor parameters travel on the type:

    - ``elem_bits`` -- element bit width (8/16/32)
    - ``tile_dim0`` -- inner tile extent in elements (innermost / row-major fastest dim)
    - ``tile_dim1`` -- outer tile extent in elements
    - ``pad_interval`` -- LDS padding interval in elements (0 disables padding)
    - ``pad_amount`` -- padding amount in elements
    - ``num_warps`` -- number of waves cooperating on the load (for warp distribution)
    - ``atomic_barrier_enable`` -- set to True to route the TDM completion to
      the ``s_barrier_signal`` rather than ``TENSORcnt``

    Dynamic state fields (set via ``atom.set_value``):

    - ``workgroup_mask`` (i32) -- MCAST mask over peer workgroups
    - ``cache_policy`` (i32, must fold to compile-time constant)
    - ``gmem_byte_offset`` (i32) -- per-k-tile gmem base offset, in bytes
    - ``lds_byte_offset`` (i32) -- LDS destination offset, in bytes
    - ``tensor_dim0`` / ``tensor_dim1`` (i32) -- full tensor shape
    - ``stride0`` (i32) -- row stride in elements
    """
    return CopyOpGFX1250TDMLoad2DType.get(
        elem_bits=elem_bits,
        tile_dim0=tile_dim0,
        tile_dim1=tile_dim1,
        pad_interval=pad_interval,
        pad_amount=pad_amount,
        num_warps=num_warps,
        atomic_barrier_enable=atomic_barrier_enable,
    )


def TDMStore2D(
    elem_bits,
    tile_dim0,
    tile_dim1,
    *,
    pad_interval=0,
    pad_amount=0,
    num_warps=1,
    atomic_barrier_enable=False,
):
    """Create a gfx1250 TDM 2D tensor store atom (LDS -> global).

    Same static / dynamic parameters as :func:`TDMLoad2D` but the data flows
    in the opposite direction. Pad encoding is not applied on the store
    descriptor; if ``pad_interval > 0`` it is folded into the effective
    tile_dim0 (matching the legacy behaviour in ``tdm_ops.py``).
    """
    return CopyOpGFX1250TDMStore2DType.get(
        elem_bits=elem_bits,
        tile_dim0=tile_dim0,
        tile_dim1=tile_dim1,
        pad_interval=pad_interval,
        pad_amount=pad_amount,
        num_warps=num_warps,
        atomic_barrier_enable=atomic_barrier_enable,
    )


def TDMGather(
    elem_bits,
    row_width,
    max_indices,
    *,
    index_size=32,
    pad_interval=0,
    pad_amount=0,
    is_store=False,
):
    """Create a gfx1250 TDM gather/scatter atom (global <-> LDS via row indices).

    Dynamic state fields (set via ``atom.set_value``):

    - ``workgroup_mask`` (i32) -- MCAST mask
    - ``cache_policy`` (i32)
    - ``gmem_byte_offset`` / ``lds_byte_offset`` (i32)
    - ``tensor_dim0`` / ``tensor_dim1`` (i32)
    - ``stride0`` (i32)
    - ``gather_row_indices`` -- vector<8xi32> with the row index set
    - ``gather_index_count`` (i32) -- number of active indices
    """
    return CopyOpGFX1250TDMGatherType.get(
        elem_bits=elem_bits,
        row_width=row_width,
        index_size=index_size,
        max_indices=max_indices,
        pad_interval=pad_interval,
        pad_amount=pad_amount,
        is_store=is_store,
    )


def tdm_wait(count=0):
    """Emit ``rocdl.s.wait.tensorcnt count``.

    ``count`` follows the AMD convention: wait until the number of
    outstanding TDM operations is less-than-or-equal-to ``count``.
    ``count=0`` is a full fence; ``count=N`` lets up to ``N`` operations
    remain outstanding to keep pipelining across loop iterations.

    This is a thin helper: it just emits the upstream ROCDL op directly
    (no intermediate ``fly_rocdl`` wrapper) because the op does not
    participate in the layout / copy-atom lowering pipeline.
    """
    from ..._mlir import ir
    from ..._mlir.dialects import rocdl as _rocdl

    count_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(16), int(count))
    _rocdl.s_wait_tensorcnt(count_attr)
