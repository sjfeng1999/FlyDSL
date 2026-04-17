# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for the gfx1250 TDM native CopyOp API.

Covers:
  * Constructing TDMLoad2D / TDMStore2D / TDMGather types.
  * tdm_wait emission.
  * The new types are importable as ``flydsl.expr.rocdl.gfx1250``.
  * The legacy ``flydsl.expr.rocdl.tdm_ops`` module still imports and
    emits a DeprecationWarning.

Full end-to-end IR lowering is exercised by
``tests/mlir/Conversion/tdm_copy_atom.mlir`` (FileCheck).
"""
from __future__ import annotations

import warnings

import pytest

from flydsl._mlir import ir
from flydsl.expr.rocdl import gfx1250


@pytest.fixture()
def ir_ctx():
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    ctx.load_all_available_dialects()
    with ctx, ir.Location.unknown():
        yield ctx


def _roundtrip_type(ty):
    text = str(ty)
    return ir.Type.parse(text)


def test_tdm_load_2d_type_roundtrip(ir_ctx):
    t = gfx1250.TDMLoad2D(
        elem_bits=16,
        tile_dim0=64,
        tile_dim1=64,
        pad_interval=0,
        pad_amount=0,
        num_warps=8,
        atomic_barrier_enable=False,
    )
    s = str(t)
    assert "!fly_rocdl.gfx1250.tdm_load_2d" in s
    assert "elem_bits = 16" in s
    assert "tile = (64, 64)" in s
    assert "num_warps = 8" in s
    assert "abe = false" in s
    _roundtrip_type(t)


def test_tdm_store_2d_type_roundtrip(ir_ctx):
    t = gfx1250.TDMStore2D(
        elem_bits=16,
        tile_dim0=64,
        tile_dim1=64,
        pad_interval=0,
        pad_amount=0,
        num_warps=4,
    )
    s = str(t)
    assert "!fly_rocdl.gfx1250.tdm_store_2d" in s
    assert "num_warps = 4" in s
    _roundtrip_type(t)


def test_tdm_gather_type_roundtrip(ir_ctx):
    t = gfx1250.TDMGather(elem_bits=16, row_width=128, max_indices=8)
    s = str(t)
    assert "!fly_rocdl.gfx1250.tdm_gather" in s
    assert "row_width = 128" in s
    assert "index_size = 32" in s
    assert "max_indices = 8" in s
    _roundtrip_type(t)


def test_tdm_gather_16bit_indices(ir_ctx):
    t = gfx1250.TDMGather(
        elem_bits=16, row_width=128, index_size=16, max_indices=16, is_store=True
    )
    s = str(t)
    assert "is_store = true" in s
    assert "index_size = 16" in s
    assert "max_indices = 16" in s


def test_tdm_wait_emits_op(ir_ctx):
    """gfx1250.tdm_wait is a thin Python helper that emits the upstream
    rocdl.s.wait.tensorcnt op directly (no intermediate fly_rocdl wrapper),
    because the wait does not participate in the layout / copy-atom
    lowering pipeline.
    """
    mod = ir.Module.create()
    with ir.InsertionPoint(mod.body):
        f_type = ir.FunctionType.get([], [])
        from flydsl._mlir.dialects import func

        f = func.FuncOp("tdm_wait_smoke", f_type)
        entry = f.add_entry_block()
        with ir.InsertionPoint(entry):
            gfx1250.tdm_wait(count=0)
            gfx1250.tdm_wait(count=2)
            func.ReturnOp([])

    text = str(mod)
    assert "fly_rocdl.tdm_wait" not in text
    assert text.count("rocdl.s.wait.tensorcnt 0") == 1
    assert text.count("rocdl.s.wait.tensorcnt 2") == 1


def test_legacy_tdm_ops_emits_deprecation_warning(ir_ctx):
    """The legacy tdm_ops module must still be importable and expose the
    original public API surface, but any *use* of it must emit a
    DeprecationWarning so downstream kernels get a visible hint to migrate
    to the new gfx1250.TDMLoad2D / TDMStore2D / TDMGather API.
    """
    from flydsl.expr.rocdl import tdm_ops as legacy

    for name in (
        "make_tensor_descriptor_2d",
        "make_tensor_gather_descriptor",
        "tensor_load_2d",
        "tensor_store_2d",
        "tensor_load_gather",
        "tensor_store_gather",
        "tensor_wait",
        "l2_prefetch_tile",
    ):
        assert hasattr(legacy, name), f"legacy tdm_ops.{name} is gone after refactor"

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        legacy._warn_legacy_explicit()

    assert any(
        issubclass(w.category, DeprecationWarning) and "tdm_ops" in str(w.message)
        for w in captured
    ), f"expected DeprecationWarning from tdm_ops, got {[str(w.message) for w in captured]}"


def test_tdm_load_2d_default_pad(ir_ctx):
    t = gfx1250.TDMLoad2D(elem_bits=16, tile_dim0=64, tile_dim1=64)
    s = str(t)
    assert "pad = (0, 0)" in s
    assert "num_warps = 1" in s


def test_tdm_load_2d_with_pad(ir_ctx):
    t = gfx1250.TDMLoad2D(
        elem_bits=16, tile_dim0=64, tile_dim1=32, pad_interval=64, pad_amount=8
    )
    s = str(t)
    assert "pad = (64, 8)" in s
