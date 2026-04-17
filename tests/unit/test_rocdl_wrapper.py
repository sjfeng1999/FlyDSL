#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for flydsl.expr.rocdl elementwise vector wrappers.

Verifies that:
1. ROCDL-only fast-approx unary ops (``rcp``/``rsq``) accept both scalar
   and vector inputs, unrolling vector inputs lane-by-lane. (Generic math
   ops like cos/exp/log/sin/sqrt/tanh are covered by flydsl.expr.math.)
2. ``cvt_pk_{fp8,bf8}_f32_v`` packs a ``vector<NxF32>`` (N % 4 == 0) into
   ``vector<(N/4)xI32>`` chaining word_sel=0 then word_sel=1.
3. ``cvt_pkrtz_v`` expands ``vector<NxF32>`` (N even) to ``vector<NxF16>``.
4. ``cvt_pk_f32_{fp8,bf8}_v`` unpacks ``vector<MxI32>`` into ``vector<(4*M)xF32>``.
5. ``cvt_f32_{fp8,bf8}_v`` byte-expands ``vector<MxI32>`` into ``vector<(4*M)xF32>``.
6. ``cvt_sr_{fp8,bf8}_f32_v`` stochastic-rounds ``vector<NxF32>`` (N % 4 == 0)
   into ``vector<(N/4)xI32>``, threading byte_sel=0..3 per destination.
7. ``fmed3`` accepts scalar or vector f32/f16 natively.
"""

import pytest

from flydsl._mlir import ir
from flydsl._mlir.dialects import func
from flydsl.expr import rocdl as fx_rocdl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_module(build_fn, arg_types):
    """Build an MLIR module with a function taking *arg_types* and calling build_fn.

    *arg_types* is a list of callables ``() -> ir.Type``.
    Returns the IR text as a string.
    """
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        ctx.allow_unregistered_dialects = True
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            types = [t() for t in arg_types]
            ftype = ir.FunctionType.get(types, [])
            fn = func.FuncOp("test", ftype)
            with ir.InsertionPoint(fn.add_entry_block()):
                build_fn(*fn.entry_block.arguments)
                func.ReturnOp([])
        module.operation.verify()
        return str(module)


def _v(n, elem):
    return lambda: ir.VectorType.get([n], elem())


def _f32():
    return ir.F32Type.get()


def _f16():
    return ir.F16Type.get()


def _i32():
    return ir.IntegerType.get_signless(32)


# ---------------------------------------------------------------------------
# 1. Unary elementwise math (scalar/vector auto-dispatch)
# ---------------------------------------------------------------------------


_UNARY_OPS = ["rcp", "rsq"]


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("name", _UNARY_OPS)
def test_unary_scalar(name):
    fn = getattr(fx_rocdl, name)

    def build(x):
        fn(x)

    text = _build_module(build, arg_types=[_f32])
    assert f"rocdl.{name}" in text


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("name", _UNARY_OPS)
@pytest.mark.parametrize("n", [1, 4, 8, 16])
def test_unary_vector_unrolls_lane_by_lane(name, n):
    fn = getattr(fx_rocdl, name)

    def build(v):
        fn(v)

    text = _build_module(build, arg_types=[_v(n, _f32)])
    assert text.count(f"rocdl.{name}") == n
    assert "vector.from_elements" in text


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("name", _UNARY_OPS)
def test_unary_legacy_ods_signature(name):
    """``rocdl.{rcp,rsq}(res_type, arg)`` still works for scalar callers."""
    fn = getattr(fx_rocdl, name)

    def build(x):
        fn(ir.F32Type.get(), x)

    text = _build_module(build, arg_types=[_f32])
    assert f"rocdl.{name}" in text


# ---------------------------------------------------------------------------
# 2. cvt_pk_{fp8,bf8}_f32_v : vector<NxF32> (N%4==0) -> vector<(N/4)xI32>
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    "fn_name,expected_op",
    [
        ("cvt_pk_fp8_f32_v", "rocdl.cvt.pk.fp8.f32"),
        ("cvt_pk_bf8_f32_v", "rocdl.cvt.pk.bf8.f32"),
    ],
)
@pytest.mark.parametrize("n", [4, 8, 16])
def test_cvt_pk_f32_to_i32_vector(fn_name, expected_op, n):
    fn = getattr(fx_rocdl, fn_name)

    def build(v):
        fn(v)

    text = _build_module(build, arg_types=[_v(n, _f32)])
    assert text.count(expected_op) == n // 2, text
    if n > 4:
        assert f"vector<{n // 4}xi32>" in text
    assert text.count("[false]") == n // 4
    assert text.count("[true]") == n // 4


@pytest.mark.l0_backend_agnostic
def test_cvt_pk_f32_to_i32_bad_length():
    def build(v):
        fx_rocdl.cvt_pk_fp8_f32_v(v)

    with pytest.raises(ValueError, match="multiple of 4"):
        _build_module(build, arg_types=[_v(5, _f32)])


# ---------------------------------------------------------------------------
# 3. cvt_pkrtz_v : vector<NxF32> (N even) -> vector<NxF16>
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_cvt_pkrtz_vector(n):
    def build(v):
        fx_rocdl.cvt_pkrtz_v(v)

    text = _build_module(build, arg_types=[_v(n, _f32)])
    assert text.count("rocdl.cvt.pkrtz") == n // 2
    assert f"vector<{n}xf16>" in text


@pytest.mark.l0_backend_agnostic
def test_cvt_pkrtz_bad_length():
    def build(v):
        fx_rocdl.cvt_pkrtz_v(v)

    with pytest.raises(ValueError, match="even"):
        _build_module(build, arg_types=[_v(3, _f32)])


# ---------------------------------------------------------------------------
# 4. cvt_pk_f32_{fp8,bf8}_v : vector<MxI32> -> vector<(4*M)xF32>
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    "fn_name,expected_op",
    [
        ("cvt_pk_f32_fp8_v", "rocdl.cvt.pk.f32.fp8"),
        ("cvt_pk_f32_bf8_v", "rocdl.cvt.pk.f32.bf8"),
    ],
)
@pytest.mark.parametrize("m", [1, 2, 4])
def test_cvt_pk_i32_to_f32_vector(fn_name, expected_op, m):
    fn = getattr(fx_rocdl, fn_name)

    def build(v):
        fn(v)

    text = _build_module(build, arg_types=[_v(m, _i32)])
    assert text.count(expected_op) == 2 * m
    assert f"vector<{4 * m}xf32>" in text


# ---------------------------------------------------------------------------
# 5. cvt_f32_{fp8,bf8}_v : vector<MxI32> -> vector<(4*M)xF32>
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    "fn_name,expected_op",
    [
        ("cvt_f32_fp8_v", "rocdl.cvt.f32.fp8"),
        ("cvt_f32_bf8_v", "rocdl.cvt.f32.bf8"),
    ],
)
@pytest.mark.parametrize("m", [1, 2, 4])
def test_cvt_byte_i32_to_f32_vector(fn_name, expected_op, m):
    fn = getattr(fx_rocdl, fn_name)

    def build(v):
        fn(v)

    text = _build_module(build, arg_types=[_v(m, _i32)])
    assert text.count(expected_op) == 4 * m, text
    for byte in range(4):
        assert f"[{byte}]" in text
    assert f"vector<{4 * m}xf32>" in text


# ---------------------------------------------------------------------------
# 6. cvt_sr_{fp8,bf8}_f32_v : vector<NxF32> (+ i32 stoch) -> vector<(N/4)xI32>
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    "fn_name,expected_op",
    [
        ("cvt_sr_fp8_f32_v", "rocdl.cvt.sr.fp8.f32"),
        ("cvt_sr_bf8_f32_v", "rocdl.cvt.sr.bf8.f32"),
    ],
)
@pytest.mark.parametrize("n", [4, 8])
def test_cvt_sr_vector(fn_name, expected_op, n):
    fn = getattr(fx_rocdl, fn_name)

    def build(v, s):
        fn(v, s)

    text = _build_module(build, arg_types=[_v(n, _f32), _v(n, _i32)])
    assert text.count(expected_op) == n
    for byte in range(4):
        assert f"[{byte}]" in text


@pytest.mark.l0_backend_agnostic
def test_cvt_sr_scalar_stoch_broadcasts():
    """A scalar stoch i32 gets broadcast across all 4 byte positions of every dest i32."""

    def build(v, s):
        fx_rocdl.cvt_sr_fp8_f32_v(v, s)

    text = _build_module(build, arg_types=[_v(8, _f32), _i32])
    assert text.count("rocdl.cvt.sr.fp8.f32") == 8


@pytest.mark.l0_backend_agnostic
def test_cvt_sr_bad_length():
    def build(v, s):
        fx_rocdl.cvt_sr_fp8_f32_v(v, s)

    with pytest.raises(ValueError, match="multiple of 4"):
        _build_module(build, arg_types=[_v(6, _f32), _i32])


# ---------------------------------------------------------------------------
# 7. fmed3 - scalar / vector
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_fmed3_scalar_f32():
    def build(x):
        fx_rocdl.fmed3(x, x, x)

    text = _build_module(build, arg_types=[_f32])
    assert "rocdl.fmed3" in text
    assert ": f32" in text


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("n", [4, 8, 16])
def test_fmed3_vector_f32(n):
    def build(v):
        fx_rocdl.fmed3(v, v, v)

    text = _build_module(build, arg_types=[_v(n, _f32)])
    assert "rocdl.fmed3" in text
    assert f"vector<{n}xf32>" in text


@pytest.mark.l0_backend_agnostic
def test_fmed3_vector_f16():
    def build(v):
        fx_rocdl.fmed3(v, v, v)

    text = _build_module(build, arg_types=[_v(8, _f16)])
    assert "rocdl.fmed3" in text
    assert "vector<8xf16>" in text


# ---------------------------------------------------------------------------
# 8. Round-trip identity: pack-then-unpack f32 <-> fp8 preserves shape
# ---------------------------------------------------------------------------


@pytest.mark.l0_backend_agnostic
def test_pack_unpack_fp8_round_trip_shape():
    """Pack vector<16xf32> into vector<4xi32> and unpack back to vector<16xf32>."""

    def build(v):
        packed = fx_rocdl.cvt_pk_fp8_f32_v(v)
        unpacked = fx_rocdl.cvt_pk_f32_fp8_v(packed)
        return unpacked

    text = _build_module(build, arg_types=[_v(16, _f32)])
    assert text.count("rocdl.cvt.pk.fp8.f32") == 8
    assert text.count("rocdl.cvt.pk.f32.fp8") == 8
    assert "vector<4xi32>" in text
    assert "vector<16xf32>" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
