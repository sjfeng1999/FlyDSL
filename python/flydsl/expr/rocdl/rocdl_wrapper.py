# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vector-extended wrappers for elementwise ROCDL ops.

The ROCDL dialect exposes a large family of elementwise operations whose
hardware instruction is intrinsically sub-block (scalar, pk-of-2, byte-addressed,
pk-of-8 for gfx1250, ...). The Python/ODS builder for each op only exposes the
native sub-block granularity, which forces callers to hand-roll loops and to
pack/unpack vectors around each call.

This module layers a thin, uniform Python wrapper over the ROCDL ops that
accepts arbitrary-size input vectors and expands them into a compile-time
range of the underlying sub-block op. The expansion is fully static (the
range length is fixed at trace time), so the generated IR is a flat sequence
of ROCDL ops without any runtime control flow.

Categories of wrapped ops:

- *Hardware fast-approx unary* (``rcp``, ``rsq``) — ROCDL-only ops without a
  math-dialect equivalent; wrapped so vector inputs are unrolled lane-by-lane.
  For ``cos``, ``exp``, ``exp2``, ``log``, ``sin``, ``sqrt``, ``tanh`` prefer
  the ``flydsl.expr.math`` dialect wrappers, which already handle scalar and
  vector inputs uniformly.

- *Pack-2 f32 -> fp8/bf8* (``cvt_pk_fp8_f32``, ``cvt_pk_bf8_f32``): stores
  two f32 into one half-word of an ``i32``. For a ``vector<NxF32>`` input
  (``N`` divisible by 4) we emit ``N/4`` i32 destinations, chaining
  ``word_sel=0`` then ``word_sel=1`` per destination.

- *Pack-2 f32 -> f16 (``cvt_pkrtz``)*: emits two f16 per call. For
  ``vector<NxF32>`` we emit ``N/2`` calls and concatenate to
  ``vector<NxF16>``.

- *Unpack-2 fp8/bf8 -> f32* (``cvt_pk_f32_fp8``, ``cvt_pk_f32_bf8``):
  one i32 encodes 4 fp8/bf8 bytes = 2 words. From ``vector<MxI32>`` we emit
  ``2*M`` unpack calls producing ``vector<(4*M)xF32>``.

- *Byte-indexed unpack* (``cvt_f32_fp8``, ``cvt_f32_bf8``): one byte of an
  i32 -> f32. From ``vector<MxI32>`` we emit ``4*M`` calls producing
  ``vector<(4*M)xF32>``.

- *Stochastic rounding pack-into-byte* (``cvt_sr_fp8_f32``,
  ``cvt_sr_bf8_f32``): ``f32 + stoch_i32 + old_i32 + byte_sel`` stores 1 byte
  into ``old_i32``. From ``vector<NxF32> + vector<NxI32>`` we emit ``N``
  calls (``N`` divisible by 4), threading ``byte_sel=0..3`` through
  ``N/4`` destination i32 slots.

- *Ternary fmed3*: native vector support — just auto-unwrap.

Usage::

    from flydsl.expr import rocdl as fx_rocdl
    from flydsl.expr.typing import T

    v_f32 = ...  # vector<16xf32>
    v_f16 = fx_rocdl.cvt_pkrtz_v(v_f32)                    # vector<16xf16>
    v_fp8 = fx_rocdl.cvt_pk_fp8_f32_v(v_f32)               # vector<4xi32>

The original single-op wrappers (``cvt_pk_fp8_f32``, ``rcp``, ...) remain
unchanged; the new vector-extended forms have a trailing ``_v`` so that
existing call sites keep working.
"""

from __future__ import annotations

from ..._mlir import ir
from ..._mlir.dialects import rocdl as _raw_rocdl
from ..._mlir.dialects import vector as _vector
from ..meta import traced_op
from ..typing import T


def _unwrap(v, *, loc=None):
    """Coerce DSL Numeric / ArithValue to raw ir.Value."""
    from .. import arith as _arith

    return _arith.unwrap(v, loc=loc)


def _const_i32(value: int, *, loc=None):
    from .. import arith as _arith

    return _arith.unwrap(_arith.constant(int(value), type=T.i32, loc=loc), loc=loc)


def _vec_len(v) -> int:
    """Return the flat length of a vector ir.Value, or 0 for scalars."""
    raw = v if isinstance(v, ir.Value) else getattr(v, "ir_value", lambda: v)()
    if isinstance(raw.type, ir.VectorType):
        shape = list(raw.type.shape)
        n = 1
        for d in shape:
            n *= int(d)
        return n
    return 0


def _vec_elem_type(v) -> ir.Type:
    raw = v if isinstance(v, ir.Value) else getattr(v, "ir_value", lambda: v)()
    if isinstance(raw.type, ir.VectorType):
        return raw.type.element_type
    return raw.type


def _flatten_to_1d(v, *, loc=None):
    """If ``v`` is an ND vector, bitcast/shape-cast to flat 1D. Scalars pass through."""
    raw = _unwrap(v, loc=loc)
    if not isinstance(raw.type, ir.VectorType):
        return raw
    shape = list(raw.type.shape)
    if len(shape) <= 1:
        return raw
    n = 1
    for d in shape:
        n *= int(d)
    flat_ty = ir.VectorType.get([n], raw.type.element_type)
    return _vector.ShapeCastOp(flat_ty, raw, loc=loc).result


def _extract(v, idx: int, *, loc=None):
    return _vector.ExtractOp(
        v, static_position=[int(idx)], dynamic_position=[], loc=loc
    ).result


def _from_elements(dest_ty: ir.Type, elems, *, loc=None):
    return _vector.from_elements(dest_ty, list(elems), loc=loc)


# ─────────────────────────────────────────────────────────────────────────────
# Group A: hardware fast-approx unary (rcp / rsq).
#
# Only ``rcp`` and ``rsq`` are kept here because they have no math-dialect
# equivalent (math.py provides ``rsqrt`` but no reciprocal, and both map to
# different fast-approx semantics on AMD GPUs). For ``cos``, ``exp``, ``exp2``,
# ``log``, ``sin``, ``sqrt``, ``tanh`` use ``flydsl.expr.math`` instead.
#
# ROCDL's scalar math ops only accept scalar floating-point LLVM types; a
# ``vector<NxF32>`` input fails verification. To support arbitrary-size
# vectors we unroll over lanes.
# ─────────────────────────────────────────────────────────────────────────────


_UNARY_OPS = {
    "rcp": _raw_rocdl.rcp,
    "rsq": _raw_rocdl.rsq,
}


def _apply_unary_elementwise(raw_fn, arg, *, loc=None, ip=None):
    raw = _unwrap(arg, loc=loc)
    if not isinstance(raw.type, ir.VectorType):
        return raw_fn(res=raw.type, arg=raw, loc=loc, ip=ip)
    flat = _flatten_to_1d(raw, loc=loc)
    elem_ty = flat.type.element_type
    n = int(flat.type.shape[0])
    out = [raw_fn(res=elem_ty, arg=_extract(flat, i, loc=loc), loc=loc, ip=ip) for i in range(n)]
    return _from_elements(ir.VectorType.get([n], elem_ty), out, loc=loc)


def _make_unary(name: str):
    raw_fn = _UNARY_OPS[name]

    @traced_op
    def _fn(*args, loc=None, ip=None):
        if len(args) == 1:
            return _apply_unary_elementwise(raw_fn, args[0], loc=loc, ip=ip)
        if len(args) == 2 and isinstance(args[0], ir.Type):
            res_ty, arg = args
            return raw_fn(res=res_ty, arg=_unwrap(arg, loc=loc), loc=loc, ip=ip)
        raise TypeError(
            f"rocdl.{name} expects either (arg) or (res_type, arg), got {len(args)} positional args"
        )

    _fn.__name__ = name
    _fn.__qualname__ = name
    _fn.__doc__ = (
        f"Elementwise ``rocdl.{name}`` that accepts scalar or any-size vector.\n\n"
        "Two calling conventions are supported for backward compatibility:\n"
        f"  * ``{name}(arg)`` — new-style, result type auto-inferred from arg,\n"
        "    vector inputs are unrolled lane-by-lane and reassembled via\n"
        "    ``vector.from_elements``.\n"
        f"  * ``{name}(res_type, arg)`` — legacy ODS-native, scalar only."
    )
    return _fn


rcp = _make_unary("rcp")
rsq = _make_unary("rsq")


# ─────────────────────────────────────────────────────────────────────────────
# Group B: ternary fmed3 -- ROCDL accepts scalar or vector f16/f32 natively.
# ─────────────────────────────────────────────────────────────────────────────


@traced_op
def fmed3(a, b, c, *, loc=None, ip=None):
    """Hardware-accelerated median-of-three for f16/f32 scalar or vector.

    Matches ``rocdl.fmed3`` semantics: ``max(min(a,b), min(max(a,b), c))``.
    ROCDL's fmed3 accepts both scalar and vector operands directly.
    """
    a_v = _unwrap(a, loc=loc)
    b_v = _unwrap(b, loc=loc)
    c_v = _unwrap(c, loc=loc)
    return _raw_rocdl.fmed3(res=a_v.type, src0=a_v, src1=b_v, src2=c_v, loc=loc, ip=ip)


# ─────────────────────────────────────────────────────────────────────────────
# Group C: cvt.pk.{fp8,bf8}.f32 -- pack 2 f32 into 1 word of i32.
#
# Per-call signature (ODS):
#     i32 = rocdl.cvt.pk.{fp8,bf8}.f32 srcA:f32, srcB:f32, old:i32, word_sel:i1
#
# Vector extension: accept vector<NxF32> with N % 4 == 0, produce
# vector<(N/4)xi32> packing 4 f32 into each i32 (word_sel=0 then word_sel=1).
# ─────────────────────────────────────────────────────────────────────────────


def _cvt_pk_f32_to_i32(raw_fn, src, *, old=None, loc=None, ip=None):
    src_flat = _flatten_to_1d(src, loc=loc)
    if isinstance(src_flat.type, ir.VectorType):
        n = int(src_flat.type.shape[0])
        if n % 4 != 0:
            raise ValueError(
                f"vector length must be a multiple of 4 for pk-f32->i32 conversion, got {n}"
            )
        elems = [_extract(src_flat, i, loc=loc) for i in range(n)]
        out_count = n // 4
        zero_i32 = _const_i32(0, loc=loc)
        out_i32 = []
        for k in range(out_count):
            a0, a1, a2, a3 = elems[4 * k : 4 * k + 4]
            lo = raw_fn(res=T.i32, src_a=a0, src_b=a1, old=zero_i32, word_sel=False, loc=loc, ip=ip)
            wd = raw_fn(res=T.i32, src_a=a2, src_b=a3, old=lo, word_sel=True, loc=loc, ip=ip)
            out_i32.append(wd)
        if out_count == 1:
            return out_i32[0]
        return _from_elements(ir.VectorType.get([out_count], T.i32), out_i32, loc=loc)
    # Scalar: need both src_a, src_b, old and word_sel from caller.
    if old is None:
        old = _const_i32(0, loc=loc)
    return raw_fn(
        res=T.i32,
        src_a=_unwrap(src, loc=loc),
        src_b=_unwrap(src, loc=loc),
        old=_unwrap(old, loc=loc),
        word_sel=False,
        loc=loc,
        ip=ip,
    )


@traced_op
def cvt_pk_fp8_f32_v(src, *, loc=None, ip=None):
    """Pack a ``vector<NxF32>`` (``N % 4 == 0``) into ``vector<(N/4)xi32>`` of fp8.

    Each output i32 is built from 4 consecutive f32 sources via two
    ``rocdl.cvt.pk.fp8.f32`` calls (low word then high word).
    """
    return _cvt_pk_f32_to_i32(_raw_rocdl.cvt_pk_fp8_f32, src, loc=loc, ip=ip)


@traced_op
def cvt_pk_bf8_f32_v(src, *, loc=None, ip=None):
    """Pack a ``vector<NxF32>`` (``N % 4 == 0``) into ``vector<(N/4)xi32>`` of bf8."""
    return _cvt_pk_f32_to_i32(_raw_rocdl.cvt_pk_bf8_f32, src, loc=loc, ip=ip)


# ─────────────────────────────────────────────────────────────────────────────
# Group C': cvt.pkrtz -- 2 f32 -> vector<2xf16>, round-to-zero.
#
# Vector extension: vector<NxF32> (N even) -> vector<NxF16>, concatenating
# N/2 pkrtz results.
# ─────────────────────────────────────────────────────────────────────────────


@traced_op
def cvt_pkrtz_v(src, *, loc=None, ip=None):
    """Convert ``vector<NxF32>`` (``N`` even) to ``vector<NxF16>`` with RTZ.

    Emits ``N/2`` ``rocdl.cvt.pkrtz`` calls and concatenates the results.
    Falls back to the native pair builder when ``N == 2``.
    """
    src_flat = _flatten_to_1d(src, loc=loc)
    pkrtz_ty = ir.VectorType.get([2], T.f16)
    if isinstance(src_flat.type, ir.VectorType):
        n = int(src_flat.type.shape[0])
        if n % 2 != 0:
            raise ValueError(f"vector length must be even for pkrtz, got {n}")
        out_f16 = []
        for k in range(n // 2):
            a = _extract(src_flat, 2 * k, loc=loc)
            b = _extract(src_flat, 2 * k + 1, loc=loc)
            pk = _raw_rocdl.cvt_pkrtz(res=pkrtz_ty, src_a=a, src_b=b, loc=loc, ip=ip)
            out_f16.append(_extract(pk, 0, loc=loc))
            out_f16.append(_extract(pk, 1, loc=loc))
        return _from_elements(ir.VectorType.get([n], T.f16), out_f16, loc=loc)
    raise TypeError("cvt_pkrtz_v expects a vector source; use rocdl.cvt_pkrtz for scalars")


# ─────────────────────────────────────────────────────────────────────────────
# Group D: cvt.pk.f32.{fp8,bf8} -- 1 i32 + word_sel -> vector<2xf32>.
#
# Each i32 encodes 4 fp8/bf8 bytes = 2 words. For vector<MxI32>, emit 2*M
# unpacks and concatenate to vector<(4*M)xF32>.
# ─────────────────────────────────────────────────────────────────────────────


def _cvt_pk_i32_to_f32(raw_fn, src, *, loc=None, ip=None):
    src_flat = _flatten_to_1d(src, loc=loc)
    pair_ty = ir.VectorType.get([2], T.f32)
    if isinstance(src_flat.type, ir.VectorType):
        m = int(src_flat.type.shape[0])
        out_f32 = []
        for k in range(m):
            word = _extract(src_flat, k, loc=loc)
            lo = raw_fn(res=pair_ty, src=word, word_sel=False, loc=loc, ip=ip)
            hi = raw_fn(res=pair_ty, src=word, word_sel=True, loc=loc, ip=ip)
            out_f32.append(_extract(lo, 0, loc=loc))
            out_f32.append(_extract(lo, 1, loc=loc))
            out_f32.append(_extract(hi, 0, loc=loc))
            out_f32.append(_extract(hi, 1, loc=loc))
        return _from_elements(ir.VectorType.get([4 * m], T.f32), out_f32, loc=loc)
    raw = _unwrap(src, loc=loc)
    lo = raw_fn(res=pair_ty, src=raw, word_sel=False, loc=loc, ip=ip)
    hi = raw_fn(res=pair_ty, src=raw, word_sel=True, loc=loc, ip=ip)
    return _from_elements(
        ir.VectorType.get([4], T.f32),
        [
            _extract(lo, 0, loc=loc),
            _extract(lo, 1, loc=loc),
            _extract(hi, 0, loc=loc),
            _extract(hi, 1, loc=loc),
        ],
        loc=loc,
    )


@traced_op
def cvt_pk_f32_fp8_v(src, *, loc=None, ip=None):
    """Unpack a ``vector<MxI32>`` of fp8 words into ``vector<(4*M)xF32>``."""
    return _cvt_pk_i32_to_f32(_raw_rocdl.cvt_pk_f32_fp8, src, loc=loc, ip=ip)


@traced_op
def cvt_pk_f32_bf8_v(src, *, loc=None, ip=None):
    """Unpack a ``vector<MxI32>`` of bf8 words into ``vector<(4*M)xF32>``."""
    return _cvt_pk_i32_to_f32(_raw_rocdl.cvt_pk_f32_bf8, src, loc=loc, ip=ip)


# ─────────────────────────────────────────────────────────────────────────────
# Group E: cvt.f32.{fp8,bf8} -- 1 i32 + byte_sel(0..3) -> f32.
#
# For vector<MxI32> emit 4*M calls producing vector<(4*M)xF32>.
# ─────────────────────────────────────────────────────────────────────────────


def _cvt_byte_i32_to_f32(raw_fn, src, *, loc=None, ip=None):
    src_flat = _flatten_to_1d(src, loc=loc)
    if isinstance(src_flat.type, ir.VectorType):
        m = int(src_flat.type.shape[0])
        out = []
        for k in range(m):
            word = _extract(src_flat, k, loc=loc)
            for byte in range(4):
                out.append(raw_fn(res=T.f32, src_a=word, byte_sel=byte, loc=loc, ip=ip))
        return _from_elements(ir.VectorType.get([4 * m], T.f32), out, loc=loc)
    raw = _unwrap(src, loc=loc)
    out = [raw_fn(res=T.f32, src_a=raw, byte_sel=b, loc=loc, ip=ip) for b in range(4)]
    return _from_elements(ir.VectorType.get([4], T.f32), out, loc=loc)


@traced_op
def cvt_f32_fp8_v(src, *, loc=None, ip=None):
    """Expand a ``vector<MxI32>`` of fp8 bytes into ``vector<(4*M)xF32>``.

    Emits ``4*M`` ``rocdl.cvt.f32.fp8`` calls, one per byte position.
    """
    return _cvt_byte_i32_to_f32(_raw_rocdl.cvt_f32_fp8, src, loc=loc, ip=ip)


@traced_op
def cvt_f32_bf8_v(src, *, loc=None, ip=None):
    """Expand a ``vector<MxI32>`` of bf8 bytes into ``vector<(4*M)xF32>``."""
    return _cvt_byte_i32_to_f32(_raw_rocdl.cvt_f32_bf8, src, loc=loc, ip=ip)


# ─────────────────────────────────────────────────────────────────────────────
# Group F: cvt.sr.{fp8,bf8}.f32 -- stochastic-rounding pack into byte of i32.
#
# Per-call signature:
#     i32 = rocdl.cvt.sr.{fp8,bf8}.f32 val:f32, stoch:i32, old:i32, byte_sel:i32
#
# Vector extension: val = vector<NxF32>, stoch = vector<NxI32> or scalar i32,
# produce vector<(N/4)xI32>, threading byte_sel=0..3 through each 4-group.
# N must be divisible by 4.
# ─────────────────────────────────────────────────────────────────────────────


def _cvt_sr_f32(raw_fn, val, stoch, *, loc=None, ip=None):
    val_flat = _flatten_to_1d(val, loc=loc)
    stoch_raw = _unwrap(stoch, loc=loc)
    if not isinstance(val_flat.type, ir.VectorType):
        raise TypeError("cvt_sr_*_v expects a vector f32 input")

    n = int(val_flat.type.shape[0])
    if n % 4 != 0:
        raise ValueError(
            f"vector length must be a multiple of 4 for stochastic-rounding pack, got {n}"
        )

    if isinstance(stoch_raw.type, ir.VectorType):
        if int(stoch_raw.type.shape[0]) != n:
            raise ValueError(
                f"stoch vector length {int(stoch_raw.type.shape[0])} does not match value length {n}"
            )
        stoch_elems = [_extract(stoch_raw, i, loc=loc) for i in range(n)]
    else:
        stoch_elems = [stoch_raw] * n

    val_elems = [_extract(val_flat, i, loc=loc) for i in range(n)]

    out_count = n // 4
    zero_i32 = _const_i32(0, loc=loc)
    out_i32 = []
    for k in range(out_count):
        acc = zero_i32
        for byte in range(4):
            idx = 4 * k + byte
            acc = raw_fn(
                res=T.i32,
                src_a=val_elems[idx],
                src_b=stoch_elems[idx],
                old=acc,
                byte_sel=byte,
                loc=loc,
                ip=ip,
            )
        out_i32.append(acc)
    if out_count == 1:
        return out_i32[0]
    return _from_elements(ir.VectorType.get([out_count], T.i32), out_i32, loc=loc)


@traced_op
def cvt_sr_fp8_f32_v(val, stoch, *, loc=None, ip=None):
    """Stochastic-round a ``vector<NxF32>`` to fp8 bytes packed in ``vector<(N/4)xI32>``.

    ``stoch`` can be a ``vector<NxI32>`` of per-element stochastic values or a
    scalar i32 broadcast to every lane. ``N`` must be divisible by 4.
    """
    return _cvt_sr_f32(_raw_rocdl.cvt_sr_fp8_f32, val, stoch, loc=loc, ip=ip)


@traced_op
def cvt_sr_bf8_f32_v(val, stoch, *, loc=None, ip=None):
    """Stochastic-round a ``vector<NxF32>`` to bf8 bytes packed in ``vector<(N/4)xI32>``."""
    return _cvt_sr_f32(_raw_rocdl.cvt_sr_bf8_f32, val, stoch, loc=loc, ip=ip)


__all__ = [
    "rcp",
    "rsq",
    "fmed3",
    "cvt_pk_fp8_f32_v",
    "cvt_pk_bf8_f32_v",
    "cvt_pkrtz_v",
    "cvt_pk_f32_fp8_v",
    "cvt_pk_f32_bf8_v",
    "cvt_f32_fp8_v",
    "cvt_f32_bf8_v",
    "cvt_sr_fp8_f32_v",
    "cvt_sr_bf8_f32_v",
]
