# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from enum import IntEnum
from typing import overload

from .._mlir import ir
from .._mlir.dialects import arith as _arith
from .._mlir.dialects import fly
from .._mlir.dialects.fly import (
    AddressSpace,
    AtomicOp,
    CachePolicy,
    ComposedLayoutType,
    CoordTensorType,
    CopyAtomType,
    CopyOpUniversalAtomicType,
    CopyOpUniversalCopyType,
    GemmTraversalOrder,
    IntTupleType,
    LayoutType,
    MemRefType,
    MmaAtomType,
    MmaOperand,
    MmaOpUniversalFMAType,
    PointerType,
    SwizzleType,
    TiledCopyType,
    TiledMmaType,
    TileType,
    #
    has_none,
)
from .._mlir.extras import types as T
from .meta import traced_op

__all__ = [
    # Maybe remove it in the future
    "T",
    # "arith",
    # Enum Attributes
    "AtomicOp",
    "AddressSpace",
    "CachePolicy",
    "MmaOperand",
    "GemmTraversalOrder",
    # Types
    "IntTupleType",
    "TileType",
    "LayoutType",
    "SwizzleType",
    "ComposedLayoutType",
    "PointerType",
    "MemRefType",
    "CoordTensorType",
    "CopyAtomType",
    "MmaAtomType",
    "TiledCopyType",
    "TiledMmaType",
    "CopyOpUniversalCopyType",
    "CopyOpUniversalAtomicType",
    "MmaOpUniversalFMAType",
    # UniversalOps
    "UniversalCopy",
    "UniversalCopy8b",
    "UniversalCopy16b",
    "UniversalCopy32b",
    "UniversalCopy64b",
    "UniversalCopy128b",
    "UniversalAtomic",
    "UniversalAtomicAdd",
    "UniversalAtomicMax",
    "UniversalAtomicMin",
    "UniversalAtomicAnd",
    "UniversalAtomicOr",
    "UniversalAtomicInc",
    "UniversalAtomicDec",
    "UniversalFMA",
    # Constexpr functions
    "const_expr",
    "range_constexpr",
    "rank",
    "depth",
    "has_none",
    # DSL functions
    "static",
    "make_int_tuple",
    "make_shape",
    "make_stride",
    "make_coord",
    "make_layout",
    "make_layout_like",
    "make_ordered_layout",
    "make_composed_layout",
    "make_identity_layout",
    "make_view",
    "make_fragment_layout_like",
    "make_fragment_like",
    "get_scalar",
    "get_leaves",
    "get_shape",
    "get_stride",
    "get_layout",
    "get_iter",
    "composed_get_inner",
    "composed_get_offset",
    "composed_get_outer",
    "int_tuple_add",
    "int_tuple_sub",
    "int_tuple_mul",
    "int_tuple_div",
    "int_tuple_mod",
    "int_tuple_product",
    "int_tuple_product_each",
    "int_tuple_product_like",
    "shape_div",
    "ceil_div",
    "elem_less",
    "equal",
    "get",
    "get_",
    "take",
    "select",
    "group",
    "append",
    "prepend",
    "slice",
    "dice",
    "size",
    "coprofile",
    "coshape",
    "cosize",
    "crd2idx",
    "idx2crd",
    "get_flat_coord",
    "get_1d_coord",
    "coalesce",
    "composition",
    "complement",
    "right_inverse",
    "left_inverse",
    "logical_divide",
    "zipped_divide",
    "tiled_divide",
    "flat_divide",
    "logical_product",
    "zipped_product",
    "tiled_product",
    "flat_product",
    "block_product",
    "raked_product",
    "recast_layout",
    "tile_to_shape",
    "make_mma_atom",
    "make_copy_atom",
    "atom_set_value",
    "copy_atom_call",
    "mma_atom_call",
    "make_tiled_copy",
    "make_tiled_mma",
    "tiled_copy_partition_src",
    "tiled_copy_partition_dst",
    "tiled_copy_retile",
    "tiled_mma_partition",
    "tiled_mma_partition_shape",
    "mma_make_fragment",
    "copy",
    "gemm",
    "make_ptr",
    "get_dyn_shared",
    "inttoptr",
    "ptrtoint",
    "add_offset",
    "apply_swizzle",
    "ptr_load",
    "ptr_store",
    "recast_iter",
    "memref_alloca",
    "memref_load_vec",
    "memref_store_vec",
    "memref_load",
    "memref_store",
    "printf",
    "assume",
    "make_tile",
]


UniversalCopy = lambda bit_size: CopyOpUniversalCopyType.get(bit_size)
UniversalCopy8b = lambda: CopyOpUniversalCopyType.get(8)
UniversalCopy16b = lambda: CopyOpUniversalCopyType.get(16)
UniversalCopy32b = lambda: CopyOpUniversalCopyType.get(32)
UniversalCopy64b = lambda: CopyOpUniversalCopyType.get(64)
UniversalCopy128b = lambda: CopyOpUniversalCopyType.get(128)

UniversalAtomic = lambda atomic_op, val_type: CopyOpUniversalAtomicType.get(int(atomic_op), val_type.ir_type)
UniversalAtomicAdd = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Add), val_type.ir_type)
UniversalAtomicMax = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Max), val_type.ir_type)
UniversalAtomicMin = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Min), val_type.ir_type)
UniversalAtomicAnd = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.And), val_type.ir_type)
UniversalAtomicOr = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Or), val_type.ir_type)
UniversalAtomicInc = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Inc), val_type.ir_type)
UniversalAtomicDec = lambda val_type: CopyOpUniversalAtomicType.get(int(AtomicOp.Dec), val_type.ir_type)

UniversalFMA = lambda ty: MmaOpUniversalFMAType.get(ty.ir_type)


def const_expr(x):
    return x


def range_constexpr(*args):
    return range(*args)


def rank(int_or_tuple):
    """Number of top-level elements of a tuple / layout.

    A leaf integer has rank 1; each child of a nested tuple counts as one mode.

    Examples:
        rank((8, 16))        -> 2
        rank((8, (4, 2)))    -> 2   (the nested (4, 2) still counts as one mode)
    """
    return fly.rank(int_or_tuple)


def depth(int_or_tuple):
    """How deeply the tuple is nested.

    Leaf integers have depth 0; a flat tuple has depth 1; each extra level of
    nesting adds one.

    Examples:
        depth(8)             -> 0
        depth((8, 16))       -> 1
        depth((8, (4, 2)))   -> 2
    """
    return fly.depth(int_or_tuple)


# ===----------------------------------------------------------------------=== #
# Constructors
# ===----------------------------------------------------------------------=== #


@traced_op
def static(result_type, loc=None, ip=None):
    """Materialize a value whose entire content is encoded in *result_type*.

    Used for fully known compile-time objects: static tuples, tiles, swizzles, etc.
    All information lives in the type, so no runtime operands are needed.

    Examples:
        static(IntTupleType.get((4, 8)))          -> a static (4, 8) tuple
        static(SwizzleType.get(3, 3, 3))          -> a static swizzle descriptor
    """
    return fly.static(result_type, loc=loc, ip=ip)


@traced_op
def make_int_tuple(elems, loc=None, ip=None):
    """Build a (possibly nested) integer tuple from Python ints or runtime values.

    Integers become static entries; `ir.Value` operands become dynamic entries.

    Examples:
        make_int_tuple((4, 8))           -> static tuple (4, 8)
        make_int_tuple((m, 8))           -> (m, 8) where m is a runtime int
    """
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(elems)
    return fly.make_int_tuple(IntTupleTy, dyncElems, loc=loc, ip=ip)


@traced_op
def make_shape(*shape, loc=None, ip=None):
    """Build a shape tuple describing the extent of each mode.

    Supports nested shapes for hierarchical tiling.

    Examples:
        make_shape(8, 16)          -> (8, 16)
        make_shape(9, (4, 8))      -> (9, (4, 8))  (second mode is sub-structured)
    """
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(shape)
    return fly.make_shape(IntTupleTy, dyncElems, loc=loc, ip=ip)


@traced_op
def make_stride(*stride, loc=None, ip=None):
    """Build a stride tuple: the step (in elements) when moving along each mode.

    Nested structure must mirror the shape it will be paired with.

    Examples:
        make_stride(1, 8)                  -> column-major stride for (8, 16)
        make_stride(16, 1)                 -> row-major stride for (8, 16)
    """
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(stride)
    return fly.make_stride(IntTupleTy, dyncElems, loc=loc, ip=ip)


@traced_op
def make_coord(*coord, loc=None, ip=None):
    """Build a coordinate used for indexing / slicing a layout.

    Use `None` in a mode to mean "all positions of that mode" (a free axis).

    Examples:
        make_coord(3, 5)           -> point coordinate (row 3, col 5)
        make_coord(None, bid)      -> (:, bid)  keep first axis free, pick second
    """
    IntTupleTy, dyncElems = fly.infer_int_tuple_type(coord)
    return fly.make_coord(IntTupleTy, dyncElems, loc=loc, ip=ip)


@traced_op
def make_layout(shape, stride, loc=None, ip=None):
    """Pair a *shape* with a *stride* to describe how logical coords map to memory.

    Accepts Python tuples directly (auto-converted). The mapping is:
    `index = sum(coord_i * stride_i)`.

    Examples:
        make_layout((8, 16), (1, 8))     # 8x16 column-major: idx = r + c*8
        make_layout((8, 16), (16, 1))    # 8x16 row-major:    idx = r*16 + c
    """
    if not isinstance(shape, ir.Value):
        shapeTy, dyncElems = fly.infer_int_tuple_type(shape)
        shape = fly.make_shape(shapeTy, dyncElems, loc=loc, ip=ip)
    if not isinstance(stride, ir.Value):
        strideTy, dyncElems = fly.infer_int_tuple_type(stride)
        stride = fly.make_stride(strideTy, dyncElems, loc=loc, ip=ip)
    return fly.make_layout(shape, stride=stride, loc=loc, ip=ip)


@traced_op
def make_layout_like(ref, loc=None, ip=None):
    """Copy the shape of *ref* into a fresh layout (strides may be re-derived).

    Useful when you want "a layout with the same dimensions as this one" without
    repeating the shape literals.

    Examples:
        make_layout_like(src_layout)     # same shape as src_layout
    """
    return fly.make_layout_like(ref, loc=loc, ip=ip)


@traced_op
def make_ordered_layout(shape, order, loc=None, ip=None):
    """Build a compact layout whose stride order matches *order*.

    `order[i]` says where mode *i* sits when ranking strides from fastest
    (smallest value) to slowest. Lower means more contiguous.

    Examples:
        make_ordered_layout((M, N), (0, 1))  # column-major: M iterates fastest
        make_ordered_layout((M, N), (1, 0))  # row-major:    N iterates fastest
    """
    if not isinstance(shape, ir.Value):
        shapeTy, dyncElems = fly.infer_int_tuple_type(shape)
        shape = fly.make_shape(shapeTy, dyncElems, loc=loc, ip=ip)
    if not isinstance(order, ir.Value):
        orderTy, dyncElems = fly.infer_int_tuple_type(order)
        order = fly.make_int_tuple(orderTy, dyncElems, loc=loc, ip=ip)
    return fly.make_ordered_layout(shape, order, loc=loc, ip=ip)


@overload
def make_composed_layout(inner, offset, outer, loc=None, ip=None): ...
@overload
def make_composed_layout(inner, outer, loc=None, ip=None): ...
@traced_op
def make_composed_layout(inner, offset_or_outer, outer=None, loc=None, ip=None):
    """Stack two layouts: a coord is first mapped by *outer*, then by *inner*.

    An optional constant *offset* is added after the outer mapping. The common
    use is pairing a swizzle (inner) with a plain layout (outer) to permute
    addresses and avoid bank conflicts.

    Examples:
        make_composed_layout(swizzle, layout)           # no offset
        make_composed_layout(swizzle, 16, layout)       # with offset = 16
    """
    if outer is None:
        outer = offset_or_outer
        offset = make_int_tuple(0, loc=loc, ip=ip)
    else:
        offset = offset_or_outer
    return fly.make_composed_layout(inner, offset, outer, loc=loc, ip=ip)


@traced_op
def make_identity_layout(shape, loc=None, ip=None):
    """Layout with the given *shape* whose coordinates equal their own address.

    Acts like `numpy.arange`-style indexing: position `(i, j)` maps to `(i, j)`.
    Mostly used as a building block for partitioning / coordinate tensors.

    Examples:
        make_identity_layout((M, N))       # identity mapping over (M, N)
    """
    return fly.make_identity_layout(shape, loc=loc, ip=ip)


@traced_op
def make_view(iter, layout, loc=None, ip=None):
    """Combine a pointer/iterator with a layout to get an addressable tensor.

    No data is copied - the view just re-interprets the memory behind *iter*
    through *layout*'s shape and stride.

    Examples:
        make_view(smem_ptr, make_layout((BM, BK), (1, BM)))   # view LDS as BMxBK
    """
    return fly.make_view(iter, layout, loc=loc, ip=ip)


@traced_op
def make_fragment_layout_like(tensor, loc=None, ip=None):
    """Get a layout shaped like *tensor*'s logical view, suited for register fragments.

    Typically called before `make_fragment_like` when you want to further modify
    the layout (e.g. double it with `flat_product`).

    Examples:
        frag_layout = make_fragment_layout_like(thr_gB)
    """
    return fly.make_fragment_layout_like(tensor, loc=loc, ip=ip)


@traced_op
def make_fragment_like(tensor, dtype=None, loc=None, ip=None):
    """Allocate a register-level fragment whose shape matches *tensor*.

    Pass *dtype* to override the element type; omit it to inherit the source's.

    Examples:
        frag = make_fragment_like(partition_src)                   # same dtype
        frag_f16 = make_fragment_like(partition_src, Float16)      # recast to f16
    """
    if hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type
    return fly.make_fragment_like(tensor, dtype=dtype, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# Extractors
# ===----------------------------------------------------------------------=== #


@traced_op
def get_scalar(int_tuple, loc=None, ip=None):
    """Unwrap a rank-1, single-element tuple back to a plain scalar value.

    Fails if the input has more than one leaf - use this only when you know
    the tuple is a trivial wrapper.

    Examples:
        get_scalar(make_int_tuple(5))   -> 5
    """
    return fly.get_scalar(int_tuple, loc=loc, ip=ip)


@traced_op
def get_leaves(input, dynamic_only=False, loc=None, ip=None):
    """Flatten a nested tuple into a flat sequence of leaf values.

    Set *dynamic_only=True* to keep only runtime values and drop static
    constants - handy when you need the inputs that were passed at call time.

    Examples:
        get_leaves((8, (4, 2)))                -> (8, 4, 2)
        get_leaves((m, 4, n), dynamic_only=1)  -> (m, n)   # 4 is static, dropped
    """
    res_lists = fly.GetLeavesOp(input, dynamicOnly=dynamic_only, loc=loc, ip=ip)
    return tuple(res_lists.results)


@traced_op
def get_shape(layout, loc=None, ip=None):
    """Return the shape tuple of a layout.

    Examples:
        get_shape(make_layout((8, 16), (1, 8)))   -> (8, 16)
    """
    return fly.get_shape(layout, loc=loc, ip=ip)


@traced_op
def get_stride(layout, loc=None, ip=None):
    """Return the stride tuple of a layout.

    Examples:
        get_stride(make_layout((8, 16), (1, 8)))   -> (1, 8)
    """
    return fly.get_stride(layout, loc=loc, ip=ip)


@traced_op
def get_layout(memref, loc=None, ip=None):
    """Extract the layout of a memref / tensor view.

    Examples:
        L = get_layout(gA_tile)      # access its shape / stride for further algebra
    """
    return fly.get_layout(memref, loc=loc, ip=ip)


@traced_op
def get_iter(memref, loc=None, ip=None):
    """Extract the base pointer / iterator of a memref, dropping the layout.

    Useful when you want to rebind the same memory through a different layout
    via `make_view`.

    Examples:
        p = get_iter(gA_tile)
        new_tile = make_view(p, other_layout)
    """
    return fly.get_iter(memref, loc=loc, ip=ip)


@traced_op
def composed_get_inner(input, loc=None, ip=None):
    """For a composed layout `inner o outer`, return the *inner* layout.

    The inner layout is applied second (after the outer mapping + offset).

    Examples:
        composed_get_inner(make_composed_layout(swizzle, layout))   -> swizzle
    """
    return fly.composed_get_inner(input, loc=loc, ip=ip)


@traced_op
def composed_get_offset(input, loc=None, ip=None):
    """Return the constant offset tuple of a composed layout (`0` if unspecified).

    Examples:
        composed_get_offset(make_composed_layout(swizzle, 16, layout))   -> 16
    """
    return fly.composed_get_offset(input, loc=loc, ip=ip)


@traced_op
def composed_get_outer(input, loc=None, ip=None):
    """For a composed layout, return the *outer* layout applied first.

    Examples:
        composed_get_outer(make_composed_layout(swizzle, layout))   -> layout
    """
    return fly.composed_get_outer(input, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# IntTuple operations
# ===----------------------------------------------------------------------=== #


@traced_op
def int_tuple_add(lhs, rhs, loc=None, ip=None):
    """Element-wise add across tuples with the same nested structure.

    Either side may also be a plain integer broadcast over all leaves.

    Examples:
        int_tuple_add((2, 3), (1, 5))        -> (3, 8)
        int_tuple_add((2, (3, 4)), 1)        -> (3, (4, 5))   # scalar broadcast
    """
    return fly.int_tuple_add(lhs, rhs, loc=loc, ip=ip)


@traced_op
def int_tuple_sub(lhs, rhs, loc=None, ip=None):
    """Element-wise subtraction with matching structure or scalar broadcast.

    Examples:
        int_tuple_sub((5, 8), (1, 3))        -> (4, 5)
        int_tuple_sub((5, 8), 1)             -> (4, 7)
    """
    return fly.int_tuple_sub(lhs, rhs, loc=loc, ip=ip)


@traced_op
def int_tuple_mul(lhs, rhs, loc=None, ip=None):
    """Element-wise multiplication across tuples, with optional scalar broadcast.

    Examples:
        int_tuple_mul((2, 3), (4, 5))        -> (8, 15)
        int_tuple_mul((2, 3), 4)             -> (8, 12)
    """
    return fly.int_tuple_mul(lhs, rhs, loc=loc, ip=ip)


@traced_op
def int_tuple_div(lhs, rhs, loc=None, ip=None):
    """Element-wise integer division across tuples (truncation toward zero).

    Examples:
        int_tuple_div((8, 15), (4, 5))       -> (2, 3)
        int_tuple_div((9, 7), 2)             -> (4, 3)
    """
    return fly.int_tuple_div(lhs, rhs, loc=loc, ip=ip)


@traced_op
def int_tuple_mod(lhs, rhs, loc=None, ip=None):
    """Element-wise remainder across tuples.

    Examples:
        int_tuple_mod((9, 7), (4, 5))        -> (1, 2)
        int_tuple_mod((9, 10), 3)            -> (0, 1)
    """
    return fly.int_tuple_mod(lhs, rhs, loc=loc, ip=ip)


@traced_op
def int_tuple_product(int_tuple, loc=None, ip=None):
    """Multiply all leaves of a (possibly nested) tuple together.

    Examples:
        int_tuple_product((4, 8))            -> 32
        int_tuple_product((2, (3, 4)))       -> 24
    """
    return fly.int_tuple_product(int_tuple, loc=loc, ip=ip)


@traced_op
def int_tuple_product_each(int_tuple, loc=None, ip=None):
    """Reduce each top-level mode to a single product, keeping rank intact.

    Nested children collapse to one number per mode.

    Examples:
        int_tuple_product_each((4, (2, 3)))      -> (4, 6)
        int_tuple_product_each(((2, 3), (4, 5))) -> (6, 20)
    """
    return fly.int_tuple_product_each(int_tuple, loc=loc, ip=ip)


@traced_op
def int_tuple_product_like(lhs, rhs, loc=None, ip=None):
    """Reduce *lhs* to match the rank profile of *rhs*.

    Modes of *lhs* are grouped and multiplied so the result mirrors *rhs*'s
    outer structure.

    Examples:
        int_tuple_product_like((2, 3, 4), (10, 20))         -> (2, 12)
        int_tuple_product_like((2, 3, 4, 5), (0, (0, 0)))   -> (2, (3, 20))
    """
    return fly.int_tuple_product_like(lhs, rhs, loc=loc, ip=ip)


@traced_op
def shape_div(lhs, rhs, loc=None, ip=None):
    """Divide a shape by another along matching modes ("shape subtraction").

    Computes how many copies of *rhs* fit into *lhs* per mode. Requires that
    *rhs* cleanly divides *lhs*.

    Examples:
        shape_div((128, 64), (16, 16))       -> (8, 4)
        shape_div((M, N), (BM, BN))          -> (M/BM, N/BN) tile count
    """
    return fly.shape_div(lhs, rhs, loc=loc, ip=ip)


@traced_op
def ceil_div(lhs, rhs, loc=None, ip=None):
    """Element-wise ceiling division: `(lhs + rhs - 1) // rhs` per leaf.

    Typical use: compute the number of tiles needed to cover a shape.

    Examples:
        ceil_div(130, 16)             -> 9
        ceil_div((M, N), (BM, BN))    -> (ceil(M/BM), ceil(N/BN))
    """
    return fly.ceil_div(lhs, rhs, loc=loc, ip=ip)


@traced_op
def elem_less(lhs, rhs, loc=None, ip=None):
    """Lexicographic "less-than" comparison on tuples; returns a single i1 bool.

    Compares tuples left-to-right, treating them like multi-digit numbers.

    Examples:
        elem_less((3, 1), (3, 2))            -> True
        elem_less((3, 5), (3, 2))            -> False
    """
    return fly.elem_less(lhs, rhs, loc=loc, ip=ip)


@traced_op
def equal(lhs, rhs, loc=None, ip=None):
    """Structural equality of two int tuples; returns a single i1 bool.

    Both structure and every leaf value must match.

    Examples:
        equal((4, (2, 3)), (4, (2, 3)))      -> True
        equal((4, 6), (4, (2, 3)))           -> False   # different structure
    """
    return fly.equal(lhs, rhs, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# IntTupleLike operations
# ===----------------------------------------------------------------------=== #


@traced_op
def get(int_tuple, mode, loc=None, ip=None):
    """Pick a single mode from a tuple and return it as a plain scalar.

    Works on Python tuples too (falls back to Python indexing). Prefer `get_`
    if you want to keep the tuple wrapper around the result.

    Examples:
        get((4, 8, 16), 1)                   -> 8
        get(make_int_tuple((4, 8)), 0)       -> 4 (as a scalar value)
    """
    if isinstance(int_tuple, (list, tuple)):
        return int_tuple[mode]
    selected = fly.select(int_tuple, indices=[mode], loc=loc, ip=ip)
    result = fly.get_scalar(selected, loc=loc, ip=ip)
    if isinstance(result, ir.Value) and not isinstance(result.type, ir.IndexType):
        result = _arith.IndexCastOp(T.index(), result).result
    return result


@traced_op
def get_(int_tuple, mode, loc=None, ip=None):
    """Like `get`, but keep the result wrapped as a tuple.

    Use when you need the child to stay a tuple (e.g. because it's nested).

    Examples:
        get_((4, (2, 3)), 1)     -> (2, 3)          # keeps the sub-tuple
        get_((4, 8, 16), [0, 2]) -> (4, 16)         # pick multiple modes
    """
    if isinstance(mode, int):
        mode = [mode]
    return fly.get(int_tuple, mode, loc=loc, ip=ip)


@traced_op
def take(int_tuple, begin: int, end: int, loc=None, ip=None):
    """Take a contiguous slice of modes, like Python list slicing.

    Returns modes `[begin, end)` unchanged in structure.

    Examples:
        take((4, 8, 16, 32), 1, 3)           -> (8, 16)
        take(((2, 3), 5, 7), 0, 2)           -> ((2, 3), 5)
    """
    return fly.take(int_tuple, begin=begin, end=end, loc=loc, ip=ip)


@traced_op
def select(int_tuple, indices, loc=None, ip=None):
    """Pick a subset of modes by index, in any order (possibly with repeats).

    Examples:
        select((4, 8, 16), [2, 0])           -> (16, 4)
        select((4, 8, 16, 32), [1, 1, 3])    -> (8, 8, 32)
    """
    return fly.select(int_tuple, indices=indices, loc=loc, ip=ip)


@traced_op
def group(int_tuple, begin: int, end: int, loc=None, ip=None):
    """Wrap modes `[begin, end)` into one nested mode, increasing nesting depth.

    Used to logically merge adjacent modes without changing the total element count.

    Examples:
        group((4, 8, 16, 32), 1, 3)          -> (4, (8, 16), 32)
        group((2, 3, 5), 0, 2)               -> ((2, 3), 5)
    """
    return fly.group(int_tuple, begin=begin, end=end, loc=loc, ip=ip)


@traced_op
def append(base, elem, n: int | None = None, loc=None, ip=None):
    """Append *elem* to the tail of *base*, optionally padding to length *n*.

    If *n* is given and `rank(base) < n`, fills the missing positions with *elem*.

    Examples:
        append((4, 8), 16)                   -> (4, 8, 16)
        append((4,), 1, n=3)                 -> (4, 1, 1)   # pad to length 3
    """
    return fly.append(base, elem, n=n, loc=loc, ip=ip)


@traced_op
def prepend(base, elem, n: int | None = None, loc=None, ip=None):
    """Prepend *elem* to the head of *base*, optionally padding to length *n*.

    Examples:
        prepend((4, 8), 2)                   -> (2, 4, 8)
        prepend((8,), 1, n=3)                -> (1, 1, 8)   # pad to length 3
    """
    return fly.prepend(base, elem, n=n, loc=loc, ip=ip)


@traced_op
def slice(src, coord, loc=None, ip=None):
    """Keep the modes where *coord* has `None` (wildcard), drop the rest.

    A None in coord means "all of this axis"; a fixed integer picks that index
    and the mode disappears from the result.

    Examples:
        slice((4, 8, 16), (None, 3, None))   -> (4, 16)   # mode 1 fixed, dropped
        slice(layout, make_coord(None, bid)) -> sub-layout for column `bid`
    """
    if not isinstance(coord, ir.Value):
        coordTy, dyncElems = fly.infer_int_tuple_type(coord)
        coord = fly.make_coord(coordTy, dyncElems, loc=loc, ip=ip)
    return fly.slice(src, coord, loc=loc, ip=ip)


@traced_op
def dice(src, coord, loc=None, ip=None):
    """Complement of `slice`: keep the *fixed* modes, drop the `None` (wildcard) ones.

    Useful for extracting the per-tile / per-thread coordinate from a partitioned layout.

    Examples:
        dice((4, 8, 16), (None, 3, None))    -> (8,)
        dice(coord_tensor, make_coord(tid, None)) -> the thread-only part
    """
    if not isinstance(coord, ir.Value):
        coordTy, dyncElems = fly.infer_int_tuple_type(coord)
        coord = fly.make_coord(coordTy, dyncElems, loc=loc, ip=ip)
    return fly.dice(src, coord, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# LayoutLike operations
# ===----------------------------------------------------------------------=== #


@traced_op
def size(int_tuple, loc=None, ip=None):
    """Total number of elements covered by a tuple / shape / layout (product of leaves).

    Examples:
        size((8, 16))                -> 128
        size(make_layout((4, (2, 3)), ...))  -> 24
    """
    return fly.size(int_tuple, loc=loc, ip=ip)


@traced_op
def coprofile(layout, loc=None, ip=None):
    """Return the profile (nesting structure) of the layout's output index tuple.

    Describes *how* the output indices are grouped, with all leaf sizes zeroed
    out - only the nesting shape is kept.

    Examples:
        coprofile(make_layout((4, 8), (1, 4)))         -> (0,)
        coprofile(make_layout((4, 8), ((1, 2), 4)))    -> ((0, 0),)
    """
    return fly.coprofile(layout, loc=loc, ip=ip)


@traced_op
def coshape(layout, loc=None, ip=None):
    """Per-mode upper bound on the output index of a layout (like a shape of the codomain).

    Each mode's upper bound equals `(max_coord_i * stride_i) + 1` aggregated;
    nesting matches `coprofile`.

    Examples:
        coshape(make_layout((4, 8), (1, 4)))     -> (32,)   # max index + 1
        coshape(make_layout((4, 8), (8, 1)))     -> (32,)
    """
    return fly.coshape(layout, loc=loc, ip=ip)


@traced_op
def cosize(layout, loc=None, ip=None):
    """Size of the smallest contiguous range needed to hold every index the layout produces.

    Effectively `size(coshape(layout))` - useful to figure out how much memory
    a tile really occupies.

    Examples:
        cosize(make_layout((4, 8), (1, 4)))      -> 32   # spans indices [0, 32)
        cosize(make_layout((4, 4), (1, 8)))      -> 25   # spans [0, 25), has gaps
    """
    return fly.cosize(layout, loc=loc, ip=ip)


def _to_i32(v):
    """Cast index-type ir.Value to i32 (required by fly.make_int_tuple)."""
    if isinstance(v, ir.Value) and isinstance(v.type, ir.IndexType):
        return _arith.IndexCastOp(T.i32(), v).result
    return v


@traced_op
def crd2idx(crd, layout, loc=None, ip=None):
    """Map a coordinate tuple to the flat memory index through *layout*.

    Applies `layout(crd) = sum(crd_i * stride_i)`, matching nested structure
    between *crd* and the layout's shape.

    Examples:
        crd2idx((1, 2), make_layout((4, 8), (1, 4)))   -> 1 + 2*4 = 9
        crd2idx(7, make_layout((4, 8), (1, 4)))        # 1-D idx -> multi-D -> idx
    """
    if not isinstance(crd, ir.Value):
        if isinstance(crd, (list, tuple)):
            crd = tuple(_to_i32(c) for c in crd)
        crdTy, dyncElems = fly.infer_int_tuple_type(crd)
        crd = fly.make_coord(crdTy, dyncElems, loc=loc, ip=ip)
    return fly.crd2idx(crd, layout, loc=loc, ip=ip)


@traced_op
def idx2crd(index, layout, loc=None, ip=None):
    """Inverse of `crd2idx`: map a 1-D index back to a coordinate tuple.

    The result matches the nested structure of the layout's shape.

    Examples:
        idx2crd(9, make_layout((4, 8), (1, 4)))        -> (1, 2)
        idx2crd(5, make_layout((4, 8), (8, 1)))        -> (0, 5)
    """
    if isinstance(index, ir.Value) and not str(index.type).startswith("!fly.int_tuple"):
        index = _to_i32(index)
        IntTupleTy, dyncElems = fly.infer_int_tuple_type(index)
        index = fly.make_int_tuple(IntTupleTy, dyncElems, loc=loc, ip=ip)
    if not isinstance(index, ir.Value):
        indexTy, dyncElems = fly.infer_int_tuple_type(index)
        index = fly.make_int_tuple(indexTy, dyncElems, loc=loc, ip=ip)
    return fly.idx2crd(index, layout, loc=loc, ip=ip)


@traced_op
def get_flat_coord(index, layout, loc=None, ip=None):
    """Map a 1-D index to a *fully flattened* coordinate, ignoring nested grouping.

    Unlike `idx2crd`, the result is always a flat tuple of length `rank` of
    shape's flattened form - convenient when you want per-axis coordinates.

    Examples:
        get_flat_coord(9, make_layout((4, 8), (1, 4)))            -> (1, 2)
        get_flat_coord(3, make_layout(((2, 2), 4), ((1, 2), 4)))  -> (1, 1, 0)
    """
    if not isinstance(index, ir.Value):
        indexTy, dyncElems = fly.infer_int_tuple_type(index)
        index = fly.make_int_tuple(indexTy, dyncElems, loc=loc, ip=ip)
    return fly.get_flat_coord(index, layout, loc=loc, ip=ip)


@traced_op
def get_1d_coord(index, layout, loc=None, ip=None):
    """Normalize any coordinate (or 1-D index) into a single "logical" 1-D coord.

    Handy when you have a multi-dim coordinate and want to pass it to a kernel
    that expects a flat index in row-major traversal order.

    Examples:
        get_1d_coord((1, 2), make_layout((4, 8)))   -> 2*4 + 1 = 9 (layout-dep.)
        get_1d_coord(5, layout)                      -> 5
    """
    if not isinstance(index, ir.Value):
        indexTy, dyncElems = fly.infer_int_tuple_type(index)
        index = fly.make_int_tuple(indexTy, dyncElems, loc=loc, ip=ip)
    return fly.get_1d_coord(index, layout, loc=loc, ip=ip)


@traced_op
def coalesce(layout, pattern=None, loc=None, ip=None):
    """Merge adjacent modes that are contiguous in memory into a single mode.

    With no *pattern*: produces a minimal-rank equivalent layout. With *pattern*:
    coalesces only far enough to match the pattern's rank profile.

    Examples:
        coalesce(make_layout((2, 4), (1, 2)))             -> ((8,), (1,))
        coalesce(make_layout((4, 1, 8), (1, 0, 4)), (0,)) -> (32, 1) after collapse
    """
    return fly.coalesce(layout, pattern=pattern, loc=loc, ip=ip)


@traced_op
def composition(layout, tiler, loc=None, ip=None):
    """Layout function composition: `(layout o tiler)(c) = layout(tiler(c))`.

    Conceptually: "feed the output of *tiler* as the input to *layout*". This
    is the core operator that `divide` / `product` are built from.

    Examples:
        composition(A, B)      # apply B first, then A
        composition(A, make_layout((4, 4), (4, 1)))   # reshape A's access order
    """
    return fly.composition(layout, tiler, loc=loc, ip=ip)


@traced_op
def complement(layout, codomain_size=None, loc=None, ip=None):
    """Build the "gap-filling" layout so that `layout + complement(layout, N)`
    together cover the full range `[0, N)` exactly once.

    If *codomain_size* is omitted, uses the layout's own cosize.

    Examples:
        complement(make_layout(4, 1), 24)              -> layout of size 24/4=6
        complement(make_layout((2, 4), (1, 4)), 24)    -> complement into stride 2
    """
    if codomain_size is not None and not isinstance(codomain_size, ir.Value):
        codomain_sizeTy, dyncElems = fly.infer_int_tuple_type(codomain_size)
        codomain_size = fly.make_shape(codomain_sizeTy, dyncElems, loc=loc, ip=ip)
    return fly.complement(layout, codomain_size=codomain_size, loc=loc, ip=ip)


@traced_op
def right_inverse(layout, loc=None, ip=None):
    """Build a layout R such that `layout(R(i)) == i` for every `i` in the codomain.

    Works when *layout* is injective; used to invert an address permutation.

    Examples:
        right_inverse(make_layout((4, 8), (8, 1)))    # row-major <-> column-major
    """
    return fly.right_inverse(layout, loc=loc, ip=ip)


@traced_op
def left_inverse(layout, loc=None, ip=None):
    """Build a layout L such that `L(layout(c)) == c` for every coordinate `c`.

    The dual of `right_inverse`; useful when mapping indices back to input coords.

    Examples:
        left_inverse(make_layout((4, 8), (1, 4)))    # undo the shape/stride mapping
    """
    return fly.left_inverse(layout, loc=loc, ip=ip)


@traced_op
def logical_divide(layout, divisor, loc=None, ip=None):
    """Split each mode of *layout* into `(inside_tile, outside_tile)` per *divisor*.

    Result has the same rank as *layout*, but each mode becomes a nested
    `(tile, rest)` pair. Used as the common foundation for tiling and partitioning.

    Examples:
        logical_divide((16, 32), (4, 8))   -> ((4, 4), (8, 4))   # per-mode nesting
        logical_divide((24, 8), (6, 2))    -> ((6, 4), (2, 4))
    """
    if not isinstance(divisor, ir.Value):
        divisor = make_tile(*divisor, loc=loc, ip=ip)
    return fly.logical_divide(layout, divisor, loc=loc, ip=ip)


@traced_op
def zipped_divide(layout, divisor, loc=None, ip=None):
    """Divide then zip: return `(tile_layout, rest_layout)` as two siblings.

    Re-packages `logical_divide`'s output into a clean "tile x rest" tensor,
    which is what almost every tiled kernel actually uses.

    Examples:
        zipped_divide((16, 32), (4, 8))     -> ((4, 8), (4, 4))
        # -> tensor[tile_coord, rest_coord] indexing
    """
    if not isinstance(divisor, ir.Value):
        divisor = make_tile(*divisor, loc=loc, ip=ip)
    return fly.zipped_divide(layout, divisor, loc=loc, ip=ip)


@traced_op
def tiled_divide(layout, divisor, loc=None, ip=None):
    """Like `zipped_divide` but keeps the rest mode flat (un-grouped).

    Result has rank `1 + rank(layout)`: first mode is the tile, remaining
    modes are the original rest axes.

    Examples:
        tiled_divide((16, 32), (4, 8))       -> ((4, 8), 4, 4)
        # -> tensor[tile, rest_row, rest_col]
    """
    if not isinstance(divisor, ir.Value):
        divisor = make_tile(*divisor, loc=loc, ip=ip)
    return fly.tiled_divide(layout, divisor, loc=loc, ip=ip)


@traced_op
def flat_divide(layout, divisor, loc=None, ip=None):
    """Fully flatten the result of a logical divide: tile modes first, then rest modes.

    Useful when every axis should be a top-level iterable (e.g. for a simple `for`
    over all tiles without nested unpacking).

    Examples:
        flat_divide((16, 32), (4, 8))        -> (4, 8, 4, 4)
        # tile_row, tile_col, rest_row, rest_col
    """
    if not isinstance(divisor, ir.Value):
        divisor = make_tile(*divisor, loc=loc, ip=ip)
    return fly.flat_divide(layout, divisor, loc=loc, ip=ip)


@traced_op
def logical_product(layout, tiler, loc=None, ip=None):
    """Repeat *layout* for every position in *tiler* (Cartesian product of layouts).

    Produces a per-mode nested `(original, repeats)` structure - inverse spirit
    of `logical_divide`.

    Examples:
        logical_product((4, 4), (2, 3))     -> ((4, 2), (4, 3))
    """
    return fly.logical_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def zipped_product(layout, tiler, loc=None, ip=None):
    """Repeat *layout* tiler-many times and zip into `(tile, rest)` tensor shape.

    Mirror of `zipped_divide`: builds a larger tensor out of many copies of a tile.

    Examples:
        zipped_product(tile_layout, (grid_m, grid_n))   # grid of tiles
    """
    return fly.zipped_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def tiled_product(layout, tiler, loc=None, ip=None):
    """Repeat *layout* while keeping the repetition modes flat.

    Result is `layout` followed by the tiler's modes as separate axes.

    Examples:
        tiled_product(tile, (grid_m, grid_n))    -> (tile, grid_m, grid_n)
    """
    return fly.tiled_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def flat_product(layout, tiler, loc=None, ip=None):
    """Fully flatten the product: original modes, then repetition modes.

    Examples:
        flat_product(tile_layout, (2, 3))   # 2*3 copies, flattened axes
    """
    return fly.flat_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def block_product(layout, tiler, loc=None, ip=None):
    """Repeat *layout* in block-layout order: each block sits contiguously.

    Analogous to a block-Kronecker product. Produces a bigger layout where
    copies of the tile are placed back-to-back.

    Examples:
        block_product(tile, (2, 3))     # 2x3 grid of tile blocks
    """
    return fly.block_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def raked_product(layout, tiler, loc=None, ip=None):
    """Repeat *layout* in interleaved ("raked") order, not block order.

    Rows of each repeated copy are interleaved - the counterpart of `block_product`.

    Examples:
        raked_product(tile, (4,))    # interleaved 4 copies across the fast axis
    """
    return fly.raked_product(layout, tiler, loc=loc, ip=ip)


@traced_op
def recast_layout(layout, old_type_bits, new_type_bits, loc=None, ip=None):
    """Re-scale a layout's strides when reinterpreting a buffer as a different dtype.

    Typical use: recast an f16 (16-bit) tensor view as f32 (32-bit) - the strides
    are divided by 2 and element counts are halved along the fastest axis.

    Examples:
        recast_layout(layout_f16, 16, 32)   # interpret f16 view as f32
        recast_layout(layout_f8,   8, 16)   # pack pairs of f8 into f16
    """

    def _to_static_bits(v):
        if isinstance(v, int):
            return v
        if isinstance(v, ir.Type):
            if hasattr(v, "width"):
                return int(v.width)
            raise TypeError(f"recast_layout only supports int/type-with-width, got type {v}")
        raise TypeError(f"recast_layout only supports int/Type, got {type(v)}")

    old_type_bits = _to_static_bits(old_type_bits)
    new_type_bits = _to_static_bits(new_type_bits)
    return fly.recast_layout(new_type_bits=new_type_bits, old_type_bits=old_type_bits, src=layout, loc=loc, ip=ip)


@traced_op
def tile_to_shape(block, trg_shape, ord_shape, loc=None, ip=None):
    """Repeat *block* just enough to fill *trg_shape*, laying out copies by *ord_shape*.

    Think: "tile a small pattern across a bigger canvas". *ord_shape* controls
    the priority (fastest-to-slowest) order among the repetition axes.

    Examples:
        tile_to_shape(block=(2, 2) layout,
                      trg_shape=(8, 8),
                      ord_shape=(0, 1))     # grid-fill the 8x8 with 2x2 blocks
    """
    return fly.tile_to_shape(block, trg_shape, ord_shape, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# Atom and Tiled Mma/Copy ops
# ===----------------------------------------------------------------------=== #


@traced_op
def make_mma_atom(mma_op_type, loc=None, ip=None):
    """Pick the per-thread MMA (matrix multiply-accumulate) capability.

    The *mma_op_type* names a hardware instruction like an MFMA/WMMA variant
    (e.g. `MFMA_F32_16x16x16_F16`). The returned atom describes one lane's
    contribution and is later tiled across a wave.

    Examples:
        atom = make_mma_atom(MFMA_F32_16x16x16_F16)
    """
    mma_atom_ty = MmaAtomType.get(mma_op=mma_op_type)
    return fly.make_mma_atom(mma_atom_ty, loc=loc, ip=ip)


@traced_op
def make_copy_atom(copy_op_type, elem_type, loc=None, ip=None):
    """Pick the per-thread copy capability for the given element dtype.

    *copy_op_type* names the instruction (e.g. `UniversalCopy32b`,
    `BufferCopy128b`); *elem_type* tells FlyDSL how wide each value is, so the
    atom knows how many elements one thread moves per issue.

    Examples:
        atom = make_copy_atom(UniversalCopy128b, Float16)   # 8 x f16 per lane
        atom = make_copy_atom(BufferCopy32b, Float32)       # 1 x f32 per lane
    """
    from .numeric import NumericMeta

    if isinstance(elem_type, NumericMeta):
        val_bits = elem_type.width
    elif isinstance(elem_type, ir.Type):
        if hasattr(elem_type, "width"):
            val_bits = int(elem_type.width)
        else:
            raise TypeError(f"make_copy_atom: elem_type must have a width, got {elem_type}")
    elif isinstance(elem_type, int):
        val_bits = elem_type
    else:
        raise TypeError(f"make_copy_atom: elem_type must be NumericType, ir.Type, or int, got {type(elem_type)}")
    copy_atom_ty = CopyAtomType.get(copy_op=copy_op_type, val_bits=val_bits)
    return fly.make_copy_atom(copy_atom_ty, val_bits=val_bits, loc=loc, ip=ip)


@traced_op
def atom_set_value(atom, field, value, loc=None, ip=None):
    """Attach metadata to an atom (e.g. the buffer resource for buffer-based copies).

    *field* is a string or enum selecting which attribute to set; *value* is
    the runtime operand to bind.

    Examples:
        copy_atom = atom_set_value(copy_atom, "BufferResource", buffer_res)
    """
    if isinstance(field, IntEnum):
        field = str(field)
    return fly.atom_set_value(atom, field, value, loc=loc, ip=ip)


@traced_op
def copy_atom_call(copy_atom, src, dst, *, pred=None, loc=None, ip=None):
    """Issue a single, per-thread copy described by *copy_atom* from *src* to *dst*.

    Each lane moves its share of elements. Pass *pred* to mask off threads
    (e.g. for out-of-bounds tails).

    Examples:
        copy_atom_call(atom, gmem_view, smem_view)
        copy_atom_call(atom, gmem_view, smem_view, pred=in_bounds_mask)
    """
    return fly.copy_atom_call(copy_atom, src, dst, pred=pred, loc=loc, ip=ip)


@traced_op
def mma_atom_call(mma_atom, d, a, b, c, loc=None, ip=None):
    """Run one MMA instruction: `d = a @ b + c` at the wave level.

    *a*, *b*, *c*, *d* are fragments (register tiles) whose layouts match the
    atom's expected shape. `c` is the input accumulator, `d` is the result.

    Examples:
        mma_atom_call(mma_atom, D_frag, A_frag, B_frag, C_frag)
    """
    return fly.mma_atom_call(mma_atom, d, a, b, c, loc=loc, ip=ip)


@traced_op
def make_tiled_copy(copy_atom, layout_thr_val, tile_mn, loc=None, ip=None):
    """Distribute *copy_atom* across threads to cover a tile of shape *tile_mn*.

    *layout_thr_val* is a (thread, value) layout telling which thread owns which
    elements. The result knows how to partition source / destination tiles for
    every thread.

    Examples:
        tiled = make_tiled_copy(atom, layout_tv=((32, 4), ((4, 1),(1, 128))), tile_mn=(128, 32))
    """
    if not isinstance(tile_mn, ir.Value):
        tile_mn = make_tile(*tile_mn, loc=loc, ip=ip)
    return fly.make_tiled_copy(copy_atom, layout_thr_val, tile_mn, loc=loc, ip=ip)


@traced_op
def make_tiled_mma(mma_atom, atom_layout, permutation=None, loc=None, ip=None):
    """Stack multiple *mma_atom* instances so one wave computes a bigger tile in one call.

    *atom_layout* says how many atoms to issue along M / N / K. Optional
    *permutation* reorders axes of the resulting tile shape.

    Examples:
        tmma = make_tiled_mma(mma_atom, atom_layout=(2, 2, 1))  # 2x2 atoms in MxN
    """
    if permutation is not None and not isinstance(permutation, ir.Value):
        permutation = make_tile(*permutation, loc=loc, ip=ip)
    return fly.make_tiled_mma(mma_atom, atom_layout, permutation=permutation, loc=loc, ip=ip)


@traced_op
def tiled_copy_partition_src(tiled_copy, src, thr_int_tuple, loc=None, ip=None):
    """For the given thread (*thr_int_tuple*), slice out its portion of the *src* tile.

    Returns the thread-local source tensor shaped "(values per issue, number of
    issues)" that `copy_atom_call` / `copy` can then consume.

    Examples:
        thr_src = tiled_copy_partition_src(tiled_copy, src_view, tid)
    """
    return fly.tiled_copy_partition_src(tiled_copy, src, thr_int_tuple, loc=loc, ip=ip)


@traced_op
def tiled_copy_partition_dst(tiled_copy, dst, thr_int_tuple, loc=None, ip=None):
    """Destination counterpart of `tiled_copy_partition_src` - same logic for writes.

    Examples:
        thr_dst = tiled_copy_partition_dst(tiled_copy, smem_view, tid)
    """
    return fly.tiled_copy_partition_dst(tiled_copy, dst, thr_int_tuple, loc=loc, ip=ip)


@traced_op
def tiled_copy_retile(tiled_copy, t, loc=None, ip=None):
    """Rearrange a fragment *t* to match the shape the next copy expects.

    Commonly used between LDS->register and register->LDS copies whose
    tile layouts don't line up directly.

    Examples:
        reg_src = tiled_copy_retile(tiled_copy, frag)
    """
    return fly.tiled_copy_retile(tiled_copy, t, loc=loc, ip=ip)


@traced_op
def tiled_mma_partition(operand_id, tiled_mma, t, coord, loc=None, ip=None):
    """Extract one operand's (A/B/C/D) slice of *t* for a specific MMA coordinate.

    *operand_id* selects which operand (e.g. 0=A, 1=B, 2=C/D). *coord* is the
    MMA iteration index within the tiled grid.

    Examples:
        a_slice = tiled_mma_partition(0, tiled_mma, gA, (tid,))
    """
    return fly.tiled_mma_partition(operand_id, tiled_mma, t, coord, loc=loc, ip=ip)


@traced_op
def tiled_mma_partition_shape(operand_id, tiled_mma, shape, loc=None, ip=None):
    """Return the per-thread shape of *operand_id* when the MMA acts on *shape*.

    Helper for sizing register fragments before allocating them.

    Examples:
        thr_shape_A = tiled_mma_partition_shape(0, tiled_mma, (BM, BK))
    """
    return fly.tiled_mma_partition_shape(operand_id, tiled_mma, shape, loc=loc, ip=ip)


@traced_op
def mma_make_fragment(operand_id, tiled_mma, input, loc=None, ip=None):
    """Create a register fragment of the right shape to hold *operand_id* for *tiled_mma*.

    Used to prepare A/B/C/D tensors with layouts that exactly match the atom's
    requirements.

    Examples:
        frag_C = mma_make_fragment(2, tiled_mma, gC_partition)   # 2 == C operand
    """
    return fly.mma_make_fragment(operand_id, tiled_mma, input, loc=loc, ip=ip)


@traced_op
def copy(copy_atom, src, dst, *, pred=None, loc=None, ip=None):
    """High-level tiled copy: issue all atom calls to move *src* -> *dst*.

    Unlike `copy_atom_call` (one atom), this iterates over all atoms in the
    tiled copy's grid. *pred* optionally masks off boundary threads/elements.

    Examples:
        copy(tiled_copy_atom, gA_partition, sA_partition)
        copy(tiled_copy_atom, gA_partition, sA_partition, pred=in_bounds)
    """
    return fly.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip)


@traced_op
def gemm(mma_atom, d, a, b, c, *, traversal_order=None, traversal_layout=None, loc=None, ip=None):
    """High-level tiled GEMM: `d = a @ b + c` using *mma_atom* across all its atoms.

    Pass either *traversal_order* (per-axis priority) or *traversal_layout* (a
    layout describing iteration order over MMA tiles) - but not both.

    Examples:
        gemm(mma_atom, D_frag, A_frag, B_frag, C_frag)
        gemm(mma_atom, D, A, B, C, traversal_order=(0, 1, 2))   # M, N, K order
    """
    if traversal_order is not None and traversal_layout is not None:
        raise ValueError("Only one of 'traversal_order' or 'traversal_layout' can be specified, not both")
    return fly.gemm(
        mma_atom, d, a, b, c, traversal_order=traversal_order, traversal_layout=traversal_layout, loc=loc, ip=ip
    )


# ===----------------------------------------------------------------------=== #
# MemRef and Ptr operations
# ===----------------------------------------------------------------------=== #


@traced_op
def make_ptr(result_type, args, loc=None, ip=None):
    """Construct a typed pointer from raw arguments (e.g. base + swizzle info).

    Mostly used internally; end-user code usually gets pointers from tensor views
    or kernel arguments instead.

    Examples:
        ptr = make_ptr(ptr_type, [base_addr, swizzle])
    """
    return fly.make_ptr(result_type, args, loc=loc, ip=ip)


@traced_op
def get_dyn_shared(loc=None, ip=None):
    """Return a pointer to the start of the kernel's dynamic shared-memory buffer.

    Pair with `recast_iter` / `make_view` to carve named tiles out of LDS.

    Examples:
        smem_base = get_dyn_shared()
        sA = make_view(recast_iter(f16_ptr, smem_base), sA_layout)
    """
    return fly.get_dyn_shared(loc=loc, ip=ip)


@traced_op
def inttoptr(result_type, src, loc=None, ip=None):
    """Interpret an integer address *src* as a pointer of *result_type*.

    Fails for register address space (registers can't be addressed numerically).

    Examples:
        ptr = inttoptr(PtrType.global_f16(), addr_i64)
    """
    if result_type.address_space == AddressSpace.Register:
        raise ValueError("inttoptr is not supported for register address space")
    return fly.inttoptr(result_type, src, loc=loc, ip=ip)


@traced_op
def ptrtoint(ptr, loc=None, ip=None):
    """Get the raw integer address underlying *ptr*.

    Fails for register address space (no numeric address).

    Examples:
        addr = ptrtoint(global_ptr)
    """
    if ptr.address_space == AddressSpace.Register:
        raise ValueError("ptrtoint is not supported for register address space")
    return fly.ptrtoint(ptr, loc=loc, ip=ip)


@traced_op
def add_offset(ptr, offset, loc=None, ip=None):
    """Shift *ptr* by *offset* elements (not bytes), preserving its element type.

    Examples:
        ptr2 = add_offset(ptr, 16)            # move forward 16 elements
        ptr2 = add_offset(ptr, tile_id * BM)  # runtime offset
    """
    if not isinstance(offset, ir.Value):
        offset = make_int_tuple(offset, loc=loc, ip=ip)
    return fly.add_offset(ptr, offset, loc=loc, ip=ip)


@traced_op
def apply_swizzle(ptr, swizzle, loc=None, ip=None):
    """Permute *ptr*'s low bits via *swizzle* to avoid LDS bank conflicts.

    The returned pointer has the swizzle "attached"; later loads/stores go
    through the permuted address.

    Examples:
        ptr_sw = apply_swizzle(sA_ptr, SwizzleType.get(3, 3, 3))
    """
    return fly.apply_swizzle(ptr, swizzle, loc=loc, ip=ip)


@traced_op
def ptr_load(ptr, result_type=None, loc=None, ip=None):
    """Load one value (scalar or vector) from *ptr*; dtype defaults to ptr's element type.

    Examples:
        v = ptr_load(ptr)                     # matches element type
        v4 = ptr_load(ptr, result_type=vec4f) # load 4 elems as a vector
    """
    if result_type is None:
        result_type = ptr.element_type
    return fly.ptr_load(result_type.ir_type, ptr, loc=loc, ip=ip)


@traced_op
def ptr_store(value, ptr, loc=None, ip=None):
    """Store *value* into *ptr*. Types must match the pointer's element type.

    Examples:
        ptr_store(val, ptr)
    """
    return fly.ptr_store(value, ptr, loc=loc, ip=ip)


@traced_op
def recast_iter(result_type, src, loc=None, ip=None):
    """Reinterpret a pointer / iterator as another element type (like `reinterpret_cast`).

    Layout metadata is kept; only the declared element type changes.

    Examples:
        smem_f16 = recast_iter(f16_ptr_type, get_dyn_shared())
    """
    return fly.recast_iter(result_type, src, loc=loc, ip=ip)


@traced_op
def memref_alloca(memref_type, layout, loc=None, ip=None):
    """Allocate a stack / register buffer with the given *layout*; returns a typed tensor.

    Typical use is for on-chip fragments / temporaries that don't need LDS.

    Examples:
        reg_buf = memref_alloca(reg_f32_type, make_layout((4, 4), (1, 4)))
    """
    return fly.memref_alloca(memref_type, layout, loc=loc, ip=ip)


@traced_op
def memref_load_vec(memref, loc=None, ip=None):
    """Load the *entire* memref as a single vector value in one go.

    Works when the memref is small and contiguous (e.g. a per-thread register
    fragment). Useful for passing as an MMA operand.

    Examples:
        vA = memref_load_vec(A_frag)          # A_frag -> flat vector
    """
    return fly.memref_load_vec(memref, loc=loc, ip=ip)


@traced_op
def memref_store_vec(vector, memref, loc=None, ip=None):
    """Inverse of `memref_load_vec`: splatter a vector back into the whole memref.

    Examples:
        memref_store_vec(vD, D_frag)
    """
    return fly.memref_store_vec(vector, memref, loc=loc, ip=ip)


@traced_op
def memref_load(memref, indices, loc=None, ip=None):
    """Load a single element from *memref* at the given coordinate.

    *indices* can be a plain int/tuple or a coordinate tensor. Scalar index is
    interpreted via the memref's layout.

    Examples:
        v = memref_load(reg_buf, (1, 2))
        v = memref_load(reg_buf, k)           # 1-D index
    """
    if isinstance(indices, ir.Value):
        if str(indices.type).startswith("!fly.int_tuple"):
            return fly.memref_load(memref, indices, loc=loc, ip=ip)
        if str(indices.type) == "index":
            indices = _arith.IndexCastOp(T.i32(), indices)
        indices = make_int_tuple(indices, loc=loc, ip=ip)
        return fly.memref_load(memref, indices, loc=loc, ip=ip)

    indices = make_int_tuple(indices, loc=loc, ip=ip)
    return fly.memref_load(memref, indices, loc=loc, ip=ip)


@traced_op
def memref_store(value, memref, indices, loc=None, ip=None):
    """Write *value* into *memref* at the given coordinate (dual of `memref_load`).

    Examples:
        memref_store(0.0, reg_buf, (0, 0))
        memref_store(v, reg_buf, k)
    """
    if isinstance(indices, ir.Value):
        if str(indices.type).startswith("!fly.int_tuple"):
            return fly.memref_store(value, memref, indices, loc=loc, ip=ip)
        if str(indices.type) == "index":
            indices = _arith.IndexCastOp(T.i32(), indices)
        indices = make_int_tuple(indices, loc=loc, ip=ip)
        return fly.memref_store(value, memref, indices, loc=loc, ip=ip)

    indices = make_int_tuple(indices, loc=loc, ip=ip)
    return fly.memref_store(value, memref, indices, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# Utility ops
# ===----------------------------------------------------------------------=== #


@traced_op
def printf(*args, format_str="", loc=None, ip=None):
    """Device-side printf for debugging kernels. Supports `{}` placeholders.

    Accepts ints, floats, bools, strings, types, and any FlyDSL value; each
    `{}` in *format_str* is substituted in order. Handy for dumping per-thread
    values during development.

    Examples:
        printf("tid={} val={}\\n", tid, v)
        printf("shape={} dtype={}\\n", (M, N), Float16)
    """

    def _convert_printf_value(val):
        if isinstance(val, ir.Value):
            return (False, val)
        elif isinstance(val, type):
            return (True, val.__name__)
        elif isinstance(val, str):
            return (True, val)
        elif isinstance(val, bool):
            return (False, _arith.constant(T.bool(), int(val)))
        elif isinstance(val, int):
            return (False, _arith.constant(T.i32(), val))
        elif isinstance(val, float):
            return (False, _arith.constant(T.f64(), val))
        elif hasattr(val, "__fly_values__"):
            ir_values = val.__fly_values__()
            if len(ir_values) == 1:
                return (False, ir_values[0])
            raise ValueError(f"Cannot use multi-value type in printf: {type(val)}")
        elif hasattr(val, "value") and isinstance(val.value, ir.Value):
            return (False, val.value)
        else:
            raise ValueError(f"Cannot convert {type(val)} to MLIR Value for printf")

    if len(args) > 0 and isinstance(args[0], str):
        format_str = args[0]
        raw_values = list(args[1:])
    else:
        raw_values = list(args)

    converted = [_convert_printf_value(v) for v in raw_values]

    final_format = format_str
    ir_values = []
    placeholder_idx = 0
    result_parts = []
    i = 0
    while i < len(final_format):
        if i + 1 < len(final_format) and final_format[i : i + 2] == "{}":
            if placeholder_idx < len(converted):
                is_static, val = converted[placeholder_idx]
                if is_static:
                    result_parts.append(str(val))
                else:
                    result_parts.append("{}")
                    ir_values.append(val)
                placeholder_idx += 1
            else:
                result_parts.append("{}")
            i += 2
        else:
            result_parts.append(final_format[i])
            i += 1

    final_format = "".join(result_parts)
    return fly.print_(final_format, ir_values, loc=loc, ip=ip)


@traced_op
def assume(result_type, dst, src, loc=None, ip=None):
    """Assert that *src* already has the properties implied by *result_type*.

    Acts like a checked cast / annotation - no runtime work, but lets the
    optimizer rely on extra information (e.g. "this pointer is aligned").

    Examples:
        aligned_ptr = assume(aligned_ptr_ty, aligned_ptr, raw_ptr)
    """
    return fly.assume(result_type, dst, src, loc=loc, ip=ip)


# ===----------------------------------------------------------------------=== #
# Deprecated
# ===----------------------------------------------------------------------=== #


@traced_op
def make_tile(*args, loc=None, ip=None):
    """Build a "tile" static value: a heterogeneous bag of ints / layouts / None.

    Used internally by divide / product / tiled-copy helpers. End users usually
    don't call this directly - pass plain tuples to those APIs instead.

    Examples:
        make_tile(4, 8)                        # plain divisor
        make_tile(layout_A, None)              # layout + wildcard modes
    """
    from .typing import Layout

    def _resolve(m):
        if isinstance(m, int) or m is None:
            return m
        if isinstance(m, tuple):
            return tuple(_resolve(e) for e in m)
        if isinstance(m, Layout):
            return m.type
        raise ValueError(f"make_tile: expected int, None, tuple, or Layout, got {type(m)}")

    resolved = [_resolve(m) for m in args]
    if len(resolved) == 1:
        tile_type = TileType.get(resolved[0])
    else:
        tile_type = TileType.get(resolved)
    return static(tile_type, loc=loc, ip=ip)
