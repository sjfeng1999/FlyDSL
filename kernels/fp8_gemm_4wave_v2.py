# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""4-wave FP8 matmul with row-wise scaling for AMD CDNA4 (layout-API version).

Algorithm derived from HipKittens FP8_4wave
(https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L1).

Identical schedule to the manual-offset version in ``fp8_gemm_4wave.py``
(8-buffer LDS ping-pong with 2x2 wave layout and interleaved MFMA/G→LDS
clusters), but allocation flows through the FlyDSL ``@fx.struct`` /
``SharedAllocator`` syntax so all 8 LDS sub-buffers live in a single
``@__dynamic_shared__0`` global:

* LDS storage is declared as a ``@fx.struct SharedStorage`` of 8
  ``fx.Array[Float8E4M3FN, ...]`` fields and bound via
  ``fx.SharedAllocator(...).allocate(SharedStorage).peek()``.
* The XOR-128 swizzle is encoded once as
  ``fx.CoordSwizzleType.get(3, 1, [0], 4, [1])`` (``(row >> 1) & 7`` XOR-ed
  into col bits 4..6) and shared by the global view (``_gl_swz_A_layout``)
  and the LDS view (``sA_layout`` / ``sB_layout``) via
  ``make_composed_layout``.
* Global G→LDS loads use ``BufferCopyLDS128b``. Per-lane src offset comes
  from the swizzled global view; LDS dst is a wave-uniform base pointer
  computed against the *non-swizzled* row-major LDS layout
  (``sA_dst_layout`` / ``sB_dst_layout``) since ``buffer_load_lds`` writes
  ``dst + lane * 16`` in raw lane order.
* The LDS destination pointer is built with an explicit
  ``ptrtoint`` + ``addi`` + ``inttoptr`` sequence (the "B form" — see
  ANALYSIS.md §5.2 / §8.2). This is necessary because all 8 sub-buffers
  share the single ``@__dynamic_shared__0`` global: a plain
  ``getelementptr`` ("D form") confuses AMDGPU's ``RewriteAGPRCopyMFMA``
  pass and forces ~25% of the LDS reads through a VGPR detour, costing
  ~3% E2E perf. The same trick is applied to the LDS→reg read pointer.

MFMA accumulators flow through ``fly.gemm`` on a tiled MMA atom so the
chained Vec(4, f32) accumulator stays on AGPRs.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec


def _divmod(a, b):
    return (a // b, a % b)


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle(num_pid_m, num_pid_n):
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x

    num_wg = num_pid_m * num_pid_n

    if num_wg <= SWIZZLE_THRESHOLD or num_wg % NUM_XCDS != 0:
        return _divmod(wgid, num_pid_n)

    intra_xcd, xcd = _divmod(wgid, NUM_XCDS)
    wgid = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = _divmod(wgid, num_wgid_in_group)
    first_pid_m = group_id * WGM
    group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = _divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m
    return (pid_m, pid_n)


def compile_fp8_gemm(*, M: int, N: int, K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, use_xcd_remap: bool = True):
    # MFMA atom is 16x16x128; 4 waves in a 2x2 config require BLOCK >= 64.
    BLOCK_K = 128
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    assert BLOCK_M >= 64 and BLOCK_N >= 64
    assert N % BLOCK_N == 0 and M % BLOCK_M == 0 and K % BLOCK_K == 0

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K
    # Number of 16-row 16x128 tiles per wave per A/B partition.
    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    _use_interleaved_block = BLOCK_M == 256 and BLOCK_N == 256

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        RT_C_i = Vec.filled(4, 0.0, fx.Float32)

        lds = fx.SharedAllocator(base_alignment=16).allocate(SharedStorage).peek()

        _SWZ = fx.CoordSwizzleType.get(3, 1, [0], 4, [1])
        _coord_swz_A = fx.make_composed_layout(fx.static(_SWZ), fx.make_identity_layout((LDS_BLOCK_M, BLOCK_K)))
        _coord_swz_B = fx.make_composed_layout(fx.static(_SWZ), fx.make_identity_layout((LDS_BLOCK_N, BLOCK_K)))

        sA_layout = fx.make_composed_layout(
            fx.make_layout((LDS_BLOCK_M, BLOCK_K), (BLOCK_K, 1)),
            _coord_swz_A,
        )
        sB_layout = fx.make_composed_layout(
            fx.make_layout((LDS_BLOCK_N, BLOCK_K), (BLOCK_K, 1)),
            _coord_swz_B,
        )
        sA_dst_layout = fx.make_layout((LDS_BLOCK_M, BLOCK_K), (BLOCK_K, 1))
        sB_dst_layout = fx.make_layout((LDS_BLOCK_N, BLOCK_K), (BLOCK_K, 1))

        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(M // BLOCK_M, N // BLOCK_N)
        else:
            tile_i, tile_j = _divmod(fx.block_idx.x, N_BLOCKS)

        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A0_gl_offset = (tile_i * BLOCK_M) * K
        A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K
        B0_gl_offset = (tile_j * BLOCK_N) * K
        B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K

        def _make_fp8_buf_tensor(arg_i8):
            t_i8 = fx.rocdl.make_buffer_tensor(arg_i8)
            iter_i8 = fx.get_iter(t_i8)
            iter_f8 = fx.recast_iter(fx.Float8E4M3FN, iter_i8)
            return fx.make_view(iter_f8, fx.get_layout(t_i8))

        gA = _make_fp8_buf_tensor(A)
        gB = _make_fp8_buf_tensor(B_T)
        gC = fx.rocdl.make_buffer_tensor(C)
        gSA = fx.rocdl.make_buffer_tensor(A_scale)
        gSB = fx.rocdl.make_buffer_tensor(B_scale)

        ga_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        gb_div = fx.logical_divide(gB, fx.make_layout(1, 1))
        c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        sa_div = fx.logical_divide(gSA, fx.make_layout(1, 1))
        sb_div = fx.logical_divide(gSB, fx.make_layout(1, 1))

        _gl_swz_A_layout = fx.make_composed_layout(
            fx.make_layout((LDS_BLOCK_M, BLOCK_K), (K, 1)),
            _coord_swz_A,
        )

        g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)

        def _compute_global_offsets():
            offsets = []
            for step in range_constexpr(max(N_TILES_A, N_TILES_B)):
                row = lane_id // 8 + wave_id * 8 + step * 32
                col = (lane_id % 8) * 16
                offsets.append(fx.crd2idx((row, col), _gl_swz_A_layout).to_py_value())
            return offsets

        def _lds_offset_i32(lds_layout, row, col):
            # `crd2idx` returns an `IntTuple<?>` (single leaf); `to_py_value`
            # unwraps it to a runtime Int32 wrapper that we can feed into
            # raw arith.* ops below.
            return fx.crd2idx((fx.Int32(row), fx.Int32(col)), lds_layout).to_py_value()

        def _lds_addr_inttoptr(lds_array, byte_offset_i32):
            # Build an LDS pointer via `ptrtoint + addi + inttoptr` instead
            # of a single `getelementptr`. See ANALYSIS.md §5.2 / §8.2:
            # because all 8 sub-buffers share the single
            # `@__dynamic_shared__0` global, the GEP form causes AMDGPU's
            # `RewriteAGPRCopyMFMA` pass to fall back to a VGPR detour for
            # ~25% of LDS loads (~3% E2E perf hit). The int-ptr form keeps
            # the same underlying value but hides the chain from the GEP-
            # based AGPR-folding heuristic.
            #
            # `fly.ptrtoint` infers an `i32` result for shared-address-space
            # pointers (see `PtrToIntOp::inferReturnType`); the byte offset
            # is already i32, so the whole arithmetic stays in i32.
            base_i32 = fx.Int32(fx.ptrtoint(lds_array.ptr))
            sum_i32 = base_i32 + byte_offset_i32
            return fx.inttoptr(lds_array.ptr.type, sum_i32)

        def _lds_dst_at(lds_array, lds_layout, lds_step_row):
            off_i32 = _lds_offset_i32(lds_layout, lds_step_row, 0)
            ptr = _lds_addr_inttoptr(lds_array, off_i32)
            return fx.make_view(ptr, fx.make_layout(1, 1))

        def _load_lds(gl_src_div, lds_array, lds_layout, k_offset, gl_offsets, n_tiles):
            assert len(gl_offsets) >= n_tiles
            for step in range_constexpr(n_tiles):
                src = fx.slice(gl_src_div, (None, fx.Int32(gl_offsets[step])))
                dst = _lds_dst_at(lds_array, lds_layout, wave_id * 8 + step * 32)
                fx.copy(g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

        def _load_one_lds(gl_src_div, lds_array, lds_layout, k_offset, gl_offsets, tile_idx):
            assert len(gl_offsets) > tile_idx
            src = fx.slice(gl_src_div, (None, fx.Int32(gl_offsets[tile_idx])))
            dst = _lds_dst_at(lds_array, lds_layout, wave_id * 8 + tile_idx * 32)
            fx.copy(g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

        def _pack_i32x4_i32x8(lo, hi):
            return lo.shuffle(hi, list(range(8)))

        def _vec_load_lds_i32x4(lds_array, lds_layout, row, col):
            off_i32 = _lds_offset_i32(lds_layout, row, col)
            f8_ptr = _lds_addr_inttoptr(lds_array, off_i32)
            i32_iter = fx.recast_iter(fx.Int32, f8_ptr)
            view = fx.make_view(i32_iter, fx.make_layout(4, 1))
            return Vec(fx.memref_load_vec(view))

        def _load_rt(lds_array, lds_layout, wave_idx, n_tiles):
            frag = []
            for i in range_constexpr(n_tiles):
                row = wave_idx * (n_tiles * 16) + i * 16 + lane_id % 16
                halves = []
                for step in range_constexpr(2):
                    col = (lane_id // 16) * 16 + step * 64
                    halves.append(_vec_load_lds_i32x4(lds_array, lds_layout, row, col))
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
            return frag

        def _load_one_rt(lds_array, lds_layout, lds_rows, lds_cols, row_idx, k_idx):
            return _vec_load_lds_i32x4(lds_array, lds_layout, lds_rows[row_idx], lds_cols[row_idx][k_idx])

        def _c_idx(i, j):
            return i * N_TILES_B + j

        _AS_REG = 3
        scale_atom_4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        scale_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        _reg_f32_4_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(4, 1), _AS_REG)
        _reg_f32_1_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), _AS_REG)
        _reg_bf16_1_ty = fx.MemRefType.get(fx.T.bf16(), fx.LayoutType.get(1, 1), _AS_REG)

        def _store_C_scaled(c_frag, base_row, base_col):
            def _load_scale_vec4(row):
                r = fx.memref_alloca(_reg_f32_4_ty, fx.make_layout(4, 1))
                fx.copy(scale_atom_4, fx.slice(sa_div, (None, fx.Int32(row))), r)
                return Vec(fx.memref_load_vec(r))

            def _load_scale_scalar(col):
                r = fx.memref_alloca(_reg_f32_1_ty, fx.make_layout(1, 1))
                fx.copy(scale_atom_1, fx.slice(sb_div, (None, fx.Int32(col))), r)
                return Vec(fx.memref_load_vec(r))[0]

            def _store_bf16(value_bf16, c_index):
                r = fx.memref_alloca(_reg_bf16_1_ty, fx.make_layout(1, 1))
                fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), r)
                fx.copy(out_atom_1, r, fx.slice(c_div, (None, fx.Int32(c_index))))

            a_scales = [_load_scale_vec4(base_row + i * 16 + (lane_id // 16) * 4) for i in range_constexpr(N_TILES_A)]
            b_scales = [_load_scale_scalar(base_col + i * 16 + lane_id % 16) for i in range_constexpr(N_TILES_B)]
            for ti in range_constexpr(N_TILES_A):
                row = base_row + ti * 16 + (lane_id // 16) * 4
                for tj in range_constexpr(N_TILES_B):
                    col = base_col + tj * 16 + lane_id % 16
                    vec_f32 = Vec(c_frag[_c_idx(ti, tj)])
                    for i in range_constexpr(4):
                        scaled = (vec_f32[i] * (a_scales[ti][i] * b_scales[tj])).to(fx.BFloat16)
                        _store_bf16(scaled, (row + i) * N + col)

        def _wait_barrier(count):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

        mma_atom = fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN))
        tiled_mma_single = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((2, 2, 1), (1, 2, 0)),
        )
        _reg_i32_8_ty = fx.MemRefType.get(fx.T.i32(), fx.LayoutType.get(8, 1), _AS_REG)

        def _mfma(a_vec, b_vec, c_vec):
            a_mem = fx.memref_alloca(_reg_i32_8_ty, fx.make_layout(8, 1))
            b_mem = fx.memref_alloca(_reg_i32_8_ty, fx.make_layout(8, 1))
            c_mem = fx.memref_alloca(_reg_f32_4_ty, fx.make_layout(4, 1))
            fx.memref_store_vec(a_vec, a_mem)
            fx.memref_store_vec(b_vec, b_mem)
            fx.memref_store_vec(c_vec, c_mem)
            fx.gemm(tiled_mma_single, c_mem, a_mem, b_mem, c_mem)
            return Vec(fx.memref_load_vec(c_mem))

        def _mfma_ABt_all(a, b, c):
            assert len(a) == N_TILES_A
            assert len(b) == N_TILES_B
            assert len(c) == N_TILES_A * N_TILES_B

            for i in range_constexpr(N_TILES_A):
                for j in range_constexpr(N_TILES_B):
                    c[_c_idx(i, j)] = _mfma(a[i], b[j], c[_c_idx(i, j)])
            return c

        def _mfma_ABt_one(a, b, c, m, n):
            assert m < N_TILES_A and n < N_TILES_B

            c[_c_idx(m, n)] = _mfma(a[m], b[n], c[_c_idx(m, n)])
            return c

        def _precompute_lds_rt_coords(wave_idx, n_tiles):
            rows = []
            cols_per_row = []
            for row_offset in range_constexpr(n_tiles):
                rows.append(wave_idx * (n_tiles * 16) + row_offset * 16 + lane_id % 16)
                row_cols = []
                for i in range_constexpr(2):
                    row_cols.append((lane_id // 16) * 16 + i * 64)
                cols_per_row.append(row_cols)
            return rows, cols_per_row

        def _interleaved_cluster(
            lds_dst,
            lds_dst_layout,
            gl_src,
            k_offset,
            gl_offsets,
            wave_idx,
            lds_src,
            lds_src_layout,
            n_tiles_lds,
            a,
            b,
            c,
        ):
            rt_dst = []

            c = _mfma_ABt_one(a, b, c, 0, 0)
            c = _mfma_ABt_one(a, b, c, 0, 1)

            lds_rows, lds_cols = _precompute_lds_rt_coords(wave_idx, n_tiles_lds)
            _load_one_lds(gl_src, lds_dst, lds_dst_layout, k_offset, gl_offsets, 0)
            rt_dst_0 = _load_one_rt(lds_src, lds_src_layout, lds_rows, lds_cols, 0, 0)

            c = _mfma_ABt_one(a, b, c, 0, 2)

            rt_dst_1 = _load_one_rt(lds_src, lds_src_layout, lds_rows, lds_cols, 0, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 0, 3)

            _load_one_lds(gl_src, lds_dst, lds_dst_layout, k_offset, gl_offsets, 1)
            rt_dst_0 = _load_one_rt(lds_src, lds_src_layout, lds_rows, lds_cols, 1, 0)

            c = _mfma_ABt_one(a, b, c, 1, 0)
            c = _mfma_ABt_one(a, b, c, 1, 1)

            rt_dst_1 = _load_one_rt(lds_src, lds_src_layout, lds_rows, lds_cols, 1, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 1, 2)
            c = _mfma_ABt_one(a, b, c, 1, 3)

            _load_one_lds(gl_src, lds_dst, lds_dst_layout, k_offset, gl_offsets, 2)
            rt_dst_0 = _load_one_rt(lds_src, lds_src_layout, lds_rows, lds_cols, 2, 0)

            c = _mfma_ABt_one(a, b, c, 2, 0)
            c = _mfma_ABt_one(a, b, c, 2, 1)

            rt_dst_1 = _load_one_rt(lds_src, lds_src_layout, lds_rows, lds_cols, 2, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 2, 2)
            c = _mfma_ABt_one(a, b, c, 2, 3)

            _load_one_lds(gl_src, lds_dst, lds_dst_layout, k_offset, gl_offsets, 3)
            rt_dst_0 = _load_one_rt(lds_src, lds_src_layout, lds_rows, lds_cols, 3, 0)

            c = _mfma_ABt_one(a, b, c, 3, 0)
            c = _mfma_ABt_one(a, b, c, 3, 1)

            rt_dst_1 = _load_one_rt(lds_src, lds_src_layout, lds_rows, lds_cols, 3, 1)
            rt_dst.append(_pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c = _mfma_ABt_one(a, b, c, 3, 2)
            c = _mfma_ABt_one(a, b, c, 3, 3)

            return c, rt_dst

        def _compute_cluster(
            lds_dst,
            lds_dst_layout,
            gl_src,
            k_offset,
            gl_offsets,
            wave_idx,
            lds_src,
            lds_src_layout,
            n_tiles_lds,
            n_tiles_rt,
            a,
            b,
            c,
        ):
            _load_lds(gl_src, lds_dst, lds_dst_layout, k_offset, gl_offsets, n_tiles_lds)
            rt_dst = _load_rt(lds_src, lds_src_layout, wave_idx, n_tiles_rt)
            c = _mfma_ABt_all(a, b, c)
            return c, rt_dst

        def _compute_block(
            lds_dst,
            lds_dst_layout,
            gl_src,
            k_offset,
            gl_offsets,
            wave_idx,
            lds_src,
            lds_src_layout,
            n_tiles_lds,
            n_tiles_rt,
            a,
            b,
            c,
        ):
            if const_expr(_use_interleaved_block):
                return _interleaved_cluster(
                    lds_dst,
                    lds_dst_layout,
                    gl_src,
                    k_offset,
                    gl_offsets,
                    wave_idx,
                    lds_src,
                    lds_src_layout,
                    n_tiles_lds,
                    a,
                    b,
                    c,
                )
            else:
                return _compute_cluster(
                    lds_dst,
                    lds_dst_layout,
                    gl_src,
                    k_offset,
                    gl_offsets,
                    wave_idx,
                    lds_src,
                    lds_src_layout,
                    n_tiles_lds,
                    n_tiles_rt,
                    a,
                    b,
                    c,
                )

        c00_frag = [RT_C_i] * N_ACCUMS
        c01_frag = [RT_C_i] * N_ACCUMS
        c10_frag = [RT_C_i] * N_ACCUMS
        c11_frag = [RT_C_i] * N_ACCUMS

        global_offsets = _compute_global_offsets()

        # Prologue: 8-buffer LDS pipeline pre-fill.
        _load_lds(ga_div, a_cur0, sA_dst_layout, A0_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_A)
        _load_lds(gb_div, b_cur0, sB_dst_layout, B0_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(gb_div, b_cur1, sB_dst_layout, B1_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(ga_div, a_cur1, sA_dst_layout, A1_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_A)

        _load_lds(ga_div, a_next0, sA_dst_layout, A0_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_A)
        _load_lds(gb_div, b_next0, sB_dst_layout, B0_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(gb_div, b_next1, sB_dst_layout, B1_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(ga_div, a_next1, sA_dst_layout, A1_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_A)

        _wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))

        a0_frag = _load_rt(a_cur0, sA_layout, wave_i, N_TILES_A)

        _wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))

        b0_frag = _load_rt(b_cur0, sB_layout, wave_j, N_TILES_B)

        for k in range_constexpr(K_ITERS - 2):
            _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c00_frag, b1_frag = _compute_block(
                a_cur0,
                sA_dst_layout,
                ga_div,
                A0_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
                wave_j,
                b_cur1,
                sB_layout,
                N_TILES_A,
                N_TILES_B,
                a0_frag,
                b0_frag,
                c00_frag,
            )

            c01_frag, a1_frag = _compute_block(
                b_cur0,
                sB_dst_layout,
                gb_div,
                B0_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
                wave_i,
                a_cur1,
                sA_layout,
                N_TILES_B,
                N_TILES_A,
                a0_frag,
                b1_frag,
                c01_frag,
            )

            _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c10_frag, a0_frag = _compute_block(
                b_cur1,
                sB_dst_layout,
                gb_div,
                B1_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
                wave_i,
                a_next0,
                sA_layout,
                N_TILES_B,
                N_TILES_A,
                a1_frag,
                b0_frag,
                c10_frag,
            )

            c11_frag, b0_frag = _compute_block(
                a_cur1,
                sA_dst_layout,
                ga_div,
                A1_gl_offset + (k + 2) * BLOCK_K,
                global_offsets,
                wave_j,
                b_next0,
                sB_layout,
                N_TILES_A,
                N_TILES_B,
                a1_frag,
                b1_frag,
                c11_frag,
            )

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 2.
        _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        b1_frag = _load_rt(b_cur1, sB_layout, wave_j, N_TILES_B)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        a1_frag = _load_rt(a_cur1, sA_layout, wave_i, N_TILES_A)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        _wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))
        a0_frag = _load_rt(a_next0, sA_layout, wave_i, N_TILES_A)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        b0_frag = _load_rt(b_next0, sB_layout, wave_j, N_TILES_B)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Tail step k_iters - 1.
        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)
        _wait_barrier(0)
        b1_frag = _load_rt(b_cur1, sB_layout, wave_j, N_TILES_B)
        a1_frag = _load_rt(a_cur1, sA_layout, wave_i, N_TILES_A)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)

        _store_C_scaled(c00_frag, base_row + 0, base_col + 0)
        _store_C_scaled(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        _store_C_scaled(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        _store_C_scaled(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        stream: fx.Stream,
    ):
        grid_x = (M * N) // (BLOCK_M * BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_gemm
