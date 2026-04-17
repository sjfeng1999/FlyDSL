// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// gfx1250 TDM (Tensor Data Mover) copy atom lowering tests.
//
// TDM atoms live in !fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<..>, valBits>
// and emit `rocdl.tensor.load.to.lds` / `rocdl.tensor.store.from.lds` after
// --convert-fly-to-rocdl. Static descriptor metadata (tile shape, padding,
// wave count) rides on the CopyOp type, dynamic per-issue values (tensor
// dims, gmem/lds offsets, mask, cache policy, gather indices) ride on the
// atom state struct. Completion is waited on with the upstream
// `rocdl.s.wait.tensorcnt` op directly (no separate fly_rocdl wrapper).

// -----

// CHECK-LABEL: @test_tdm_load_2d_lowers_to_rocdl
// CHECK: rocdl.tensor.load.to.lds
// CHECK-NOT: rocdl.tensor.store.from.lds
// CHECK: rocdl.s.wait.tensorcnt 0
func.func @test_tdm_load_2d_lowers_to_rocdl(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 8, abe = false>, 16>,
    %src: !fly.memref<f16, buffer_desc, 1:1>,
    %dst: !fly.memref<f16, shared, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 8, abe = false>, 16>, !fly.memref<f16, buffer_desc, 1:1>, !fly.memref<f16, shared, 1:1>) -> ()
  rocdl.s.wait.tensorcnt 0
  return
}

// -----

// CHECK-LABEL: @test_tdm_store_2d_lowers_to_rocdl
// CHECK: rocdl.tensor.store.from.lds
// CHECK: rocdl.s.wait.tensorcnt 0
func.func @test_tdm_store_2d_lowers_to_rocdl(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_store_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 4, abe = false>, 16>,
    %src: !fly.memref<f16, shared, 1:1>,
    %dst: !fly.memref<f16, buffer_desc, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_store_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 4, abe = false>, 16>, !fly.memref<f16, shared, 1:1>, !fly.memref<f16, buffer_desc, 1:1>) -> ()
  rocdl.s.wait.tensorcnt 0
  return
}

// -----

// The state struct must have 13 i32 fields in the exact order of
// AtomStateField (Soffset, ImmOffset, ScaleA, ScaleB, WorkgroupMask,
// CachePolicy, GmemByteOffset, LdsByteOffset, TensorDim0, TensorDim1,
// Stride0, GatherRowIndices, GatherIndexCount).

// CHECK-LABEL: @test_tdm_load_2d_state_struct_shape
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>)
func.func @test_tdm_load_2d_state_struct_shape(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>) {
  return
}

// -----

// TDMGather's state struct replaces the GatherRowIndices field with a
// vector<8xi32> (room for up to 16 x i16 or 8 x i32 indices).

// CHECK-LABEL: @test_tdm_gather_state_struct_shape
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, vector<8xi32>, i32)>)
func.func @test_tdm_gather_state_struct_shape(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<elem_bits = 16, row_width = 128, index_size = 32, max_indices = 8, pad = (0, 0), is_store = false>, 16>) {
  return
}

// -----

// Making a default TDMLoad2D atom should initialize all 13 state fields
// to zero (undef-then-insert pattern).

// CHECK-LABEL: @test_tdm_load_2d_default_state
// CHECK-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
// CHECK-DAG: %[[Z:.*]] = arith.constant 0 : i32
// CHECK: llvm.insertvalue %[[Z]], %[[UNDEF]][0]
func.func @test_tdm_load_2d_default_state(
    %src: !fly.memref<f16, buffer_desc, 1:1>,
    %dst: !fly.memref<f16, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>, !fly.memref<f16, buffer_desc, 1:1>, !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// -----

// set_value on AtomStateField names must update the corresponding
// state struct slot by index, matching the enum ordering.

// CHECK-LABEL: @test_tdm_load_2d_set_value_fields
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<{{.*}}>, %[[TD0:.*]]: i32, %[[TD1:.*]]: i32, %[[S0:.*]]: i32
// tensor_dim0 is field 8, tensor_dim1 is field 9, stride0 is field 10.
// CHECK: %[[A1:.*]] = llvm.insertvalue %[[TD0]], %[[ATOM]][8]
// CHECK: %[[A2:.*]] = llvm.insertvalue %[[TD1]], %[[A1]][9]
// CHECK: %[[A3:.*]] = llvm.insertvalue %[[S0]], %[[A2]][10]
func.func @test_tdm_load_2d_set_value_fields(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>,
    %td0: i32, %td1: i32, %s0: i32,
    %src: !fly.memref<f16, buffer_desc, 1:1>,
    %dst: !fly.memref<f16, shared, 1:1>) {
  %a1 = fly.atom.set_value(%atom, "tensor_dim0", %td0) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>, i32) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>
  %a2 = fly.atom.set_value(%a1, "tensor_dim1", %td1) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>, i32) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>
  %a3 = fly.atom.set_value(%a2, "stride0", %s0) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>, i32) -> !fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>
  fly.copy_atom_call(%a3, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 1, abe = false>, 16>, !fly.memref<f16, buffer_desc, 1:1>, !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// -----

// With num_warps=1 there should be no rocdl.wave.id read, nor any per-
// wave remainder/divide logic — waveOffInner/Outer are folded to 0.

// CHECK-LABEL: @test_tdm_load_2d_single_warp_no_wave_id
// CHECK-NOT: rocdl.wave.id
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.divui
// CHECK: rocdl.tensor.load.to.lds
func.func @test_tdm_load_2d_single_warp_no_wave_id(
    %src: !fly.memref<f16, buffer_desc, 1:1>,
    %dst: !fly.memref<f16, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (32, 32), pad = (0, 0), num_warps = 1, abe = false>, 16>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (32, 32), pad = (0, 0), num_warps = 1, abe = false>, 16>, !fly.memref<f16, buffer_desc, 1:1>, !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// -----

// With num_warps>1 wave distribution code materializes rocdl.wave.id plus
// modulo/divide by warps_outer (= warps_per_dim[0]).

// CHECK-LABEL: @test_tdm_load_2d_multi_warp_uses_wave_id
// CHECK: %[[WAVE:.*]] = rocdl.wave.id
// CHECK-DAG: arith.remui %[[WAVE]]
// CHECK-DAG: arith.divui %[[WAVE]]
// CHECK: rocdl.tensor.load.to.lds
func.func @test_tdm_load_2d_multi_warp_uses_wave_id(
    %src: !fly.memref<f16, buffer_desc, 1:1>,
    %dst: !fly.memref<f16, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 4, abe = false>, 16>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_load_2d<elem_bits = 16, tile = (64, 64), pad = (0, 0), num_warps = 4, abe = false>, 16>, !fly.memref<f16, buffer_desc, 1:1>, !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// -----

// TDM wait is just a direct use of the upstream ROCDL op; the count
// attribute is preserved verbatim through --convert-fly-to-rocdl.

// CHECK-LABEL: @test_tdm_wait_preserves_count
// CHECK: rocdl.s.wait.tensorcnt 2
func.func @test_tdm_wait_preserves_count() {
  rocdl.s.wait.tensorcnt 2
  return
}

// -----

// Gather atom lowers to rocdl.tensor.load.to.lds with 5 descriptor groups;
// isStore=true flavor goes to rocdl.tensor.store.from.lds.

// CHECK-LABEL: @test_tdm_gather_load_lowers_to_rocdl
// CHECK: rocdl.tensor.load.to.lds
func.func @test_tdm_gather_load_lowers_to_rocdl(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<elem_bits = 16, row_width = 128, index_size = 32, max_indices = 8, pad = (0, 0), is_store = false>, 16>,
    %src: !fly.memref<f16, buffer_desc, 1:1>,
    %dst: !fly.memref<f16, shared, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<elem_bits = 16, row_width = 128, index_size = 32, max_indices = 8, pad = (0, 0), is_store = false>, 16>, !fly.memref<f16, buffer_desc, 1:1>, !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_tdm_gather_store_lowers_to_rocdl
// CHECK: rocdl.tensor.store.from.lds
func.func @test_tdm_gather_store_lowers_to_rocdl(
    %atom: !fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<elem_bits = 16, row_width = 128, index_size = 32, max_indices = 8, pad = (0, 0), is_store = true>, 16>,
    %src: !fly.memref<f16, shared, 1:1>,
    %dst: !fly.memref<f16, buffer_desc, 1:1>) {
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_rocdl.gfx1250.tdm_gather<elem_bits = 16, row_width = 128, index_size = 32, max_indices = 8, pad = (0, 0), is_store = true>, 16>, !fly.memref<f16, shared, 1:1>, !fly.memref<f16, buffer_desc, 1:1>) -> ()
  return
}
