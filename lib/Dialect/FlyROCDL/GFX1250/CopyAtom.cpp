// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Lowering for gfx1250 TDM (Tensor Data Mover) copy atoms.
//
// These atoms mirror (and subsume) the descriptor bitfield packing that
// used to live in python/flydsl/expr/rocdl/tdm_ops.py. They turn
// ``fly.copy_atom_call(TDMLoad2D/Store2D/Gather atom, src, dst)`` into
// ``rocdl.tensor.load.to.lds`` / ``rocdl.tensor.store.from.lds`` with
// GROUP0..GROUP4 assembled here.
//
// Static descriptor metadata (elem bits, tile shape, padding interval,
// per-workgroup wave count) travels on the CopyOp type parameters.
// Dynamic per-issue values (workgroup multicast mask, cache policy,
// K-offset, LDS offset, full tensor dims, row indices for gather) travel
// through the stateful atom value via ``atom.set_value("<field>", ...)``.
//
// The pair ``{Soffset, ImmOffset}`` of AtomStateField is reused as a
// generic 32-bit SGPR scratch that feeds ``gmem_byte_offset`` /
// ``lds_byte_offset`` etc.; see the field table in emitAtomCall below.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Utils/BufferFatPtr.h"

#include <cmath>

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

namespace {

constexpr int64_t kNumTDMStateFields = 13;

int fieldIndex(AtomStateField f) { return static_cast<int>(f); }

LLVM::LLVMStructType getTDMStateStructTy(MLIRContext *ctx, bool withIndices) {
  SmallVector<Type> fieldTypes(kNumTDMStateFields, IntegerType::get(ctx, 32));
  if (withIndices) {
    // Replace GatherRowIndices field with a vector<8xi32> (room for up to
    // 16 x i16 or 8 x i32). We allocate the max packed size here; the
    // emit code looks at `indexSize` / `maxIndices` to know how many
    // lanes are active.
    fieldTypes[fieldIndex(AtomStateField::GatherRowIndices)] =
        VectorType::get({8}, IntegerType::get(ctx, 32));
  }
  return LLVM::LLVMStructType::getLiteral(ctx, fieldTypes);
}

Value insertField(OpBuilder &b, Location loc, Value structVal, AtomStateField field,
                  Value fieldValue) {
  return LLVM::InsertValueOp::create(b, loc, structVal, fieldValue,
                                     ArrayRef<int64_t>{fieldIndex(field)});
}

Value extractField(OpBuilder &b, Location loc, Value structVal, AtomStateField field) {
  return LLVM::ExtractValueOp::create(b, loc, structVal, ArrayRef<int64_t>{fieldIndex(field)});
}

Value i32Const(OpBuilder &b, Location loc, int64_t v) {
  return arith::ConstantIntOp::create(b, loc, b.getI32Type(), v);
}

Value defaultZeroStruct(OpBuilder &b, Location loc, LLVM::LLVMStructType sty, bool withIndices) {
  Value undef = LLVM::UndefOp::create(b, loc, sty);
  Value zero = i32Const(b, loc, 0);
  for (int i = 0; i < kNumTDMStateFields; ++i) {
    auto fieldTy = sty.getBody()[i];
    Value v;
    if (isa<VectorType>(fieldTy)) {
      (void)withIndices;
      auto vecTy = cast<VectorType>(fieldTy);
      SmallVector<Value> zeros(vecTy.getNumElements(), zero);
      v = vector::FromElementsOp::create(b, loc, vecTy, zeros);
    } else {
      v = zero;
    }
    undef = LLVM::InsertValueOp::create(b, loc, undef, v, ArrayRef<int64_t>{i});
  }
  return undef;
}

// ---------------------------------------------------------------------------
// Compile-time helpers mirroring tdm_ops.compute_* so the emitted IR is
// bit-identical to the legacy python path.
// ---------------------------------------------------------------------------

struct PadEncoding {
  int32_t encInterval = 0;
  int32_t encAmount = 0;
  bool enable = false;
};

PadEncoding computePadEncoding(int32_t padIntervalElems, int32_t padAmountElems, int32_t elemBits) {
  PadEncoding p;
  if (padIntervalElems <= 0 || padAmountElems <= 0)
    return p;
  int32_t intervalDwords = padIntervalElems * elemBits / 32;
  int32_t amountDwords = padAmountElems * elemBits / 32;
  if (intervalDwords <= 0 || amountDwords <= 0)
    return p;
  // intervalDwords must be power of two.
  if ((intervalDwords & (intervalDwords - 1)) != 0)
    return p;
  int log2Interval = 0;
  while ((1 << log2Interval) < intervalDwords)
    ++log2Interval;
  p.encInterval = log2Interval - 1;
  p.encAmount = amountDwords - 1;
  p.enable = true;
  return p;
}

struct WarpDistribution {
  int32_t warpsDim0 = 1;
  int32_t warpsDim1 = 1;
  int32_t bpwDim0 = 0;
  int32_t bpwDim1 = 0;
};

WarpDistribution computeWarpDistribution(int32_t tileD0, int32_t tileD1, int32_t numWarps) {
  // Mirror python: iterate dims in order, double along each dim as long
  // as it fits in tile_shape; leftover warps land on last dim.
  // In tdm_ops.py the distribution is over (outer_tile, inner_tile) =
  // (tileD1_outer, tileD0_inner). We honour the same "outer then inner"
  // order here.
  WarpDistribution wd;
  // Python uses [outer, inner] = [tileD1, tileD0]; reuse that order.
  int32_t warps[2] = {1, 1};
  int32_t dims[2] = {tileD1, tileD0};
  int32_t remaining = numWarps;
  for (int i = 0; i < 2; ++i) {
    while (remaining > 1 && warps[i] * 2 <= dims[i]) {
      warps[i] *= 2;
      remaining /= 2;
    }
  }
  if (remaining > 1)
    warps[1] *= remaining;
  wd.warpsDim1 = warps[0]; // outer / tileD1
  wd.warpsDim0 = warps[1]; // inner / tileD0
  wd.bpwDim1 = (tileD1 + wd.warpsDim1 - 1) / wd.warpsDim1;
  wd.bpwDim0 = (tileD0 + wd.warpsDim0 - 1) / wd.warpsDim0;
  return wd;
}

// ---------------------------------------------------------------------------
// Assembly of GROUP0 / GROUP1 for 2D load/store, matching
// tdm_ops.make_tensor_descriptor_2d exactly.
// ---------------------------------------------------------------------------

struct Addr64 {
  Value lo; // i32
  Value hi; // i32
};

// Convert a BufferFatPtr (global / buffer_desc) into (lo, hi) of a 64-bit
// global byte address. buffer_desc memref in gfx1250 TDM still needs the raw
// global base; we get it by reading struct field 0 of the resource
// descriptor packed by make_buffer_rsrc: the first 64 bits are `base` ptr.
//
// But BufferFatPtr only exposes (rsrc, offset). The raw base pointer is a
// ROCDL::MakeBufferRsrcOp operand. We recover it by tracing the defining
// op; if unavailable we fall back to ptrtoint on the rsrc pointer (which
// reads the dword[0..1] field of the resource descriptor at runtime).
//
// For simplicity and robustness we use the runtime form: the address-of-
// pointer of the rsrc struct as i64. Since a buffer rsrc in MLIR is a
// pointer in addrspace(8) and the physical descriptor's first 64 bits
// *are* the base, emitting ptrtoint on it yields the correct global
// base pointer value.
Addr64 gmemAddrFromBufferRsrc(OpBuilder &b, Location loc, fly::MemRefType memTy,
                              Value packedFatPtr) {
  BufferFatPtr bp(memTy.getPointerType(), packedFatPtr);
  Value rsrc = bp.bufferRsrc(b, loc);
  Type i64 = b.getI64Type();
  Value basePtrI64 = LLVM::PtrToIntOp::create(b, loc, i64, rsrc);
  Value byteOff = bp.swizzleByteOffset(b, loc);
  Value byteOffI64 = arith::ExtUIOp::create(b, loc, i64, byteOff);
  Value total = arith::AddIOp::create(b, loc, basePtrI64, byteOffI64);

  Value shift32 = arith::ConstantIntOp::create(b, loc, i64, 32);
  Value hi64 = arith::ShRUIOp::create(b, loc, total, shift32);
  Addr64 a;
  a.lo = arith::TruncIOp::create(b, loc, b.getI32Type(), total);
  a.hi = arith::TruncIOp::create(b, loc, b.getI32Type(), hi64);
  return a;
}

// Get LDS byte base as i32, from a shared memref value that has been type-
// converted to llvm.ptr<3>.
Value ldsByteAddrI32(OpBuilder &b, Location loc, Value sharedPtr) {
  Type i32 = b.getI32Type();
  return LLVM::PtrToIntOp::create(b, loc, i32, sharedPtr);
}

} // namespace

//===----------------------------------------------------------------------===//
// CopyOpGFX1250TDMLoad2DType
//===----------------------------------------------------------------------===//

LogicalResult CopyOpGFX1250TDMLoad2DType::verify(function_ref<InFlightDiagnostic()> emitError,
                                                 int32_t elemBits, int32_t tileDim0,
                                                 int32_t tileDim1, int32_t padInterval,
                                                 int32_t padAmount, int32_t numWarps,
                                                 bool atomicBarrierEnable) {
  if (!(elemBits == 8 || elemBits == 16 || elemBits == 32))
    return emitError() << "TDMLoad2D: elemBits must be 8/16/32, got " << elemBits;
  if (tileDim0 <= 0 || tileDim1 <= 0)
    return emitError() << "TDMLoad2D: tile dims must be positive";
  if (numWarps <= 0)
    return emitError() << "TDMLoad2D: numWarps must be positive";
  if ((padInterval > 0) ^ (padAmount > 0))
    return emitError() << "TDMLoad2D: padInterval/padAmount must be both set or both zero";
  return success();
}

std::optional<unsigned> CopyOpGFX1250TDMLoad2DType::getFieldIndex(AtomStateField field) {
  // All TDM atoms share the same state struct layout defined by the enum.
  if (fieldIndex(field) < kNumTDMStateFields)
    return fieldIndex(field);
  return std::nullopt;
}

Type CopyOpGFX1250TDMLoad2DType::getConvertedType(MLIRContext *ctx) const {
  return getTDMStateStructTy(ctx, /*withIndices=*/false);
}

Value CopyOpGFX1250TDMLoad2DType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto sty = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  return defaultZeroStruct(builder, loc, sty, /*withIndices=*/false);
}

Value CopyOpGFX1250TDMLoad2DType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                               Attribute fieldAttr, Value fieldValue) const {
  auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
  if (!fieldStr)
    return nullptr;
  auto field = symbolizeAtomStateField(fieldStr.getValue());
  if (!field)
    return nullptr;
  return insertField(builder, loc, atomStruct, *field, fieldValue);
}

Attribute CopyOpGFX1250TDMLoad2DType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }
Attribute CopyOpGFX1250TDMLoad2DType::getThrBitLayoutSrc() const {
  int64_t totalBits = int64_t{getTileDim0()} * getTileDim1() * getElemBits();
  return FxLayout(FxShape(FxC(1), FxC(totalBits)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpGFX1250TDMLoad2DType::getThrBitLayoutDst() const { return getThrBitLayoutSrc(); }
Attribute CopyOpGFX1250TDMLoad2DType::getThrBitLayoutRef() const { return getThrBitLayoutSrc(); }

namespace {

// Assemble GROUP0/GROUP1 vectors shared by load2d and store2d paths.
struct Desc2D {
  Value dgroup0; // vector<4xi32>
  Value dgroup1; // vector<8xi32>
};

Desc2D buildDesc2D(OpBuilder &b, Location loc, fly::MemRefType gmemMemTy, Value gmemVal,
                   fly::MemRefType /*ldsMemTy*/, Value ldsVal, Value atomVal, int32_t elemBits,
                   int32_t tileD0, int32_t tileD1, int32_t padInterval, int32_t padAmount,
                   int32_t numWarps, bool abe, bool forStore) {
  Type i32 = b.getI32Type();

  // -- Warp distribution --
  WarpDistribution wd = computeWarpDistribution(tileD0, tileD1, numWarps);
  int32_t bpwD0 = wd.bpwDim0;
  int32_t bpwD1 = wd.bpwDim1;

  // -- Per-wave offsets --
  Value zeroI32 = i32Const(b, loc, 0);
  Value waveOffOuter = zeroI32;
  Value waveOffInner = zeroI32;
  if (numWarps > 1) {
    // Python ordering (tdm_ops.compute_warp_distribution):
    //   block_shape = [outer_tile, inner_tile] = [tileD1, tileD0]
    //   warps_per_dim[0] = warps along outer -> wd.warpsDim1
    //   warp_coord_outer = wave %  warps_per_dim[0]
    //   warp_coord_inner = wave // warps_per_dim[0]
    Value wave = ROCDL::WaveId::create(b, loc, i32, /*range=*/LLVM::ConstantRangeAttr{});
    Value warpsOuter = i32Const(b, loc, wd.warpsDim1);
    Value warpCoordOuter = arith::RemUIOp::create(b, loc, wave, warpsOuter);
    Value warpCoordInner = arith::DivUIOp::create(b, loc, wave, warpsOuter);
    Value bpwOuter = i32Const(b, loc, bpwD1);
    Value bpwInner = i32Const(b, loc, bpwD0);
    waveOffOuter = arith::MulIOp::create(b, loc, warpCoordOuter, bpwOuter);
    waveOffInner = arith::MulIOp::create(b, loc, warpCoordInner, bpwInner);
  }

  // -- State struct extracts --
  Value gmemByteOff = extractField(b, loc, atomVal, AtomStateField::GmemByteOffset);
  Value ldsByteOff = extractField(b, loc, atomVal, AtomStateField::LdsByteOffset);
  Value workgroupMask = extractField(b, loc, atomVal, AtomStateField::WorkgroupMask);
  Value tensorDim0 = extractField(b, loc, atomVal, AtomStateField::TensorDim0);
  Value tensorDim1 = extractField(b, loc, atomVal, AtomStateField::TensorDim1);
  Value stride0 = extractField(b, loc, atomVal, AtomStateField::Stride0);

  // -- Compute per-wave GROUP0 addresses --
  // Global side: base = gmem_addr + gmem_byte_off + wave_off
  Addr64 gmemBase = gmemAddrFromBufferRsrc(b, loc, gmemMemTy, gmemVal);

  // Fold gmem_byte_offset (SGPR) into addr64.
  Value gmemByteOffI64 = arith::ExtUIOp::create(b, loc, b.getI64Type(), gmemByteOff);
  Value baseI64 = arith::AddIOp::create(
      b, loc,
      arith::OrIOp::create(
          b, loc,
          arith::ShLIOp::create(b, loc, arith::ExtUIOp::create(b, loc, b.getI64Type(), gmemBase.hi),
                                arith::ConstantIntOp::create(b, loc, b.getI64Type(), 32)),
          arith::ExtUIOp::create(b, loc, b.getI64Type(), gmemBase.lo)),
      gmemByteOffI64);

  // Wave-level gmem offset mirrors tdm_ops.py (row-major layout):
  //   wave_delta_elems = warp_off_outer * stride0 + warp_off_inner * 1
  //   wave_delta_bytes = wave_delta_elems * elem_bytes
  // stride0 is atom state field "stride0" (outer stride in elements).
  // The TDM descriptor model assumes inner_stride = 1 (contiguous row);
  // non-row-major access must precompute gmem_byte_offset itself.
  Value waveElemOffOuter = arith::MulIOp::create(b, loc, waveOffOuter, stride0);
  Value waveElemOff = arith::AddIOp::create(b, loc, waveElemOffOuter, waveOffInner);
  int64_t elemBytes = std::max<int64_t>(1, elemBits / 8);
  Value elemBytesC = i32Const(b, loc, elemBytes);
  Value waveByteOff = arith::MulIOp::create(b, loc, waveElemOff, elemBytesC);
  Value waveByteOffI64 = arith::ExtUIOp::create(b, loc, b.getI64Type(), waveByteOff);
  Value addr64 = arith::AddIOp::create(b, loc, baseI64, waveByteOffI64);

  Value addrLo = arith::TruncIOp::create(b, loc, i32, addr64);
  Value shift32 = arith::ConstantIntOp::create(b, loc, b.getI64Type(), 32);
  Value addrHiRaw =
      arith::TruncIOp::create(b, loc, i32, arith::ShRUIOp::create(b, loc, addr64, shift32));
  // Set type field = 2 in [31:30] of hi word.
  Value addrHi = arith::OrIOp::create(b, loc, addrHiRaw, i32Const(b, loc, 1LL << 31));

  // LDS side:
  //   lds_total = lds_base_ptr + lds_byte_offset + wave_lds_byte_off
  // where lds_inner_stride = inner_tile (+ pad_amount if padding enabled for load)
  int32_t ldsInnerStride = bpwD0;
  if (padInterval > 0 && padAmount > 0 && !forStore)
    ldsInnerStride = bpwD0 + padAmount;
  Value ldsInnerStrideC = i32Const(b, loc, ldsInnerStride);
  Value waveLdsElemOff = arith::AddIOp::create(
      b, loc, arith::MulIOp::create(b, loc, waveOffOuter, ldsInnerStrideC), waveOffInner);
  Value waveLdsByteOff = arith::MulIOp::create(b, loc, waveLdsElemOff, elemBytesC);

  Value ldsBaseI32 = ldsByteAddrI32(b, loc, ldsVal);
  Value ldsAddrI32 = arith::AddIOp::create(
      b, loc, arith::AddIOp::create(b, loc, ldsBaseI32, ldsByteOff), waveLdsByteOff);

  // GROUP0 assembly: [pred, lds_addr, addr_lo, addr_hi|type]
  Value predConst = i32Const(b, loc, 1);
  auto vec4i32 = VectorType::get({4}, i32);
  SmallVector<Value> g0{predConst, ldsAddrI32, addrLo, addrHi};
  Value dgroup0 = vector::FromElementsOp::create(b, loc, vec4i32, g0);

  // GROUP1 assembly:
  int32_t tdim0 = bpwD0;
  int32_t tdim1 = bpwD1;
  int32_t tileDesc0 = bpwD0;
  int32_t tileDesc1 = bpwD1;

  // Store path cannot de-pad, so fold pad_amount into tile extent.
  int32_t effPadInterval = padInterval;
  int32_t effPadAmount = padAmount;
  if (forStore && padInterval > 0 && padAmount > 0) {
    tileDesc0 += padAmount;
    effPadInterval = 0;
    effPadAmount = 0;
  }

  int32_t dataSizeCode = 0;
  if (elemBytes > 1) {
    int32_t bytes = static_cast<int32_t>(elemBytes);
    while (bytes > 1) {
      ++dataSizeCode;
      bytes >>= 1;
    }
  }
  PadEncoding pe = computePadEncoding(effPadInterval, effPadAmount, elemBits);

  int32_t abeBit = abe ? 1 : 0;
  int32_t g1_s0_upper = (dataSizeCode << 16) | (abeBit << 18) | (0 << 19) // iterate_enable
                        | ((pe.enable ? 1 : 0) << 20) | (0 << 21)         // early_timeout
                        | (pe.encInterval << 22) | (pe.encAmount << 25);

  // g1_s0: workgroup_mask[15:0] | upper
  Value upperC = i32Const(b, loc, g1_s0_upper);
  Value maskLow = arith::AndIOp::create(b, loc, workgroupMask, i32Const(b, loc, 0xFFFF));
  Value g1_s0 = arith::OrIOp::create(b, loc, upperC, maskLow);

  // g1_s1..s4 pack tensor_dim0/dim1 and tile_dim0/dim1.
  // Layout (from python tdm_ops):
  //   sgpr1[31:16] = tensor_dim0[15:0]
  //   sgpr2[15:0]  = tensor_dim0[31:16]
  //   sgpr2[31:16] = tensor_dim1[15:0]
  //   sgpr3[15:0]  = tensor_dim1[31:16]
  //   sgpr3[31:16] = tile_dim0[15:0]
  //   sgpr4[15:0]  = tile_dim1[15:0]
  //   sgpr4[31:16] = 0
  //   sgpr5        = stride0[31:0]
  Value sixteen = i32Const(b, loc, 16);
  Value ffff = i32Const(b, loc, 0xFFFF);

  // Use dynamic tensor_dim0/tensor_dim1/stride0 from state (falls back to
  // zero when user did not set them).
  Value td0Lo = arith::AndIOp::create(b, loc, tensorDim0, ffff);
  Value td0Hi =
      arith::AndIOp::create(b, loc, arith::ShRUIOp::create(b, loc, tensorDim0, sixteen), ffff);
  Value td1Lo = arith::AndIOp::create(b, loc, tensorDim1, ffff);
  Value td1Hi =
      arith::AndIOp::create(b, loc, arith::ShRUIOp::create(b, loc, tensorDim1, sixteen), ffff);

  Value td0LoShl = arith::ShLIOp::create(b, loc, td0Lo, sixteen);
  Value g1_s1 = td0LoShl;

  Value td1LoShl = arith::ShLIOp::create(b, loc, td1Lo, sixteen);
  Value g1_s2 = arith::OrIOp::create(b, loc, td0Hi, td1LoShl);

  // tile_dim0[15:0] is static, pack with td1Hi
  Value tile0Shl = i32Const(b, loc, (tileDesc0 & 0xFFFF) << 16);
  Value g1_s3 = arith::OrIOp::create(b, loc, td1Hi, tile0Shl);

  // tile_dim1[15:0], tile_dim2[31:16]=0
  Value g1_s4 = i32Const(b, loc, tileDesc1 & 0xFFFF);

  // stride0 (low 32 bits)
  Value g1_s5 = stride0;

  Value g1_s6 = zeroI32;
  Value g1_s7 = zeroI32;

  auto vec8i32 = VectorType::get({8}, i32);
  SmallVector<Value> g1{g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, g1_s6, g1_s7};
  Value dgroup1 = vector::FromElementsOp::create(b, loc, vec8i32, g1);

  (void)tdim0;
  (void)tdim1;
  (void)shift32;
  return Desc2D{dgroup0, dgroup1};
}

Value zeroVec4I32(OpBuilder &b, Location loc) {
  Value z = i32Const(b, loc, 0);
  SmallVector<Value> zs{z, z, z, z};
  return vector::FromElementsOp::create(b, loc, VectorType::get({4}, b.getI32Type()), zs);
}
Value zeroVec8I32(OpBuilder &b, Location loc) {
  Value z = i32Const(b, loc, 0);
  SmallVector<Value> zs(8, z);
  return vector::FromElementsOp::create(b, loc, VectorType::get({8}, b.getI32Type()), zs);
}

} // namespace

LogicalResult CopyOpGFX1250TDMLoad2DType::emitAtomCall(OpBuilder &builder, Location loc,
                                                       Type /*copyAtomTyArg*/, Type srcMemTyArg,
                                                       Type dstMemTyArg, Value atomVal, Value src,
                                                       Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);
  if (srcMemTy.getAddressSpace().getValue() != AddressSpace::BufferDesc)
    return failure();
  if (dstMemTy.getAddressSpace().getValue() != AddressSpace::Shared)
    return failure();

  Desc2D desc = buildDesc2D(builder, loc, srcMemTy, src, dstMemTy, dst, atomVal, getElemBits(),
                            getTileDim0(), getTileDim1(), getPadInterval(), getPadAmount(),
                            getNumWarps(), getAtomicBarrierEnable(), /*forStore=*/false);

  Value dg2 = zeroVec4I32(builder, loc);
  Value dg3 = zeroVec4I32(builder, loc);
  Value dg4 = zeroVec8I32(builder, loc);

  Value cachePolicy = extractField(builder, loc, atomVal, AtomStateField::CachePolicy);
  // The intrinsic takes cachePolicy as an attribute, not an SSA value.
  // Materialize it as a constant-or-zero attr. If the user set a non-constant
  // runtime value we fall back to 0 (matching legacy behaviour; non-constant
  // cache policy is unusual for TDM).
  int64_t cpAttr = 0;
  if (auto def = cachePolicy.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(def.getValue()))
      cpAttr = ia.getInt();
  }

  ROCDL::TensorLoadToLDSOp::create(builder, loc, desc.dgroup0, desc.dgroup1, dg2, dg3, dg4,
                                   builder.getI32IntegerAttr(cpAttr), ArrayAttr{}, ArrayAttr{},
                                   ArrayAttr{});
  return success();
}

LogicalResult CopyOpGFX1250TDMLoad2DType::emitAtomCall(OpBuilder &builder, Location loc,
                                                       Type copyAtomTyArg, Type srcMemTyArg,
                                                       Type dstMemTyArg, Type predMemTyArg,
                                                       Value atomVal, Value src, Value dst,
                                                       Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
}

FailureOr<Value> CopyOpGFX1250TDMLoad2DType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                             Type /*resultTy*/, Type copyAtomTyArg,
                                                             Type srcTyArg, Type dstTyArg,
                                                             Value atomVal, Value src,
                                                             Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpGFX1250TDMLoad2DType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type /*resultTy*/, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

//===----------------------------------------------------------------------===//
// CopyOpGFX1250TDMStore2DType
//===----------------------------------------------------------------------===//

LogicalResult CopyOpGFX1250TDMStore2DType::verify(function_ref<InFlightDiagnostic()> emitError,
                                                  int32_t elemBits, int32_t tileDim0,
                                                  int32_t tileDim1, int32_t padInterval,
                                                  int32_t padAmount, int32_t numWarps,
                                                  bool atomicBarrierEnable) {
  if (!(elemBits == 8 || elemBits == 16 || elemBits == 32))
    return emitError() << "TDMStore2D: elemBits must be 8/16/32, got " << elemBits;
  if (tileDim0 <= 0 || tileDim1 <= 0)
    return emitError() << "TDMStore2D: tile dims must be positive";
  if (numWarps <= 0)
    return emitError() << "TDMStore2D: numWarps must be positive";
  return success();
}

std::optional<unsigned> CopyOpGFX1250TDMStore2DType::getFieldIndex(AtomStateField field) {
  if (fieldIndex(field) < kNumTDMStateFields)
    return fieldIndex(field);
  return std::nullopt;
}

Type CopyOpGFX1250TDMStore2DType::getConvertedType(MLIRContext *ctx) const {
  return getTDMStateStructTy(ctx, /*withIndices=*/false);
}

Value CopyOpGFX1250TDMStore2DType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto sty = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  return defaultZeroStruct(builder, loc, sty, /*withIndices=*/false);
}

Value CopyOpGFX1250TDMStore2DType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                                Attribute fieldAttr, Value fieldValue) const {
  auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
  if (!fieldStr)
    return nullptr;
  auto field = symbolizeAtomStateField(fieldStr.getValue());
  if (!field)
    return nullptr;
  return insertField(builder, loc, atomStruct, *field, fieldValue);
}

Attribute CopyOpGFX1250TDMStore2DType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }
Attribute CopyOpGFX1250TDMStore2DType::getThrBitLayoutSrc() const {
  int64_t totalBits = int64_t{getTileDim0()} * getTileDim1() * getElemBits();
  return FxLayout(FxShape(FxC(1), FxC(totalBits)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpGFX1250TDMStore2DType::getThrBitLayoutDst() const { return getThrBitLayoutSrc(); }
Attribute CopyOpGFX1250TDMStore2DType::getThrBitLayoutRef() const { return getThrBitLayoutSrc(); }

LogicalResult CopyOpGFX1250TDMStore2DType::emitAtomCall(OpBuilder &builder, Location loc,
                                                        Type /*copyAtomTyArg*/, Type srcMemTyArg,
                                                        Type dstMemTyArg, Value atomVal, Value src,
                                                        Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);
  // Store direction: src = LDS (Shared), dst = global (BufferDesc).
  if (srcMemTy.getAddressSpace().getValue() != AddressSpace::Shared)
    return failure();
  if (dstMemTy.getAddressSpace().getValue() != AddressSpace::BufferDesc)
    return failure();

  // buildDesc2D treats the first memref pair as (gmem, lds), and for 2D
  // store the *global destination* still carries the addr64, *shared source*
  // carries the LDS byte address -- we pass them in that role swap below.
  Desc2D desc = buildDesc2D(builder, loc, dstMemTy, dst, srcMemTy, src, atomVal, getElemBits(),
                            getTileDim0(), getTileDim1(), getPadInterval(), getPadAmount(),
                            getNumWarps(), getAtomicBarrierEnable(), /*forStore=*/true);

  Value dg2 = zeroVec4I32(builder, loc);
  Value dg3 = zeroVec4I32(builder, loc);
  Value dg4 = zeroVec8I32(builder, loc);

  Value cachePolicy = extractField(builder, loc, atomVal, AtomStateField::CachePolicy);
  int64_t cpAttr = 0;
  if (auto def = cachePolicy.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(def.getValue()))
      cpAttr = ia.getInt();
  }

  ROCDL::TensorStoreFromLDSOp::create(builder, loc, desc.dgroup0, desc.dgroup1, dg2, dg3, dg4,
                                      builder.getI32IntegerAttr(cpAttr), ArrayAttr{}, ArrayAttr{},
                                      ArrayAttr{});
  return success();
}

LogicalResult CopyOpGFX1250TDMStore2DType::emitAtomCall(OpBuilder &builder, Location loc,
                                                        Type copyAtomTyArg, Type srcMemTyArg,
                                                        Type dstMemTyArg, Type predMemTyArg,
                                                        Value atomVal, Value src, Value dst,
                                                        Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
}

FailureOr<Value> CopyOpGFX1250TDMStore2DType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                              Type, Type copyAtomTyArg,
                                                              Type srcTyArg, Type dstTyArg,
                                                              Value atomVal, Value src,
                                                              Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpGFX1250TDMStore2DType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type, Type copyAtomTyArg, Type srcTyArg, Type dstTyArg,
    Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

//===----------------------------------------------------------------------===//
// CopyOpGFX1250TDMGatherType
//===----------------------------------------------------------------------===//

LogicalResult CopyOpGFX1250TDMGatherType::verify(function_ref<InFlightDiagnostic()> emitError,
                                                 int32_t elemBits, int32_t rowWidth,
                                                 int32_t indexSize, int32_t maxIndices,
                                                 int32_t padInterval, int32_t padAmount,
                                                 bool /*isStore*/) {
  if (!(elemBits == 8 || elemBits == 16 || elemBits == 32 || elemBits == 64))
    return emitError() << "TDMGather: elemBits must be 8/16/32/64, got " << elemBits;
  if (indexSize != 16 && indexSize != 32)
    return emitError() << "TDMGather: indexSize must be 16 or 32";
  if (maxIndices <= 0 || maxIndices > (indexSize == 32 ? 8 : 16))
    return emitError() << "TDMGather: maxIndices out of range";
  if (rowWidth * elemBits % 32 != 0)
    return emitError() << "TDMGather: row_width * elemBits must be a multiple of 32 bits";
  return success();
}

std::optional<unsigned> CopyOpGFX1250TDMGatherType::getFieldIndex(AtomStateField field) {
  if (fieldIndex(field) < kNumTDMStateFields)
    return fieldIndex(field);
  return std::nullopt;
}

Type CopyOpGFX1250TDMGatherType::getConvertedType(MLIRContext *ctx) const {
  return getTDMStateStructTy(ctx, /*withIndices=*/true);
}

Value CopyOpGFX1250TDMGatherType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto sty = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  return defaultZeroStruct(builder, loc, sty, /*withIndices=*/true);
}

Value CopyOpGFX1250TDMGatherType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                               Attribute fieldAttr, Value fieldValue) const {
  auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
  if (!fieldStr)
    return nullptr;
  auto field = symbolizeAtomStateField(fieldStr.getValue());
  if (!field)
    return nullptr;
  return insertField(builder, loc, atomStruct, *field, fieldValue);
}

Attribute CopyOpGFX1250TDMGatherType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }
Attribute CopyOpGFX1250TDMGatherType::getThrBitLayoutSrc() const {
  int64_t totalBits = int64_t{getRowWidth()} * getElemBits() * getMaxIndices();
  return FxLayout(FxShape(FxC(1), FxC(totalBits)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpGFX1250TDMGatherType::getThrBitLayoutDst() const { return getThrBitLayoutSrc(); }
Attribute CopyOpGFX1250TDMGatherType::getThrBitLayoutRef() const { return getThrBitLayoutSrc(); }

LogicalResult CopyOpGFX1250TDMGatherType::emitAtomCall(OpBuilder &builder, Location loc,
                                                       Type /*copyAtomTyArg*/, Type srcMemTyArg,
                                                       Type dstMemTyArg, Value atomVal, Value src,
                                                       Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);
  fly::MemRefType gmemTy;
  fly::MemRefType ldsTy;
  Value gmemVal, ldsVal;
  if (getIsStore()) {
    // src=LDS, dst=buffer_desc
    if (srcMemTy.getAddressSpace().getValue() != AddressSpace::Shared ||
        dstMemTy.getAddressSpace().getValue() != AddressSpace::BufferDesc)
      return failure();
    gmemTy = dstMemTy;
    gmemVal = dst;
    ldsTy = srcMemTy;
    ldsVal = src;
  } else {
    if (srcMemTy.getAddressSpace().getValue() != AddressSpace::BufferDesc ||
        dstMemTy.getAddressSpace().getValue() != AddressSpace::Shared)
      return failure();
    gmemTy = srcMemTy;
    gmemVal = src;
    ldsTy = dstMemTy;
    ldsVal = dst;
  }
  (void)ldsTy;

  Type i32 = builder.getI32Type();

  // ------- GROUP 0 (gather flavor) -------
  Value gmemByteOff = extractField(builder, loc, atomVal, AtomStateField::GmemByteOffset);
  Value ldsByteOff = extractField(builder, loc, atomVal, AtomStateField::LdsByteOffset);
  Value workgroupMask = extractField(builder, loc, atomVal, AtomStateField::WorkgroupMask);
  Value tensorDim0 = extractField(builder, loc, atomVal, AtomStateField::TensorDim0);
  Value tensorDim1 = extractField(builder, loc, atomVal, AtomStateField::TensorDim1);
  Value stride0 = extractField(builder, loc, atomVal, AtomStateField::Stride0);
  Value indicesVec = extractField(builder, loc, atomVal, AtomStateField::GatherRowIndices);
  Value indexCount = extractField(builder, loc, atomVal, AtomStateField::GatherIndexCount);

  Addr64 gmemBase = gmemAddrFromBufferRsrc(builder, loc, gmemTy, gmemVal);
  Value gmemByteOffI64 = arith::ExtUIOp::create(builder, loc, builder.getI64Type(), gmemByteOff);
  Value baseI64 = arith::AddIOp::create(
      builder, loc,
      arith::OrIOp::create(
          builder, loc,
          arith::ShLIOp::create(
              builder, loc, arith::ExtUIOp::create(builder, loc, builder.getI64Type(), gmemBase.hi),
              arith::ConstantIntOp::create(builder, loc, builder.getI64Type(), 32)),
          arith::ExtUIOp::create(builder, loc, builder.getI64Type(), gmemBase.lo)),
      gmemByteOffI64);
  Value addrLo = arith::TruncIOp::create(builder, loc, i32, baseI64);
  Value shift32 = arith::ConstantIntOp::create(builder, loc, builder.getI64Type(), 32);
  Value addrHiRaw = arith::TruncIOp::create(builder, loc, i32,
                                            arith::ShRUIOp::create(builder, loc, baseI64, shift32));
  Value addrHi = arith::OrIOp::create(builder, loc, addrHiRaw, i32Const(builder, loc, 1LL << 31));

  // gather pred word = 1 | (gather_index_bit << 30) | (1 << 31)
  int32_t gatherIndexBit = (getIndexSize() == 32) ? 1 : 0;
  int32_t predValue = 1 | (gatherIndexBit << 30) | (1 << 31);
  Value predConst = i32Const(builder, loc, predValue);

  Value ldsBaseI32 = ldsByteAddrI32(builder, loc, ldsVal);
  Value ldsAddrI32 = arith::AddIOp::create(builder, loc, ldsBaseI32, ldsByteOff);

  auto vec4i32 = VectorType::get({4}, i32);
  SmallVector<Value> g0{predConst, ldsAddrI32, addrLo, addrHi};
  Value dgroup0 = vector::FromElementsOp::create(builder, loc, vec4i32, g0);

  // ------- GROUP 1 -------
  int64_t elemBytes = std::max<int64_t>(1, getElemBits() / 8);
  int32_t dataSizeCode = 0;
  for (int32_t b = (int32_t)elemBytes; b > 1; b >>= 1)
    ++dataSizeCode;
  PadEncoding pe = computePadEncoding(getPadInterval(), getPadAmount(), getElemBits());

  int32_t g1_s0_upper = (dataSizeCode << 16) | (0 << 18) // atomic_barrier_enable
                        | (0 << 19)                      // iterate_enable ignored for gather
                        | ((pe.enable ? 1 : 0) << 20) | (0 << 21) | (pe.encInterval << 22) |
                        (pe.encAmount << 25);

  Value upperC = i32Const(builder, loc, g1_s0_upper);
  Value ffff = i32Const(builder, loc, 0xFFFF);
  Value maskLow = arith::AndIOp::create(builder, loc, workgroupMask, ffff);
  Value g1_s0 = arith::OrIOp::create(builder, loc, upperC, maskLow);

  // tensor_dim0 (32b) -> sgpr1[31:16] | sgpr2[15:0]
  Value sixteen = i32Const(builder, loc, 16);
  Value td0Lo = arith::AndIOp::create(builder, loc, tensorDim0, ffff);
  Value td0Hi = arith::AndIOp::create(
      builder, loc, arith::ShRUIOp::create(builder, loc, tensorDim0, sixteen), ffff);
  Value td1Lo = arith::AndIOp::create(builder, loc, tensorDim1, ffff);
  Value td1Hi = arith::AndIOp::create(
      builder, loc, arith::ShRUIOp::create(builder, loc, tensorDim1, sixteen), ffff);

  Value g1_s1 = arith::ShLIOp::create(builder, loc, td0Lo, sixteen);
  Value g1_s2 = arith::OrIOp::create(builder, loc, td0Hi,
                                     arith::ShLIOp::create(builder, loc, td1Lo, sixteen));

  Value rowWidthShl = i32Const(builder, loc, (getRowWidth() & 0xFFFF) << 16);
  Value g1_s3 = arith::OrIOp::create(builder, loc, td1Hi, rowWidthShl);

  // gather tile_dim1 = index_count (runtime, masked to 16 bits).
  Value g1_s4 = arith::AndIOp::create(builder, loc, indexCount, ffff);
  Value g1_s5 = stride0;
  Value zeroI32 = i32Const(builder, loc, 0);

  auto vec8i32 = VectorType::get({8}, i32);
  SmallVector<Value> g1{g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, zeroI32, zeroI32};
  Value dgroup1 = vector::FromElementsOp::create(builder, loc, vec8i32, g1);

  // ------- GROUP 2 / GROUP 3 : row indices -------
  // Split indicesVec (vector<8xi32>) into two vector<4xi32>.
  SmallVector<Value> g2vals(4), g3vals(4);
  for (int i = 0; i < 4; ++i) {
    Value idxC = i32Const(builder, loc, i);
    (void)idxC;
    g2vals[i] = vector::ExtractOp::create(builder, loc, indicesVec, ArrayRef<int64_t>{i});
    g3vals[i] = vector::ExtractOp::create(builder, loc, indicesVec, ArrayRef<int64_t>{4 + i});
  }
  Value dgroup2 = vector::FromElementsOp::create(builder, loc, vec4i32, g2vals);
  Value dgroup3 = vector::FromElementsOp::create(builder, loc, vec4i32, g3vals);
  Value dgroup4 = zeroVec8I32(builder, loc);

  Value cachePolicy = extractField(builder, loc, atomVal, AtomStateField::CachePolicy);
  int64_t cpAttr = 0;
  if (auto def = cachePolicy.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(def.getValue()))
      cpAttr = ia.getInt();
  }

  if (getIsStore()) {
    ROCDL::TensorStoreFromLDSOp::create(builder, loc, dgroup0, dgroup1, dgroup2, dgroup3, dgroup4,
                                        builder.getI32IntegerAttr(cpAttr), ArrayAttr{}, ArrayAttr{},
                                        ArrayAttr{});
  } else {
    ROCDL::TensorLoadToLDSOp::create(builder, loc, dgroup0, dgroup1, dgroup2, dgroup3, dgroup4,
                                     builder.getI32IntegerAttr(cpAttr), ArrayAttr{}, ArrayAttr{},
                                     ArrayAttr{});
  }
  return success();
}

LogicalResult CopyOpGFX1250TDMGatherType::emitAtomCall(OpBuilder &builder, Location loc,
                                                       Type copyAtomTyArg, Type srcMemTyArg,
                                                       Type dstMemTyArg, Type predMemTyArg,
                                                       Value atomVal, Value src, Value dst,
                                                       Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
}

FailureOr<Value> CopyOpGFX1250TDMGatherType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type,
                                                             Type copyAtomTyArg, Type srcTyArg,
                                                             Type dstTyArg, Value atomVal,
                                                             Value src, Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpGFX1250TDMGatherType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type,
                                                             Type copyAtomTyArg, Type srcTyArg,
                                                             Type dstTyArg, Type predTyArg,
                                                             Value atomVal, Value src, Value dst,
                                                             Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

} // namespace mlir::fly_rocdl
