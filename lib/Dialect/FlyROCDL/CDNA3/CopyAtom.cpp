// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Utils/BufferFatPtr.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

// --- CopyOpCDNA3BufferCopy (stateful: soffset) ---

static constexpr unsigned kBufferCopySoffsetIdx = 0;

bool CopyOpCDNA3BufferCopyType::isStatic() const { return true; }

Value CopyOpCDNA3BufferCopyType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                    Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, getBitSize()), getBitSize());
}

Attribute CopyOpCDNA3BufferCopyType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpCDNA3BufferCopyType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferCopyType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferCopyType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

Type CopyOpCDNA3BufferCopyType::getConvertedType(MLIRContext *ctx) const {
  return LLVM::LLVMStructType::getLiteral(ctx, {IntegerType::get(ctx, 32)});
}

Value CopyOpCDNA3BufferCopyType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                              Attribute fieldAttr, Value fieldValue) const {
  return LLVM::InsertValueOp::create(builder, loc, atomStruct, fieldValue, kBufferCopySoffsetIdx);
}

LogicalResult CopyOpCDNA3BufferCopyType::emitAtomCall(OpBuilder &builder, Location loc,
                                                      Type copyAtomTyArg, Type srcMemTyArg,
                                                      Type dstMemTyArg, Value atomVal, Value src,
                                                      Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);

  IntegerType copyTy = builder.getIntegerType(getBitSize());

  AddressSpace srcAS = srcMemTy.getAddressSpace().getValue();
  AddressSpace dstAS = dstMemTy.getAddressSpace().getValue();

  bool srcIsBuffer = (srcAS == AddressSpace::BufferDesc);
  bool dstIsBuffer = (dstAS == AddressSpace::BufferDesc);

  if (!(srcIsBuffer || dstIsBuffer))
    return failure();

  Value soffset =
      LLVM::ExtractValueOp::create(builder, loc, atomVal, kBufferCopySoffsetIdx);
  ArrayAttr noAttrs;
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);

  auto unpackBuffer = [&](Value val, fly::MemRefType flyTy) -> std::pair<Value, Value> {
    BufferFatPtr bp(flyTy.getPointerType(), val);
    return {bp.bufferRsrc(builder, loc), bp.swizzleByteOffset(builder, loc)};
  };

  if (srcIsBuffer && !dstIsBuffer) {
    auto [srcRsrc, srcOff] = unpackBuffer(src, srcMemTy);
    Value loaded = ROCDL::RawPtrBufferLoadOp::create(builder, loc, copyTy, srcRsrc, srcOff,
                                                     soffset, zero, noAttrs, noAttrs, noAttrs);
    LLVM::StoreOp::create(builder, loc, loaded, dst);
  } else if (!srcIsBuffer && dstIsBuffer) {
    auto [dstRsrc, dstOff] = unpackBuffer(dst, dstMemTy);
    Value loaded = LLVM::LoadOp::create(builder, loc, copyTy, src);
    ROCDL::RawPtrBufferStoreOp::create(builder, loc, loaded, dstRsrc, dstOff, soffset, zero,
                                       noAttrs, noAttrs, noAttrs);
  } else {
    auto [srcRsrc, srcOff] = unpackBuffer(src, srcMemTy);
    auto [dstRsrc, dstOff] = unpackBuffer(dst, dstMemTy);
    Value loaded = ROCDL::RawPtrBufferLoadOp::create(builder, loc, copyTy, srcRsrc, srcOff,
                                                     soffset, zero, noAttrs, noAttrs, noAttrs);
    ROCDL::RawPtrBufferStoreOp::create(builder, loc, loaded, dstRsrc, dstOff, soffset, zero,
                                       noAttrs, noAttrs, noAttrs);
  }
  return success();
}

LogicalResult CopyOpCDNA3BufferCopyType::emitAtomCall(OpBuilder &builder, Location loc,
                                                      Type copyAtomTyArg, Type srcMemTyArg,
                                                      Type dstMemTyArg, Type predMemTyArg,
                                                      Value atomVal, Value src, Value dst,
                                                      Value pred) const {
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
}

// --- CopyOpCDNA3BufferAtomic ---

bool CopyOpCDNA3BufferAtomicType::isStatic() const { return true; }

Value CopyOpCDNA3BufferAtomicType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                      Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, bits), bits);
}

Attribute CopyOpCDNA3BufferAtomicType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpCDNA3BufferAtomicType::getThrBitLayoutSrc() const {
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferAtomicType::getThrBitLayoutDst() const {
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferAtomicType::getThrBitLayoutRef() const {
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(1), FxC(1)));
}

static LogicalResult emitBufferAtomicOp(OpBuilder &builder, Location loc, fly::AtomicOp atomicOp,
                                        Value loaded, Value rsrc, Value offset, Value zero,
                                        ArrayAttr noAttrs, bool isFloat) {
  switch (atomicOp) {
  case fly::AtomicOp::Add:
    if (!isFloat)
      return failure();
    ROCDL::RawPtrBufferAtomicFaddOp::create(builder, loc, loaded, rsrc, offset, zero, zero,
                                            noAttrs, noAttrs, noAttrs);
    return success();
  case fly::AtomicOp::Max:
    if (isFloat)
      ROCDL::RawPtrBufferAtomicFmaxOp::create(builder, loc, loaded, rsrc, offset, zero, zero,
                                              noAttrs, noAttrs, noAttrs);
    else
      ROCDL::RawPtrBufferAtomicSmaxOp::create(builder, loc, loaded, rsrc, offset, zero, zero,
                                              noAttrs, noAttrs, noAttrs);
    return success();
  case fly::AtomicOp::Min:
    if (isFloat)
      return failure();
    ROCDL::RawPtrBufferAtomicUminOp::create(builder, loc, loaded, rsrc, offset, zero, zero,
                                            noAttrs, noAttrs, noAttrs);
    return success();
  default:
    return failure();
  }
}

LogicalResult CopyOpCDNA3BufferAtomicType::emitAtomCall(OpBuilder &builder, Location loc,
                                                        Type copyAtomTyArg, Type srcMemTyArg,
                                                        Type dstMemTyArg, Value atomVal, Value src,
                                                        Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);

  AddressSpace dstAS = dstMemTy.getAddressSpace().getValue();
  if (dstAS != AddressSpace::BufferDesc)
    return failure();

  Type elemTy = getValType();
  int32_t bits = elemTy.getIntOrFloatBitWidth();
  bool isFloat = isa<FloatType>(elemTy);

  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  ArrayAttr noAttrs;

  Value loaded;
  AddressSpace srcAS = srcMemTy.getAddressSpace().getValue();
  if (srcAS == AddressSpace::BufferDesc) {
    IntegerType loadTy = builder.getIntegerType(bits);
    BufferFatPtr srcBp(srcMemTy.getPointerType(), src);
    loaded = ROCDL::RawPtrBufferLoadOp::create(builder, loc, loadTy, srcBp.bufferRsrc(builder, loc),
                                               srcBp.swizzleByteOffset(builder, loc), zero, zero,
                                               noAttrs, noAttrs, noAttrs);
    if (isFloat)
      loaded = arith::BitcastOp::create(builder, loc, elemTy, loaded);
  } else {
    loaded = LLVM::LoadOp::create(builder, loc, elemTy, src);
  }

  BufferFatPtr dstBp(dstMemTy.getPointerType(), dst);
  return emitBufferAtomicOp(builder, loc, getAtomicOp(), loaded, dstBp.bufferRsrc(builder, loc),
                            dstBp.swizzleByteOffset(builder, loc), zero, noAttrs, isFloat);
}

LogicalResult CopyOpCDNA3BufferAtomicType::emitAtomCall(OpBuilder &builder, Location loc,
                                                        Type copyAtomTyArg, Type srcMemTyArg,
                                                        Type dstMemTyArg, Type predMemTyArg,
                                                        Value atomVal, Value src, Value dst,
                                                        Value pred) const {
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
}

} // namespace mlir::fly_rocdl
