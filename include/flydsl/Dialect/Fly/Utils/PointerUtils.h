// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_UTILS_POINTERUTILS_H
#define FLYDSL_DIALECT_FLY_UTILS_POINTERUTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

TypedValue<LLVM::LLVMPointerType> applySwizzleOnPtr(OpBuilder &b, Location loc,
                                                    TypedValue<LLVM::LLVMPointerType> ptr,
                                                    SwizzleAttr swizzle);

/// Project small-float element types (Float8/Float6/Float4) to integer types of
/// the same bit-width. Non-small-float types are returned unchanged. Used to
/// map Fly register memref values to types that are legal within the LLVM
/// dialect (which does not support `vector<N x f8>`).
Type projectToLLVMCompatibleElemTy(Type elemTy);

/// Compute the SSA-value type corresponding to a Fly register memref.
///
/// \p llvmCompatibleType controls whether small-float element types (e.g.
/// f8E4M3FNUZ/f8E5M2FNUZ) are projected to their same-width integer
/// counterpart. Callers that feed the resulting type into `LLVM::LoadOp`,
/// `LLVM::StoreOp`, or any SSA value that must survive into the LLVM
/// dialect MUST set \p llvmCompatibleType to true.
Type RegMem2SSAType(fly::MemRefType memRefTy, bool llvmCompatibleType = false);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_POINTERUTILS_H
