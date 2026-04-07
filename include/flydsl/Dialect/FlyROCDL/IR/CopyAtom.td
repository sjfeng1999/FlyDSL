// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_COPYATOM
#define FLYROCDL_COPYATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_CopyOpBufferCopy : FlyxROCL_StatefulCopyOp<"CopyOpCDNA3BufferCopy", "cdna3.buffer_copy",[]> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

def FlyROCDL_CopyOpBufferAtomic : FlyxROCL_CopyOp<"CopyOpCDNA3BufferAtomic", "cdna3.buffer_atomic",[]> {
  let parameters = (ins
    "::mlir::fly::AtomicOp":$atomicOp,
    "Type":$valType
  );
  let assemblyFormat = "`<` $atomicOp `,` qualified($valType) `>`";
}

#endif // FLYROCDL_COPYATOM
