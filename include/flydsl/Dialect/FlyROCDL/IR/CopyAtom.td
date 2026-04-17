// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_COPYATOM
#define FLYROCDL_COPYATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_CopyOpBufferCopy : FlyROCL_StatefulCopyOp<"CopyOpCDNA3BufferCopy", "cdna3.buffer_copy",[]> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

def FlyROCDL_CopyOpBufferCopyLDS : FlyROCL_StatefulCopyOp<"CopyOpCDNA3BufferCopyLDS", "cdna3.buffer_copy_lds",[]> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
  let genVerifyDecl = 1;
}

def FlyROCDL_CopyOpBufferAtomic : FlyROCL_StatefulCopyOp<"CopyOpCDNA3BufferAtomic", "cdna3.buffer_atomic",[]> {
  let parameters = (ins
    "::mlir::fly::AtomicOpAttr":$atomicOp,
    "Type":$valType
  );
  let assemblyFormat = "`<` $atomicOp `,` $valType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::fly::AtomicOpAttr":$atomicOp, "Type":$valType), [{
      return $_get(atomicOp.getContext(), atomicOp, valType);
    }]>
  ];
}

def FlyROCDL_CopyOpLdsReadTranspose : FlyROCL_CopyOp<"CopyOpCDNA4LdsReadTranspose", "cdna4.lds_read_trans",[]> {
  let parameters = (ins
    "int32_t":$transGranularity, "int32_t":$bitSize
  );
  let assemblyFormat = "`<` `trans` `=` $transGranularity `` `b` `,` $bitSize  `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// gfx1250 TDM (Tensor Data Mover) copy atoms
//===----------------------------------------------------------------------===//
//
// These atoms encapsulate the ``rocdl.tensor.load.to.lds`` /
// ``rocdl.tensor.store.from.lds`` descriptor-driven DMA on gfx1250.
//
// The tile / padding / number-of-issuing-warps metadata is captured as
// *static* type parameters, while per-issue dynamic bits (workgroup
// multicast mask, cache policy, per-wave gmem / LDS byte offsets, and
// — for gather mode — the row index vector) flow through the stateful
// atom value by ``atom.set_value``.
//
// ``emitAtomCall`` assembles GROUP0/GROUP1(/GROUP2/GROUP3) vectors from
// the memref operands + atom state and issues the underlying ROCDL op.
// This moves all the descriptor bitfield packing that used to live in
// Python (``python/flydsl/expr/rocdl/tdm_ops.py``) into the compiler.

def FlyROCDL_CopyOpTDMLoad2D : FlyROCL_StatefulCopyOp<
    "CopyOpGFX1250TDMLoad2D", "gfx1250.tdm_load_2d", []> {
  let parameters = (ins
    "int32_t":$elemBits,
    "int32_t":$tileDim0,
    "int32_t":$tileDim1,
    "int32_t":$padInterval,
    "int32_t":$padAmount,
    "int32_t":$numWarps,
    "bool":$atomicBarrierEnable
  );
  let assemblyFormat = [{
    `<` `elem_bits` `=` $elemBits `,`
        `tile` `=` `(` $tileDim0 `,` $tileDim1 `)` `,`
        `pad` `=` `(` $padInterval `,` $padAmount `)` `,`
        `num_warps` `=` $numWarps `,`
        `abe` `=` $atomicBarrierEnable `>`
  }];
  let genVerifyDecl = 1;
}

def FlyROCDL_CopyOpTDMStore2D : FlyROCL_StatefulCopyOp<
    "CopyOpGFX1250TDMStore2D", "gfx1250.tdm_store_2d", []> {
  let parameters = (ins
    "int32_t":$elemBits,
    "int32_t":$tileDim0,
    "int32_t":$tileDim1,
    "int32_t":$padInterval,
    "int32_t":$padAmount,
    "int32_t":$numWarps,
    "bool":$atomicBarrierEnable
  );
  let assemblyFormat = [{
    `<` `elem_bits` `=` $elemBits `,`
        `tile` `=` `(` $tileDim0 `,` $tileDim1 `)` `,`
        `pad` `=` `(` $padInterval `,` $padAmount `)` `,`
        `num_warps` `=` $numWarps `,`
        `abe` `=` $atomicBarrierEnable `>`
  }];
  let genVerifyDecl = 1;
}

def FlyROCDL_CopyOpTDMGather : FlyROCL_StatefulCopyOp<
    "CopyOpGFX1250TDMGather", "gfx1250.tdm_gather", []> {
  let parameters = (ins
    "int32_t":$elemBits,
    "int32_t":$rowWidth,
    "int32_t":$indexSize,
    "int32_t":$maxIndices,
    "int32_t":$padInterval,
    "int32_t":$padAmount,
    "bool":$isStore
  );
  let assemblyFormat = [{
    `<` `elem_bits` `=` $elemBits `,`
        `row_width` `=` $rowWidth `,`
        `index_size` `=` $indexSize `,`
        `max_indices` `=` $maxIndices `,`
        `pad` `=` `(` $padInterval `,` $padAmount `)` `,`
        `is_store` `=` $isStore `>`
  }];
  let genVerifyDecl = 1;
}

#endif // FLYROCDL_COPYATOM
