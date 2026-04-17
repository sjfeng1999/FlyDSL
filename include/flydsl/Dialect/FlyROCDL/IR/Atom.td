// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_ATOM
#define FLYROCDL_ATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_AtomStateField : I32EnumAttr<"AtomStateField", "", [
  I32EnumAttrCase<"Soffset",          0, "soffset">,
  I32EnumAttrCase<"ImmOffset",        1, "imm_offset">,
  I32EnumAttrCase<"ScaleA",           2, "scale_a">,
  I32EnumAttrCase<"ScaleB",           3, "scale_b">,
  I32EnumAttrCase<"WorkgroupMask",    4, "workgroup_mask">,
  I32EnumAttrCase<"CachePolicy",      5, "cache_policy">,
  I32EnumAttrCase<"GmemByteOffset",   6, "gmem_byte_offset">,
  I32EnumAttrCase<"LdsByteOffset",    7, "lds_byte_offset">,
  I32EnumAttrCase<"TensorDim0",       8, "tensor_dim0">,
  I32EnumAttrCase<"TensorDim1",       9, "tensor_dim1">,
  I32EnumAttrCase<"Stride0",         10, "stride0">,
  I32EnumAttrCase<"GatherRowIndices",11, "gather_row_indices">,
  I32EnumAttrCase<"GatherIndexCount",12, "gather_index_count">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyROCDL_Dialect.cppNamespace;
}

include "flydsl/Dialect/FlyROCDL/IR/CopyAtom.td"
include "flydsl/Dialect/FlyROCDL/IR/MmaAtom.td"

#endif // FLYROCDL_ATOM
