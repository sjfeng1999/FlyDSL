// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/TiledOpUtils.h"

#include "DLTensorAdaptor.h"

#include <cstdint>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace ::mlir;
using namespace ::mlir::fly;

namespace {

// Unwrap a PyConcreteType to the underlying C++ MLIR type.
template <typename CppTy, typename PySelf>
CppTy unwrapSelf(PySelf &self) {
  return cast<CppTy>(unwrap(static_cast<MlirType>(self)));
}

struct IntTupleAttrBuilder {
  MLIRContext *ctx;
  std::vector<nb::handle> dyncElems{};

  IntTupleAttrBuilder(MLIRContext *ctx) : ctx(ctx) {}

  void clear() { dyncElems.clear(); }

  IntTupleAttr operator()(nb::handle args) {
    if (PyTuple_Check(args.ptr())) {
      SmallVector<Attribute> elements;
      for (auto item : args) {
        elements.push_back((*this)(item));
      }
      return IntTupleAttr::get(ArrayAttr::get(ctx, elements));
    } else if (PyLong_Check(args.ptr())) {
      int32_t cInt = PyLong_AsLong(args.ptr());
      return IntTupleAttr::get(IntAttr::getStatic(ctx, cInt));
    } else if (args.is_none()) {
      return IntTupleAttr::getLeafNone(ctx);
    } else {
      if (!nb::hasattr(args, "_CAPIPtr")) {
        throw std::invalid_argument("Expected I32, got: " +
                                    std::string(nb::str(nb::type_name(args)).c_str()));
      }
      dyncElems.push_back(args);
      return IntTupleAttr::get(IntAttr::getDynamic(ctx));
    }
  }
};

int32_t rank(MlirValue int_or_tuple) {
  Value val = unwrap(int_or_tuple);
  Type ty = val.getType();
  if (auto t = dyn_cast<IntTupleType>(ty))
    return t.getAttr().rank();
  if (auto t = dyn_cast<LayoutType>(ty))
    return t.getAttr().rank();
  if (auto t = dyn_cast<ComposedLayoutType>(ty))
    return t.getAttr().rank();
  if (auto t = dyn_cast<CoordTensorType>(ty))
    return cast<NestedAttrInterface>(t.getLayout()).rank();
  if (auto t = dyn_cast<fly::MemRefType>(ty))
    return cast<NestedAttrInterface>(t.getLayout()).rank();
  throw std::invalid_argument("Unsupported type for rank()");
}

int32_t depth(MlirValue int_or_tuple) {
  Value val = unwrap(int_or_tuple);
  Type ty = val.getType();
  if (auto t = dyn_cast<IntTupleType>(ty))
    return t.getAttr().depth();
  if (auto t = dyn_cast<LayoutType>(ty))
    return t.getAttr().depth();
  if (auto t = dyn_cast<ComposedLayoutType>(ty))
    return t.getAttr().depth();
  if (auto t = dyn_cast<CoordTensorType>(ty))
    return cast<NestedAttrInterface>(t.getLayout()).depth();
  if (auto t = dyn_cast<fly::MemRefType>(ty))
    return cast<NestedAttrInterface>(t.getLayout()).depth();
  throw std::invalid_argument("Unsupported type for depth()");
}

/// Convert nb::handle (Python int|tuple|IntTupleType) to IntTupleAttr.
IntTupleAttr toIntTupleAttr(nb::handle h, MLIRContext *ctx) {
  if (nb::hasattr(h, MLIR_PYTHON_CAPI_PTR_ATTR)) {
    auto capsule = nb::cast<nb::capsule>(h.attr(MLIR_PYTHON_CAPI_PTR_ATTR));
    MlirType mlirTy = mlirPythonCapsuleToType(capsule.ptr());
    auto intTupleType = dyn_cast<IntTupleType>(unwrap(mlirTy));
    if (!intTupleType)
      throw std::invalid_argument("Expected IntTupleType, got other MlirType");
    return intTupleType.getAttr();
  }
  IntTupleAttrBuilder builder{ctx};
  return builder(h);
}

} // namespace

// =============================================================================
// PyConcreteType definitions in the MLIR Python domain
// =============================================================================

// Generate isaFunction / getTypeIdFunction lambdas for PyConcreteType.
// Must use fully-qualified names because we are inside mlir::python::mlir::fly.
#define FLY_ISA_AND_TYPEID(CppType)                                                                \
  static constexpr IsAFunctionTy isaFunction =                                                     \
      +[](MlirType type) { return ::mlir::isa<CppType>(unwrap(type)); };                          \
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =                                         \
      +[]() { return wrap(CppType::getTypeID()); }

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace fly {

// ---------------------------------------------------------------------------
// IntTupleType
// ---------------------------------------------------------------------------
struct PyIntTupleType : PyConcreteType<PyIntTupleType> {
  FLY_ISA_AND_TYPEID(::mlir::fly::IntTupleType);
  static constexpr const char *pyClassName = "IntTupleType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::handle int_or_tuple, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          IntTupleAttrBuilder builder{ctx};
          auto attr = builder(int_or_tuple);
          return PyIntTupleType(context->getRef(), wrap(IntTupleType::get(attr)));
        },
        "int_or_tuple"_a, nb::kw_only(), "context"_a = nb::none(),
        // clang-format off
        nb::sig("def get(int_or_tuple, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None)"),
        // clang-format on
        "Create an IntTupleType from Python int or tuple");

    c.def_prop_ro("rank",
                  [](PyIntTupleType &self) { return unwrapSelf<IntTupleType>(self).rank(); });
    c.def_prop_ro("depth",
                  [](PyIntTupleType &self) { return unwrapSelf<IntTupleType>(self).depth(); });
    c.def_prop_ro("is_leaf",
                  [](PyIntTupleType &self) { return unwrapSelf<IntTupleType>(self).isLeaf(); });
    c.def_prop_ro("is_static",
                  [](PyIntTupleType &self) { return unwrapSelf<IntTupleType>(self).isStatic(); });
    c.def_prop_ro("static_value", [](PyIntTupleType &self) {
      auto ty = unwrapSelf<IntTupleType>(self);
      assert(ty.isLeaf() && ty.isStatic());
      return ty.getAttr().getLeafAsInt().getValue();
    });
  }
};

// ---------------------------------------------------------------------------
// LayoutType
// ---------------------------------------------------------------------------
struct PyLayoutType : PyConcreteType<PyLayoutType> {
  FLY_ISA_AND_TYPEID(::mlir::fly::LayoutType);
  static constexpr const char *pyClassName = "LayoutType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](nb::handle shape, nb::handle stride, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto shapeAttr = toIntTupleAttr(shape, ctx);
          auto strideAttr = toIntTupleAttr(stride, ctx);
          auto layoutAttr = LayoutAttr::get(ctx, shapeAttr, strideAttr);
          return PyLayoutType(context->getRef(), wrap(LayoutType::get(layoutAttr)));
        },
        "shape"_a, "stride"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a LayoutType with shape and stride");

    c.def_prop_ro("shape", [](PyLayoutType &self) -> MlirType {
      return wrap(IntTupleType::get(unwrapSelf<LayoutType>(self).getAttr().getShape()));
    });
    c.def_prop_ro("stride", [](PyLayoutType &self) -> MlirType {
      return wrap(IntTupleType::get(unwrapSelf<LayoutType>(self).getAttr().getStride()));
    });
    c.def_prop_ro("rank",
                  [](PyLayoutType &self) { return unwrapSelf<LayoutType>(self).rank(); });
    c.def_prop_ro("depth",
                  [](PyLayoutType &self) { return unwrapSelf<LayoutType>(self).depth(); });
    c.def_prop_ro("is_leaf",
                  [](PyLayoutType &self) { return unwrapSelf<LayoutType>(self).isLeaf(); });
    c.def_prop_ro("is_static",
                  [](PyLayoutType &self) { return unwrapSelf<LayoutType>(self).isStatic(); });
    c.def_prop_ro("is_static_shape",
                  [](PyLayoutType &self) { return unwrapSelf<LayoutType>(self).isStaticShape(); });
    c.def_prop_ro("is_static_stride", [](PyLayoutType &self) {
      return unwrapSelf<LayoutType>(self).isStaticStride();
    });
  }
};

// ---------------------------------------------------------------------------
// SwizzleType
// ---------------------------------------------------------------------------
struct PySwizzleType : PyConcreteType<PySwizzleType> {
  FLY_ISA_AND_TYPEID(::mlir::fly::SwizzleType);
  static constexpr const char *pyClassName = "SwizzleType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t mask, int32_t base, int32_t shift, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto attr = SwizzleAttr::get(ctx, mask, base, shift);
          return PySwizzleType(context->getRef(), wrap(SwizzleType::get(attr)));
        },
        "mask"_a, "base"_a, "shift"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a SwizzleType");

    c.def_prop_ro("mask", [](PySwizzleType &self) {
      return unwrapSelf<SwizzleType>(self).getAttr().getMask();
    });
    c.def_prop_ro("base", [](PySwizzleType &self) {
      return unwrapSelf<SwizzleType>(self).getAttr().getBase();
    });
    c.def_prop_ro("shift", [](PySwizzleType &self) {
      return unwrapSelf<SwizzleType>(self).getAttr().getShift();
    });
  }
};

// ---------------------------------------------------------------------------
// PointerType
// ---------------------------------------------------------------------------
struct PyPointerType : PyConcreteType<PyPointerType> {
  FLY_ISA_AND_TYPEID(::mlir::fly::PointerType);
  static constexpr const char *pyClassName = "PointerType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elemTyObj, std::optional<int32_t> addressSpace, std::optional<int32_t> alignment,
           DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto elemType = unwrap(static_cast<MlirType>(elemTyObj));

          auto addr = AddressSpace::Register;
          if (addressSpace.has_value())
            addr = static_cast<AddressSpace>(addressSpace.value());

          int32_t alignSize =
              alignment.value_or(AlignAttr::getTrivialAlignment(elemType).getAlignment());
          int32_t elemByte = (elemType.getIntOrFloatBitWidth() + 7) / 8;
          if (alignSize <= 0 || alignSize % elemByte != 0)
            throw std::invalid_argument(
                "alignment must be a positive multiple of element byte size (" +
                std::to_string(elemByte) + "), got " + std::to_string(alignSize));

          return PyPointerType(context->getRef(),
                               wrap(PointerType::get(elemType, AddressSpaceAttr::get(ctx, addr),
                                                     AlignAttr::get(ctx, alignSize))));
        },
        "elem_ty"_a, "address_space"_a = nb::none(), "alignment"_a = nb::none(), nb::kw_only(),
        "context"_a = nb::none(), "Create a PointerType with element type and address space");

    c.def_prop_ro("element_type", [](PyPointerType &self) -> MlirType {
      return wrap(unwrapSelf<PointerType>(self).getElemTy());
    });
    c.def_prop_ro("address_space", [](PyPointerType &self) -> int32_t {
      return static_cast<int32_t>(unwrapSelf<PointerType>(self).getAddressSpace().getValue());
    });
    c.def_prop_ro("alignment", [](PyPointerType &self) -> int32_t {
      return unwrapSelf<PointerType>(self).getAlignment().getAlignment();
    });
    c.def_prop_ro("swizzle", [](PyPointerType &self) -> MlirType {
      return wrap(SwizzleType::get(unwrapSelf<PointerType>(self).getSwizzle()));
    });
  }
};

// ---------------------------------------------------------------------------
// MemRefType
// ---------------------------------------------------------------------------
struct PyMemRefType : PyConcreteType<PyMemRefType> {
  FLY_ISA_AND_TYPEID(::mlir::fly::MemRefType);
  static constexpr const char *pyClassName = "MemRefType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elemTyObj, PyType &layoutObj, std::optional<int32_t> addressSpace,
           std::optional<int32_t> alignment, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          auto layoutType =
              ::mlir::dyn_cast<LayoutType>(unwrap(static_cast<MlirType>(layoutObj)));
          if (!layoutType)
            throw std::invalid_argument("layout must be a LayoutType");

          auto addr = AddressSpace::Register;
          if (addressSpace.has_value())
            addr = static_cast<AddressSpace>(addressSpace.value());

          auto elemType = unwrap(static_cast<MlirType>(elemTyObj));
          int32_t alignSize =
              alignment.value_or(AlignAttr::getTrivialAlignment(elemType).getAlignment());
          int32_t elemByte = (elemType.getIntOrFloatBitWidth() + 7) / 8;
          if (alignSize <= 0 || alignSize % elemByte != 0)
            throw std::invalid_argument(
                "alignment must be a positive multiple of element byte size (" +
                std::to_string(elemByte) + "), got " + std::to_string(alignSize));

          return PyMemRefType(
              context->getRef(),
              wrap(::mlir::fly::MemRefType::get(elemType, AddressSpaceAttr::get(ctx, addr),
                                                   layoutType.getAttr(),
                                                   AlignAttr::get(ctx, alignSize))));
        },
        "elem_ty"_a, "layout"_a, "address_space"_a = 0, "alignment"_a = nb::none(), nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MemRefType with element type, layout, address space and "
        "alignment");

    c.def_prop_ro("element_type", [](PyMemRefType &self) -> MlirType {
      return wrap(unwrapSelf<::mlir::fly::MemRefType>(self).getElemTy());
    });
    c.def_prop_ro("layout", [](PyMemRefType &self) -> MlirType {
      auto memrefType = unwrapSelf<::mlir::fly::MemRefType>(self);
      ::mlir::Attribute layout = memrefType.getLayout();
      if (auto la = ::mlir::dyn_cast<LayoutAttr>(layout))
        return wrap(LayoutType::get(la));
      return wrap(ComposedLayoutType::get(::mlir::cast<ComposedLayoutAttr>(layout)));
    });
    c.def_prop_ro("address_space", [](PyMemRefType &self) -> int32_t {
      return static_cast<int32_t>(
          unwrapSelf<::mlir::fly::MemRefType>(self).getAddressSpace().getValue());
    });
    c.def_prop_ro("alignment", [](PyMemRefType &self) -> int32_t {
      return unwrapSelf<::mlir::fly::MemRefType>(self).getAlignment().getAlignment();
    });
    c.def_prop_ro("swizzle", [](PyMemRefType &self) -> MlirType {
      return wrap(SwizzleType::get(unwrapSelf<::mlir::fly::MemRefType>(self).getSwizzle()));
    });
  }
};

// ---------------------------------------------------------------------------
// CopyOpUniversalCopyType
// ---------------------------------------------------------------------------
struct PyCopyOpUniversalCopyType : PyConcreteType<PyCopyOpUniversalCopyType> {
  FLY_ISA_AND_TYPEID(::mlir::fly::CopyOpUniversalCopyType);
  static constexpr const char *pyClassName = "CopyOpUniversalCopyType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpUniversalCopyType(context->getRef(),
                                           wrap(CopyOpUniversalCopyType::get(ctx, bitSize)));
        },
        "bitSize"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpUniversalCopyType with bit size");

    c.def_prop_ro("bit_size", [](PyCopyOpUniversalCopyType &self) {
      return unwrapSelf<CopyOpUniversalCopyType>(self).getBitSize();
    });
  }
};

// ---------------------------------------------------------------------------
// CopyAtomType
// ---------------------------------------------------------------------------
struct PyCopyAtomType : PyConcreteType<PyCopyAtomType> {
  FLY_ISA_AND_TYPEID(::mlir::fly::CopyAtomType);
  static constexpr const char *pyClassName = "CopyAtomType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &copyOp, int32_t valBits) {
          return PyCopyAtomType(copyOp.getContext(),
                                wrap(CopyAtomType::get(unwrap(static_cast<MlirType>(copyOp)),
                                                       valBits)));
        },
        "copy_op"_a, "val_bits"_a,
        "Create a CopyAtomType with the given copy op type and value bits");

    c.def_prop_ro("copy_op", [](PyCopyAtomType &self) -> MlirType {
      return wrap(unwrapSelf<CopyAtomType>(self).getCopyOp());
    });
    c.def_prop_ro("val_bits",
                  [](PyCopyAtomType &self) { return unwrapSelf<CopyAtomType>(self).getValBits(); });
    c.def_prop_ro("thr_layout", [](PyCopyAtomType &self) -> MlirType {
      return wrap(LayoutType::get(cast<LayoutAttr>(unwrapSelf<CopyAtomType>(self).getThrLayout())));
    });
    c.def_prop_ro("tv_layout_src", [](PyCopyAtomType &self) -> MlirType {
      return wrap(
          LayoutType::get(cast<LayoutAttr>(unwrapSelf<CopyAtomType>(self).getThrValLayoutSrc())));
    });
    c.def_prop_ro("tv_layout_dst", [](PyCopyAtomType &self) -> MlirType {
      return wrap(
          LayoutType::get(cast<LayoutAttr>(unwrapSelf<CopyAtomType>(self).getThrValLayoutDst())));
    });
    c.def_prop_ro("tv_layout_ref", [](PyCopyAtomType &self) -> MlirType {
      return wrap(
          LayoutType::get(cast<LayoutAttr>(unwrapSelf<CopyAtomType>(self).getThrValLayoutRef())));
    });
  }
};

// ---------------------------------------------------------------------------
// MmaAtomUniversalFMAType
// ---------------------------------------------------------------------------
struct PyMmaAtomUniversalFMAType : PyConcreteType<PyMmaAtomUniversalFMAType> {
  FLY_ISA_AND_TYPEID(::mlir::fly::MmaAtomUniversalFMAType);
  static constexpr const char *pyClassName = "MmaAtomUniversalFMAType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elemTyObj, DefaultingPyMlirContext context) {
          return PyMmaAtomUniversalFMAType(
              context->getRef(),
              wrap(MmaAtomUniversalFMAType::get(unwrap(static_cast<MlirType>(elemTyObj)))));
        },
        "elem_ty"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a MmaAtomUniversalFMAType with element type");

    c.def_prop_ro("elem_ty", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      return wrap(unwrapSelf<MmaAtomUniversalFMAType>(self).getElemTy());
    });
    c.def_prop_ro("thr_layout", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      auto ty = unwrapSelf<MmaAtomTypeInterface>(self);
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrLayout())));
    });
    c.def_prop_ro("shape_mnk", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      auto ty = unwrapSelf<MmaAtomTypeInterface>(self);
      return wrap(IntTupleType::get(cast<IntTupleAttr>(ty.getShapeMNK())));
    });
    c.def_prop_ro("tv_layout_a", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      auto ty = unwrapSelf<MmaAtomTypeInterface>(self);
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrValLayoutA())));
    });
    c.def_prop_ro("tv_layout_b", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      auto ty = unwrapSelf<MmaAtomTypeInterface>(self);
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrValLayoutB())));
    });
    c.def_prop_ro("tv_layout_c", [](PyMmaAtomUniversalFMAType &self) -> MlirType {
      auto ty = unwrapSelf<MmaAtomTypeInterface>(self);
      return wrap(LayoutType::get(cast<LayoutAttr>(ty.getThrValLayoutC())));
    });
  }
};

struct PyTiledCopyType : PyConcreteType<PyTiledCopyType> {
  FLY_ISA_AND_TYPEID(::mlir::fly::TiledCopyType);
  static constexpr const char *pyClassName = "TiledCopyType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("copy_atom", [](PyTiledCopyType &self) -> MlirType {
      return wrap(unwrapSelf<TiledCopyType>(self).getCopyAtom());
    });
    c.def_prop_ro("layout_thr_val", [](PyTiledCopyType &self) -> MlirType {
      return wrap(static_cast<Type>(unwrapSelf<TiledCopyType>(self).getLayoutThrVal()));
    });
    c.def_prop_ro("tile_mn", [](PyTiledCopyType &self) -> MlirType {
      return wrap(static_cast<Type>(unwrapSelf<TiledCopyType>(self).getTileMN()));
    });
    c.def_prop_ro("tiled_tv_layout_src", [](PyTiledCopyType &self) -> MlirType {
      auto ty = unwrapSelf<TiledCopyType>(self);
      auto copyAtom = cast<CopyAtomType>(ty.getCopyAtom());
      auto result = tiledCopyGetTiledTVLayoutSrc(
          copyAtom, ty.getLayoutThrVal().getAttr(), ty.getTileMN().getAttr());
      return wrap(LayoutType::get(result));
    });
    c.def_prop_ro("tiled_tv_layout_dst", [](PyTiledCopyType &self) -> MlirType {
      auto ty = unwrapSelf<TiledCopyType>(self);
      auto copyAtom = cast<CopyAtomType>(ty.getCopyAtom());
      auto result = tiledCopyGetTiledTVLayoutDst(
          copyAtom, ty.getLayoutThrVal().getAttr(), ty.getTileMN().getAttr());
      return wrap(LayoutType::get(result));
    });
  }
};

struct PyTiledMmaType : PyConcreteType<PyTiledMmaType> {
  FLY_ISA_AND_TYPEID(::mlir::fly::TiledMmaType);
  static constexpr const char *pyClassName = "TiledMmaType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_prop_ro("mma_atom", [](PyTiledMmaType &self) -> MlirType {
      return wrap(unwrapSelf<TiledMmaType>(self).getMmaAtom());
    });
    c.def_prop_ro("atom_layout", [](PyTiledMmaType &self) -> MlirType {
      return wrap(static_cast<Type>(unwrapSelf<TiledMmaType>(self).getAtomLayout()));
    });
    c.def_prop_ro("permutation", [](PyTiledMmaType &self) -> MlirType {
      return wrap(static_cast<Type>(unwrapSelf<TiledMmaType>(self).getPermutation()));
    });
    c.def_prop_ro("tile_size_mnk", [](PyTiledMmaType &self) -> MlirType {
      auto ty = unwrapSelf<TiledMmaType>(self);
      auto mmaAtom = cast<MmaAtomTypeInterface>(ty.getMmaAtom());
      auto result = tiledMmaGetTileSizeMNK(
          mmaAtom, ty.getAtomLayout().getAttr(), ty.getPermutation().getAttr());
      return wrap(IntTupleType::get(result));
    });
    c.def_prop_ro("thr_layout_vmnk", [](PyTiledMmaType &self) -> MlirType {
      auto ty = unwrapSelf<TiledMmaType>(self);
      auto mmaAtom = cast<MmaAtomTypeInterface>(ty.getMmaAtom());
      auto result = tiledMmaGetThrLayoutVMNK(mmaAtom, ty.getAtomLayout().getAttr());
      return wrap(LayoutType::get(result));
    });
    c.def_prop_ro("tiled_tv_layout_a", [](PyTiledMmaType &self) -> MlirType {
      auto ty = unwrapSelf<TiledMmaType>(self);
      auto mmaAtom = cast<MmaAtomTypeInterface>(ty.getMmaAtom());
      auto result = tiledMmaGetTiledTVLayout(
          mmaAtom, ty.getAtomLayout().getAttr(), ty.getPermutation().getAttr(), MmaOperand::A);
      return wrap(LayoutType::get(result));
    });
    c.def_prop_ro("tiled_tv_layout_b", [](PyTiledMmaType &self) -> MlirType {
      auto ty = unwrapSelf<TiledMmaType>(self);
      auto mmaAtom = cast<MmaAtomTypeInterface>(ty.getMmaAtom());
      auto result = tiledMmaGetTiledTVLayout(
          mmaAtom, ty.getAtomLayout().getAttr(), ty.getPermutation().getAttr(), MmaOperand::B);
      return wrap(LayoutType::get(result));
    });
    c.def_prop_ro("tiled_tv_layout_c", [](PyTiledMmaType &self) -> MlirType {
      auto ty = unwrapSelf<TiledMmaType>(self);
      auto mmaAtom = cast<MmaAtomTypeInterface>(ty.getMmaAtom());
      auto result = tiledMmaGetTiledTVLayout(
          mmaAtom, ty.getAtomLayout().getAttr(), ty.getPermutation().getAttr(), MmaOperand::C);
      return wrap(LayoutType::get(result));
    });
  }
};

} // namespace fly
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

// =============================================================================
// Module definition
// =============================================================================

NB_MODULE(_fly, m) {
  m.doc() = "MLIR Python FlyDSL Extension";

  // -------------------------------------------------------------------------
  // DLTensorAdaptor (standalone, not an MLIR type)
  // -------------------------------------------------------------------------
  using DLTensorAdaptor = utils::DLTensorAdaptor;

  nb::class_<DLTensorAdaptor>(m, "DLTensorAdaptor")
      .def(nb::init<nb::object, std::optional<int32_t>, bool>(), "dlpack_capsule"_a,
           "alignment"_a = nb::none(), "use_32bit_stride"_a = false,
           "Create a DLTensorAdaptor from a DLPack capsule. "
           "If alignment is None, defaults to element size in bytes (minimum "
           "1). ")
      .def_prop_ro("shape", &DLTensorAdaptor::getShape, "Get tensor shape as tuple")
      .def_prop_ro("stride", &DLTensorAdaptor::getStride, "Get tensor stride as tuple")
      .def_prop_ro("data_ptr", &DLTensorAdaptor::getDataPtr, "Get data pointer as int64")
      .def_prop_ro("address_space", &DLTensorAdaptor::getAddressSpace,
                   "Get address space (0=host, 1=device)")
      .def("size_in_bytes", &DLTensorAdaptor::getSizeInBytes, "Get total size in bytes")
      .def("build_memref_desc", &DLTensorAdaptor::buildMemRefDesc,
           "Build memref descriptor based on current dynamic marks")
      .def("get_memref_type", &DLTensorAdaptor::getMemRefType,
           "Get fly.memref MLIR type based on current dynamic marks")
      .def("get_c_pointers", &DLTensorAdaptor::getCPointers, "Get list of c pointers")
      .def("mark_layout_dynamic", &DLTensorAdaptor::markLayoutDynamic, "leading_dim"_a = -1,
           "divisibility"_a = 1, "Mark entire layout as dynamic except leading dim stride")
      .def("use_32bit_stride", &DLTensorAdaptor::use32BitStride, "use_32bit_stride"_a,
           "Decide whether to use 32-bit stride");

  // -------------------------------------------------------------------------
  // Module-level helper functions
  // -------------------------------------------------------------------------
  m.def(
      "infer_int_tuple_type",
      [](nb::handle int_or_tuple, MlirContext context) {
        MLIRContext *ctx = unwrap(context);
        IntTupleAttrBuilder builder{ctx};
        auto attr = builder(int_or_tuple);
        return std::make_pair(wrap(IntTupleType::get(attr)), builder.dyncElems);
      },
      "int_or_tuple"_a, "context"_a = nb::none(),
      // clang-format off
      nb::sig("def infer_int_tuple_type(int_or_tuple, context: " MAKE_MLIR_PYTHON_QUALNAME("ir.Context") " | None = None)"),
      // clang-format on
      "infer IntTupleType for given input");

  m.def("rank", &rank, "int_or_tuple"_a,
        nb::sig("def rank(int_or_tuple: " MAKE_MLIR_PYTHON_QUALNAME("ir.Value") ") -> int"));
  m.def("depth", &depth, "int_or_tuple"_a,
        nb::sig("def depth(int_or_tuple: " MAKE_MLIR_PYTHON_QUALNAME("ir.Value") ") -> int"));

  // -------------------------------------------------------------------------
  // Bind Fly dialect types (PyConcreteType pattern)
  // -------------------------------------------------------------------------
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyIntTupleType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyLayoutType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PySwizzleType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyPointerType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyMemRefType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyCopyOpUniversalCopyType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyCopyAtomType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyMmaAtomUniversalFMAType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyTiledCopyType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly::PyTiledMmaType::bind(m);
}
