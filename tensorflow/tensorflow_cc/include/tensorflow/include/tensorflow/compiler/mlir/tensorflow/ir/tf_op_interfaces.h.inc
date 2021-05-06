/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct ContractionFusableInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual Optional<ContractionFusion> GetContractionFusion(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Optional<ContractionFusion> GetContractionFusion(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetContractionFusion();
    }
  };
};
} // end namespace detail
class ContractionFusableInterface : public ::mlir::OpInterface<ContractionFusableInterface, detail::ContractionFusableInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ContractionFusableInterface, detail::ContractionFusableInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ContractionFusableInterfaceTrait : public ::mlir::OpInterface<ContractionFusableInterface, detail::ContractionFusableInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public ContractionFusableInterfaceTrait<ConcreteOp> {};
  Optional<ContractionFusion> GetContractionFusion();
};
namespace detail {
struct FoldOperandsTransposeInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual SmallVector<unsigned, 4> GetLayoutDependentArgs(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual SmallVector<unsigned, 4> GetLayoutDependentResults(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual LogicalResult FoldOperandsPermutation(::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    SmallVector<unsigned, 4> GetLayoutDependentArgs(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentArgs();
    }
    SmallVector<unsigned, 4> GetLayoutDependentResults(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentResults();
    }
    LogicalResult FoldOperandsPermutation(::mlir::Operation *tablegen_opaque_val, ArrayRef<int64_t> permutation) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).FoldOperandsPermutation(permutation);
    }
  };
};
} // end namespace detail
class FoldOperandsTransposeInterface : public ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct FoldOperandsTransposeInterfaceTrait : public ::mlir::OpInterface<FoldOperandsTransposeInterface, detail::FoldOperandsTransposeInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return VerifyFoldOperandsTransposeInterface(op);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public FoldOperandsTransposeInterfaceTrait<ConcreteOp> {};
  SmallVector<unsigned, 4> GetLayoutDependentArgs();
  SmallVector<unsigned, 4> GetLayoutDependentResults();
  LogicalResult FoldOperandsPermutation(ArrayRef<int64_t> permutation);
};
namespace detail {
struct LayoutSensitiveInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual StringRef data_format(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual SmallVector<unsigned, 4> GetLayoutDependentArgs(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual SmallVector<unsigned, 4> GetLayoutDependentResults(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual StringRef GetOptimalLayout(::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices) const = 0;
    virtual LogicalResult UpdateDataFormat(::mlir::Operation *tablegen_opaque_val, StringRef data_format) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    StringRef data_format(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).data_format();
    }
    SmallVector<unsigned, 4> GetLayoutDependentArgs(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentArgs();
    }
    SmallVector<unsigned, 4> GetLayoutDependentResults(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetLayoutDependentResults();
    }
    StringRef GetOptimalLayout(::mlir::Operation *tablegen_opaque_val, const RuntimeDevices& devices) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetOptimalLayout(devices);
    }
    LogicalResult UpdateDataFormat(::mlir::Operation *tablegen_opaque_val, StringRef data_format) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).UpdateDataFormat(data_format);
    }
  };
};
} // end namespace detail
class LayoutSensitiveInterface : public ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct LayoutSensitiveInterfaceTrait : public ::mlir::OpInterface<LayoutSensitiveInterface, detail::LayoutSensitiveInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return VerifyLayoutSensitiveInterface(op);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public LayoutSensitiveInterfaceTrait<ConcreteOp> {};
  StringRef data_format();
  SmallVector<unsigned, 4> GetLayoutDependentArgs();
  SmallVector<unsigned, 4> GetLayoutDependentResults();
  StringRef GetOptimalLayout(const RuntimeDevices& devices);
  LogicalResult UpdateDataFormat(StringRef data_format);
};
namespace detail {
struct ResourceHandleAllocatorInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual ResourceHandleValueAndId GetResourceHandleValueAndId(::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    ResourceHandleValueAndId GetResourceHandleValueAndId(::mlir::Operation *tablegen_opaque_val, llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).GetResourceHandleValueAndId(resource_handle_id_map, next_id);
    }
  };
};
} // end namespace detail
class ResourceHandleAllocatorInterface : public ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct ResourceHandleAllocatorInterfaceTrait : public ::mlir::OpInterface<ResourceHandleAllocatorInterface, detail::ResourceHandleAllocatorInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public ResourceHandleAllocatorInterfaceTrait<ConcreteOp> {};
  ResourceHandleValueAndId GetResourceHandleValueAndId(llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map, int64_t& next_id);
};
