/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace detail {
struct VectorTransferOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual StringRef getMaskedAttrName() const = 0;
    virtual StringRef getPermutationMapAttrName() const = 0;
    virtual bool isMaskedDim(::mlir::Operation *tablegen_opaque_val, unsigned dim) const = 0;
    virtual Value memref(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Value vector(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual ValueRange indices(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual AffineMap permutation_map(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Optional<ArrayAttr> masked(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual VectorType getVectorType(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual unsigned getTransferRank(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual unsigned getLeadingMemRefRank(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual bool hasMaskedDim(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual void zipResultAndIndexing(::mlir::Operation *tablegen_opaque_val, llvm::function_ref<void(int64_t, int64_t)> fun) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    StringRef getMaskedAttrName() const final {
      return ConcreteOp::getMaskedAttrName();
    }
    StringRef getPermutationMapAttrName() const final {
      return ConcreteOp::getPermutationMapAttrName();
    }
    bool isMaskedDim(::mlir::Operation *tablegen_opaque_val, unsigned dim) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isMaskedDim(dim);
    }
    Value memref(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).memref();
    }
    Value vector(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).vector();
    }
    ValueRange indices(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).indices();
    }
    AffineMap permutation_map(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).permutation_map();
    }
    Optional<ArrayAttr> masked(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).masked();
    }
    MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefType();
    }
    VectorType getVectorType(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getVectorType();
    }
    unsigned getTransferRank(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTransferRank();
    }
    unsigned getLeadingMemRefRank(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLeadingMemRefRank();
    }
    bool hasMaskedDim(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasMaskedDim();
    }
    void zipResultAndIndexing(::mlir::Operation *tablegen_opaque_val, llvm::function_ref<void(int64_t, int64_t)> fun) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).zipResultAndIndexing(fun);
    }
  };
};
} // end namespace detail
class VectorTransferOpInterface : public ::mlir::OpInterface<VectorTransferOpInterface, detail::VectorTransferOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<VectorTransferOpInterface, detail::VectorTransferOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct VectorTransferOpInterfaceTrait : public ::mlir::OpInterface<VectorTransferOpInterface, detail::VectorTransferOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static StringRef getMaskedAttrName() {
      return "masked";
    }
    static StringRef getPermutationMapAttrName() {
      return "permutation_map";
    }
    bool isMaskedDim(unsigned dim) {
      return !(*static_cast<ConcreteOp *>(this)).masked() ||
          (*static_cast<ConcreteOp *>(this)).masked()->template cast<ArrayAttr>()[dim]
                        .template cast<BoolAttr>().getValue();
    }
    MemRefType getMemRefType() {
      return (*static_cast<ConcreteOp *>(this)).memref().getType().template cast<MemRefType>();
    }
    VectorType getVectorType() {
      return (*static_cast<ConcreteOp *>(this)).vector().getType().template dyn_cast<VectorType>();
    }
    unsigned getTransferRank() {
      return (*static_cast<ConcreteOp *>(this)).permutation_map().getNumResults();
    }
    unsigned getLeadingMemRefRank() {
      return (*static_cast<ConcreteOp *>(this)).getMemRefType().getRank() - (*static_cast<ConcreteOp *>(this)).getTransferRank();
    }
    bool hasMaskedDim() {
      for (unsigned idx = 0, e = (*static_cast<ConcreteOp *>(this)).getTransferRank(); idx < e; ++idx)
          if ((*static_cast<ConcreteOp *>(this)).isMaskedDim(idx))
            return true;
        return false;
    }
    void zipResultAndIndexing(llvm::function_ref<void(int64_t, int64_t)> fun) {
      for (int64_t resultIdx = 0,
                   indicesIdx = (*static_cast<ConcreteOp *>(this)).getLeadingMemRefRank(),
                   eResult = (*static_cast<ConcreteOp *>(this)).getTransferRank();
           resultIdx < eResult;
           ++resultIdx, ++indicesIdx)
        fun(resultIdx, indicesIdx);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public VectorTransferOpInterfaceTrait<ConcreteOp> {};
  StringRef getMaskedAttrName();
  StringRef getPermutationMapAttrName();
  bool isMaskedDim(unsigned dim);
  Value memref();
  Value vector();
  ValueRange indices();
  AffineMap permutation_map();
  Optional<ArrayAttr> masked();
  MemRefType getMemRefType();
  VectorType getVectorType();
  unsigned getTransferRank();
  unsigned getLeadingMemRefRank();
  bool hasMaskedDim();
  void zipResultAndIndexing(llvm::function_ref<void(int64_t, int64_t)> fun);
};
} // namespace mlir
namespace mlir {
namespace detail {
struct VectorUnrollOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual Optional<SmallVector<int64_t, 4>> getShapeForUnroll(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Optional<SmallVector<int64_t, 4>> getShapeForUnroll(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapeForUnroll();
    }
  };
};
} // end namespace detail
class VectorUnrollOpInterface : public ::mlir::OpInterface<VectorUnrollOpInterface, detail::VectorUnrollOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<VectorUnrollOpInterface, detail::VectorUnrollOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct VectorUnrollOpInterfaceTrait : public ::mlir::OpInterface<VectorUnrollOpInterface, detail::VectorUnrollOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Optional<SmallVector<int64_t, 4>> getShapeForUnroll() {
      assert((*static_cast<ConcreteOp *>(this)).getOperation()->getNumResults() == 1);
        auto vt = (*static_cast<ConcreteOp *>(this)).getResult().getType().
          template dyn_cast<VectorType>();
        if (!vt)
          return None;
        SmallVector<int64_t, 4> res(vt.getShape().begin(), vt.getShape().end());
        return res;
    }
  };
  template <typename ConcreteOp>
  struct Trait : public VectorUnrollOpInterfaceTrait<ConcreteOp> {};
  Optional<SmallVector<int64_t, 4>> getShapeForUnroll();
};
} // namespace mlir
