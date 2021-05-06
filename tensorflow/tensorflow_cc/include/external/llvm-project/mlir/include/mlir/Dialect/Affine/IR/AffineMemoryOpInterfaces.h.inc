/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct AffineReadOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual Value getMemRef(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Operation::operand_range getMapOperands(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual AffineMap getAffineMap(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual NamedAttribute getAffineMapAttrForMemRef(::mlir::Operation *tablegen_opaque_val, Value memref) const = 0;
    virtual Value getValue(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Value getMemRef(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef();
    }
    MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefType();
    }
    Operation::operand_range getMapOperands(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapOperands();
    }
    AffineMap getAffineMap(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMap();
    }
    NamedAttribute getAffineMapAttrForMemRef(::mlir::Operation *tablegen_opaque_val, Value memref) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMapAttrForMemRef(memref);
    }
    Value getValue(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getValue();
    }
  };
};
} // end namespace detail
class AffineReadOpInterface : public ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct AffineReadOpInterfaceTrait : public ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Value getMemRef() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getMemRefOperandIndex());
    }
    MemRefType getMemRefType() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getMemRef().getType().template cast<MemRefType>();
    }
    Operation::operand_range getMapOperands() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return llvm::drop_begin(op.getOperands(), 1);
    }
    AffineMap getAffineMap() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAffineMapAttr().getValue();
    }
    NamedAttribute getAffineMapAttrForMemRef(Value memref) {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        assert(memref == getMemRef());
        return {Identifier::get(op.getMapAttrName(), op.getContext()),
                op.getAffineMapAttr()};
    }
    Value getValue() {
      return cast<ConcreteOp>(this->getOperation());
    }
  };
  template <typename ConcreteOp>
  struct Trait : public AffineReadOpInterfaceTrait<ConcreteOp> {};
  Value getMemRef();
  MemRefType getMemRefType();
  Operation::operand_range getMapOperands();
  AffineMap getAffineMap();
  NamedAttribute getAffineMapAttrForMemRef(Value memref);
  Value getValue();
};
namespace detail {
struct AffineWriteOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual Value getMemRef(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Operation::operand_range getMapOperands(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual AffineMap getAffineMap(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual NamedAttribute getAffineMapAttrForMemRef(::mlir::Operation *tablegen_opaque_val, Value memref) const = 0;
    virtual Value getValueToStore(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Value getMemRef(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef();
    }
    MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefType();
    }
    Operation::operand_range getMapOperands(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapOperands();
    }
    AffineMap getAffineMap(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMap();
    }
    NamedAttribute getAffineMapAttrForMemRef(::mlir::Operation *tablegen_opaque_val, Value memref) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMapAttrForMemRef(memref);
    }
    Value getValueToStore(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getValueToStore();
    }
  };
};
} // end namespace detail
class AffineWriteOpInterface : public ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct AffineWriteOpInterfaceTrait : public ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Value getMemRef() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getMemRefOperandIndex());
    }
    MemRefType getMemRefType() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getMemRef().getType().template cast<MemRefType>();
    }
    Operation::operand_range getMapOperands() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return llvm::drop_begin(op.getOperands(), 2);
    }
    AffineMap getAffineMap() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAffineMapAttr().getValue();
    }
    NamedAttribute getAffineMapAttrForMemRef(Value memref) {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        assert(memref == getMemRef());
        return {Identifier::get(op.getMapAttrName(), op.getContext()),
                op.getAffineMapAttr()};
    }
    Value getValueToStore() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getStoredValOperandIndex());
    }
  };
  template <typename ConcreteOp>
  struct Trait : public AffineWriteOpInterfaceTrait<ConcreteOp> {};
  Value getMemRef();
  MemRefType getMemRefType();
  Operation::operand_range getMapOperands();
  AffineMap getAffineMap();
  NamedAttribute getAffineMapAttrForMemRef(Value memref);
  Value getValueToStore();
};
