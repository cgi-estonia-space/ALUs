/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace detail {
struct LoopLikeOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual bool isDefinedOutsideOfLoop(::mlir::Operation *tablegen_opaque_val, Value  value) const = 0;
    virtual Region &getLoopBody(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual LogicalResult moveOutOfLoop(::mlir::Operation *tablegen_opaque_val, ArrayRef<Operation *> ops) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    bool isDefinedOutsideOfLoop(::mlir::Operation *tablegen_opaque_val, Value  value) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDefinedOutsideOfLoop(value);
    }
    Region &getLoopBody(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopBody();
    }
    LogicalResult moveOutOfLoop(::mlir::Operation *tablegen_opaque_val, ArrayRef<Operation *> ops) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).moveOutOfLoop(ops);
    }
  };
};
} // end namespace detail
class LoopLikeOpInterface : public ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct LoopLikeOpInterfaceTrait : public ::mlir::OpInterface<LoopLikeOpInterface, detail::LoopLikeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public LoopLikeOpInterfaceTrait<ConcreteOp> {};
  bool isDefinedOutsideOfLoop(Value  value);
  Region &getLoopBody();
  LogicalResult moveOutOfLoop(ArrayRef<Operation *> ops);
};
} // namespace mlir
