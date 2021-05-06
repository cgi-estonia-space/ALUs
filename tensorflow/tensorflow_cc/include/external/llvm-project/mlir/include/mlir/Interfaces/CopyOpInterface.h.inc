/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace detail {
struct CopyOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual Value getSource(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Value getTarget(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    Value getSource(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSource();
    }
    Value getTarget(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTarget();
    }
  };
};
} // end namespace detail
class CopyOpInterface : public ::mlir::OpInterface<CopyOpInterface, detail::CopyOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CopyOpInterface, detail::CopyOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct CopyOpInterfaceTrait : public ::mlir::OpInterface<CopyOpInterface, detail::CopyOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public CopyOpInterfaceTrait<ConcreteOp> {};
  Value getSource();
  Value getTarget();
};
} // namespace mlir
