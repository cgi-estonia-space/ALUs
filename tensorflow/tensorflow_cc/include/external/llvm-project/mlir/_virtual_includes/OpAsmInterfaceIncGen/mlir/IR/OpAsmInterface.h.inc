/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace detail {
struct OpAsmOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual void getAsmResultNames(::mlir::Operation *tablegen_opaque_val, OpAsmSetValueNameFn setNameFn) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    void getAsmResultNames(::mlir::Operation *tablegen_opaque_val, OpAsmSetValueNameFn setNameFn) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsmResultNames(setNameFn);
    }
  };
};
} // end namespace detail
class OpAsmOpInterface : public ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct OpAsmOpInterfaceTrait : public ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public OpAsmOpInterfaceTrait<ConcreteOp> {};
  void getAsmResultNames(OpAsmSetValueNameFn setNameFn);
};
} // namespace mlir
