/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace detail {
struct DerivedAttributeOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual bool isDerivedAttribute(StringRef name) const = 0;
    virtual DictionaryAttr materializeDerivedAttributes(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    bool isDerivedAttribute(StringRef name) const final {
      return ConcreteOp::isDerivedAttribute(name);
    }
    DictionaryAttr materializeDerivedAttributes(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).materializeDerivedAttributes();
    }
  };
};
} // end namespace detail
class DerivedAttributeOpInterface : public ::mlir::OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct DerivedAttributeOpInterfaceTrait : public ::mlir::OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public DerivedAttributeOpInterfaceTrait<ConcreteOp> {};
  bool isDerivedAttribute(StringRef name);
  DictionaryAttr materializeDerivedAttributes();
};
} // namespace mlir
