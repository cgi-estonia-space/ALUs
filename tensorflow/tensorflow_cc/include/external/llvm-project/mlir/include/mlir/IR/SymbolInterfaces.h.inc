/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace detail {
struct SymbolOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual StringRef getName(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual void setName(::mlir::Operation *tablegen_opaque_val, StringRef name) const = 0;
    virtual mlir::SymbolTable::Visibility getVisibility(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual void setVisibility(::mlir::Operation *tablegen_opaque_val, mlir::SymbolTable::Visibility vis) const = 0;
    virtual bool isNested(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual bool isPrivate(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual bool isPublic(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual Optional<::mlir::SymbolTable::UseRange> getSymbolUses(::mlir::Operation *tablegen_opaque_val, Operation * from) const = 0;
    virtual bool symbolKnownUseEmpty(::mlir::Operation *tablegen_opaque_val, Operation * from) const = 0;
    virtual LogicalResult replaceAllSymbolUses(::mlir::Operation *tablegen_opaque_val, StringRef newSymbol, Operation * from) const = 0;
    virtual bool isOptionalSymbol(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual bool canDiscardOnUseEmpty(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    StringRef getName(::mlir::Operation *tablegen_opaque_val) const final {
      // Don't rely on the trait implementation as optional symbol operations
        // may override this.
        return mlir::SymbolTable::getSymbolName((llvm::cast<ConcreteOp>(tablegen_opaque_val)));
    }
    void setName(::mlir::Operation *tablegen_opaque_val, StringRef name) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setName(name);
    }
    mlir::SymbolTable::Visibility getVisibility(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getVisibility();
    }
    void setVisibility(::mlir::Operation *tablegen_opaque_val, mlir::SymbolTable::Visibility vis) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setVisibility(vis);
    }
    bool isNested(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isNested();
    }
    bool isPrivate(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isPrivate();
    }
    bool isPublic(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isPublic();
    }
    Optional<::mlir::SymbolTable::UseRange> getSymbolUses(::mlir::Operation *tablegen_opaque_val, Operation * from) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getSymbolUses(from);
    }
    bool symbolKnownUseEmpty(::mlir::Operation *tablegen_opaque_val, Operation * from) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).symbolKnownUseEmpty(from);
    }
    LogicalResult replaceAllSymbolUses(::mlir::Operation *tablegen_opaque_val, StringRef newSymbol, Operation * from) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).replaceAllSymbolUses(newSymbol, from);
    }
    bool isOptionalSymbol(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isOptionalSymbol();
    }
    bool canDiscardOnUseEmpty(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).canDiscardOnUseEmpty();
    }
  };
};
} // end namespace detail
class SymbolOpInterface : public ::mlir::OpInterface<SymbolOpInterface, detail::SymbolOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SymbolOpInterface, detail::SymbolOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct SymbolOpInterfaceTrait : public ::mlir::OpInterface<SymbolOpInterface, detail::SymbolOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    StringRef getName() {
      return mlir::SymbolTable::getSymbolName(this->getOperation());
    }
    void setName(StringRef name) {
      this->getOperation()->setAttr(
            mlir::SymbolTable::getSymbolAttrName(),
            StringAttr::get(name, this->getOperation()->getContext()));
    }
    mlir::SymbolTable::Visibility getVisibility() {
      return mlir::SymbolTable::getSymbolVisibility(this->getOperation());
    }
    void setVisibility(mlir::SymbolTable::Visibility vis) {
      mlir::SymbolTable::setSymbolVisibility(this->getOperation(), vis);
    }
    bool isNested() {
      return getVisibility() == mlir::SymbolTable::Visibility::Nested;
    }
    bool isPrivate() {
      return getVisibility() == mlir::SymbolTable::Visibility::Private;
    }
    bool isPublic() {
      return getVisibility() == mlir::SymbolTable::Visibility::Public;
    }
    Optional<::mlir::SymbolTable::UseRange> getSymbolUses(Operation * from) {
      return ::mlir::SymbolTable::getSymbolUses(this->getOperation(), from);
    }
    bool symbolKnownUseEmpty(Operation * from) {
      return ::mlir::SymbolTable::symbolKnownUseEmpty(this->getOperation(),
                                                        from);
    }
    LogicalResult replaceAllSymbolUses(StringRef newSymbol, Operation * from) {
      return ::mlir::SymbolTable::replaceAllSymbolUses(this->getOperation(),
                                                         newSymbol, from);
    }
    bool isOptionalSymbol() {
      return false;
    }
    bool canDiscardOnUseEmpty() {
      // By default, base this on the visibility alone. A symbol can be
        // discarded as long as it is not public. Only public symbols may be
        // visible from outside of the IR.
        return getVisibility() != ::mlir::SymbolTable::Visibility::Public;
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      // If this is an optional symbol, bail out early if possible.
    auto concreteOp = cast<ConcreteOp>(op);
    if (concreteOp.isOptionalSymbol()) {
      if(!concreteOp.getAttr(::mlir::SymbolTable::getSymbolAttrName()))
        return success();
    }
    return ::mlir::detail::verifySymbol(op);
    }

    using Visibility = mlir::SymbolTable::Visibility;
  
  };
  template <typename ConcreteOp>
  struct Trait : public SymbolOpInterfaceTrait<ConcreteOp> {};
  StringRef getName();
  void setName(StringRef name);
  mlir::SymbolTable::Visibility getVisibility();
  void setVisibility(mlir::SymbolTable::Visibility vis);
  bool isNested();
  bool isPrivate();
  bool isPublic();
  Optional<::mlir::SymbolTable::UseRange> getSymbolUses(Operation * from);
  bool symbolKnownUseEmpty(Operation * from);
  LogicalResult replaceAllSymbolUses(StringRef newSymbol, Operation * from);
  bool isOptionalSymbol();
  bool canDiscardOnUseEmpty();

    /// Custom classof that handles the case where the symbol is optional.
    static bool classof(Operation *op) {
      auto *concept = getInterfaceFor(op);
      if (!concept)
        return false;
      return !concept->isOptionalSymbol(op) ||
             op->getAttr(::mlir::SymbolTable::getSymbolAttrName());
    }
  
};
} // namespace mlir
namespace mlir {
namespace detail {
struct SymbolUserOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual LogicalResult verifySymbolUses(::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection & symbolTable) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    LogicalResult verifySymbolUses(::mlir::Operation *tablegen_opaque_val, ::mlir::SymbolTableCollection & symbolTable) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifySymbolUses(symbolTable);
    }
  };
};
} // end namespace detail
class SymbolUserOpInterface : public ::mlir::OpInterface<SymbolUserOpInterface, detail::SymbolUserOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<SymbolUserOpInterface, detail::SymbolUserOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct SymbolUserOpInterfaceTrait : public ::mlir::OpInterface<SymbolUserOpInterface, detail::SymbolUserOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  template <typename ConcreteOp>
  struct Trait : public SymbolUserOpInterfaceTrait<ConcreteOp> {};
  LogicalResult verifySymbolUses(::mlir::SymbolTableCollection & symbolTable);
};
} // namespace mlir
