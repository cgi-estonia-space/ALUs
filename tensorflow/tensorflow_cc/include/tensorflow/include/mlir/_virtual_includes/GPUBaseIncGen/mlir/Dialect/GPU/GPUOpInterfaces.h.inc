/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace gpu {
namespace detail {
struct AsyncOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual OperandRange getAsyncDependencies(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual void addAsyncDependency(::mlir::Operation *tablegen_opaque_val, Value token) const = 0;
    virtual OpResult getAsyncToken(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    OperandRange getAsyncDependencies(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsyncDependencies();
    }
    void addAsyncDependency(::mlir::Operation *tablegen_opaque_val, Value token) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).addAsyncDependency(token);
    }
    OpResult getAsyncToken(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsyncToken();
    }
  };
};
} // end namespace detail
class AsyncOpInterface : public ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct AsyncOpInterfaceTrait : public ::mlir::OpInterface<AsyncOpInterface, detail::AsyncOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    OperandRange getAsyncDependencies() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncDependencies();
    }
    void addAsyncDependency(Value token) {
      ::mlir::gpu::addAsyncDependency(this->getOperation(), token);
    }
    OpResult getAsyncToken() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncToken().template dyn_cast_or_null<OpResult>();
    }
  };
  template <typename ConcreteOp>
  struct Trait : public AsyncOpInterfaceTrait<ConcreteOp> {};
  OperandRange getAsyncDependencies();
  void addAsyncDependency(Value token);
  OpResult getAsyncToken();
};
} // namespace gpu
} // namespace mlir
