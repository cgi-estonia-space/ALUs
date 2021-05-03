/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace detail {
struct InferShapedTypeOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual LogicalResult inferReturnTypeComponents(MLIRContext* context, Optional<Location> location, ValueRange operands, DictionaryAttr attributes, RegionRange regions, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) const = 0;
    virtual LogicalResult reifyReturnTypeShapes(::mlir::Operation *tablegen_opaque_val, OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    LogicalResult inferReturnTypeComponents(MLIRContext* context, Optional<Location> location, ValueRange operands, DictionaryAttr attributes, RegionRange regions, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) const final {
      return ConcreteOp::inferReturnTypeComponents(context, location, operands, attributes, regions, inferredReturnShapes);
    }
    LogicalResult reifyReturnTypeShapes(::mlir::Operation *tablegen_opaque_val, OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).reifyReturnTypeShapes(builder, reifiedReturnShapes);
    }
  };
};
} // end namespace detail
class InferShapedTypeOpInterface : public ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct InferShapedTypeOpInterfaceTrait : public ::mlir::OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    LogicalResult reifyReturnTypeShapes(OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) {
      return failure();
    }
  };
  template <typename ConcreteOp>
  struct Trait : public InferShapedTypeOpInterfaceTrait<ConcreteOp> {};
  LogicalResult inferReturnTypeComponents(MLIRContext* context, Optional<Location> location, ValueRange operands, DictionaryAttr attributes, RegionRange regions, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);
  LogicalResult reifyReturnTypeShapes(OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes);
};
} // namespace mlir
namespace mlir {
namespace detail {
struct InferTypeOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual LogicalResult inferReturnTypes(MLIRContext* context, Optional<Location> location, ValueRange operands, DictionaryAttr attributes, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) const = 0;
    virtual bool isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    LogicalResult inferReturnTypes(MLIRContext* context, Optional<Location> location, ValueRange operands, DictionaryAttr attributes, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) const final {
      return ConcreteOp::inferReturnTypes(context, location, operands, attributes, regions, inferredReturnTypes);
    }
    bool isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs) const final {
      return ConcreteOp::isCompatibleReturnTypes(lhs, rhs);
    }
  };
};
} // end namespace detail
class InferTypeOpInterface : public ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct InferTypeOpInterfaceTrait : public ::mlir::OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static bool isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs) {
      /// Returns whether two arrays are equal as strongest check for
        /// compatibility by default.
        return lhs == rhs;
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return detail::verifyInferredResultTypes(op);
    }
  };
  template <typename ConcreteOp>
  struct Trait : public InferTypeOpInterfaceTrait<ConcreteOp> {};
  LogicalResult inferReturnTypes(MLIRContext* context, Optional<Location> location, ValueRange operands, DictionaryAttr attributes, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes);
  bool isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs);
};
} // namespace mlir
