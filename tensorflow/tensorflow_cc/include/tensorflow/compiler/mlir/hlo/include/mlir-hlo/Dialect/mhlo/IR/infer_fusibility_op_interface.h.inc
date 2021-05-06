/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct InferFusibilityOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual bool isFusibleWithOperand(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual bool isFusibleWithConsumer(::mlir::Operation *tablegen_opaque_val) const = 0;
    virtual bool inferInputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) const = 0;
    virtual bool inferOutputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) const = 0;
    virtual bool inferInputOutputShapeEquality(::mlir::Operation *tablegen_opaque_val, int input, int output) const = 0;
    virtual llvm::Optional<Value> inferEffectiveWorkloadShape(::mlir::Operation *tablegen_opaque_val) const = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    bool isFusibleWithOperand(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isFusibleWithOperand();
    }
    bool isFusibleWithConsumer(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isFusibleWithConsumer();
    }
    bool inferInputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferInputsShapeEquality(lhs, rhs);
    }
    bool inferOutputsShapeEquality(::mlir::Operation *tablegen_opaque_val, int lhs, int rhs) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferOutputsShapeEquality(lhs, rhs);
    }
    bool inferInputOutputShapeEquality(::mlir::Operation *tablegen_opaque_val, int input, int output) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferInputOutputShapeEquality(input, output);
    }
    llvm::Optional<Value> inferEffectiveWorkloadShape(::mlir::Operation *tablegen_opaque_val) const final {
      return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inferEffectiveWorkloadShape();
    }
  };
};
} // end namespace detail
class InferFusibilityOpInterface : public ::mlir::OpInterface<InferFusibilityOpInterface, detail::InferFusibilityOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<InferFusibilityOpInterface, detail::InferFusibilityOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct InferFusibilityOpInterfaceTrait : public ::mlir::OpInterface<InferFusibilityOpInterface, detail::InferFusibilityOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    bool isFusibleWithOperand() {
      /// Returns whether this op can be fused with its operands
        return true;
    }
    bool isFusibleWithConsumer() {
      /// Return whether this op can be fused withh its consumers
        return true;
    }
    bool inferInputsShapeEquality(int lhs, int rhs) {
      /// Return whether two inputs have the same shape.
        Operation *op = this->getOperation();
        assert(lhs < op->getNumOperands() && lhs >= 0 &&
               rhs < op->getNumOperands() && rhs >= 0);
        if (lhs == rhs) return true;

        // if both lhs and rhs have static shapes, check them directly
        Type lhs_ty = op->getOperand(lhs).getType();
        Type rhs_ty = op->getOperand(rhs).getType();
        auto lhs_shape_type = lhs_ty.dyn_cast_or_null<RankedTensorType>();
        auto rhs_shape_type = rhs_ty.dyn_cast_or_null<RankedTensorType>();
        if (!lhs_shape_type || !lhs_shape_type.hasStaticShape() ||
            !rhs_shape_type || !rhs_shape_type.hasStaticShape() ||
            lhs_shape_type.getRank() != rhs_shape_type.getRank()) {
          return false;
        }
        return lhs_shape_type.getShape() == rhs_shape_type.getShape();
    }
    bool inferOutputsShapeEquality(int lhs, int rhs) {
      /// Return whether two outputs have the same shape.
        Operation *op = this->getOperation();
        assert(lhs < op->getNumResults() && lhs >= 0 &&
               rhs < op->getNumResults() && rhs >= 0);
        if (lhs == rhs) return true;

        // if both lhs and rhs have static shapes, check them directly
        Type lhs_ty = op->getResult(lhs).getType();
        Type rhs_ty = op->getResult(rhs).getType();
        auto lhs_shape_type = lhs_ty.dyn_cast_or_null<RankedTensorType>();
        auto rhs_shape_type = rhs_ty.dyn_cast_or_null<RankedTensorType>();
        if (!lhs_shape_type || !lhs_shape_type.hasStaticShape() ||
            !rhs_shape_type || !rhs_shape_type.hasStaticShape() ||
            lhs_shape_type.getRank() != rhs_shape_type.getRank()) {
          return false;
        }
        return lhs_shape_type.getShape() == rhs_shape_type.getShape();
    }
    bool inferInputOutputShapeEquality(int input, int output) {
      /// Return whether the input and the output have the same shape.
        Operation *op = this->getOperation();
        assert(input < op->getNumOperands() && input >= 0 &&
               output < op->getNumResults() && output >= 0);

        // if both input and output have static shapes, check them directly
        Type input_ty = op->getOperand(input).getType();
        Type output_ty = op->getResult(output).getType();
        auto input_shape_type = input_ty.dyn_cast_or_null<RankedTensorType>();
        auto output_shape_type = output_ty.dyn_cast_or_null<RankedTensorType>();
        if (!input_shape_type || !input_shape_type.hasStaticShape() ||
            !output_shape_type || !output_shape_type.hasStaticShape() ||
            input_shape_type.getRank() != output_shape_type.getRank()) {
          return false;
        }
        return input_shape_type.getShape() == output_shape_type.getShape();
    }
    llvm::Optional<Value> inferEffectiveWorkloadShape() {
      /// Return effective workload size if possible, otherwise None.
        return {};
    }
  };
  template <typename ConcreteOp>
  struct Trait : public InferFusibilityOpInterfaceTrait<ConcreteOp> {};
  bool isFusibleWithOperand();
  bool isFusibleWithConsumer();
  bool inferInputsShapeEquality(int lhs, int rhs);
  bool inferOutputsShapeEquality(int lhs, int rhs);
  bool inferInputOutputShapeEquality(int input, int output);
  llvm::Optional<Value> inferEffectiveWorkloadShape();
};
