/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool InferFusibilityOpInterface::isFusibleWithOperand() {
      return getImpl()->isFusibleWithOperand(getOperation());
  }
bool InferFusibilityOpInterface::isFusibleWithConsumer() {
      return getImpl()->isFusibleWithConsumer(getOperation());
  }
bool InferFusibilityOpInterface::inferInputsShapeEquality(int lhs, int rhs) {
      return getImpl()->inferInputsShapeEquality(getOperation(), lhs, rhs);
  }
bool InferFusibilityOpInterface::inferOutputsShapeEquality(int lhs, int rhs) {
      return getImpl()->inferOutputsShapeEquality(getOperation(), lhs, rhs);
  }
bool InferFusibilityOpInterface::inferInputOutputShapeEquality(int input, int output) {
      return getImpl()->inferInputOutputShapeEquality(getOperation(), input, output);
  }
llvm::Optional<Value> InferFusibilityOpInterface::inferEffectiveWorkloadShape() {
      return getImpl()->inferEffectiveWorkloadShape(getOperation());
  }
