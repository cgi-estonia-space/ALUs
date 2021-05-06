/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

LogicalResult mlir::InferShapedTypeOpInterface::inferReturnTypeComponents(MLIRContext* context, Optional<Location> location, ValueRange operands, DictionaryAttr attributes, RegionRange regions, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
      return getImpl()->inferReturnTypeComponents(context, location, operands, attributes, regions, inferredReturnShapes);
  }
LogicalResult mlir::InferShapedTypeOpInterface::reifyReturnTypeShapes(OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) {
      return getImpl()->reifyReturnTypeShapes(getOperation(), builder, reifiedReturnShapes);
  }
LogicalResult mlir::InferTypeOpInterface::inferReturnTypes(MLIRContext* context, Optional<Location> location, ValueRange operands, DictionaryAttr attributes, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
      return getImpl()->inferReturnTypes(context, location, operands, attributes, regions, inferredReturnTypes);
  }
bool mlir::InferTypeOpInterface::isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs) {
      return getImpl()->isCompatibleReturnTypes(lhs, rhs);
  }
