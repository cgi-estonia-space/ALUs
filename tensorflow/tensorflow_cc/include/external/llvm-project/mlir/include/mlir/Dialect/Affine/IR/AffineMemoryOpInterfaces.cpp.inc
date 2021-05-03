/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

Value AffineReadOpInterface::getMemRef() {
      return getImpl()->getMemRef(getOperation());
  }
MemRefType AffineReadOpInterface::getMemRefType() {
      return getImpl()->getMemRefType(getOperation());
  }
Operation::operand_range AffineReadOpInterface::getMapOperands() {
      return getImpl()->getMapOperands(getOperation());
  }
AffineMap AffineReadOpInterface::getAffineMap() {
      return getImpl()->getAffineMap(getOperation());
  }
NamedAttribute AffineReadOpInterface::getAffineMapAttrForMemRef(Value memref) {
      return getImpl()->getAffineMapAttrForMemRef(getOperation(), memref);
  }
Value AffineReadOpInterface::getValue() {
      return getImpl()->getValue(getOperation());
  }
Value AffineWriteOpInterface::getMemRef() {
      return getImpl()->getMemRef(getOperation());
  }
MemRefType AffineWriteOpInterface::getMemRefType() {
      return getImpl()->getMemRefType(getOperation());
  }
Operation::operand_range AffineWriteOpInterface::getMapOperands() {
      return getImpl()->getMapOperands(getOperation());
  }
AffineMap AffineWriteOpInterface::getAffineMap() {
      return getImpl()->getAffineMap(getOperation());
  }
NamedAttribute AffineWriteOpInterface::getAffineMapAttrForMemRef(Value memref) {
      return getImpl()->getAffineMapAttrForMemRef(getOperation(), memref);
  }
Value AffineWriteOpInterface::getValueToStore() {
      return getImpl()->getValueToStore(getOperation());
  }
