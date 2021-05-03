/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

unsigned mlir::linalg::LinalgOp::getNumParallelLoops() {
      return getImpl()->getNumParallelLoops(getOperation());
  }
void mlir::linalg::LinalgOp::getParallelDims(SmallVectorImpl<AffineExpr> & res) {
      return getImpl()->getParallelDims(getOperation(), res);
  }
unsigned mlir::linalg::LinalgOp::getNumReductionLoops() {
      return getImpl()->getNumReductionLoops(getOperation());
  }
void mlir::linalg::LinalgOp::getReductionDims(SmallVectorImpl<AffineExpr> & res) {
      return getImpl()->getReductionDims(getOperation(), res);
  }
unsigned mlir::linalg::LinalgOp::getNumWindowLoops() {
      return getImpl()->getNumWindowLoops(getOperation());
  }
void mlir::linalg::LinalgOp::getWindowDims(SmallVectorImpl<AffineExpr> & res) {
      return getImpl()->getWindowDims(getOperation(), res);
  }
unsigned mlir::linalg::LinalgOp::getNumLoops() {
      return getImpl()->getNumLoops(getOperation());
  }
bool mlir::linalg::LinalgOp::hasSingleReductionLoop() {
      return getImpl()->hasSingleReductionLoop(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumInputs() {
      return getImpl()->getNumInputs(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumInitTensors() {
      return getImpl()->getNumInitTensors(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumOutputs() {
      return getImpl()->getNumOutputs(getOperation());
  }
Value mlir::linalg::LinalgOp::getInput(unsigned i) {
      return getImpl()->getInput(getOperation(), i);
  }
llvm::Optional<unsigned> mlir::linalg::LinalgOp::getIndexOfInput(Value value) {
      return getImpl()->getIndexOfInput(getOperation(), value);
  }
ShapedType mlir::linalg::LinalgOp::getInputShapedType(unsigned i) {
      return getImpl()->getInputShapedType(getOperation(), i);
  }
Operation::operand_range mlir::linalg::LinalgOp::getInputs() {
      return getImpl()->getInputs(getOperation());
  }
SmallVector<RankedTensorType, 4> mlir::linalg::LinalgOp::getInputTensorTypes() {
      return getImpl()->getInputTensorTypes(getOperation());
  }
Value mlir::linalg::LinalgOp::getOutputBuffer(unsigned i) {
      return getImpl()->getOutputBuffer(getOperation(), i);
  }
llvm::Optional<unsigned> mlir::linalg::LinalgOp::getIndexOfOutputBuffer(Value value) {
      return getImpl()->getIndexOfOutputBuffer(getOperation(), value);
  }
MemRefType mlir::linalg::LinalgOp::getOutputBufferType(unsigned i) {
      return getImpl()->getOutputBufferType(getOperation(), i);
  }
ShapedType mlir::linalg::LinalgOp::getOutputShapedType(unsigned i) {
      return getImpl()->getOutputShapedType(getOperation(), i);
  }
SmallVector<RankedTensorType, 4> mlir::linalg::LinalgOp::getOutputTensorTypes() {
      return getImpl()->getOutputTensorTypes(getOperation());
  }
Operation::operand_range mlir::linalg::LinalgOp::getOutputBuffers() {
      return getImpl()->getOutputBuffers(getOperation());
  }
Value mlir::linalg::LinalgOp::getBuffer(unsigned i) {
      return getImpl()->getBuffer(getOperation(), i);
  }
unsigned mlir::linalg::LinalgOp::getNumInputsAndOutputs() {
      return getImpl()->getNumInputsAndOutputs(getOperation());
  }
unsigned mlir::linalg::LinalgOp::getNumInputsAndOutputBuffers() {
      return getImpl()->getNumInputsAndOutputBuffers(getOperation());
  }
Operation::operand_range mlir::linalg::LinalgOp::getInputsAndOutputBuffers() {
      return getImpl()->getInputsAndOutputBuffers(getOperation());
  }
Operation::operand_range mlir::linalg::LinalgOp::getInitTensors() {
      return getImpl()->getInitTensors(getOperation());
  }
Value mlir::linalg::LinalgOp::getInitTensor(unsigned i) {
      return getImpl()->getInitTensor(getOperation(), i);
  }
unsigned mlir::linalg::LinalgOp::getNumShapedOperands() {
      return getImpl()->getNumShapedOperands(getOperation());
  }
Operation::operand_range mlir::linalg::LinalgOp::getShapedOperands() {
      return getImpl()->getShapedOperands(getOperation());
  }
ShapedType mlir::linalg::LinalgOp::getShapedType(unsigned i) {
      return getImpl()->getShapedType(getOperation(), i);
  }
SmallVector<ShapedType, 4> mlir::linalg::LinalgOp::getInputOutputShapedTypes() {
      return getImpl()->getInputOutputShapedTypes(getOperation());
  }
Optional<unsigned> mlir::linalg::LinalgOp::getIndexOfInputAndOutputBuffer(Value value) {
      return getImpl()->getIndexOfInputAndOutputBuffer(getOperation(), value);
  }
ArrayAttr mlir::linalg::LinalgOp::iterator_types() {
      return getImpl()->iterator_types(getOperation());
  }
ArrayAttr mlir::linalg::LinalgOp::indexing_maps() {
      return getImpl()->indexing_maps(getOperation());
  }
SmallVector<AffineMap, 4> mlir::linalg::LinalgOp::getIndexingMaps() {
      return getImpl()->getIndexingMaps(getOperation());
  }
AffineMap mlir::linalg::LinalgOp::getIndexingMap(unsigned i) {
      return getImpl()->getIndexingMap(getOperation(), i);
  }
AffineMap mlir::linalg::LinalgOp::getInputIndexingMap(unsigned i) {
      return getImpl()->getInputIndexingMap(getOperation(), i);
  }
AffineMap mlir::linalg::LinalgOp::getOutputIndexingMap(unsigned i) {
      return getImpl()->getOutputIndexingMap(getOperation(), i);
  }
bool mlir::linalg::LinalgOp::hasBufferSemantics() {
      return getImpl()->hasBufferSemantics(getOperation());
  }
bool mlir::linalg::LinalgOp::hasTensorSemantics() {
      return getImpl()->hasTensorSemantics(getOperation());
  }
std::string mlir::linalg::LinalgOp::getLibraryCallName() {
      return getImpl()->getLibraryCallName(getOperation());
  }
Operation *mlir::linalg::LinalgOp::create(OpBuilder & builder, Location loc, TypeRange resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
      return getImpl()->create(builder, loc, resultTypes, operands, attributes);
  }
Operation *mlir::linalg::LinalgOp::clone(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
      return getImpl()->clone(getOperation(), b, loc, resultTypes, operands);
  }
