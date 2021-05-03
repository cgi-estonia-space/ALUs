/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::lmhlo_gpu::BatchNormGradOp,
::mlir::lmhlo_gpu::BatchNormInferenceOp,
::mlir::lmhlo_gpu::BatchNormTrainingOp,
::mlir::lmhlo_gpu::CholeskyOp,
::mlir::lmhlo_gpu::ConvBackwardFilterOp,
::mlir::lmhlo_gpu::ConvBackwardInputOp,
::mlir::lmhlo_gpu::ConvForwardFusedOp,
::mlir::lmhlo_gpu::ConvForwardOp,
::mlir::lmhlo_gpu::GEMMOp,
::mlir::lmhlo_gpu::GEMM_BiasOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::BatchNormGradOp definitions
//===----------------------------------------------------------------------===//

BatchNormGradOpAdaptor::BatchNormGradOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BatchNormGradOpAdaptor::BatchNormGradOpAdaptor(BatchNormGradOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BatchNormGradOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BatchNormGradOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BatchNormGradOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value BatchNormGradOpAdaptor::scale() {
  return *getODSOperands(1).begin();
}

::mlir::Value BatchNormGradOpAdaptor::mean() {
  return *getODSOperands(2).begin();
}

::mlir::Value BatchNormGradOpAdaptor::stddev() {
  return *getODSOperands(3).begin();
}

::mlir::Value BatchNormGradOpAdaptor::grad_output() {
  return *getODSOperands(4).begin();
}

::mlir::Value BatchNormGradOpAdaptor::grad_operand() {
  return *getODSOperands(5).begin();
}

::mlir::Value BatchNormGradOpAdaptor::grad_scale() {
  return *getODSOperands(6).begin();
}

::mlir::Value BatchNormGradOpAdaptor::grad_offset() {
  return *getODSOperands(7).begin();
}

::mlir::FloatAttr BatchNormGradOpAdaptor::epsilon() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("epsilon").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::IntegerAttr BatchNormGradOpAdaptor::feature_index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("feature_index").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult BatchNormGradOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_epsilon = odsAttrs.get("epsilon");
  if (!tblgen_epsilon) return emitError(loc, "'lmhlo_gpu.batch_norm_grad' op ""requires attribute 'epsilon'");
    if (!(((tblgen_epsilon.isa<::mlir::FloatAttr>())) && ((tblgen_epsilon.cast<::mlir::FloatAttr>().getType().isF32())))) return emitError(loc, "'lmhlo_gpu.batch_norm_grad' op ""attribute 'epsilon' failed to satisfy constraint: 32-bit float attribute");
  }
  {
  auto tblgen_feature_index = odsAttrs.get("feature_index");
  if (!tblgen_feature_index) return emitError(loc, "'lmhlo_gpu.batch_norm_grad' op ""requires attribute 'feature_index'");
    if (!(((tblgen_feature_index.isa<::mlir::IntegerAttr>())) && ((tblgen_feature_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.batch_norm_grad' op ""attribute 'feature_index' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef BatchNormGradOp::getOperationName() {
  return "lmhlo_gpu.batch_norm_grad";
}

std::pair<unsigned, unsigned> BatchNormGradOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BatchNormGradOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BatchNormGradOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value BatchNormGradOp::scale() {
  return *getODSOperands(1).begin();
}

::mlir::Value BatchNormGradOp::mean() {
  return *getODSOperands(2).begin();
}

::mlir::Value BatchNormGradOp::stddev() {
  return *getODSOperands(3).begin();
}

::mlir::Value BatchNormGradOp::grad_output() {
  return *getODSOperands(4).begin();
}

::mlir::Value BatchNormGradOp::grad_operand() {
  return *getODSOperands(5).begin();
}

::mlir::Value BatchNormGradOp::grad_scale() {
  return *getODSOperands(6).begin();
}

::mlir::Value BatchNormGradOp::grad_offset() {
  return *getODSOperands(7).begin();
}

::mlir::MutableOperandRange BatchNormGradOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormGradOp::scaleMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormGradOp::meanMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormGradOp::stddevMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormGradOp::grad_outputMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormGradOp::grad_operandMutable() {
  auto range = getODSOperandIndexAndLength(5);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormGradOp::grad_scaleMutable() {
  auto range = getODSOperandIndexAndLength(6);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormGradOp::grad_offsetMutable() {
  auto range = getODSOperandIndexAndLength(7);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BatchNormGradOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BatchNormGradOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::FloatAttr BatchNormGradOp::epsilonAttr() {
  return this->getAttr("epsilon").cast<::mlir::FloatAttr>();
}

::llvm::APFloat BatchNormGradOp::epsilon() {
  auto attr = epsilonAttr();
  return attr.getValue();
}

::mlir::IntegerAttr BatchNormGradOp::feature_indexAttr() {
  return this->getAttr("feature_index").cast<::mlir::IntegerAttr>();
}

uint64_t BatchNormGradOp::feature_index() {
  auto attr = feature_indexAttr();
  return attr.getValue().getZExtValue();
}

void BatchNormGradOp::epsilonAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("epsilon", attr);
}

void BatchNormGradOp::feature_indexAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("feature_index", attr);
}

void BatchNormGradOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(mean);
  odsState.addOperands(stddev);
  odsState.addOperands(grad_output);
  odsState.addOperands(grad_operand);
  odsState.addOperands(grad_scale);
  odsState.addOperands(grad_offset);
  odsState.addAttribute("epsilon", epsilon);
  odsState.addAttribute("feature_index", feature_index);
}

void BatchNormGradOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(mean);
  odsState.addOperands(stddev);
  odsState.addOperands(grad_output);
  odsState.addOperands(grad_operand);
  odsState.addOperands(grad_scale);
  odsState.addOperands(grad_offset);
  odsState.addAttribute("epsilon", epsilon);
  odsState.addAttribute("feature_index", feature_index);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BatchNormGradOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::llvm::APFloat epsilon, uint64_t feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(mean);
  odsState.addOperands(stddev);
  odsState.addOperands(grad_output);
  odsState.addOperands(grad_operand);
  odsState.addOperands(grad_scale);
  odsState.addOperands(grad_offset);
  odsState.addAttribute("epsilon", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), epsilon));
  odsState.addAttribute("feature_index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_index));
}

void BatchNormGradOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value grad_output, ::mlir::Value grad_operand, ::mlir::Value grad_scale, ::mlir::Value grad_offset, ::llvm::APFloat epsilon, uint64_t feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(mean);
  odsState.addOperands(stddev);
  odsState.addOperands(grad_output);
  odsState.addOperands(grad_operand);
  odsState.addOperands(grad_scale);
  odsState.addOperands(grad_offset);
  odsState.addAttribute("epsilon", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), epsilon));
  odsState.addAttribute("feature_index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_index));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BatchNormGradOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 8u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BatchNormGradOp::verify() {
  if (failed(BatchNormGradOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup6 = getODSOperands(6);
    for (::mlir::Value v : valueGroup6) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup7 = getODSOperands(7);
    for (::mlir::Value v : valueGroup7) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void BatchNormGradOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(3))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(4))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(5))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(6))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(7))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::BatchNormInferenceOp definitions
//===----------------------------------------------------------------------===//

BatchNormInferenceOpAdaptor::BatchNormInferenceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BatchNormInferenceOpAdaptor::BatchNormInferenceOpAdaptor(BatchNormInferenceOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BatchNormInferenceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BatchNormInferenceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BatchNormInferenceOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value BatchNormInferenceOpAdaptor::scale() {
  return *getODSOperands(1).begin();
}

::mlir::Value BatchNormInferenceOpAdaptor::offset() {
  return *getODSOperands(2).begin();
}

::mlir::Value BatchNormInferenceOpAdaptor::mean() {
  return *getODSOperands(3).begin();
}

::mlir::Value BatchNormInferenceOpAdaptor::stddev() {
  return *getODSOperands(4).begin();
}

::mlir::Value BatchNormInferenceOpAdaptor::output() {
  return *getODSOperands(5).begin();
}

::mlir::FloatAttr BatchNormInferenceOpAdaptor::epsilon() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("epsilon").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::IntegerAttr BatchNormInferenceOpAdaptor::feature_index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("feature_index").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult BatchNormInferenceOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_epsilon = odsAttrs.get("epsilon");
  if (!tblgen_epsilon) return emitError(loc, "'lmhlo_gpu.batch_norm_inference' op ""requires attribute 'epsilon'");
    if (!(((tblgen_epsilon.isa<::mlir::FloatAttr>())) && ((tblgen_epsilon.cast<::mlir::FloatAttr>().getType().isF32())))) return emitError(loc, "'lmhlo_gpu.batch_norm_inference' op ""attribute 'epsilon' failed to satisfy constraint: 32-bit float attribute");
  }
  {
  auto tblgen_feature_index = odsAttrs.get("feature_index");
  if (!tblgen_feature_index) return emitError(loc, "'lmhlo_gpu.batch_norm_inference' op ""requires attribute 'feature_index'");
    if (!(((tblgen_feature_index.isa<::mlir::IntegerAttr>())) && ((tblgen_feature_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.batch_norm_inference' op ""attribute 'feature_index' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef BatchNormInferenceOp::getOperationName() {
  return "lmhlo_gpu.batch_norm_inference";
}

std::pair<unsigned, unsigned> BatchNormInferenceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BatchNormInferenceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BatchNormInferenceOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value BatchNormInferenceOp::scale() {
  return *getODSOperands(1).begin();
}

::mlir::Value BatchNormInferenceOp::offset() {
  return *getODSOperands(2).begin();
}

::mlir::Value BatchNormInferenceOp::mean() {
  return *getODSOperands(3).begin();
}

::mlir::Value BatchNormInferenceOp::stddev() {
  return *getODSOperands(4).begin();
}

::mlir::Value BatchNormInferenceOp::output() {
  return *getODSOperands(5).begin();
}

::mlir::MutableOperandRange BatchNormInferenceOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormInferenceOp::scaleMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormInferenceOp::offsetMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormInferenceOp::meanMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormInferenceOp::stddevMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormInferenceOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(5);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BatchNormInferenceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BatchNormInferenceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::FloatAttr BatchNormInferenceOp::epsilonAttr() {
  return this->getAttr("epsilon").cast<::mlir::FloatAttr>();
}

::llvm::APFloat BatchNormInferenceOp::epsilon() {
  auto attr = epsilonAttr();
  return attr.getValue();
}

::mlir::IntegerAttr BatchNormInferenceOp::feature_indexAttr() {
  return this->getAttr("feature_index").cast<::mlir::IntegerAttr>();
}

uint64_t BatchNormInferenceOp::feature_index() {
  auto attr = feature_indexAttr();
  return attr.getValue().getZExtValue();
}

void BatchNormInferenceOp::epsilonAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("epsilon", attr);
}

void BatchNormInferenceOp::feature_indexAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("feature_index", attr);
}

void BatchNormInferenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(offset);
  odsState.addOperands(mean);
  odsState.addOperands(stddev);
  odsState.addOperands(output);
  odsState.addAttribute("epsilon", epsilon);
  odsState.addAttribute("feature_index", feature_index);
}

void BatchNormInferenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(offset);
  odsState.addOperands(mean);
  odsState.addOperands(stddev);
  odsState.addOperands(output);
  odsState.addAttribute("epsilon", epsilon);
  odsState.addAttribute("feature_index", feature_index);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BatchNormInferenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::llvm::APFloat epsilon, uint64_t feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(offset);
  odsState.addOperands(mean);
  odsState.addOperands(stddev);
  odsState.addOperands(output);
  odsState.addAttribute("epsilon", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), epsilon));
  odsState.addAttribute("feature_index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_index));
}

void BatchNormInferenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value stddev, ::mlir::Value output, ::llvm::APFloat epsilon, uint64_t feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(offset);
  odsState.addOperands(mean);
  odsState.addOperands(stddev);
  odsState.addOperands(output);
  odsState.addAttribute("epsilon", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), epsilon));
  odsState.addAttribute("feature_index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_index));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BatchNormInferenceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BatchNormInferenceOp::verify() {
  if (failed(BatchNormInferenceOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void BatchNormInferenceOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(3))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(4))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(5))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::BatchNormTrainingOp definitions
//===----------------------------------------------------------------------===//

BatchNormTrainingOpAdaptor::BatchNormTrainingOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BatchNormTrainingOpAdaptor::BatchNormTrainingOpAdaptor(BatchNormTrainingOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BatchNormTrainingOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BatchNormTrainingOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BatchNormTrainingOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value BatchNormTrainingOpAdaptor::scale() {
  return *getODSOperands(1).begin();
}

::mlir::Value BatchNormTrainingOpAdaptor::offset() {
  return *getODSOperands(2).begin();
}

::mlir::Value BatchNormTrainingOpAdaptor::output() {
  return *getODSOperands(3).begin();
}

::mlir::Value BatchNormTrainingOpAdaptor::batch_mean() {
  return *getODSOperands(4).begin();
}

::mlir::Value BatchNormTrainingOpAdaptor::batch_stddev() {
  return *getODSOperands(5).begin();
}

::mlir::FloatAttr BatchNormTrainingOpAdaptor::epsilon() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("epsilon").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::IntegerAttr BatchNormTrainingOpAdaptor::feature_index() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("feature_index").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult BatchNormTrainingOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_epsilon = odsAttrs.get("epsilon");
  if (!tblgen_epsilon) return emitError(loc, "'lmhlo_gpu.batch_norm_training' op ""requires attribute 'epsilon'");
    if (!(((tblgen_epsilon.isa<::mlir::FloatAttr>())) && ((tblgen_epsilon.cast<::mlir::FloatAttr>().getType().isF32())))) return emitError(loc, "'lmhlo_gpu.batch_norm_training' op ""attribute 'epsilon' failed to satisfy constraint: 32-bit float attribute");
  }
  {
  auto tblgen_feature_index = odsAttrs.get("feature_index");
  if (!tblgen_feature_index) return emitError(loc, "'lmhlo_gpu.batch_norm_training' op ""requires attribute 'feature_index'");
    if (!(((tblgen_feature_index.isa<::mlir::IntegerAttr>())) && ((tblgen_feature_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.batch_norm_training' op ""attribute 'feature_index' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef BatchNormTrainingOp::getOperationName() {
  return "lmhlo_gpu.batch_norm_training";
}

std::pair<unsigned, unsigned> BatchNormTrainingOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BatchNormTrainingOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BatchNormTrainingOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value BatchNormTrainingOp::scale() {
  return *getODSOperands(1).begin();
}

::mlir::Value BatchNormTrainingOp::offset() {
  return *getODSOperands(2).begin();
}

::mlir::Value BatchNormTrainingOp::output() {
  return *getODSOperands(3).begin();
}

::mlir::Value BatchNormTrainingOp::batch_mean() {
  return *getODSOperands(4).begin();
}

::mlir::Value BatchNormTrainingOp::batch_stddev() {
  return *getODSOperands(5).begin();
}

::mlir::MutableOperandRange BatchNormTrainingOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormTrainingOp::scaleMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormTrainingOp::offsetMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormTrainingOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormTrainingOp::batch_meanMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BatchNormTrainingOp::batch_stddevMutable() {
  auto range = getODSOperandIndexAndLength(5);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BatchNormTrainingOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BatchNormTrainingOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::FloatAttr BatchNormTrainingOp::epsilonAttr() {
  return this->getAttr("epsilon").cast<::mlir::FloatAttr>();
}

::llvm::APFloat BatchNormTrainingOp::epsilon() {
  auto attr = epsilonAttr();
  return attr.getValue();
}

::mlir::IntegerAttr BatchNormTrainingOp::feature_indexAttr() {
  return this->getAttr("feature_index").cast<::mlir::IntegerAttr>();
}

uint64_t BatchNormTrainingOp::feature_index() {
  auto attr = feature_indexAttr();
  return attr.getValue().getZExtValue();
}

void BatchNormTrainingOp::epsilonAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("epsilon", attr);
}

void BatchNormTrainingOp::feature_indexAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("feature_index", attr);
}

void BatchNormTrainingOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(offset);
  odsState.addOperands(output);
  odsState.addOperands(batch_mean);
  odsState.addOperands(batch_stddev);
  odsState.addAttribute("epsilon", epsilon);
  odsState.addAttribute("feature_index", feature_index);
}

void BatchNormTrainingOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::mlir::FloatAttr epsilon, ::mlir::IntegerAttr feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(offset);
  odsState.addOperands(output);
  odsState.addOperands(batch_mean);
  odsState.addOperands(batch_stddev);
  odsState.addAttribute("epsilon", epsilon);
  odsState.addAttribute("feature_index", feature_index);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BatchNormTrainingOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::llvm::APFloat epsilon, uint64_t feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(offset);
  odsState.addOperands(output);
  odsState.addOperands(batch_mean);
  odsState.addOperands(batch_stddev);
  odsState.addAttribute("epsilon", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), epsilon));
  odsState.addAttribute("feature_index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_index));
}

void BatchNormTrainingOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value output, ::mlir::Value batch_mean, ::mlir::Value batch_stddev, ::llvm::APFloat epsilon, uint64_t feature_index) {
  odsState.addOperands(operand);
  odsState.addOperands(scale);
  odsState.addOperands(offset);
  odsState.addOperands(output);
  odsState.addOperands(batch_mean);
  odsState.addOperands(batch_stddev);
  odsState.addAttribute("epsilon", odsBuilder.getFloatAttr(odsBuilder.getF32Type(), epsilon));
  odsState.addAttribute("feature_index", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_index));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BatchNormTrainingOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BatchNormTrainingOp::verify() {
  if (failed(BatchNormTrainingOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void BatchNormTrainingOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(3))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(4))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(5))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::CholeskyOp definitions
//===----------------------------------------------------------------------===//

CholeskyOpAdaptor::CholeskyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CholeskyOpAdaptor::CholeskyOpAdaptor(CholeskyOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CholeskyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CholeskyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CholeskyOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value CholeskyOpAdaptor::output() {
  return *getODSOperands(1).begin();
}

::mlir::Value CholeskyOpAdaptor::scratch() {
  return *getODSOperands(2).begin();
}

::mlir::Value CholeskyOpAdaptor::info() {
  return *getODSOperands(3).begin();
}

::mlir::BoolAttr CholeskyOpAdaptor::is_upper() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("is_upper").cast<::mlir::BoolAttr>();
  return attr;
}

::mlir::LogicalResult CholeskyOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_is_upper = odsAttrs.get("is_upper");
  if (!tblgen_is_upper) return emitError(loc, "'lmhlo_gpu.cholesky' op ""requires attribute 'is_upper'");
    if (!((tblgen_is_upper.isa<::mlir::BoolAttr>()))) return emitError(loc, "'lmhlo_gpu.cholesky' op ""attribute 'is_upper' failed to satisfy constraint: bool attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef CholeskyOp::getOperationName() {
  return "lmhlo_gpu.cholesky";
}

std::pair<unsigned, unsigned> CholeskyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CholeskyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CholeskyOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value CholeskyOp::output() {
  return *getODSOperands(1).begin();
}

::mlir::Value CholeskyOp::scratch() {
  return *getODSOperands(2).begin();
}

::mlir::Value CholeskyOp::info() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange CholeskyOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CholeskyOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CholeskyOp::scratchMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CholeskyOp::infoMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CholeskyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CholeskyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::BoolAttr CholeskyOp::is_upperAttr() {
  return this->getAttr("is_upper").cast<::mlir::BoolAttr>();
}

bool CholeskyOp::is_upper() {
  auto attr = is_upperAttr();
  return attr.getValue();
}

void CholeskyOp::is_upperAttr(::mlir::BoolAttr attr) {
  this->getOperation()->setAttr("is_upper", attr);
}

void CholeskyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_upper) {
  odsState.addOperands(input);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  odsState.addOperands(info);
  odsState.addAttribute("is_upper", is_upper);
}

void CholeskyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, ::mlir::BoolAttr is_upper) {
  odsState.addOperands(input);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  odsState.addOperands(info);
  odsState.addAttribute("is_upper", is_upper);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CholeskyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_upper) {
  odsState.addOperands(input);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  odsState.addOperands(info);
  odsState.addAttribute("is_upper", odsBuilder.getBoolAttr(is_upper));
}

void CholeskyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, ::mlir::Value scratch, ::mlir::Value info, bool is_upper) {
  odsState.addOperands(input);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  odsState.addOperands(info);
  odsState.addAttribute("is_upper", odsBuilder.getBoolAttr(is_upper));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CholeskyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CholeskyOp::verify() {
  if (failed(CholeskyOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8)))) && (((v.getType().cast<::mlir::ShapedType>().hasRank())) && ((v.getType().cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
        return emitOpError("operand #") << index << " must be 1D memref of 8-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))))) {
        return emitOpError("operand #") << index << " must be memref of 32-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void CholeskyOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(3))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardFilterOp definitions
//===----------------------------------------------------------------------===//

ConvBackwardFilterOpAdaptor::ConvBackwardFilterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvBackwardFilterOpAdaptor::ConvBackwardFilterOpAdaptor(ConvBackwardFilterOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvBackwardFilterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConvBackwardFilterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvBackwardFilterOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvBackwardFilterOpAdaptor::d_output() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvBackwardFilterOpAdaptor::d_filter() {
  return *getODSOperands(2).begin();
}

::mlir::Value ConvBackwardFilterOpAdaptor::scratch() {
  return *getODSOperands(3).begin();
}

::mlir::DenseIntElementsAttr ConvBackwardFilterOpAdaptor::window_strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("window_strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvBackwardFilterOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvBackwardFilterOpAdaptor::lhs_dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("lhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvBackwardFilterOpAdaptor::rhs_dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("rhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::mhlo::ConvDimensionNumbers ConvBackwardFilterOpAdaptor::dimension_numbers() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::mhlo::ConvDimensionNumbers attr = odsAttrs.get("dimension_numbers").cast<::mlir::mhlo::ConvDimensionNumbers>();
  return attr;
}

::mlir::IntegerAttr ConvBackwardFilterOpAdaptor::feature_group_count() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("feature_group_count").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr ConvBackwardFilterOpAdaptor::batch_group_count() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("batch_group_count").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::ArrayAttr ConvBackwardFilterOpAdaptor::precision_config() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("precision_config").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::FloatAttr ConvBackwardFilterOpAdaptor::result_scale() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("result_scale").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvBackwardFilterOpAdaptor::backend_config() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig attr = odsAttrs.get("backend_config").cast<::mlir::lmhlo_gpu::ConvolutionBackendConfig>();
  return attr;
}

::mlir::LogicalResult ConvBackwardFilterOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_window_strides = odsAttrs.get("window_strides");
  if (tblgen_window_strides) {
    if (!(((tblgen_window_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_window_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""attribute 'window_strides' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_lhs_dilation = odsAttrs.get("lhs_dilation");
  if (tblgen_lhs_dilation) {
    if (!(((tblgen_lhs_dilation.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_lhs_dilation.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""attribute 'lhs_dilation' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_rhs_dilation = odsAttrs.get("rhs_dilation");
  if (tblgen_rhs_dilation) {
    if (!(((tblgen_rhs_dilation.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_rhs_dilation.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""attribute 'rhs_dilation' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_dimension_numbers = odsAttrs.get("dimension_numbers");
  if (!tblgen_dimension_numbers) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""requires attribute 'dimension_numbers'");
    if (!((tblgen_dimension_numbers.isa<::mlir::mhlo::ConvDimensionNumbers>()))) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""attribute 'dimension_numbers' failed to satisfy constraint: Structure of dimension information for conv op");
  }
  {
  auto tblgen_feature_group_count = odsAttrs.get("feature_group_count");
  if (!tblgen_feature_group_count) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""requires attribute 'feature_group_count'");
    if (!(((tblgen_feature_group_count.isa<::mlir::IntegerAttr>())) && ((tblgen_feature_group_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""attribute 'feature_group_count' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_batch_group_count = odsAttrs.get("batch_group_count");
  if (!tblgen_batch_group_count) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""requires attribute 'batch_group_count'");
    if (!(((tblgen_batch_group_count.isa<::mlir::IntegerAttr>())) && ((tblgen_batch_group_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""attribute 'batch_group_count' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_precision_config = odsAttrs.get("precision_config");
  if (tblgen_precision_config) {
    if (!(((tblgen_precision_config.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_precision_config.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::StringAttr>())) && (((attr.cast<::mlir::StringAttr>().getValue() == "DEFAULT")) || ((attr.cast<::mlir::StringAttr>().getValue() == "HIGH")) || ((attr.cast<::mlir::StringAttr>().getValue() == "HIGHEST"))); })))) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""attribute 'precision_config' failed to satisfy constraint: Precision Config attribute");
  }
  }
  {
  auto tblgen_result_scale = odsAttrs.get("result_scale");
  if (!tblgen_result_scale) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""requires attribute 'result_scale'");
    if (!(((tblgen_result_scale.isa<::mlir::FloatAttr>())) && ((tblgen_result_scale.cast<::mlir::FloatAttr>().getType().isF64())))) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""attribute 'result_scale' failed to satisfy constraint: 64-bit float attribute");
  }
  {
  auto tblgen_backend_config = odsAttrs.get("backend_config");
  if (!tblgen_backend_config) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""requires attribute 'backend_config'");
    if (!((tblgen_backend_config.isa<::mlir::lmhlo_gpu::ConvolutionBackendConfig>()))) return emitError(loc, "'lmhlo_gpu.conv_backwardfilter' op ""attribute 'backend_config' failed to satisfy constraint: GPU Convolution backend configuration");
  }
  return ::mlir::success();
}

::llvm::StringRef ConvBackwardFilterOp::getOperationName() {
  return "lmhlo_gpu.conv_backwardfilter";
}

std::pair<unsigned, unsigned> ConvBackwardFilterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConvBackwardFilterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvBackwardFilterOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvBackwardFilterOp::d_output() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvBackwardFilterOp::d_filter() {
  return *getODSOperands(2).begin();
}

::mlir::Value ConvBackwardFilterOp::scratch() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange ConvBackwardFilterOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvBackwardFilterOp::d_outputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvBackwardFilterOp::d_filterMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvBackwardFilterOp::scratchMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConvBackwardFilterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConvBackwardFilterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr ConvBackwardFilterOp::window_stridesAttr() {
  return this->getAttr("window_strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvBackwardFilterOp::window_strides() {
  auto attr = window_stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvBackwardFilterOp::paddingAttr() {
  return this->getAttr("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvBackwardFilterOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvBackwardFilterOp::lhs_dilationAttr() {
  return this->getAttr("lhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvBackwardFilterOp::lhs_dilation() {
  auto attr = lhs_dilationAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvBackwardFilterOp::rhs_dilationAttr() {
  return this->getAttr("rhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvBackwardFilterOp::rhs_dilation() {
  auto attr = rhs_dilationAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::mhlo::ConvDimensionNumbers ConvBackwardFilterOp::dimension_numbersAttr() {
  return this->getAttr("dimension_numbers").cast<::mlir::mhlo::ConvDimensionNumbers>();
}

::mlir::mhlo::ConvDimensionNumbers ConvBackwardFilterOp::dimension_numbers() {
  auto attr = dimension_numbersAttr();
  return attr;
}

::mlir::IntegerAttr ConvBackwardFilterOp::feature_group_countAttr() {
  return this->getAttr("feature_group_count").cast<::mlir::IntegerAttr>();
}

uint64_t ConvBackwardFilterOp::feature_group_count() {
  auto attr = feature_group_countAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConvBackwardFilterOp::batch_group_countAttr() {
  return this->getAttr("batch_group_count").cast<::mlir::IntegerAttr>();
}

uint64_t ConvBackwardFilterOp::batch_group_count() {
  auto attr = batch_group_countAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr ConvBackwardFilterOp::precision_configAttr() {
  return this->getAttr("precision_config").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ConvBackwardFilterOp::precision_config() {
  auto attr = precision_configAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::FloatAttr ConvBackwardFilterOp::result_scaleAttr() {
  return this->getAttr("result_scale").cast<::mlir::FloatAttr>();
}

::llvm::APFloat ConvBackwardFilterOp::result_scale() {
  auto attr = result_scaleAttr();
  return attr.getValue();
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvBackwardFilterOp::backend_configAttr() {
  return this->getAttr("backend_config").cast<::mlir::lmhlo_gpu::ConvolutionBackendConfig>();
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvBackwardFilterOp::backend_config() {
  auto attr = backend_configAttr();
  return attr;
}

void ConvBackwardFilterOp::window_stridesAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("window_strides", attr);
}

void ConvBackwardFilterOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("padding", attr);
}

void ConvBackwardFilterOp::lhs_dilationAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("lhs_dilation", attr);
}

void ConvBackwardFilterOp::rhs_dilationAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("rhs_dilation", attr);
}

void ConvBackwardFilterOp::dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr) {
  this->getOperation()->setAttr("dimension_numbers", attr);
}

void ConvBackwardFilterOp::feature_group_countAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("feature_group_count", attr);
}

void ConvBackwardFilterOp::batch_group_countAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("batch_group_count", attr);
}

void ConvBackwardFilterOp::precision_configAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("precision_config", attr);
}

void ConvBackwardFilterOp::result_scaleAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("result_scale", attr);
}

void ConvBackwardFilterOp::backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr) {
  this->getOperation()->setAttr("backend_config", attr);
}

void ConvBackwardFilterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(d_output);
  odsState.addOperands(d_filter);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", feature_group_count);
  odsState.addAttribute("batch_group_count", batch_group_count);
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", result_scale);
  odsState.addAttribute("backend_config", backend_config);
}

void ConvBackwardFilterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(d_output);
  odsState.addOperands(d_filter);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", feature_group_count);
  odsState.addAttribute("batch_group_count", batch_group_count);
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", result_scale);
  odsState.addAttribute("backend_config", backend_config);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvBackwardFilterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(d_output);
  odsState.addOperands(d_filter);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_group_count));
  odsState.addAttribute("batch_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_group_count));
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), result_scale));
  odsState.addAttribute("backend_config", backend_config);
}

void ConvBackwardFilterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value d_output, ::mlir::Value d_filter, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(d_output);
  odsState.addOperands(d_filter);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_group_count));
  odsState.addAttribute("batch_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_group_count));
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), result_scale));
  odsState.addAttribute("backend_config", backend_config);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvBackwardFilterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConvBackwardFilterOp::verify() {
  if (failed(ConvBackwardFilterOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8)))) && (((v.getType().cast<::mlir::ShapedType>().hasRank())) && ((v.getType().cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
        return emitOpError("operand #") << index << " must be 1D memref of 8-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void ConvBackwardFilterOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(3))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvBackwardInputOp definitions
//===----------------------------------------------------------------------===//

ConvBackwardInputOpAdaptor::ConvBackwardInputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvBackwardInputOpAdaptor::ConvBackwardInputOpAdaptor(ConvBackwardInputOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvBackwardInputOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConvBackwardInputOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvBackwardInputOpAdaptor::d_output() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvBackwardInputOpAdaptor::filter() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvBackwardInputOpAdaptor::d_input() {
  return *getODSOperands(2).begin();
}

::mlir::Value ConvBackwardInputOpAdaptor::scratch() {
  return *getODSOperands(3).begin();
}

::mlir::DenseIntElementsAttr ConvBackwardInputOpAdaptor::window_strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("window_strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvBackwardInputOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvBackwardInputOpAdaptor::lhs_dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("lhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvBackwardInputOpAdaptor::rhs_dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("rhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::mhlo::ConvDimensionNumbers ConvBackwardInputOpAdaptor::dimension_numbers() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::mhlo::ConvDimensionNumbers attr = odsAttrs.get("dimension_numbers").cast<::mlir::mhlo::ConvDimensionNumbers>();
  return attr;
}

::mlir::IntegerAttr ConvBackwardInputOpAdaptor::feature_group_count() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("feature_group_count").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr ConvBackwardInputOpAdaptor::batch_group_count() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("batch_group_count").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::ArrayAttr ConvBackwardInputOpAdaptor::precision_config() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("precision_config").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::FloatAttr ConvBackwardInputOpAdaptor::result_scale() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("result_scale").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvBackwardInputOpAdaptor::backend_config() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig attr = odsAttrs.get("backend_config").cast<::mlir::lmhlo_gpu::ConvolutionBackendConfig>();
  return attr;
}

::mlir::LogicalResult ConvBackwardInputOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_window_strides = odsAttrs.get("window_strides");
  if (tblgen_window_strides) {
    if (!(((tblgen_window_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_window_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""attribute 'window_strides' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_lhs_dilation = odsAttrs.get("lhs_dilation");
  if (tblgen_lhs_dilation) {
    if (!(((tblgen_lhs_dilation.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_lhs_dilation.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""attribute 'lhs_dilation' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_rhs_dilation = odsAttrs.get("rhs_dilation");
  if (tblgen_rhs_dilation) {
    if (!(((tblgen_rhs_dilation.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_rhs_dilation.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""attribute 'rhs_dilation' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_dimension_numbers = odsAttrs.get("dimension_numbers");
  if (!tblgen_dimension_numbers) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""requires attribute 'dimension_numbers'");
    if (!((tblgen_dimension_numbers.isa<::mlir::mhlo::ConvDimensionNumbers>()))) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""attribute 'dimension_numbers' failed to satisfy constraint: Structure of dimension information for conv op");
  }
  {
  auto tblgen_feature_group_count = odsAttrs.get("feature_group_count");
  if (!tblgen_feature_group_count) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""requires attribute 'feature_group_count'");
    if (!(((tblgen_feature_group_count.isa<::mlir::IntegerAttr>())) && ((tblgen_feature_group_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""attribute 'feature_group_count' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_batch_group_count = odsAttrs.get("batch_group_count");
  if (!tblgen_batch_group_count) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""requires attribute 'batch_group_count'");
    if (!(((tblgen_batch_group_count.isa<::mlir::IntegerAttr>())) && ((tblgen_batch_group_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""attribute 'batch_group_count' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_precision_config = odsAttrs.get("precision_config");
  if (tblgen_precision_config) {
    if (!(((tblgen_precision_config.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_precision_config.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::StringAttr>())) && (((attr.cast<::mlir::StringAttr>().getValue() == "DEFAULT")) || ((attr.cast<::mlir::StringAttr>().getValue() == "HIGH")) || ((attr.cast<::mlir::StringAttr>().getValue() == "HIGHEST"))); })))) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""attribute 'precision_config' failed to satisfy constraint: Precision Config attribute");
  }
  }
  {
  auto tblgen_result_scale = odsAttrs.get("result_scale");
  if (!tblgen_result_scale) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""requires attribute 'result_scale'");
    if (!(((tblgen_result_scale.isa<::mlir::FloatAttr>())) && ((tblgen_result_scale.cast<::mlir::FloatAttr>().getType().isF64())))) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""attribute 'result_scale' failed to satisfy constraint: 64-bit float attribute");
  }
  {
  auto tblgen_backend_config = odsAttrs.get("backend_config");
  if (!tblgen_backend_config) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""requires attribute 'backend_config'");
    if (!((tblgen_backend_config.isa<::mlir::lmhlo_gpu::ConvolutionBackendConfig>()))) return emitError(loc, "'lmhlo_gpu.conv_backwardinput' op ""attribute 'backend_config' failed to satisfy constraint: GPU Convolution backend configuration");
  }
  return ::mlir::success();
}

::llvm::StringRef ConvBackwardInputOp::getOperationName() {
  return "lmhlo_gpu.conv_backwardinput";
}

std::pair<unsigned, unsigned> ConvBackwardInputOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConvBackwardInputOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvBackwardInputOp::d_output() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvBackwardInputOp::filter() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvBackwardInputOp::d_input() {
  return *getODSOperands(2).begin();
}

::mlir::Value ConvBackwardInputOp::scratch() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange ConvBackwardInputOp::d_outputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvBackwardInputOp::filterMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvBackwardInputOp::d_inputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvBackwardInputOp::scratchMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConvBackwardInputOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConvBackwardInputOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr ConvBackwardInputOp::window_stridesAttr() {
  return this->getAttr("window_strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvBackwardInputOp::window_strides() {
  auto attr = window_stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvBackwardInputOp::paddingAttr() {
  return this->getAttr("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvBackwardInputOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvBackwardInputOp::lhs_dilationAttr() {
  return this->getAttr("lhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvBackwardInputOp::lhs_dilation() {
  auto attr = lhs_dilationAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvBackwardInputOp::rhs_dilationAttr() {
  return this->getAttr("rhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvBackwardInputOp::rhs_dilation() {
  auto attr = rhs_dilationAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::mhlo::ConvDimensionNumbers ConvBackwardInputOp::dimension_numbersAttr() {
  return this->getAttr("dimension_numbers").cast<::mlir::mhlo::ConvDimensionNumbers>();
}

::mlir::mhlo::ConvDimensionNumbers ConvBackwardInputOp::dimension_numbers() {
  auto attr = dimension_numbersAttr();
  return attr;
}

::mlir::IntegerAttr ConvBackwardInputOp::feature_group_countAttr() {
  return this->getAttr("feature_group_count").cast<::mlir::IntegerAttr>();
}

uint64_t ConvBackwardInputOp::feature_group_count() {
  auto attr = feature_group_countAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConvBackwardInputOp::batch_group_countAttr() {
  return this->getAttr("batch_group_count").cast<::mlir::IntegerAttr>();
}

uint64_t ConvBackwardInputOp::batch_group_count() {
  auto attr = batch_group_countAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr ConvBackwardInputOp::precision_configAttr() {
  return this->getAttr("precision_config").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ConvBackwardInputOp::precision_config() {
  auto attr = precision_configAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::FloatAttr ConvBackwardInputOp::result_scaleAttr() {
  return this->getAttr("result_scale").cast<::mlir::FloatAttr>();
}

::llvm::APFloat ConvBackwardInputOp::result_scale() {
  auto attr = result_scaleAttr();
  return attr.getValue();
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvBackwardInputOp::backend_configAttr() {
  return this->getAttr("backend_config").cast<::mlir::lmhlo_gpu::ConvolutionBackendConfig>();
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvBackwardInputOp::backend_config() {
  auto attr = backend_configAttr();
  return attr;
}

void ConvBackwardInputOp::window_stridesAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("window_strides", attr);
}

void ConvBackwardInputOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("padding", attr);
}

void ConvBackwardInputOp::lhs_dilationAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("lhs_dilation", attr);
}

void ConvBackwardInputOp::rhs_dilationAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("rhs_dilation", attr);
}

void ConvBackwardInputOp::dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr) {
  this->getOperation()->setAttr("dimension_numbers", attr);
}

void ConvBackwardInputOp::feature_group_countAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("feature_group_count", attr);
}

void ConvBackwardInputOp::batch_group_countAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("batch_group_count", attr);
}

void ConvBackwardInputOp::precision_configAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("precision_config", attr);
}

void ConvBackwardInputOp::result_scaleAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("result_scale", attr);
}

void ConvBackwardInputOp::backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr) {
  this->getOperation()->setAttr("backend_config", attr);
}

void ConvBackwardInputOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(d_output);
  odsState.addOperands(filter);
  odsState.addOperands(d_input);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", feature_group_count);
  odsState.addAttribute("batch_group_count", batch_group_count);
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", result_scale);
  odsState.addAttribute("backend_config", backend_config);
}

void ConvBackwardInputOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(d_output);
  odsState.addOperands(filter);
  odsState.addOperands(d_input);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", feature_group_count);
  odsState.addAttribute("batch_group_count", batch_group_count);
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", result_scale);
  odsState.addAttribute("backend_config", backend_config);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvBackwardInputOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(d_output);
  odsState.addOperands(filter);
  odsState.addOperands(d_input);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_group_count));
  odsState.addAttribute("batch_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_group_count));
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), result_scale));
  odsState.addAttribute("backend_config", backend_config);
}

void ConvBackwardInputOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value d_output, ::mlir::Value filter, ::mlir::Value d_input, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(d_output);
  odsState.addOperands(filter);
  odsState.addOperands(d_input);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_group_count));
  odsState.addAttribute("batch_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_group_count));
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), result_scale));
  odsState.addAttribute("backend_config", backend_config);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvBackwardInputOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConvBackwardInputOp::verify() {
  if (failed(ConvBackwardInputOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8)))) && (((v.getType().cast<::mlir::ShapedType>().hasRank())) && ((v.getType().cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
        return emitOpError("operand #") << index << " must be 1D memref of 8-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void ConvBackwardInputOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(3))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardFusedOp definitions
//===----------------------------------------------------------------------===//

ConvForwardFusedOpAdaptor::ConvForwardFusedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvForwardFusedOpAdaptor::ConvForwardFusedOpAdaptor(ConvForwardFusedOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvForwardFusedOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConvForwardFusedOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvForwardFusedOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvForwardFusedOpAdaptor::filter() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvForwardFusedOpAdaptor::bias() {
  return *getODSOperands(2).begin();
}

::mlir::Value ConvForwardFusedOpAdaptor::side_input() {
  return *getODSOperands(3).begin();
}

::mlir::Value ConvForwardFusedOpAdaptor::output() {
  return *getODSOperands(4).begin();
}

::mlir::Value ConvForwardFusedOpAdaptor::scratch() {
  return *getODSOperands(5).begin();
}

::mlir::DenseIntElementsAttr ConvForwardFusedOpAdaptor::window_strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("window_strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvForwardFusedOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvForwardFusedOpAdaptor::lhs_dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("lhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvForwardFusedOpAdaptor::rhs_dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("rhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::mhlo::ConvDimensionNumbers ConvForwardFusedOpAdaptor::dimension_numbers() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::mhlo::ConvDimensionNumbers attr = odsAttrs.get("dimension_numbers").cast<::mlir::mhlo::ConvDimensionNumbers>();
  return attr;
}

::mlir::IntegerAttr ConvForwardFusedOpAdaptor::feature_group_count() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("feature_group_count").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr ConvForwardFusedOpAdaptor::batch_group_count() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("batch_group_count").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::ArrayAttr ConvForwardFusedOpAdaptor::precision_config() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("precision_config").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::FloatAttr ConvForwardFusedOpAdaptor::result_scale() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("result_scale").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::StringAttr ConvForwardFusedOpAdaptor::activation_mode() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("activation_mode").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::FloatAttr ConvForwardFusedOpAdaptor::side_input_scale() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("side_input_scale").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvForwardFusedOpAdaptor::backend_config() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig attr = odsAttrs.get("backend_config").cast<::mlir::lmhlo_gpu::ConvolutionBackendConfig>();
  return attr;
}

::mlir::LogicalResult ConvForwardFusedOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_window_strides = odsAttrs.get("window_strides");
  if (tblgen_window_strides) {
    if (!(((tblgen_window_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_window_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'window_strides' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_lhs_dilation = odsAttrs.get("lhs_dilation");
  if (tblgen_lhs_dilation) {
    if (!(((tblgen_lhs_dilation.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_lhs_dilation.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'lhs_dilation' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_rhs_dilation = odsAttrs.get("rhs_dilation");
  if (tblgen_rhs_dilation) {
    if (!(((tblgen_rhs_dilation.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_rhs_dilation.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'rhs_dilation' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_dimension_numbers = odsAttrs.get("dimension_numbers");
  if (!tblgen_dimension_numbers) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""requires attribute 'dimension_numbers'");
    if (!((tblgen_dimension_numbers.isa<::mlir::mhlo::ConvDimensionNumbers>()))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'dimension_numbers' failed to satisfy constraint: Structure of dimension information for conv op");
  }
  {
  auto tblgen_feature_group_count = odsAttrs.get("feature_group_count");
  if (!tblgen_feature_group_count) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""requires attribute 'feature_group_count'");
    if (!(((tblgen_feature_group_count.isa<::mlir::IntegerAttr>())) && ((tblgen_feature_group_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'feature_group_count' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_batch_group_count = odsAttrs.get("batch_group_count");
  if (!tblgen_batch_group_count) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""requires attribute 'batch_group_count'");
    if (!(((tblgen_batch_group_count.isa<::mlir::IntegerAttr>())) && ((tblgen_batch_group_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'batch_group_count' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_precision_config = odsAttrs.get("precision_config");
  if (tblgen_precision_config) {
    if (!(((tblgen_precision_config.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_precision_config.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::StringAttr>())) && (((attr.cast<::mlir::StringAttr>().getValue() == "DEFAULT")) || ((attr.cast<::mlir::StringAttr>().getValue() == "HIGH")) || ((attr.cast<::mlir::StringAttr>().getValue() == "HIGHEST"))); })))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'precision_config' failed to satisfy constraint: Precision Config attribute");
  }
  }
  {
  auto tblgen_result_scale = odsAttrs.get("result_scale");
  if (!tblgen_result_scale) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""requires attribute 'result_scale'");
    if (!(((tblgen_result_scale.isa<::mlir::FloatAttr>())) && ((tblgen_result_scale.cast<::mlir::FloatAttr>().getType().isF64())))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'result_scale' failed to satisfy constraint: 64-bit float attribute");
  }
  {
  auto tblgen_activation_mode = odsAttrs.get("activation_mode");
  if (!tblgen_activation_mode) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""requires attribute 'activation_mode'");
    if (!(((tblgen_activation_mode.isa<::mlir::StringAttr>())) && (((tblgen_activation_mode.cast<::mlir::StringAttr>().getValue() == "None")) || ((tblgen_activation_mode.cast<::mlir::StringAttr>().getValue() == "Sigmoid")) || ((tblgen_activation_mode.cast<::mlir::StringAttr>().getValue() == "Relu")) || ((tblgen_activation_mode.cast<::mlir::StringAttr>().getValue() == "Relu")) || ((tblgen_activation_mode.cast<::mlir::StringAttr>().getValue() == "Relu6")) || ((tblgen_activation_mode.cast<::mlir::StringAttr>().getValue() == "ReluX")) || ((tblgen_activation_mode.cast<::mlir::StringAttr>().getValue() == "BandPass"))))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'activation_mode' failed to satisfy constraint: Activation applied with fused convolution");
  }
  {
  auto tblgen_side_input_scale = odsAttrs.get("side_input_scale");
  if (!tblgen_side_input_scale) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""requires attribute 'side_input_scale'");
    if (!(((tblgen_side_input_scale.isa<::mlir::FloatAttr>())) && ((tblgen_side_input_scale.cast<::mlir::FloatAttr>().getType().isF64())))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'side_input_scale' failed to satisfy constraint: 64-bit float attribute");
  }
  {
  auto tblgen_backend_config = odsAttrs.get("backend_config");
  if (!tblgen_backend_config) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""requires attribute 'backend_config'");
    if (!((tblgen_backend_config.isa<::mlir::lmhlo_gpu::ConvolutionBackendConfig>()))) return emitError(loc, "'lmhlo_gpu.conv_forward_fused' op ""attribute 'backend_config' failed to satisfy constraint: GPU Convolution backend configuration");
  }
  return ::mlir::success();
}

::llvm::StringRef ConvForwardFusedOp::getOperationName() {
  return "lmhlo_gpu.conv_forward_fused";
}

std::pair<unsigned, unsigned> ConvForwardFusedOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConvForwardFusedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvForwardFusedOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvForwardFusedOp::filter() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvForwardFusedOp::bias() {
  return *getODSOperands(2).begin();
}

::mlir::Value ConvForwardFusedOp::side_input() {
  return *getODSOperands(3).begin();
}

::mlir::Value ConvForwardFusedOp::output() {
  return *getODSOperands(4).begin();
}

::mlir::Value ConvForwardFusedOp::scratch() {
  return *getODSOperands(5).begin();
}

::mlir::MutableOperandRange ConvForwardFusedOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvForwardFusedOp::filterMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvForwardFusedOp::biasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvForwardFusedOp::side_inputMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvForwardFusedOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvForwardFusedOp::scratchMutable() {
  auto range = getODSOperandIndexAndLength(5);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConvForwardFusedOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConvForwardFusedOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr ConvForwardFusedOp::window_stridesAttr() {
  return this->getAttr("window_strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvForwardFusedOp::window_strides() {
  auto attr = window_stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvForwardFusedOp::paddingAttr() {
  return this->getAttr("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvForwardFusedOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvForwardFusedOp::lhs_dilationAttr() {
  return this->getAttr("lhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvForwardFusedOp::lhs_dilation() {
  auto attr = lhs_dilationAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvForwardFusedOp::rhs_dilationAttr() {
  return this->getAttr("rhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvForwardFusedOp::rhs_dilation() {
  auto attr = rhs_dilationAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::mhlo::ConvDimensionNumbers ConvForwardFusedOp::dimension_numbersAttr() {
  return this->getAttr("dimension_numbers").cast<::mlir::mhlo::ConvDimensionNumbers>();
}

::mlir::mhlo::ConvDimensionNumbers ConvForwardFusedOp::dimension_numbers() {
  auto attr = dimension_numbersAttr();
  return attr;
}

::mlir::IntegerAttr ConvForwardFusedOp::feature_group_countAttr() {
  return this->getAttr("feature_group_count").cast<::mlir::IntegerAttr>();
}

uint64_t ConvForwardFusedOp::feature_group_count() {
  auto attr = feature_group_countAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConvForwardFusedOp::batch_group_countAttr() {
  return this->getAttr("batch_group_count").cast<::mlir::IntegerAttr>();
}

uint64_t ConvForwardFusedOp::batch_group_count() {
  auto attr = batch_group_countAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr ConvForwardFusedOp::precision_configAttr() {
  return this->getAttr("precision_config").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ConvForwardFusedOp::precision_config() {
  auto attr = precision_configAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::FloatAttr ConvForwardFusedOp::result_scaleAttr() {
  return this->getAttr("result_scale").cast<::mlir::FloatAttr>();
}

::llvm::APFloat ConvForwardFusedOp::result_scale() {
  auto attr = result_scaleAttr();
  return attr.getValue();
}

::mlir::StringAttr ConvForwardFusedOp::activation_modeAttr() {
  return this->getAttr("activation_mode").cast<::mlir::StringAttr>();
}

::llvm::StringRef ConvForwardFusedOp::activation_mode() {
  auto attr = activation_modeAttr();
  return attr.getValue();
}

::mlir::FloatAttr ConvForwardFusedOp::side_input_scaleAttr() {
  return this->getAttr("side_input_scale").cast<::mlir::FloatAttr>();
}

::llvm::APFloat ConvForwardFusedOp::side_input_scale() {
  auto attr = side_input_scaleAttr();
  return attr.getValue();
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvForwardFusedOp::backend_configAttr() {
  return this->getAttr("backend_config").cast<::mlir::lmhlo_gpu::ConvolutionBackendConfig>();
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvForwardFusedOp::backend_config() {
  auto attr = backend_configAttr();
  return attr;
}

void ConvForwardFusedOp::window_stridesAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("window_strides", attr);
}

void ConvForwardFusedOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("padding", attr);
}

void ConvForwardFusedOp::lhs_dilationAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("lhs_dilation", attr);
}

void ConvForwardFusedOp::rhs_dilationAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("rhs_dilation", attr);
}

void ConvForwardFusedOp::dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr) {
  this->getOperation()->setAttr("dimension_numbers", attr);
}

void ConvForwardFusedOp::feature_group_countAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("feature_group_count", attr);
}

void ConvForwardFusedOp::batch_group_countAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("batch_group_count", attr);
}

void ConvForwardFusedOp::precision_configAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("precision_config", attr);
}

void ConvForwardFusedOp::result_scaleAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("result_scale", attr);
}

void ConvForwardFusedOp::activation_modeAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("activation_mode", attr);
}

void ConvForwardFusedOp::side_input_scaleAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("side_input_scale", attr);
}

void ConvForwardFusedOp::backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr) {
  this->getOperation()->setAttr("backend_config", attr);
}

void ConvForwardFusedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::StringAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.addOperands(side_input);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", feature_group_count);
  odsState.addAttribute("batch_group_count", batch_group_count);
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", result_scale);
  odsState.addAttribute("activation_mode", activation_mode);
  odsState.addAttribute("side_input_scale", side_input_scale);
  odsState.addAttribute("backend_config", backend_config);
}

void ConvForwardFusedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::StringAttr activation_mode, ::mlir::FloatAttr side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.addOperands(side_input);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", feature_group_count);
  odsState.addAttribute("batch_group_count", batch_group_count);
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", result_scale);
  odsState.addAttribute("activation_mode", activation_mode);
  odsState.addAttribute("side_input_scale", side_input_scale);
  odsState.addAttribute("backend_config", backend_config);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvForwardFusedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::llvm::StringRef activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.addOperands(side_input);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_group_count));
  odsState.addAttribute("batch_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_group_count));
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), result_scale));
  odsState.addAttribute("activation_mode", odsBuilder.getStringAttr(activation_mode));
  odsState.addAttribute("side_input_scale", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), side_input_scale));
  odsState.addAttribute("backend_config", backend_config);
}

void ConvForwardFusedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value bias, ::mlir::Value side_input, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::llvm::StringRef activation_mode, ::llvm::APFloat side_input_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(bias);
  odsState.addOperands(side_input);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_group_count));
  odsState.addAttribute("batch_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_group_count));
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), result_scale));
  odsState.addAttribute("activation_mode", odsBuilder.getStringAttr(activation_mode));
  odsState.addAttribute("side_input_scale", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), side_input_scale));
  odsState.addAttribute("backend_config", backend_config);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvForwardFusedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 6u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConvForwardFusedOp::verify() {
  if (failed(ConvForwardFusedOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup5 = getODSOperands(5);
    for (::mlir::Value v : valueGroup5) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8)))) && (((v.getType().cast<::mlir::ShapedType>().hasRank())) && ((v.getType().cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
        return emitOpError("operand #") << index << " must be 1D memref of 8-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void ConvForwardFusedOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(3))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(4))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(5))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::ConvForwardOp definitions
//===----------------------------------------------------------------------===//

ConvForwardOpAdaptor::ConvForwardOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvForwardOpAdaptor::ConvForwardOpAdaptor(ConvForwardOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvForwardOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConvForwardOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvForwardOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvForwardOpAdaptor::filter() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvForwardOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::Value ConvForwardOpAdaptor::scratch() {
  return *getODSOperands(3).begin();
}

::mlir::DenseIntElementsAttr ConvForwardOpAdaptor::window_strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("window_strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvForwardOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvForwardOpAdaptor::lhs_dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("lhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvForwardOpAdaptor::rhs_dilation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("rhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::mhlo::ConvDimensionNumbers ConvForwardOpAdaptor::dimension_numbers() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::mhlo::ConvDimensionNumbers attr = odsAttrs.get("dimension_numbers").cast<::mlir::mhlo::ConvDimensionNumbers>();
  return attr;
}

::mlir::IntegerAttr ConvForwardOpAdaptor::feature_group_count() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("feature_group_count").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr ConvForwardOpAdaptor::batch_group_count() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("batch_group_count").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::ArrayAttr ConvForwardOpAdaptor::precision_config() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("precision_config").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::FloatAttr ConvForwardOpAdaptor::result_scale() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("result_scale").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvForwardOpAdaptor::backend_config() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::lmhlo_gpu::ConvolutionBackendConfig attr = odsAttrs.get("backend_config").cast<::mlir::lmhlo_gpu::ConvolutionBackendConfig>();
  return attr;
}

::mlir::LogicalResult ConvForwardOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_window_strides = odsAttrs.get("window_strides");
  if (tblgen_window_strides) {
    if (!(((tblgen_window_strides.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_window_strides.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""attribute 'window_strides' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_lhs_dilation = odsAttrs.get("lhs_dilation");
  if (tblgen_lhs_dilation) {
    if (!(((tblgen_lhs_dilation.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_lhs_dilation.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""attribute 'lhs_dilation' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_rhs_dilation = odsAttrs.get("rhs_dilation");
  if (tblgen_rhs_dilation) {
    if (!(((tblgen_rhs_dilation.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_rhs_dilation.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""attribute 'rhs_dilation' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_dimension_numbers = odsAttrs.get("dimension_numbers");
  if (!tblgen_dimension_numbers) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""requires attribute 'dimension_numbers'");
    if (!((tblgen_dimension_numbers.isa<::mlir::mhlo::ConvDimensionNumbers>()))) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""attribute 'dimension_numbers' failed to satisfy constraint: Structure of dimension information for conv op");
  }
  {
  auto tblgen_feature_group_count = odsAttrs.get("feature_group_count");
  if (!tblgen_feature_group_count) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""requires attribute 'feature_group_count'");
    if (!(((tblgen_feature_group_count.isa<::mlir::IntegerAttr>())) && ((tblgen_feature_group_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""attribute 'feature_group_count' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_batch_group_count = odsAttrs.get("batch_group_count");
  if (!tblgen_batch_group_count) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""requires attribute 'batch_group_count'");
    if (!(((tblgen_batch_group_count.isa<::mlir::IntegerAttr>())) && ((tblgen_batch_group_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""attribute 'batch_group_count' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_precision_config = odsAttrs.get("precision_config");
  if (tblgen_precision_config) {
    if (!(((tblgen_precision_config.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_precision_config.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::StringAttr>())) && (((attr.cast<::mlir::StringAttr>().getValue() == "DEFAULT")) || ((attr.cast<::mlir::StringAttr>().getValue() == "HIGH")) || ((attr.cast<::mlir::StringAttr>().getValue() == "HIGHEST"))); })))) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""attribute 'precision_config' failed to satisfy constraint: Precision Config attribute");
  }
  }
  {
  auto tblgen_result_scale = odsAttrs.get("result_scale");
  if (!tblgen_result_scale) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""requires attribute 'result_scale'");
    if (!(((tblgen_result_scale.isa<::mlir::FloatAttr>())) && ((tblgen_result_scale.cast<::mlir::FloatAttr>().getType().isF64())))) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""attribute 'result_scale' failed to satisfy constraint: 64-bit float attribute");
  }
  {
  auto tblgen_backend_config = odsAttrs.get("backend_config");
  if (!tblgen_backend_config) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""requires attribute 'backend_config'");
    if (!((tblgen_backend_config.isa<::mlir::lmhlo_gpu::ConvolutionBackendConfig>()))) return emitError(loc, "'lmhlo_gpu.conv_forward' op ""attribute 'backend_config' failed to satisfy constraint: GPU Convolution backend configuration");
  }
  return ::mlir::success();
}

::llvm::StringRef ConvForwardOp::getOperationName() {
  return "lmhlo_gpu.conv_forward";
}

std::pair<unsigned, unsigned> ConvForwardOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConvForwardOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvForwardOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvForwardOp::filter() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvForwardOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::Value ConvForwardOp::scratch() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange ConvForwardOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvForwardOp::filterMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvForwardOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvForwardOp::scratchMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConvForwardOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConvForwardOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr ConvForwardOp::window_stridesAttr() {
  return this->getAttr("window_strides").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvForwardOp::window_strides() {
  auto attr = window_stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvForwardOp::paddingAttr() {
  return this->getAttr("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvForwardOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvForwardOp::lhs_dilationAttr() {
  return this->getAttr("lhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvForwardOp::lhs_dilation() {
  auto attr = lhs_dilationAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvForwardOp::rhs_dilationAttr() {
  return this->getAttr("rhs_dilation").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvForwardOp::rhs_dilation() {
  auto attr = rhs_dilationAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::mhlo::ConvDimensionNumbers ConvForwardOp::dimension_numbersAttr() {
  return this->getAttr("dimension_numbers").cast<::mlir::mhlo::ConvDimensionNumbers>();
}

::mlir::mhlo::ConvDimensionNumbers ConvForwardOp::dimension_numbers() {
  auto attr = dimension_numbersAttr();
  return attr;
}

::mlir::IntegerAttr ConvForwardOp::feature_group_countAttr() {
  return this->getAttr("feature_group_count").cast<::mlir::IntegerAttr>();
}

uint64_t ConvForwardOp::feature_group_count() {
  auto attr = feature_group_countAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConvForwardOp::batch_group_countAttr() {
  return this->getAttr("batch_group_count").cast<::mlir::IntegerAttr>();
}

uint64_t ConvForwardOp::batch_group_count() {
  auto attr = batch_group_countAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr ConvForwardOp::precision_configAttr() {
  return this->getAttr("precision_config").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ConvForwardOp::precision_config() {
  auto attr = precision_configAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::FloatAttr ConvForwardOp::result_scaleAttr() {
  return this->getAttr("result_scale").cast<::mlir::FloatAttr>();
}

::llvm::APFloat ConvForwardOp::result_scale() {
  auto attr = result_scaleAttr();
  return attr.getValue();
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvForwardOp::backend_configAttr() {
  return this->getAttr("backend_config").cast<::mlir::lmhlo_gpu::ConvolutionBackendConfig>();
}

::mlir::lmhlo_gpu::ConvolutionBackendConfig ConvForwardOp::backend_config() {
  auto attr = backend_configAttr();
  return attr;
}

void ConvForwardOp::window_stridesAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("window_strides", attr);
}

void ConvForwardOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("padding", attr);
}

void ConvForwardOp::lhs_dilationAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("lhs_dilation", attr);
}

void ConvForwardOp::rhs_dilationAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("rhs_dilation", attr);
}

void ConvForwardOp::dimension_numbersAttr(::mlir::mhlo::ConvDimensionNumbers attr) {
  this->getOperation()->setAttr("dimension_numbers", attr);
}

void ConvForwardOp::feature_group_countAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("feature_group_count", attr);
}

void ConvForwardOp::batch_group_countAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("batch_group_count", attr);
}

void ConvForwardOp::precision_configAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("precision_config", attr);
}

void ConvForwardOp::result_scaleAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("result_scale", attr);
}

void ConvForwardOp::backend_configAttr(::mlir::lmhlo_gpu::ConvolutionBackendConfig attr) {
  this->getOperation()->setAttr("backend_config", attr);
}

void ConvForwardOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", feature_group_count);
  odsState.addAttribute("batch_group_count", batch_group_count);
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", result_scale);
  odsState.addAttribute("backend_config", backend_config);
}

void ConvForwardOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, ::mlir::IntegerAttr feature_group_count, ::mlir::IntegerAttr batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::mlir::FloatAttr result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", feature_group_count);
  odsState.addAttribute("batch_group_count", batch_group_count);
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", result_scale);
  odsState.addAttribute("backend_config", backend_config);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvForwardOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_group_count));
  odsState.addAttribute("batch_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_group_count));
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), result_scale));
  odsState.addAttribute("backend_config", backend_config);
}

void ConvForwardOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::Value output, ::mlir::Value scratch, /*optional*/::mlir::DenseIntElementsAttr window_strides, /*optional*/::mlir::DenseIntElementsAttr padding, /*optional*/::mlir::DenseIntElementsAttr lhs_dilation, /*optional*/::mlir::DenseIntElementsAttr rhs_dilation, ::mlir::mhlo::ConvDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, /*optional*/::mlir::ArrayAttr precision_config, ::llvm::APFloat result_scale, ::mlir::lmhlo_gpu::ConvolutionBackendConfig backend_config) {
  odsState.addOperands(input);
  odsState.addOperands(filter);
  odsState.addOperands(output);
  odsState.addOperands(scratch);
  if (window_strides) {
  odsState.addAttribute("window_strides", window_strides);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  if (lhs_dilation) {
  odsState.addAttribute("lhs_dilation", lhs_dilation);
  }
  if (rhs_dilation) {
  odsState.addAttribute("rhs_dilation", rhs_dilation);
  }
  odsState.addAttribute("dimension_numbers", dimension_numbers);
  odsState.addAttribute("feature_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), feature_group_count));
  odsState.addAttribute("batch_group_count", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_group_count));
  if (precision_config) {
  odsState.addAttribute("precision_config", precision_config);
  }
  odsState.addAttribute("result_scale", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), result_scale));
  odsState.addAttribute("backend_config", backend_config);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvForwardOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConvForwardOp::verify() {
  if (failed(ConvForwardOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8)))) && (((v.getType().cast<::mlir::ShapedType>().hasRank())) && ((v.getType().cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
        return emitOpError("operand #") << index << " must be 1D memref of 8-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void ConvForwardOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(3))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::GEMMOp definitions
//===----------------------------------------------------------------------===//

GEMMOpAdaptor::GEMMOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

GEMMOpAdaptor::GEMMOpAdaptor(GEMMOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> GEMMOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GEMMOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEMMOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value GEMMOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value GEMMOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::mhlo::DotDimensionNumbers GEMMOpAdaptor::dot_dimension_numbers() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::mhlo::DotDimensionNumbers attr = odsAttrs.get("dot_dimension_numbers").cast<::mlir::mhlo::DotDimensionNumbers>();
  return attr;
}

::mlir::FloatAttr GEMMOpAdaptor::alpha() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("alpha").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::IntegerAttr GEMMOpAdaptor::batch_size() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("batch_size").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr GEMMOpAdaptor::algorithm() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("algorithm").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult GEMMOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_dot_dimension_numbers = odsAttrs.get("dot_dimension_numbers");
  if (!tblgen_dot_dimension_numbers) return emitError(loc, "'lmhlo_gpu.gemm' op ""requires attribute 'dot_dimension_numbers'");
    if (!((tblgen_dot_dimension_numbers.isa<::mlir::mhlo::DotDimensionNumbers>()))) return emitError(loc, "'lmhlo_gpu.gemm' op ""attribute 'dot_dimension_numbers' failed to satisfy constraint: Structure of dimension information for dot product");
  }
  {
  auto tblgen_alpha = odsAttrs.get("alpha");
  if (!tblgen_alpha) return emitError(loc, "'lmhlo_gpu.gemm' op ""requires attribute 'alpha'");
    if (!(((tblgen_alpha.isa<::mlir::FloatAttr>())) && ((tblgen_alpha.cast<::mlir::FloatAttr>().getType().isF64())))) return emitError(loc, "'lmhlo_gpu.gemm' op ""attribute 'alpha' failed to satisfy constraint: 64-bit float attribute");
  }
  {
  auto tblgen_batch_size = odsAttrs.get("batch_size");
  if (!tblgen_batch_size) return emitError(loc, "'lmhlo_gpu.gemm' op ""requires attribute 'batch_size'");
    if (!(((tblgen_batch_size.isa<::mlir::IntegerAttr>())) && ((tblgen_batch_size.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.gemm' op ""attribute 'batch_size' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_algorithm = odsAttrs.get("algorithm");
  if (!tblgen_algorithm) return emitError(loc, "'lmhlo_gpu.gemm' op ""requires attribute 'algorithm'");
    if (!(((tblgen_algorithm.isa<::mlir::IntegerAttr>())) && ((tblgen_algorithm.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.gemm' op ""attribute 'algorithm' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef GEMMOp::getOperationName() {
  return "lmhlo_gpu.gemm";
}

std::pair<unsigned, unsigned> GEMMOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GEMMOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEMMOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value GEMMOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value GEMMOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange GEMMOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GEMMOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GEMMOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GEMMOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GEMMOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::mhlo::DotDimensionNumbers GEMMOp::dot_dimension_numbersAttr() {
  return this->getAttr("dot_dimension_numbers").cast<::mlir::mhlo::DotDimensionNumbers>();
}

::mlir::mhlo::DotDimensionNumbers GEMMOp::dot_dimension_numbers() {
  auto attr = dot_dimension_numbersAttr();
  return attr;
}

::mlir::FloatAttr GEMMOp::alphaAttr() {
  return this->getAttr("alpha").cast<::mlir::FloatAttr>();
}

::llvm::APFloat GEMMOp::alpha() {
  auto attr = alphaAttr();
  return attr.getValue();
}

::mlir::IntegerAttr GEMMOp::batch_sizeAttr() {
  return this->getAttr("batch_size").cast<::mlir::IntegerAttr>();
}

uint64_t GEMMOp::batch_size() {
  auto attr = batch_sizeAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr GEMMOp::algorithmAttr() {
  return this->getAttr("algorithm").cast<::mlir::IntegerAttr>();
}

uint64_t GEMMOp::algorithm() {
  auto attr = algorithmAttr();
  return attr.getValue().getZExtValue();
}

void GEMMOp::dot_dimension_numbersAttr(::mlir::mhlo::DotDimensionNumbers attr) {
  this->getOperation()->setAttr("dot_dimension_numbers", attr);
}

void GEMMOp::alphaAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("alpha", attr);
}

void GEMMOp::batch_sizeAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("batch_size", attr);
}

void GEMMOp::algorithmAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("algorithm", attr);
}

void GEMMOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr algorithm) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(output);
  odsState.addAttribute("dot_dimension_numbers", dot_dimension_numbers);
  odsState.addAttribute("alpha", alpha);
  odsState.addAttribute("batch_size", batch_size);
  odsState.addAttribute("algorithm", algorithm);
}

void GEMMOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr algorithm) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(output);
  odsState.addAttribute("dot_dimension_numbers", dot_dimension_numbers);
  odsState.addAttribute("alpha", alpha);
  odsState.addAttribute("batch_size", batch_size);
  odsState.addAttribute("algorithm", algorithm);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GEMMOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha, uint64_t batch_size, uint64_t algorithm) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(output);
  odsState.addAttribute("dot_dimension_numbers", dot_dimension_numbers);
  odsState.addAttribute("alpha", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), alpha));
  odsState.addAttribute("batch_size", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_size));
  odsState.addAttribute("algorithm", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), algorithm));
}

void GEMMOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha, uint64_t batch_size, uint64_t algorithm) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(output);
  odsState.addAttribute("dot_dimension_numbers", dot_dimension_numbers);
  odsState.addAttribute("alpha", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), alpha));
  odsState.addAttribute("batch_size", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_size));
  odsState.addAttribute("algorithm", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), algorithm));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GEMMOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GEMMOp::verify() {
  if (failed(GEMMOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void GEMMOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace lmhlo_gpu
} // namespace mlir
namespace mlir {
namespace lmhlo_gpu {

//===----------------------------------------------------------------------===//
// ::mlir::lmhlo_gpu::GEMM_BiasOp definitions
//===----------------------------------------------------------------------===//

GEMM_BiasOpAdaptor::GEMM_BiasOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

GEMM_BiasOpAdaptor::GEMM_BiasOpAdaptor(GEMM_BiasOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> GEMM_BiasOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GEMM_BiasOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEMM_BiasOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value GEMM_BiasOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value GEMM_BiasOpAdaptor::bias() {
  return *getODSOperands(2).begin();
}

::mlir::Value GEMM_BiasOpAdaptor::output() {
  return *getODSOperands(3).begin();
}

::mlir::mhlo::DotDimensionNumbers GEMM_BiasOpAdaptor::dot_dimension_numbers() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::mhlo::DotDimensionNumbers attr = odsAttrs.get("dot_dimension_numbers").cast<::mlir::mhlo::DotDimensionNumbers>();
  return attr;
}

::mlir::FloatAttr GEMM_BiasOpAdaptor::alpha() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("alpha").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::FloatAttr GEMM_BiasOpAdaptor::beta() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FloatAttr attr = odsAttrs.get("beta").cast<::mlir::FloatAttr>();
  return attr;
}

::mlir::IntegerAttr GEMM_BiasOpAdaptor::batch_size() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("batch_size").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::IntegerAttr GEMM_BiasOpAdaptor::algorithm() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("algorithm").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult GEMM_BiasOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_dot_dimension_numbers = odsAttrs.get("dot_dimension_numbers");
  if (!tblgen_dot_dimension_numbers) return emitError(loc, "'lmhlo_gpu.gemm_bias' op ""requires attribute 'dot_dimension_numbers'");
    if (!((tblgen_dot_dimension_numbers.isa<::mlir::mhlo::DotDimensionNumbers>()))) return emitError(loc, "'lmhlo_gpu.gemm_bias' op ""attribute 'dot_dimension_numbers' failed to satisfy constraint: Structure of dimension information for dot product");
  }
  {
  auto tblgen_alpha = odsAttrs.get("alpha");
  if (!tblgen_alpha) return emitError(loc, "'lmhlo_gpu.gemm_bias' op ""requires attribute 'alpha'");
    if (!(((tblgen_alpha.isa<::mlir::FloatAttr>())) && ((tblgen_alpha.cast<::mlir::FloatAttr>().getType().isF64())))) return emitError(loc, "'lmhlo_gpu.gemm_bias' op ""attribute 'alpha' failed to satisfy constraint: 64-bit float attribute");
  }
  {
  auto tblgen_beta = odsAttrs.get("beta");
  if (!tblgen_beta) return emitError(loc, "'lmhlo_gpu.gemm_bias' op ""requires attribute 'beta'");
    if (!(((tblgen_beta.isa<::mlir::FloatAttr>())) && ((tblgen_beta.cast<::mlir::FloatAttr>().getType().isF64())))) return emitError(loc, "'lmhlo_gpu.gemm_bias' op ""attribute 'beta' failed to satisfy constraint: 64-bit float attribute");
  }
  {
  auto tblgen_batch_size = odsAttrs.get("batch_size");
  if (!tblgen_batch_size) return emitError(loc, "'lmhlo_gpu.gemm_bias' op ""requires attribute 'batch_size'");
    if (!(((tblgen_batch_size.isa<::mlir::IntegerAttr>())) && ((tblgen_batch_size.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.gemm_bias' op ""attribute 'batch_size' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
  auto tblgen_algorithm = odsAttrs.get("algorithm");
  if (!tblgen_algorithm) return emitError(loc, "'lmhlo_gpu.gemm_bias' op ""requires attribute 'algorithm'");
    if (!(((tblgen_algorithm.isa<::mlir::IntegerAttr>())) && ((tblgen_algorithm.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'lmhlo_gpu.gemm_bias' op ""attribute 'algorithm' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef GEMM_BiasOp::getOperationName() {
  return "lmhlo_gpu.gemm_bias";
}

std::pair<unsigned, unsigned> GEMM_BiasOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GEMM_BiasOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEMM_BiasOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value GEMM_BiasOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value GEMM_BiasOp::bias() {
  return *getODSOperands(2).begin();
}

::mlir::Value GEMM_BiasOp::output() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange GEMM_BiasOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GEMM_BiasOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GEMM_BiasOp::biasMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GEMM_BiasOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GEMM_BiasOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GEMM_BiasOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::mhlo::DotDimensionNumbers GEMM_BiasOp::dot_dimension_numbersAttr() {
  return this->getAttr("dot_dimension_numbers").cast<::mlir::mhlo::DotDimensionNumbers>();
}

::mlir::mhlo::DotDimensionNumbers GEMM_BiasOp::dot_dimension_numbers() {
  auto attr = dot_dimension_numbersAttr();
  return attr;
}

::mlir::FloatAttr GEMM_BiasOp::alphaAttr() {
  return this->getAttr("alpha").cast<::mlir::FloatAttr>();
}

::llvm::APFloat GEMM_BiasOp::alpha() {
  auto attr = alphaAttr();
  return attr.getValue();
}

::mlir::FloatAttr GEMM_BiasOp::betaAttr() {
  return this->getAttr("beta").cast<::mlir::FloatAttr>();
}

::llvm::APFloat GEMM_BiasOp::beta() {
  auto attr = betaAttr();
  return attr.getValue();
}

::mlir::IntegerAttr GEMM_BiasOp::batch_sizeAttr() {
  return this->getAttr("batch_size").cast<::mlir::IntegerAttr>();
}

uint64_t GEMM_BiasOp::batch_size() {
  auto attr = batch_sizeAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr GEMM_BiasOp::algorithmAttr() {
  return this->getAttr("algorithm").cast<::mlir::IntegerAttr>();
}

uint64_t GEMM_BiasOp::algorithm() {
  auto attr = algorithmAttr();
  return attr.getValue().getZExtValue();
}

void GEMM_BiasOp::dot_dimension_numbersAttr(::mlir::mhlo::DotDimensionNumbers attr) {
  this->getOperation()->setAttr("dot_dimension_numbers", attr);
}

void GEMM_BiasOp::alphaAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("alpha", attr);
}

void GEMM_BiasOp::betaAttr(::mlir::FloatAttr attr) {
  this->getOperation()->setAttr("beta", attr);
}

void GEMM_BiasOp::batch_sizeAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("batch_size", attr);
}

void GEMM_BiasOp::algorithmAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("algorithm", attr);
}

void GEMM_BiasOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha, ::mlir::FloatAttr beta, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr algorithm) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(bias);
  odsState.addOperands(output);
  odsState.addAttribute("dot_dimension_numbers", dot_dimension_numbers);
  odsState.addAttribute("alpha", alpha);
  odsState.addAttribute("beta", beta);
  odsState.addAttribute("batch_size", batch_size);
  odsState.addAttribute("algorithm", algorithm);
}

void GEMM_BiasOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::mlir::FloatAttr alpha, ::mlir::FloatAttr beta, ::mlir::IntegerAttr batch_size, ::mlir::IntegerAttr algorithm) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(bias);
  odsState.addOperands(output);
  odsState.addAttribute("dot_dimension_numbers", dot_dimension_numbers);
  odsState.addAttribute("alpha", alpha);
  odsState.addAttribute("beta", beta);
  odsState.addAttribute("batch_size", batch_size);
  odsState.addAttribute("algorithm", algorithm);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GEMM_BiasOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha, ::llvm::APFloat beta, uint64_t batch_size, uint64_t algorithm) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(bias);
  odsState.addOperands(output);
  odsState.addAttribute("dot_dimension_numbers", dot_dimension_numbers);
  odsState.addAttribute("alpha", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), alpha));
  odsState.addAttribute("beta", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), beta));
  odsState.addAttribute("batch_size", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_size));
  odsState.addAttribute("algorithm", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), algorithm));
}

void GEMM_BiasOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value bias, ::mlir::Value output, ::mlir::mhlo::DotDimensionNumbers dot_dimension_numbers, ::llvm::APFloat alpha, ::llvm::APFloat beta, uint64_t batch_size, uint64_t algorithm) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(bias);
  odsState.addOperands(output);
  odsState.addAttribute("dot_dimension_numbers", dot_dimension_numbers);
  odsState.addAttribute("alpha", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), alpha));
  odsState.addAttribute("beta", odsBuilder.getFloatAttr(odsBuilder.getF64Type(), beta));
  odsState.addAttribute("batch_size", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), batch_size));
  odsState.addAttribute("algorithm", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), algorithm));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GEMM_BiasOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GEMM_BiasOp::verify() {
  if (failed(GEMM_BiasOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>()))))) {
        return emitOpError("operand #") << index << " must be memref of floating-point or signless integer or complex-type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

void GEMM_BiasOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(3))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace lmhlo_gpu
} // namespace mlir

#endif  // GET_OP_CLASSES

