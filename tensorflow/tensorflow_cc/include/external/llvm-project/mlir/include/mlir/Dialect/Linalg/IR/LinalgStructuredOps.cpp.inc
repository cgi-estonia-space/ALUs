/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::linalg::BatchMatmulOp,
::mlir::linalg::ConvDHWOp,
::mlir::linalg::ConvHWOp,
::mlir::linalg::ConvNCDHWOp,
::mlir::linalg::ConvNCHWOp,
::mlir::linalg::ConvNCWOp,
::mlir::linalg::ConvNDHWCOp,
::mlir::linalg::ConvNHWCOp,
::mlir::linalg::ConvNWCOp,
::mlir::linalg::ConvOp,
::mlir::linalg::ConvWOp,
::mlir::linalg::CopyOp,
::mlir::linalg::DotOp,
::mlir::linalg::FillOp,
::mlir::linalg::GenericOp,
::mlir::linalg::IndexedGenericOp,
::mlir::linalg::MatmulOp,
::mlir::linalg::MatvecOp,
::mlir::linalg::PoolingMaxOp,
::mlir::linalg::PoolingMinOp,
::mlir::linalg::PoolingSumOp,
::mlir::linalg::VecmatOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulOp definitions
//===----------------------------------------------------------------------===//

BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BatchMatmulOpAdaptor::BatchMatmulOpAdaptor(BatchMatmulOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BatchMatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange BatchMatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BatchMatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange BatchMatmulOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange BatchMatmulOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult BatchMatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef BatchMatmulOp::getOperationName() {
  return "linalg.batch_matmul";
}

std::pair<unsigned, unsigned> BatchMatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range BatchMatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BatchMatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range BatchMatmulOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range BatchMatmulOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange BatchMatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange BatchMatmulOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange BatchMatmulOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> BatchMatmulOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range BatchMatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range BatchMatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &BatchMatmulOp::region() {
  return this->getOperation()->getRegion(0);
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<BatchMatmulOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<BatchMatmulOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void BatchMatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult BatchMatmulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<BatchMatmulOp>(parser, result);
}

void BatchMatmulOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult BatchMatmulOp::verify() {
  if (failed(BatchMatmulOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvDHWOp definitions
//===----------------------------------------------------------------------===//

ConvDHWOpAdaptor::ConvDHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvDHWOpAdaptor::ConvDHWOpAdaptor(ConvDHWOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvDHWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ConvDHWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvDHWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvDHWOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange ConvDHWOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult ConvDHWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef ConvDHWOp::getOperationName() {
  return "linalg.conv_3d";
}

std::pair<unsigned, unsigned> ConvDHWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ConvDHWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvDHWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvDHWOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ConvDHWOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ConvDHWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvDHWOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvDHWOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvDHWOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ConvDHWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvDHWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvDHWOp::region() {
  return this->getOperation()->getRegion(0);
}

void ConvDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvDHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void ConvDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvDHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void ConvDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvDHWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvDHWOp>(parser, result);
}

void ConvDHWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvDHWOp::verify() {
  if (failed(ConvDHWOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvHWOp definitions
//===----------------------------------------------------------------------===//

ConvHWOpAdaptor::ConvHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvHWOpAdaptor::ConvHWOpAdaptor(ConvHWOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvHWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ConvHWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvHWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvHWOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange ConvHWOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult ConvHWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef ConvHWOp::getOperationName() {
  return "linalg.conv_2d";
}

std::pair<unsigned, unsigned> ConvHWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ConvHWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvHWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvHWOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ConvHWOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ConvHWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvHWOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvHWOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvHWOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ConvHWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvHWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvHWOp::region() {
  return this->getOperation()->getRegion(0);
}

void ConvHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void ConvHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void ConvHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvHWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvHWOp>(parser, result);
}

void ConvHWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvHWOp::verify() {
  if (failed(ConvHWOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNCDHWOp definitions
//===----------------------------------------------------------------------===//

ConvNCDHWOpAdaptor::ConvNCDHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvNCDHWOpAdaptor::ConvNCDHWOpAdaptor(ConvNCDHWOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvNCDHWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ConvNCDHWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNCDHWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNCDHWOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange ConvNCDHWOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult ConvNCDHWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef ConvNCDHWOp::getOperationName() {
  return "linalg.conv_3d_ncdhw";
}

std::pair<unsigned, unsigned> ConvNCDHWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ConvNCDHWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNCDHWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNCDHWOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ConvNCDHWOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ConvNCDHWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNCDHWOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNCDHWOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNCDHWOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ConvNCDHWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNCDHWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNCDHWOp::region() {
  return this->getOperation()->getRegion(0);
}

void ConvNCDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCDHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void ConvNCDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCDHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void ConvNCDHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNCDHWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNCDHWOp>(parser, result);
}

void ConvNCDHWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNCDHWOp::verify() {
  if (failed(ConvNCDHWOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNCHWOp definitions
//===----------------------------------------------------------------------===//

ConvNCHWOpAdaptor::ConvNCHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvNCHWOpAdaptor::ConvNCHWOpAdaptor(ConvNCHWOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvNCHWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ConvNCHWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNCHWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNCHWOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange ConvNCHWOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult ConvNCHWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef ConvNCHWOp::getOperationName() {
  return "linalg.conv_2d_nchw";
}

std::pair<unsigned, unsigned> ConvNCHWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ConvNCHWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNCHWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNCHWOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ConvNCHWOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ConvNCHWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNCHWOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNCHWOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNCHWOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ConvNCHWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNCHWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNCHWOp::region() {
  return this->getOperation()->getRegion(0);
}

void ConvNCHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void ConvNCHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCHWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void ConvNCHWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNCHWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNCHWOp>(parser, result);
}

void ConvNCHWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNCHWOp::verify() {
  if (failed(ConvNCHWOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNCWOp definitions
//===----------------------------------------------------------------------===//

ConvNCWOpAdaptor::ConvNCWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvNCWOpAdaptor::ConvNCWOpAdaptor(ConvNCWOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvNCWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ConvNCWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNCWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNCWOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange ConvNCWOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult ConvNCWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef ConvNCWOp::getOperationName() {
  return "linalg.conv_1d_ncw";
}

std::pair<unsigned, unsigned> ConvNCWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ConvNCWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNCWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNCWOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ConvNCWOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ConvNCWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNCWOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNCWOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNCWOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ConvNCWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNCWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNCWOp::region() {
  return this->getOperation()->getRegion(0);
}

void ConvNCWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void ConvNCWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNCWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void ConvNCWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNCWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNCWOp>(parser, result);
}

void ConvNCWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNCWOp::verify() {
  if (failed(ConvNCWOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNDHWCOp definitions
//===----------------------------------------------------------------------===//

ConvNDHWCOpAdaptor::ConvNDHWCOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvNDHWCOpAdaptor::ConvNDHWCOpAdaptor(ConvNDHWCOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvNDHWCOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ConvNDHWCOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNDHWCOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNDHWCOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange ConvNDHWCOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult ConvNDHWCOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef ConvNDHWCOp::getOperationName() {
  return "linalg.conv_3d_ndhwc";
}

std::pair<unsigned, unsigned> ConvNDHWCOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ConvNDHWCOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNDHWCOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNDHWCOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ConvNDHWCOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ConvNDHWCOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNDHWCOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNDHWCOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNDHWCOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ConvNDHWCOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNDHWCOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNDHWCOp::region() {
  return this->getOperation()->getRegion(0);
}

void ConvNDHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNDHWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void ConvNDHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNDHWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void ConvNDHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNDHWCOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNDHWCOp>(parser, result);
}

void ConvNDHWCOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNDHWCOp::verify() {
  if (failed(ConvNDHWCOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNHWCOp definitions
//===----------------------------------------------------------------------===//

ConvNHWCOpAdaptor::ConvNHWCOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvNHWCOpAdaptor::ConvNHWCOpAdaptor(ConvNHWCOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvNHWCOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ConvNHWCOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNHWCOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNHWCOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange ConvNHWCOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult ConvNHWCOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef ConvNHWCOp::getOperationName() {
  return "linalg.conv_2d_nhwc";
}

std::pair<unsigned, unsigned> ConvNHWCOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ConvNHWCOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNHWCOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNHWCOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ConvNHWCOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ConvNHWCOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNHWCOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNHWCOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNHWCOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ConvNHWCOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNHWCOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNHWCOp::region() {
  return this->getOperation()->getRegion(0);
}

void ConvNHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNHWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void ConvNHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNHWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void ConvNHWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNHWCOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNHWCOp>(parser, result);
}

void ConvNHWCOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNHWCOp::verify() {
  if (failed(ConvNHWCOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNWCOp definitions
//===----------------------------------------------------------------------===//

ConvNWCOpAdaptor::ConvNWCOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvNWCOpAdaptor::ConvNWCOpAdaptor(ConvNWCOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvNWCOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ConvNWCOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvNWCOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvNWCOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange ConvNWCOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult ConvNWCOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef ConvNWCOp::getOperationName() {
  return "linalg.conv_1d_nwc";
}

std::pair<unsigned, unsigned> ConvNWCOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ConvNWCOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvNWCOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvNWCOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ConvNWCOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ConvNWCOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNWCOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvNWCOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvNWCOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ConvNWCOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvNWCOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvNWCOp::region() {
  return this->getOperation()->getRegion(0);
}

void ConvNWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvNWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void ConvNWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvNWCOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void ConvNWCOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvNWCOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvNWCOp>(parser, result);
}

void ConvNWCOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvNWCOp::verify() {
  if (failed(ConvNWCOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvOp definitions
//===----------------------------------------------------------------------===//

ConvOpAdaptor::ConvOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvOpAdaptor::ConvOpAdaptor(ConvOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConvOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvOpAdaptor::filter() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvOpAdaptor::input() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::ArrayAttr ConvOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ConvOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr ConvOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult ConvOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.conv' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.conv' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.conv' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef ConvOp::getOperationName() {
  return "linalg.conv";
}

std::pair<unsigned, unsigned> ConvOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConvOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConvOp::filter() {
  return *getODSOperands(0).begin();
}

::mlir::Value ConvOp::input() {
  return *getODSOperands(1).begin();
}

::mlir::Value ConvOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange ConvOp::filterMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ConvOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConvOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConvOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr ConvOp::stridesAttr() {
  return this->getAttr("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ConvOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr ConvOp::dilationsAttr() {
  return this->getAttr("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ConvOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr ConvOp::paddingAttr() {
  return this->getAttr("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > ConvOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void ConvOp::stridesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("strides", attr);
}

void ConvOp::dilationsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("dilations", attr);
}

void ConvOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("padding", attr);
}

void ConvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value filter, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(filter);
  odsState.addOperands(input);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
}

void ConvOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value filter, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(filter);
  odsState.addOperands(input);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConvOp::verify() {
  if (failed(ConvOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult ConvOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConvOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.conv";
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvWOp definitions
//===----------------------------------------------------------------------===//

ConvWOpAdaptor::ConvWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConvWOpAdaptor::ConvWOpAdaptor(ConvWOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConvWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ConvWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ConvWOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ConvWOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange ConvWOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult ConvWOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef ConvWOp::getOperationName() {
  return "linalg.conv_1d";
}

std::pair<unsigned, unsigned> ConvWOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ConvWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConvWOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ConvWOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ConvWOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ConvWOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvWOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ConvWOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ConvWOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ConvWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ConvWOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &ConvWOp::region() {
  return this->getOperation()->getRegion(0);
}

void ConvWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<ConvWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void ConvWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<ConvWOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void ConvWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult ConvWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<ConvWOp>(parser, result);
}

void ConvWOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult ConvWOp::verify() {
  if (failed(ConvWOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::CopyOp definitions
//===----------------------------------------------------------------------===//

CopyOpAdaptor::CopyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CopyOpAdaptor::CopyOpAdaptor(CopyOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CopyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CopyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopyOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopyOpAdaptor::output() {
  return *getODSOperands(1).begin();
}

::mlir::AffineMapAttr CopyOpAdaptor::inputPermutation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("inputPermutation").dyn_cast_or_null<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMapAttr CopyOpAdaptor::outputPermutation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("outputPermutation").dyn_cast_or_null<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::LogicalResult CopyOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_inputPermutation = odsAttrs.get("inputPermutation");
  if (tblgen_inputPermutation) {
    if (!((tblgen_inputPermutation.isa<::mlir::AffineMapAttr>()))) return emitError(loc, "'linalg.copy' op ""attribute 'inputPermutation' failed to satisfy constraint: AffineMap attribute");
  }
  }
  {
  auto tblgen_outputPermutation = odsAttrs.get("outputPermutation");
  if (tblgen_outputPermutation) {
    if (!((tblgen_outputPermutation.isa<::mlir::AffineMapAttr>()))) return emitError(loc, "'linalg.copy' op ""attribute 'outputPermutation' failed to satisfy constraint: AffineMap attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef CopyOp::getOperationName() {
  return "linalg.copy";
}

std::pair<unsigned, unsigned> CopyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CopyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopyOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopyOp::output() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CopyOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CopyOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CopyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CopyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::AffineMapAttr CopyOp::inputPermutationAttr() {
  return this->getAttr("inputPermutation").dyn_cast_or_null<::mlir::AffineMapAttr>();
}

::llvm::Optional< ::mlir::AffineMap > CopyOp::inputPermutation() {
  auto attr = inputPermutationAttr();
  return attr ? ::llvm::Optional< ::mlir::AffineMap >(attr.getValue()) : (::llvm::None);
}

::mlir::AffineMapAttr CopyOp::outputPermutationAttr() {
  return this->getAttr("outputPermutation").dyn_cast_or_null<::mlir::AffineMapAttr>();
}

::llvm::Optional< ::mlir::AffineMap > CopyOp::outputPermutation() {
  auto attr = outputPermutationAttr();
  return attr ? ::llvm::Optional< ::mlir::AffineMap >(attr.getValue()) : (::llvm::None);
}

void CopyOp::inputPermutationAttr(::mlir::AffineMapAttr attr) {
  this->getOperation()->setAttr("inputPermutation", attr);
}

void CopyOp::outputPermutationAttr(::mlir::AffineMapAttr attr) {
  this->getOperation()->setAttr("outputPermutation", attr);
}

void CopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value input, Value output) {
    return build(
      odsBuilder, odsState, input, output, AffineMapAttr(), AffineMapAttr());
  
}

void CopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::AffineMapAttr inputPermutation, /*optional*/::mlir::AffineMapAttr outputPermutation) {
  odsState.addOperands(input);
  odsState.addOperands(output);
  if (inputPermutation) {
  odsState.addAttribute("inputPermutation", inputPermutation);
  }
  if (outputPermutation) {
  odsState.addAttribute("outputPermutation", outputPermutation);
  }
}

void CopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::AffineMapAttr inputPermutation, /*optional*/::mlir::AffineMapAttr outputPermutation) {
  odsState.addOperands(input);
  odsState.addOperands(output);
  if (inputPermutation) {
  odsState.addAttribute("inputPermutation", inputPermutation);
  }
  if (outputPermutation) {
  odsState.addAttribute("outputPermutation", outputPermutation);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CopyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CopyOp::verify() {
  if (failed(CopyOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult CopyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CopyOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.copy";
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotOp definitions
//===----------------------------------------------------------------------===//

DotOpAdaptor::DotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

DotOpAdaptor::DotOpAdaptor(DotOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> DotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange DotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DotOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange DotOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange DotOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult DotOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef DotOp::getOperationName() {
  return "linalg.dot";
}

std::pair<unsigned, unsigned> DotOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range DotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DotOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range DotOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DotOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange DotOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DotOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange DotOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> DotOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range DotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range DotOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &DotOp::region() {
  return this->getOperation()->getRegion(0);
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<DotOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<DotOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void DotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult DotOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<DotOp>(parser, result);
}

void DotOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult DotOp::verify() {
  if (failed(DotOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillOp definitions
//===----------------------------------------------------------------------===//

FillOpAdaptor::FillOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

FillOpAdaptor::FillOpAdaptor(FillOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> FillOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FillOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FillOpAdaptor::output() {
  return *getODSOperands(0).begin();
}

::mlir::Value FillOpAdaptor::value() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult FillOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef FillOp::getOperationName() {
  return "linalg.fill";
}

std::pair<unsigned, unsigned> FillOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FillOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FillOp::output() {
  return *getODSOperands(0).begin();
}

::mlir::Value FillOp::value() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FillOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange FillOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FillOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FillOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value output, ::mlir::Value value) {
  odsState.addOperands(output);
  odsState.addOperands(value);
}

void FillOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value output, ::mlir::Value value) {
  odsState.addOperands(output);
  odsState.addOperands(value);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FillOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FillOp::verify() {
  if (failed(FillOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || ((v.getType().isSignlessInteger())) || (((v.getType().isa<::mlir::VectorType>())) && ((true))))) {
        return emitOpError("operand #") << index << " must be floating-point or signless integer or vector of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult FillOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FillOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.fill";
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::GenericOp definitions
//===----------------------------------------------------------------------===//

GenericOpAdaptor::GenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

GenericOpAdaptor::GenericOpAdaptor(GenericOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> GenericOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange GenericOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange GenericOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange GenericOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange GenericOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::ArrayAttr GenericOpAdaptor::indexing_maps() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("indexing_maps").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr GenericOpAdaptor::iterator_types() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("iterator_types").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::StringAttr GenericOpAdaptor::doc() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("doc").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::StringAttr GenericOpAdaptor::library_call() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("library_call").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::IntegerAttr GenericOpAdaptor::symbol_source() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("symbol_source").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult GenericOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    {
  auto tblgen_indexing_maps = odsAttrs.get("indexing_maps");
  if (!tblgen_indexing_maps) return emitError(loc, "'linalg.generic' op ""requires attribute 'indexing_maps'");
    if (!(((tblgen_indexing_maps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_indexing_maps.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return (attr.isa<::mlir::AffineMapAttr>()); })))) return emitError(loc, "'linalg.generic' op ""attribute 'indexing_maps' failed to satisfy constraint: AffineMap array attribute");
  }
  {
  auto tblgen_iterator_types = odsAttrs.get("iterator_types");
  if (!tblgen_iterator_types) return emitError(loc, "'linalg.generic' op ""requires attribute 'iterator_types'");
    if (!((tblgen_iterator_types.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'linalg.generic' op ""attribute 'iterator_types' failed to satisfy constraint: array attribute");
  }
  {
  auto tblgen_doc = odsAttrs.get("doc");
  if (tblgen_doc) {
    if (!((tblgen_doc.isa<::mlir::StringAttr>()))) return emitError(loc, "'linalg.generic' op ""attribute 'doc' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_library_call = odsAttrs.get("library_call");
  if (tblgen_library_call) {
    if (!((tblgen_library_call.isa<::mlir::StringAttr>()))) return emitError(loc, "'linalg.generic' op ""attribute 'library_call' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_symbol_source = odsAttrs.get("symbol_source");
  if (tblgen_symbol_source) {
    if (!((((tblgen_symbol_source.isa<::mlir::IntegerAttr>())) && ((tblgen_symbol_source.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_symbol_source.cast<::mlir::IntegerAttr>().getInt() >= 0)))) return emitError(loc, "'linalg.generic' op ""attribute 'symbol_source' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef GenericOp::getOperationName() {
  return "linalg.generic";
}

std::pair<unsigned, unsigned> GenericOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range GenericOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GenericOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range GenericOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range GenericOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange GenericOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange GenericOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange GenericOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> GenericOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range GenericOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range GenericOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &GenericOp::region() {
  return this->getOperation()->getRegion(0);
}

::mlir::ArrayAttr GenericOp::indexing_mapsAttr() {
  return this->getAttr("indexing_maps").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return attr;
}

::mlir::ArrayAttr GenericOp::iterator_typesAttr() {
  return this->getAttr("iterator_types").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GenericOp::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::StringAttr GenericOp::docAttr() {
  return this->getAttr("doc").dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::doc() {
  auto attr = docAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GenericOp::library_callAttr() {
  return this->getAttr("library_call").dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GenericOp::library_call() {
  auto attr = library_callAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr GenericOp::symbol_sourceAttr() {
  return this->getAttr("symbol_source").dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> GenericOp::symbol_source() {
  auto attr = symbol_sourceAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void GenericOp::indexing_mapsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("indexing_maps", attr);
}

void GenericOp::iterator_typesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("iterator_types", attr);
}

void GenericOp::docAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("doc", attr);
}

void GenericOp::library_callAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("library_call", attr);
}

void GenericOp::symbol_sourceAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("symbol_source", attr);
}









void GenericOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange output_buffers, ::mlir::ValueRange init_tensors, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call, /*optional*/::mlir::IntegerAttr symbol_source) {
  odsState.addOperands(inputs);
  odsState.addOperands(output_buffers);
  odsState.addOperands(init_tensors);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(output_buffers.size()), static_cast<int32_t>(init_tensors.size())}));
  odsState.addAttribute("indexing_maps", indexing_maps);
  odsState.addAttribute("iterator_types", iterator_types);
  if (doc) {
  odsState.addAttribute("doc", doc);
  }
  if (library_call) {
  odsState.addAttribute("library_call", library_call);
  }
  if (symbol_source) {
  odsState.addAttribute("symbol_source", symbol_source);
  }
  (void)odsState.addRegion();
  odsState.addTypes(result_tensors);
}

void GenericOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult GenericOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseGenericOp(parser, result);
}

void GenericOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult GenericOp::verify() {
  if (failed(GenericOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::IndexedGenericOp definitions
//===----------------------------------------------------------------------===//

IndexedGenericOpAdaptor::IndexedGenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

IndexedGenericOpAdaptor::IndexedGenericOpAdaptor(IndexedGenericOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> IndexedGenericOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange IndexedGenericOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange IndexedGenericOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange IndexedGenericOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange IndexedGenericOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::ArrayAttr IndexedGenericOpAdaptor::indexing_maps() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("indexing_maps").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr IndexedGenericOpAdaptor::iterator_types() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("iterator_types").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::StringAttr IndexedGenericOpAdaptor::doc() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("doc").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::StringAttr IndexedGenericOpAdaptor::library_call() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("library_call").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::IntegerAttr IndexedGenericOpAdaptor::symbol_source() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("symbol_source").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult IndexedGenericOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    {
  auto tblgen_indexing_maps = odsAttrs.get("indexing_maps");
  if (!tblgen_indexing_maps) return emitError(loc, "'linalg.indexed_generic' op ""requires attribute 'indexing_maps'");
    if (!(((tblgen_indexing_maps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_indexing_maps.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return (attr.isa<::mlir::AffineMapAttr>()); })))) return emitError(loc, "'linalg.indexed_generic' op ""attribute 'indexing_maps' failed to satisfy constraint: AffineMap array attribute");
  }
  {
  auto tblgen_iterator_types = odsAttrs.get("iterator_types");
  if (!tblgen_iterator_types) return emitError(loc, "'linalg.indexed_generic' op ""requires attribute 'iterator_types'");
    if (!((tblgen_iterator_types.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'linalg.indexed_generic' op ""attribute 'iterator_types' failed to satisfy constraint: array attribute");
  }
  {
  auto tblgen_doc = odsAttrs.get("doc");
  if (tblgen_doc) {
    if (!((tblgen_doc.isa<::mlir::StringAttr>()))) return emitError(loc, "'linalg.indexed_generic' op ""attribute 'doc' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_library_call = odsAttrs.get("library_call");
  if (tblgen_library_call) {
    if (!((tblgen_library_call.isa<::mlir::StringAttr>()))) return emitError(loc, "'linalg.indexed_generic' op ""attribute 'library_call' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_symbol_source = odsAttrs.get("symbol_source");
  if (tblgen_symbol_source) {
    if (!((((tblgen_symbol_source.isa<::mlir::IntegerAttr>())) && ((tblgen_symbol_source.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_symbol_source.cast<::mlir::IntegerAttr>().getInt() >= 0)))) return emitError(loc, "'linalg.indexed_generic' op ""attribute 'symbol_source' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef IndexedGenericOp::getOperationName() {
  return "linalg.indexed_generic";
}

std::pair<unsigned, unsigned> IndexedGenericOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range IndexedGenericOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range IndexedGenericOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range IndexedGenericOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range IndexedGenericOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange IndexedGenericOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange IndexedGenericOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange IndexedGenericOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> IndexedGenericOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range IndexedGenericOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range IndexedGenericOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &IndexedGenericOp::region() {
  return this->getOperation()->getRegion(0);
}

::mlir::ArrayAttr IndexedGenericOp::indexing_mapsAttr() {
  return this->getAttr("indexing_maps").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr IndexedGenericOp::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return attr;
}

::mlir::ArrayAttr IndexedGenericOp::iterator_typesAttr() {
  return this->getAttr("iterator_types").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr IndexedGenericOp::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::StringAttr IndexedGenericOp::docAttr() {
  return this->getAttr("doc").dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > IndexedGenericOp::doc() {
  auto attr = docAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr IndexedGenericOp::library_callAttr() {
  return this->getAttr("library_call").dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > IndexedGenericOp::library_call() {
  auto attr = library_callAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr IndexedGenericOp::symbol_sourceAttr() {
  return this->getAttr("symbol_source").dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> IndexedGenericOp::symbol_source() {
  auto attr = symbol_sourceAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void IndexedGenericOp::indexing_mapsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("indexing_maps", attr);
}

void IndexedGenericOp::iterator_typesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("iterator_types", attr);
}

void IndexedGenericOp::docAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("doc", attr);
}

void IndexedGenericOp::library_callAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("library_call", attr);
}

void IndexedGenericOp::symbol_sourceAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("symbol_source", attr);
}









void IndexedGenericOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange output_buffers, ::mlir::ValueRange init_tensors, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call, /*optional*/::mlir::IntegerAttr symbol_source) {
  odsState.addOperands(inputs);
  odsState.addOperands(output_buffers);
  odsState.addOperands(init_tensors);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(output_buffers.size()), static_cast<int32_t>(init_tensors.size())}));
  odsState.addAttribute("indexing_maps", indexing_maps);
  odsState.addAttribute("iterator_types", iterator_types);
  if (doc) {
  odsState.addAttribute("doc", doc);
  }
  if (library_call) {
  odsState.addAttribute("library_call", library_call);
  }
  if (symbol_source) {
  odsState.addAttribute("symbol_source", symbol_source);
  }
  (void)odsState.addRegion();
  odsState.addTypes(result_tensors);
}

void IndexedGenericOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult IndexedGenericOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseGenericOp(parser, result);
}

void IndexedGenericOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult IndexedGenericOp::verify() {
  if (failed(IndexedGenericOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulOp definitions
//===----------------------------------------------------------------------===//

MatmulOpAdaptor::MatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

MatmulOpAdaptor::MatmulOpAdaptor(MatmulOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> MatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange MatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatmulOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatmulOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange MatmulOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult MatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef MatmulOp::getOperationName() {
  return "linalg.matmul";
}

std::pair<unsigned, unsigned> MatmulOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range MatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatmulOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatmulOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range MatmulOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange MatmulOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatmulOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatmulOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatmulOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range MatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatmulOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatmulOp::region() {
  return this->getOperation()->getRegion(0);
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<MatmulOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<MatmulOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult MatmulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatmulOp>(parser, result);
}

void MatmulOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatmulOp::verify() {
  if (failed(MatmulOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecOp definitions
//===----------------------------------------------------------------------===//

MatvecOpAdaptor::MatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

MatvecOpAdaptor::MatvecOpAdaptor(MatvecOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> MatvecOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange MatvecOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MatvecOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange MatvecOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange MatvecOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult MatvecOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef MatvecOp::getOperationName() {
  return "linalg.matvec";
}

std::pair<unsigned, unsigned> MatvecOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range MatvecOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MatvecOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MatvecOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range MatvecOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange MatvecOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatvecOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange MatvecOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> MatvecOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range MatvecOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range MatvecOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &MatvecOp::region() {
  return this->getOperation()->getRegion(0);
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<MatvecOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<MatvecOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void MatvecOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult MatvecOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<MatvecOp>(parser, result);
}

void MatvecOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult MatvecOp::verify() {
  if (failed(MatvecOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingMaxOp definitions
//===----------------------------------------------------------------------===//

PoolingMaxOpAdaptor::PoolingMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

PoolingMaxOpAdaptor::PoolingMaxOpAdaptor(PoolingMaxOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> PoolingMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PoolingMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMaxOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMaxOpAdaptor::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMaxOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::ArrayAttr PoolingMaxOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PoolingMaxOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingMaxOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult PoolingMaxOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_max' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_max' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.pooling_max' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef PoolingMaxOp::getOperationName() {
  return "linalg.pooling_max";
}

std::pair<unsigned, unsigned> PoolingMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PoolingMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMaxOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMaxOp::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMaxOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange PoolingMaxOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingMaxOp::windowDimsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingMaxOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PoolingMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PoolingMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr PoolingMaxOp::stridesAttr() {
  return this->getAttr("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMaxOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr PoolingMaxOp::dilationsAttr() {
  return this->getAttr("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMaxOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr PoolingMaxOp::paddingAttr() {
  return this->getAttr("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > PoolingMaxOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void PoolingMaxOp::stridesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("strides", attr);
}

void PoolingMaxOp::dilationsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("dilations", attr);
}

void PoolingMaxOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("padding", attr);
}

void PoolingMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
}

void PoolingMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PoolingMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PoolingMaxOp::verify() {
  if (failed(PoolingMaxOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult PoolingMaxOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PoolingMaxOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.pooling_max";
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingMinOp definitions
//===----------------------------------------------------------------------===//

PoolingMinOpAdaptor::PoolingMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

PoolingMinOpAdaptor::PoolingMinOpAdaptor(PoolingMinOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> PoolingMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PoolingMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMinOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMinOpAdaptor::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMinOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::ArrayAttr PoolingMinOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PoolingMinOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingMinOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult PoolingMinOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_min' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_min' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.pooling_min' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef PoolingMinOp::getOperationName() {
  return "linalg.pooling_min";
}

std::pair<unsigned, unsigned> PoolingMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PoolingMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingMinOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingMinOp::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingMinOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange PoolingMinOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingMinOp::windowDimsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingMinOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PoolingMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PoolingMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr PoolingMinOp::stridesAttr() {
  return this->getAttr("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMinOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr PoolingMinOp::dilationsAttr() {
  return this->getAttr("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingMinOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr PoolingMinOp::paddingAttr() {
  return this->getAttr("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > PoolingMinOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void PoolingMinOp::stridesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("strides", attr);
}

void PoolingMinOp::dilationsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("dilations", attr);
}

void PoolingMinOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("padding", attr);
}

void PoolingMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
}

void PoolingMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PoolingMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PoolingMinOp::verify() {
  if (failed(PoolingMinOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult PoolingMinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PoolingMinOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.pooling_min";
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingSumOp definitions
//===----------------------------------------------------------------------===//

PoolingSumOpAdaptor::PoolingSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

PoolingSumOpAdaptor::PoolingSumOpAdaptor(PoolingSumOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> PoolingSumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PoolingSumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingSumOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingSumOpAdaptor::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingSumOpAdaptor::output() {
  return *getODSOperands(2).begin();
}

::mlir::ArrayAttr PoolingSumOpAdaptor::strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PoolingSumOpAdaptor::dilations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr PoolingSumOpAdaptor::padding() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult PoolingSumOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_strides = odsAttrs.get("strides");
  if (tblgen_strides) {
    if (!(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_sum' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_dilations = odsAttrs.get("dilations");
  if (tblgen_dilations) {
    if (!(((tblgen_dilations.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dilations.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'linalg.pooling_sum' op ""attribute 'dilations' failed to satisfy constraint: 64-bit integer array attribute");
  }
  }
  {
  auto tblgen_padding = odsAttrs.get("padding");
  if (tblgen_padding) {
    if (!(((tblgen_padding.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_padding.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'linalg.pooling_sum' op ""attribute 'padding' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef PoolingSumOp::getOperationName() {
  return "linalg.pooling_sum";
}

std::pair<unsigned, unsigned> PoolingSumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PoolingSumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PoolingSumOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Value PoolingSumOp::windowDims() {
  return *getODSOperands(1).begin();
}

::mlir::Value PoolingSumOp::output() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange PoolingSumOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingSumOp::windowDimsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PoolingSumOp::outputMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PoolingSumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PoolingSumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr PoolingSumOp::stridesAttr() {
  return this->getAttr("strides").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingSumOp::strides() {
  auto attr = stridesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr PoolingSumOp::dilationsAttr() {
  return this->getAttr("dilations").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > PoolingSumOp::dilations() {
  auto attr = dilationsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr PoolingSumOp::paddingAttr() {
  return this->getAttr("padding").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > PoolingSumOp::padding() {
  auto attr = paddingAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void PoolingSumOp::stridesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("strides", attr);
}

void PoolingSumOp::dilationsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("dilations", attr);
}

void PoolingSumOp::paddingAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("padding", attr);
}

void PoolingSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
}

void PoolingSumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding) {
  odsState.addOperands(input);
  odsState.addOperands(windowDims);
  odsState.addOperands(output);
  if (strides) {
  odsState.addAttribute("strides", strides);
  }
  if (dilations) {
  odsState.addAttribute("dilations", dilations);
  }
  if (padding) {
  odsState.addAttribute("padding", padding);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PoolingSumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PoolingSumOp::verify() {
  if (failed(PoolingSumOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("operand #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult PoolingSumOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  if (parser.parseLParen())
    return ::mlir::failure();
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PoolingSumOp::print(::mlir::OpAsmPrinter &p) {
  p << "linalg.pooling_sum";
  p << "(";
  p << getOperation()->getOperands();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << getOperation()->getOperandTypes();
}

} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatOp definitions
//===----------------------------------------------------------------------===//

VecmatOpAdaptor::VecmatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

VecmatOpAdaptor::VecmatOpAdaptor(VecmatOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> VecmatOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange VecmatOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange VecmatOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange VecmatOpAdaptor::output_buffers() {
  return getODSOperands(1);
}

::mlir::ValueRange VecmatOpAdaptor::init_tensors() {
  return getODSOperands(2);
}

::mlir::LogicalResult VecmatOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef VecmatOp::getOperationName() {
  return "linalg.vecmat";
}

std::pair<unsigned, unsigned> VecmatOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range VecmatOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range VecmatOp::inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range VecmatOp::output_buffers() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range VecmatOp::init_tensors() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange VecmatOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange VecmatOp::output_buffersMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange VecmatOp::init_tensorsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> VecmatOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range VecmatOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range VecmatOp::result_tensors() {
  return getODSResults(0);
}

::mlir::Region &VecmatOp::region() {
  return this->getOperation()->getRegion(0);
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(0)}));
          buildNamedStructuredOpRegionAndAttributes<VecmatOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(),
            TypeRange());
        
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors) {
          odsState.addOperands(inputs);
          odsState.addOperands(outputBuffers);
          odsState.addOperands(initTensors);
          odsState.addTypes(resultTensorTypes);
          odsState.addAttribute(
            "operand_segment_sizes",
            odsBuilder.getI32VectorAttr({
              static_cast<int32_t>(inputs.size()),
              static_cast<int32_t>(outputBuffers.size()),
              static_cast<int32_t>(initTensors.size())}));
          buildNamedStructuredOpRegionAndAttributes<VecmatOp>(
            odsBuilder,
            odsState,
            TypeRange(inputs),
            TypeRange(outputBuffers),
            TypeRange(initTensors),
            resultTensorTypes);
        
}

void VecmatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes ) {
          odsState.addOperands(operands);
          odsState.addAttributes(attributes);
          odsState.addTypes(resultTensorTypes);
          (void)odsState.addRegion();
        
}

::mlir::ParseResult VecmatOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseNamedStructuredOp<VecmatOp>(parser, result);
}

void VecmatOp::print(::mlir::OpAsmPrinter &p) {
  return ::printNamedStructuredOp(p, *this);
}

::mlir::LogicalResult VecmatOp::verify() {
  if (failed(VecmatOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ShapedType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be shaped of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('region') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verifyNamedStructuredOp(*this);
}





} // namespace linalg
} // namespace mlir

#endif  // GET_OP_CLASSES

