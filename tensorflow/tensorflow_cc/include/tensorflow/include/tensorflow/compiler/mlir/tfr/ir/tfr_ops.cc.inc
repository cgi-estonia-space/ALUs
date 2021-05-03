/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::TFR::BuildListOp,
::mlir::TFR::CallOp,
::mlir::TFR::CastOp,
::mlir::TFR::ConstOp,
::mlir::TFR::ConstantTensorOp,
::mlir::TFR::EqualOp,
::mlir::TFR::GetElementOp,
::mlir::TFR::GetElementTypeOp,
::mlir::TFR::GetShapeOp,
::mlir::TFR::TFRFuncOp,
::mlir::TFR::TFRReturnOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::BuildListOp definitions
//===----------------------------------------------------------------------===//

BuildListOpAdaptor::BuildListOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BuildListOpAdaptor::BuildListOpAdaptor(BuildListOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BuildListOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange BuildListOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BuildListOpAdaptor::tensors() {
  return getODSOperands(0);
}

::mlir::LogicalResult BuildListOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef BuildListOp::getOperationName() {
  return "tfr.build_list";
}

std::pair<unsigned, unsigned> BuildListOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range BuildListOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BuildListOp::tensors() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange BuildListOp::tensorsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BuildListOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BuildListOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BuildListOp::out() {
  return *getODSResults(0).begin();
}

void BuildListOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type out, ::mlir::ValueRange tensors) {
  odsState.addOperands(tensors);
  odsState.addTypes(out);
}

void BuildListOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BuildListOp::verify() {
  if (failed(BuildListOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::TFR::TFRTensorType>())) || (((((v.getType().isF16())) || ((v.getType().isa<mlir::TF::HalfRefType>()))) || (((v.getType().isF32())) || ((v.getType().isa<mlir::TF::FloatRefType>()))) || (((v.getType().isF64())) || ((v.getType().isa<mlir::TF::DoubleRefType>()))) || (((v.getType().isBF16())) || ((v.getType().isa<mlir::TF::Bfloat16RefType>()))) || ((v.getType().isa<mlir::TF::HalfRefType>())) || ((v.getType().isa<mlir::TF::FloatRefType>())) || ((v.getType().isa<mlir::TF::DoubleRefType>())) || ((v.getType().isa<mlir::TF::Bfloat16RefType>()))) || (((((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isF32()))) || ((v.getType().isa<mlir::TF::Complex64RefType>()))) || ((((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isF64()))) || ((v.getType().isa<mlir::TF::Complex128RefType>())))) || (((((v.getType().isSignlessInteger(8))) || ((v.getType().isa<mlir::TF::Int8RefType>()))) || (((v.getType().isSignlessInteger(16))) || ((v.getType().isa<mlir::TF::Int16RefType>()))) || (((v.getType().isSignlessInteger(32))) || ((v.getType().isa<mlir::TF::Int32RefType>()))) || (((v.getType().isSignlessInteger(64))) || ((v.getType().isa<mlir::TF::Int64RefType>())))) || ((((v.getType().isUnsignedInteger(8))) || ((v.getType().isa<mlir::TF::Uint8RefType>()))) || (((v.getType().isUnsignedInteger(16))) || ((v.getType().isa<mlir::TF::Uint16RefType>()))) || (((v.getType().isUnsignedInteger(32))) || ((v.getType().isa<mlir::TF::Uint32RefType>()))) || (((v.getType().isUnsignedInteger(64))) || ((v.getType().isa<mlir::TF::Uint64RefType>()))))) || (((v.getType().isSignlessInteger(1))) || ((v.getType().isa<mlir::TF::BoolRefType>()))) || ((v.getType().isa<mlir::TF::TensorFlowType>()))))) {
        return emitOpError("operand #") << index << " must be single tfr.tensor or tensor element type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::TFR::TFRTensorListType>())) || ((v.getType().isa<mlir::TFR::TFRAttrType>())))) {
        return emitOpError("result #") << index << " must be tfr.tensor_list or tfr.attr type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



void BuildListOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace TFR
} // namespace mlir
namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::CallOp definitions
//===----------------------------------------------------------------------===//

CallOpAdaptor::CallOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CallOpAdaptor::CallOpAdaptor(CallOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CallOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange CallOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange CallOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::FlatSymbolRefAttr CallOpAdaptor::callee() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FlatSymbolRefAttr attr = odsAttrs.get("callee").cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::mlir::LogicalResult CallOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_callee = odsAttrs.get("callee");
  if (!tblgen_callee) return emitError(loc, "'tfr.call' op ""requires attribute 'callee'");
    if (!((tblgen_callee.isa<::mlir::FlatSymbolRefAttr>()))) return emitError(loc, "'tfr.call' op ""attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef CallOp::getOperationName() {
  return "tfr.call";
}

std::pair<unsigned, unsigned> CallOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range CallOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CallOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange CallOp::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CallOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range CallOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range CallOp::outs() {
  return getODSResults(0);
}

::mlir::FlatSymbolRefAttr CallOp::calleeAttr() {
  return this->getAttr("callee").cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef CallOp::callee() {
  auto attr = calleeAttr();
  return attr.getValue();
}

void CallOp::calleeAttr(::mlir::FlatSymbolRefAttr attr) {
  this->getOperation()->setAttr("callee", attr);
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute("callee", callee);
  odsState.addTypes(outs);
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::llvm::StringRef callee, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute("callee", odsBuilder.getSymbolRefAttr(callee));
  odsState.addTypes(outs);
}

void CallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CallOp::verify() {
  if (failed(CallOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<mlir::TFR::TFRTensorType>())) || ((v.getType().isa<mlir::TFR::TFRTensorListType>()))) || (((v.getType().isa<mlir::TFR::TFRAttrType>())) || ((v.getType().isa<::mlir::IndexType>())) || (((((v.getType().isF16())) || ((v.getType().isa<mlir::TF::HalfRefType>()))) || (((v.getType().isF32())) || ((v.getType().isa<mlir::TF::FloatRefType>()))) || (((v.getType().isF64())) || ((v.getType().isa<mlir::TF::DoubleRefType>()))) || (((v.getType().isBF16())) || ((v.getType().isa<mlir::TF::Bfloat16RefType>()))) || ((v.getType().isa<mlir::TF::HalfRefType>())) || ((v.getType().isa<mlir::TF::FloatRefType>())) || ((v.getType().isa<mlir::TF::DoubleRefType>())) || ((v.getType().isa<mlir::TF::Bfloat16RefType>()))) || (((((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isF32()))) || ((v.getType().isa<mlir::TF::Complex64RefType>()))) || ((((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isF64()))) || ((v.getType().isa<mlir::TF::Complex128RefType>())))) || (((((v.getType().isSignlessInteger(8))) || ((v.getType().isa<mlir::TF::Int8RefType>()))) || (((v.getType().isSignlessInteger(16))) || ((v.getType().isa<mlir::TF::Int16RefType>()))) || (((v.getType().isSignlessInteger(32))) || ((v.getType().isa<mlir::TF::Int32RefType>()))) || (((v.getType().isSignlessInteger(64))) || ((v.getType().isa<mlir::TF::Int64RefType>())))) || ((((v.getType().isUnsignedInteger(8))) || ((v.getType().isa<mlir::TF::Uint8RefType>()))) || (((v.getType().isUnsignedInteger(16))) || ((v.getType().isa<mlir::TF::Uint16RefType>()))) || (((v.getType().isUnsignedInteger(32))) || ((v.getType().isa<mlir::TF::Uint32RefType>()))) || (((v.getType().isUnsignedInteger(64))) || ((v.getType().isa<mlir::TF::Uint64RefType>()))))) || (((v.getType().isSignlessInteger(1))) || ((v.getType().isa<mlir::TF::BoolRefType>()))) || ((v.getType().isa<mlir::TF::TensorFlowType>()))) || (((v.getType().isa<::mlir::VectorType>())) && ((((((v.getType().cast<::mlir::ShapedType>().getElementType().isF16())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::HalfRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::FloatRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::DoubleRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isBF16())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Bfloat16RefType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::HalfRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::FloatRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::DoubleRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Bfloat16RefType>()))) || (((((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Complex64RefType>()))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Complex128RefType>())))) || (((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int8RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int16RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int32RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int64RefType>())))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint8RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint16RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint32RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint64RefType>()))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::BoolRefType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::TensorFlowType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TFR::TFRAttrType>()))))))) {
        return emitOpError("operand #") << index << " must be allowed tfr.call operand types, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::TFR::TFRTensorType>())) || ((v.getType().isa<mlir::TFR::TFRTensorListType>())))) {
        return emitOpError("result #") << index << " must be allowed tfr.call result types, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CallOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr calleeAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argsTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  if (parser.parseAttribute(calleeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "callee", result.attributes))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType args__allResult_functionType;
  if (parser.parseType(args__allResult_functionType))
    return ::mlir::failure();
  argsTypes = args__allResult_functionType.getInputs();
  allResultTypes = args__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CallOp::print(::mlir::OpAsmPrinter &p) {
  p << "tfr.call";
  p << " ";
  p.printAttributeWithoutType(calleeAttr());
  p << "(";
  p << args();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " " << ":";
  p << " ";
  p.printFunctionalType(args().getTypes(), getOperation()->getResultTypes());
}

} // namespace TFR
} // namespace mlir
namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::CastOp definitions
//===----------------------------------------------------------------------===//

CastOpAdaptor::CastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CastOpAdaptor::CastOpAdaptor(CastOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult CastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef CastOp::getOperationName() {
  return "tfr.cast";
}

std::pair<unsigned, unsigned> CastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CastOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOp::out() {
  return *getODSResults(0).begin();
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type out, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(out);
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CastOp::verify() {
  if (failed(CastOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::TFR::TFRTensorType>())) || (((v.getType().isa<::mlir::TensorType>())) && (((((v.getType().cast<::mlir::ShapedType>().getElementType().isF16())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::HalfRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::FloatRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::DoubleRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isBF16())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Bfloat16RefType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::HalfRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::FloatRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::DoubleRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Bfloat16RefType>()))) || (((((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Complex64RefType>()))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Complex128RefType>())))) || (((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int8RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int16RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int32RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int64RefType>())))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint8RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint16RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint32RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint64RefType>()))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::BoolRefType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::TensorFlowType>())))))) {
        return emitOpError("operand #") << index << " must be single tensor or tfr.tensor type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::TFR::TFRTensorType>())) || (((v.getType().isa<::mlir::TensorType>())) && (((((v.getType().cast<::mlir::ShapedType>().getElementType().isF16())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::HalfRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::FloatRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::DoubleRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isBF16())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Bfloat16RefType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::HalfRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::FloatRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::DoubleRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Bfloat16RefType>()))) || (((((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Complex64RefType>()))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Complex128RefType>())))) || (((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int8RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int16RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int32RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int64RefType>())))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint8RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint16RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint32RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint64RefType>()))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::BoolRefType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::TensorFlowType>())))))) {
        return emitOpError("result #") << index << " must be single tensor or tfr.tensor type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



void CastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace TFR
} // namespace mlir
namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::ConstOp definitions
//===----------------------------------------------------------------------===//

ConstOpAdaptor::ConstOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConstOpAdaptor::ConstOpAdaptor(ConstOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConstOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

Attribute ConstOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  Attribute attr = odsAttrs.get("value").cast<Attribute>();
  return attr;
}

::mlir::LogicalResult ConstOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (!tblgen_value) return emitError(loc, "'tfr.constant' op ""requires attribute 'value'");
    if (!(((tblgen_value.isa<::mlir::FlatSymbolRefAttr>())) || (((tblgen_value.isa<::mlir::TypeAttr>())) && ((tblgen_value.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))) || ((tblgen_value.isa<::mlir::StringAttr>())) || ((tblgen_value.isa<::mlir::ArrayAttr>())))) return emitError(loc, "'tfr.constant' op ""attribute 'value' failed to satisfy constraint: allowed tfr.constant value");
  }
  return ::mlir::success();
}

::llvm::StringRef ConstOp::getOperationName() {
  return "tfr.constant";
}

std::pair<unsigned, unsigned> ConstOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstOp::out() {
  return *getODSResults(0).begin();
}

Attribute ConstOp::valueAttr() {
  return this->getAttr("value").cast<Attribute>();
}

Attribute ConstOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstOp::valueAttr(Attribute attr) {
  this->getOperation()->setAttr("value", attr);
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute value) {
      auto* ctx = value.getContext();
      odsState.addAttribute("value", value);
      odsState.addTypes(TFRAttrType::get(ctx));
    
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type out, Attribute value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(out);
}

void ConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, Attribute value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstOp::verify() {
  if (failed(ConstOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::TFR::TFRAttrType>()))) {
        return emitOpError("result #") << index << " must be TFR TFRAttr type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



::mlir::ParseResult ConstOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  Attribute valueAttr;
  ::mlir::Type outRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outTypes(outRawTypes);

  if (parser.parseAttribute(valueAttr, "value", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(outRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(outTypes);
  return ::mlir::success();
}

void ConstOp::print(::mlir::OpAsmPrinter &p) {
  p << "tfr.constant";
  p << " ";
  p.printAttribute(valueAttr());
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"value"});
  p << " " << "->";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(out().getType());
}

void ConstOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace TFR
} // namespace mlir
namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::ConstantTensorOp definitions
//===----------------------------------------------------------------------===//

ConstantTensorOpAdaptor::ConstantTensorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConstantTensorOpAdaptor::ConstantTensorOpAdaptor(ConstantTensorOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConstantTensorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstantTensorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstantTensorOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult ConstantTensorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ConstantTensorOp::getOperationName() {
  return "tfr.constant_tensor";
}

std::pair<unsigned, unsigned> ConstantTensorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantTensorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstantTensorOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ConstantTensorOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConstantTensorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantTensorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstantTensorOp::out() {
  return *getODSResults(0).begin();
}

void ConstantTensorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type out, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(out);
}

void ConstantTensorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantTensorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantTensorOp::verify() {
  if (failed(ConstantTensorOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::TFR::TFRAttrType>())) || ((v.getType().isa<::mlir::IndexType>())) || (((((v.getType().isF16())) || ((v.getType().isa<mlir::TF::HalfRefType>()))) || (((v.getType().isF32())) || ((v.getType().isa<mlir::TF::FloatRefType>()))) || (((v.getType().isF64())) || ((v.getType().isa<mlir::TF::DoubleRefType>()))) || (((v.getType().isBF16())) || ((v.getType().isa<mlir::TF::Bfloat16RefType>()))) || ((v.getType().isa<mlir::TF::HalfRefType>())) || ((v.getType().isa<mlir::TF::FloatRefType>())) || ((v.getType().isa<mlir::TF::DoubleRefType>())) || ((v.getType().isa<mlir::TF::Bfloat16RefType>()))) || (((((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isF32()))) || ((v.getType().isa<mlir::TF::Complex64RefType>()))) || ((((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isF64()))) || ((v.getType().isa<mlir::TF::Complex128RefType>())))) || (((((v.getType().isSignlessInteger(8))) || ((v.getType().isa<mlir::TF::Int8RefType>()))) || (((v.getType().isSignlessInteger(16))) || ((v.getType().isa<mlir::TF::Int16RefType>()))) || (((v.getType().isSignlessInteger(32))) || ((v.getType().isa<mlir::TF::Int32RefType>()))) || (((v.getType().isSignlessInteger(64))) || ((v.getType().isa<mlir::TF::Int64RefType>())))) || ((((v.getType().isUnsignedInteger(8))) || ((v.getType().isa<mlir::TF::Uint8RefType>()))) || (((v.getType().isUnsignedInteger(16))) || ((v.getType().isa<mlir::TF::Uint16RefType>()))) || (((v.getType().isUnsignedInteger(32))) || ((v.getType().isa<mlir::TF::Uint32RefType>()))) || (((v.getType().isUnsignedInteger(64))) || ((v.getType().isa<mlir::TF::Uint64RefType>()))))) || (((v.getType().isSignlessInteger(1))) || ((v.getType().isa<mlir::TF::BoolRefType>()))) || ((v.getType().isa<mlir::TF::TensorFlowType>()))) || (((v.getType().isa<::mlir::VectorType>())) && ((((((v.getType().cast<::mlir::ShapedType>().getElementType().isF16())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::HalfRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::FloatRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::DoubleRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isBF16())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Bfloat16RefType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::HalfRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::FloatRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::DoubleRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Bfloat16RefType>()))) || (((((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Complex64RefType>()))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Complex128RefType>())))) || (((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int8RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int16RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int32RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int64RefType>())))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint8RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint16RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint32RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint64RefType>()))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::BoolRefType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::TensorFlowType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TFR::TFRAttrType>())))))) {
        return emitOpError("operand #") << index << " must be all attribute related types, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::TFR::TFRTensorType>())) || (((v.getType().isa<::mlir::TensorType>())) && (((((v.getType().cast<::mlir::ShapedType>().getElementType().isF16())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::HalfRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::FloatRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::DoubleRefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isBF16())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Bfloat16RefType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::HalfRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::FloatRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::DoubleRefType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Bfloat16RefType>()))) || (((((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Complex64RefType>()))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Complex128RefType>())))) || (((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int8RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int16RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int32RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Int64RefType>())))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint8RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint16RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint32RefType>()))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::Uint64RefType>()))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::BoolRefType>()))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<mlir::TF::TensorFlowType>())))))) {
        return emitOpError("result #") << index << " must be single tensor or tfr.tensor type, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}



void ConstantTensorOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace TFR
} // namespace mlir
namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::EqualOp definitions
//===----------------------------------------------------------------------===//

EqualOpAdaptor::EqualOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

EqualOpAdaptor::EqualOpAdaptor(EqualOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> EqualOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange EqualOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EqualOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value EqualOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult EqualOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef EqualOp::getOperationName() {
  return "tfr.equal";
}

std::pair<unsigned, unsigned> EqualOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range EqualOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EqualOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value EqualOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange EqualOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange EqualOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> EqualOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EqualOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EqualOp::result() {
  return *getODSResults(0).begin();
}

void EqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(result);
}

void EqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EqualOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EqualOp::verify() {
  if (failed(EqualOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::TFR::TFRAttrType>()))) {
        return emitOpError("operand #") << index << " must be TFR TFRAttr type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<mlir::TFR::TFRAttrType>()))) {
        return emitOpError("operand #") << index << " must be TFR TFRAttr type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger(1))) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1)))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1)))))) {
        return emitOpError("result #") << index << " must be bool-like, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



::mlir::ParseResult EqualOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<mlir::TFR::TFRAttrType>();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(lhsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EqualOp::print(::mlir::OpAsmPrinter &p) {
  p << "tfr.equal";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << "->";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
}

void EqualOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace TFR
} // namespace mlir
namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::GetElementOp definitions
//===----------------------------------------------------------------------===//

GetElementOpAdaptor::GetElementOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

GetElementOpAdaptor::GetElementOpAdaptor(GetElementOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> GetElementOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetElementOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetElementOpAdaptor::tensor_list() {
  return *getODSOperands(0).begin();
}

::mlir::Value GetElementOpAdaptor::index() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult GetElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef GetElementOp::getOperationName() {
  return "tfr.get_element";
}

std::pair<unsigned, unsigned> GetElementOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetElementOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetElementOp::tensor_list() {
  return *getODSOperands(0).begin();
}

::mlir::Value GetElementOp::index() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange GetElementOp::tensor_listMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GetElementOp::indexMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetElementOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetElementOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetElementOp::out() {
  return *getODSResults(0).begin();
}

void GetElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type out, ::mlir::Value tensor_list, ::mlir::Value index) {
  odsState.addOperands(tensor_list);
  odsState.addOperands(index);
  odsState.addTypes(out);
}

void GetElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor_list, ::mlir::Value index) {
  odsState.addOperands(tensor_list);
  odsState.addOperands(index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetElementOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetElementOp::verify() {
  if (failed(GetElementOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::TFR::TFRTensorListType>()))) {
        return emitOpError("operand #") << index << " must be TFR TFRTensorList type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<::mlir::IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::TFR::TFRTensorType>()))) {
        return emitOpError("result #") << index << " must be TFR TFRTensor type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



::mlir::ParseResult GetElementOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType tensor_listRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> tensor_listOperands(tensor_listRawOperands);  ::llvm::SMLoc tensor_listOperandsLoc;
  (void)tensor_listOperandsLoc;
  ::mlir::OpAsmParser::OperandType indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> indexOperands(indexRawOperands);  ::llvm::SMLoc indexOperandsLoc;
  (void)indexOperandsLoc;
  ::mlir::Type tensor_listRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensor_listTypes(tensor_listRawTypes);
  ::mlir::Type indexRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> indexTypes(indexRawTypes);
  ::mlir::Type outRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outTypes(outRawTypes);

  tensor_listOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensor_listRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(indexRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(tensor_listRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(indexRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(outRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(outTypes);
  if (parser.resolveOperands(tensor_listOperands, tensor_listTypes, tensor_listOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indexOperands, indexTypes, indexOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetElementOp::print(::mlir::OpAsmPrinter &p) {
  p << "tfr.get_element";
  p << " ";
  p << tensor_list();
  p << "[";
  p << index();
  p << "]";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(tensor_list().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(index().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(out().getType());
}

void GetElementOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace TFR
} // namespace mlir
namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::GetElementTypeOp definitions
//===----------------------------------------------------------------------===//

GetElementTypeOpAdaptor::GetElementTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

GetElementTypeOpAdaptor::GetElementTypeOpAdaptor(GetElementTypeOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> GetElementTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetElementTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetElementTypeOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult GetElementTypeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef GetElementTypeOp::getOperationName() {
  return "tfr.get_element_type";
}

std::pair<unsigned, unsigned> GetElementTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetElementTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetElementTypeOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetElementTypeOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetElementTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetElementTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetElementTypeOp::out() {
  return *getODSResults(0).begin();
}

void GetElementTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type out, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(out);
}

void GetElementTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetElementTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetElementTypeOp::verify() {
  if (failed(GetElementTypeOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::TFR::TFRTensorType>()))) {
        return emitOpError("operand #") << index << " must be TFR TFRTensor type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::TFR::TFRAttrType>()))) {
        return emitOpError("result #") << index << " must be TFR TFRAttr type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult GetElementTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type outRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outTypes(outRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(outRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<mlir::TFR::TFRTensorType>();
  result.addTypes(outTypes);
  if (parser.resolveOperands(argOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetElementTypeOp::print(::mlir::OpAsmPrinter &p) {
  p << "tfr.get_element_type";
  p << " ";
  p << arg();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << "->";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(out().getType());
}

void GetElementTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace TFR
} // namespace mlir
namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::GetShapeOp definitions
//===----------------------------------------------------------------------===//

GetShapeOpAdaptor::GetShapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

GetShapeOpAdaptor::GetShapeOpAdaptor(GetShapeOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> GetShapeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetShapeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetShapeOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult GetShapeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef GetShapeOp::getOperationName() {
  return "tfr.get_shape";
}

std::pair<unsigned, unsigned> GetShapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetShapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetShapeOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetShapeOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GetShapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetShapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetShapeOp::out() {
  return *getODSResults(0).begin();
}

void GetShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type out, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(out);
}

void GetShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetShapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetShapeOp::verify() {
  if (failed(GetShapeOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::TFR::TFRTensorType>()))) {
        return emitOpError("operand #") << index << " must be TFR TFRTensor type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("result #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



::mlir::ParseResult GetShapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type outRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outTypes(outRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(outRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<mlir::TFR::TFRTensorType>();
  result.addTypes(outTypes);
  if (parser.resolveOperands(argOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetShapeOp::print(::mlir::OpAsmPrinter &p) {
  p << "tfr.get_shape";
  p << " ";
  p << arg();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << "->";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(out().getType());
}

void GetShapeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace TFR
} // namespace mlir
namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::TFRFuncOp definitions
//===----------------------------------------------------------------------===//

TFRFuncOpAdaptor::TFRFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TFRFuncOpAdaptor::TFRFuncOpAdaptor(TFRFuncOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TFRFuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TFRFuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::TypeAttr TFRFuncOpAdaptor::type() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("type").cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::StringAttr TFRFuncOpAdaptor::sym_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult TFRFuncOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_type = odsAttrs.get("type");
  if (!tblgen_type) return emitError(loc, "'tfr.func' op ""requires attribute 'type'");
    if (!(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) return emitError(loc, "'tfr.func' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  }
  {
  auto tblgen_sym_name = odsAttrs.get("sym_name");
  if (!tblgen_sym_name) return emitError(loc, "'tfr.func' op ""requires attribute 'sym_name'");
    if (!((tblgen_sym_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'tfr.func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef TFRFuncOp::getOperationName() {
  return "tfr.func";
}

std::pair<unsigned, unsigned> TFRFuncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TFRFuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TFRFuncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TFRFuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::MutableArrayRef<Region> TFRFuncOp::body() {
  return this->getOperation()->getRegions().drop_front(0);
}

::mlir::TypeAttr TFRFuncOp::typeAttr() {
  return this->getAttr("type").cast<::mlir::TypeAttr>();
}

::mlir::Type TFRFuncOp::type() {
  auto attr = typeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::StringAttr TFRFuncOp::sym_nameAttr() {
  return this->getAttr("sym_name").cast<::mlir::StringAttr>();
}

::llvm::StringRef TFRFuncOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

void TFRFuncOp::typeAttr(::mlir::TypeAttr attr) {
  this->getOperation()->setAttr("type", attr);
}

void TFRFuncOp::sym_nameAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("sym_name", attr);
}



::mlir::ParseResult TFRFuncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ParseFuncOp(parser, &result);
}

void TFRFuncOp::print(::mlir::OpAsmPrinter &p) {
  PrintFuncOp(p, *this);
}

::mlir::LogicalResult TFRFuncOp::verify() {
  if (failed(TFRFuncOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : body()) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return Verify(*this);
}





} // namespace TFR
} // namespace mlir
namespace mlir {
namespace TFR {

//===----------------------------------------------------------------------===//
// ::mlir::TFR::TFRReturnOp definitions
//===----------------------------------------------------------------------===//

TFRReturnOpAdaptor::TFRReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TFRReturnOpAdaptor::TFRReturnOpAdaptor(TFRReturnOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TFRReturnOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange TFRReturnOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange TFRReturnOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::LogicalResult TFRReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef TFRReturnOp::getOperationName() {
  return "tfr.return";
}

std::pair<unsigned, unsigned> TFRReturnOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range TFRReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range TFRReturnOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange TFRReturnOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TFRReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TFRReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TFRReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void TFRReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TFRReturnOp::verify() {
  if (failed(TFRReturnOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::TFR::TFRTensorType>())) || ((v.getType().isa<mlir::TFR::TFRTensorListType>())))) {
        return emitOpError("operand #") << index << " must be allowed tfr.call result types, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult TFRReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> operandsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (!operandsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TFRReturnOp::print(::mlir::OpAsmPrinter &p) {
  p << "tfr.return";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
  p << " ";
  p << operands();
  p << " " << ":";
  p << " ";
  p << operands().getTypes();
  }
}

void TFRReturnOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace TFR
} // namespace mlir

#endif  // GET_OP_CLASSES

