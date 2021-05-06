/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::tf_executor::ControlTriggerOp,
::mlir::tf_executor::EnterOp,
::mlir::tf_executor::ExitOp,
::mlir::tf_executor::FetchOp,
::mlir::tf_executor::GraphOp,
::mlir::tf_executor::IslandOp,
::mlir::tf_executor::LoopCondOp,
::mlir::tf_executor::MergeOp,
::mlir::tf_executor::NextIterationSinkOp,
::mlir::tf_executor::NextIterationSourceOp,
::mlir::tf_executor::SwitchNOp,
::mlir::tf_executor::SwitchOp,
::mlir::tf_executor::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::ControlTriggerOp definitions
//===----------------------------------------------------------------------===//

ControlTriggerOpAdaptor::ControlTriggerOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ControlTriggerOpAdaptor::ControlTriggerOpAdaptor(ControlTriggerOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ControlTriggerOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ControlTriggerOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ControlTriggerOpAdaptor::controlInputs() {
  return getODSOperands(0);
}

::mlir::LogicalResult ControlTriggerOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ControlTriggerOp::getOperationName() {
  return "tf_executor.ControlTrigger";
}

std::pair<unsigned, unsigned> ControlTriggerOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ControlTriggerOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ControlTriggerOp::controlInputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ControlTriggerOp::controlInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ControlTriggerOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ControlTriggerOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ControlTriggerOp::control() {
  return *getODSResults(0).begin();
}

void ControlTriggerOp::build(OpBuilder &builder, OperationState &result, ArrayRef<Value> operands, ArrayRef<NamedAttribute> attributes ) {
      assert(operands.size() >= 1 && "tf_executor.ControlTrigger builder "
             "expects at least one operand");
      result.operands.insert(result.operands.end(), operands.begin(),
                              operands.end());
      Type control_type = ControlType::get(builder.getContext());
      result.types = {control_type};
      result.attributes.append(attributes.begin(), attributes.end());
    
}

void ControlTriggerOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type control, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(controlInputs);
  odsState.addTypes(control);
}

void ControlTriggerOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ControlTriggerOp::verify() {
  if (failed(ControlTriggerOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("operand #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("result #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



::mlir::ParseResult ControlTriggerOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> controlInputsOperands;
  ::llvm::SMLoc controlInputsOperandsLoc;
  (void)controlInputsOperandsLoc;

  controlInputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(controlInputsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<ControlType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(controlInputsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ControlTriggerOp::print(::mlir::OpAsmPrinter &p) {
  p << "tf_executor.ControlTrigger";
  p << " ";
  p << controlInputs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::EnterOp definitions
//===----------------------------------------------------------------------===//

EnterOpAdaptor::EnterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

EnterOpAdaptor::EnterOpAdaptor(EnterOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> EnterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange EnterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EnterOpAdaptor::data() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange EnterOpAdaptor::controlInputs() {
  return getODSOperands(1);
}

::mlir::StringAttr EnterOpAdaptor::frame_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("frame_name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::BoolAttr EnterOpAdaptor::is_constant() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("is_constant").dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

::mlir::IntegerAttr EnterOpAdaptor::parallel_iterations() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("parallel_iterations").dyn_cast_or_null<::mlir::IntegerAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 10);
  return attr;
}

::mlir::LogicalResult EnterOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_frame_name = odsAttrs.get("frame_name");
  if (!tblgen_frame_name) return emitError(loc, "'tf_executor.Enter' op ""requires attribute 'frame_name'");
    if (!((tblgen_frame_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'tf_executor.Enter' op ""attribute 'frame_name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_is_constant = odsAttrs.get("is_constant");
  if (tblgen_is_constant) {
    if (!((tblgen_is_constant.isa<::mlir::BoolAttr>()))) return emitError(loc, "'tf_executor.Enter' op ""attribute 'is_constant' failed to satisfy constraint: bool attribute");
  }
  }
  {
  auto tblgen_parallel_iterations = odsAttrs.get("parallel_iterations");
  if (tblgen_parallel_iterations) {
    if (!(((tblgen_parallel_iterations.isa<::mlir::IntegerAttr>())) && ((tblgen_parallel_iterations.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tf_executor.Enter' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  }
  return ::mlir::success();
}

void EnterOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "output");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "control");
}

::llvm::StringRef EnterOp::getOperationName() {
  return "tf_executor.Enter";
}

std::pair<unsigned, unsigned> EnterOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range EnterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EnterOp::data() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range EnterOp::controlInputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange EnterOp::dataMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange EnterOp::controlInputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> EnterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EnterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EnterOp::output() {
  return *getODSResults(0).begin();
}

::mlir::Value EnterOp::control() {
  return *getODSResults(1).begin();
}

::mlir::StringAttr EnterOp::frame_nameAttr() {
  return this->getAttr("frame_name").cast<::mlir::StringAttr>();
}

::llvm::StringRef EnterOp::frame_name() {
  auto attr = frame_nameAttr();
  return attr.getValue();
}

::mlir::BoolAttr EnterOp::is_constantAttr() {
  return this->getAttr("is_constant").dyn_cast_or_null<::mlir::BoolAttr>();
}

bool EnterOp::is_constant() {
  auto attr = is_constantAttr();
    if (!attr)
      return ::mlir::Builder(this->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::IntegerAttr EnterOp::parallel_iterationsAttr() {
  return this->getAttr("parallel_iterations").dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t EnterOp::parallel_iterations() {
  auto attr = parallel_iterationsAttr();
    if (!attr)
      return ::mlir::Builder(this->getContext()).getIntegerAttr(::mlir::Builder(this->getContext()).getIntegerType(64), 10).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

void EnterOp::frame_nameAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("frame_name", attr);
}

void EnterOp::is_constantAttr(::mlir::BoolAttr attr) {
  this->getOperation()->setAttr("is_constant", attr);
}

void EnterOp::parallel_iterationsAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("parallel_iterations", attr);
}

void EnterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type control, ::mlir::Value data, ::mlir::StringAttr frame_name, ::mlir::BoolAttr is_constant, ::mlir::IntegerAttr parallel_iterations, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(data);
  odsState.addOperands(controlInputs);
  odsState.addAttribute("frame_name", frame_name);
  odsState.addAttribute("is_constant", is_constant);
  odsState.addAttribute("parallel_iterations", parallel_iterations);
  odsState.addTypes(output);
  odsState.addTypes(control);
}

void EnterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::StringAttr frame_name, ::mlir::BoolAttr is_constant, ::mlir::IntegerAttr parallel_iterations, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(data);
  odsState.addOperands(controlInputs);
  odsState.addAttribute("frame_name", frame_name);
  odsState.addAttribute("is_constant", is_constant);
  odsState.addAttribute("parallel_iterations", parallel_iterations);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EnterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type control, ::mlir::Value data, ::llvm::StringRef frame_name, bool is_constant, uint64_t parallel_iterations, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(data);
  odsState.addOperands(controlInputs);
  odsState.addAttribute("frame_name", odsBuilder.getStringAttr(frame_name));
  odsState.addAttribute("is_constant", odsBuilder.getBoolAttr(is_constant));
  odsState.addAttribute("parallel_iterations", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  odsState.addTypes(output);
  odsState.addTypes(control);
}

void EnterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::llvm::StringRef frame_name, bool is_constant, uint64_t parallel_iterations, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(data);
  odsState.addOperands(controlInputs);
  odsState.addAttribute("frame_name", odsBuilder.getStringAttr(frame_name));
  odsState.addAttribute("is_constant", odsBuilder.getBoolAttr(is_constant));
  odsState.addAttribute("parallel_iterations", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EnterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult EnterOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ParseEnterOp(parser, result);
}

void EnterOp::print(::mlir::OpAsmPrinter &p) {
  return ::mlir::tf_executor::Print(*this, p);
}

::mlir::LogicalResult EnterOp::verify() {
  if (failed(EnterOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("operand #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("result #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((succeeded(VerifyControlOperandsAfterAllData(&(*this->getOperation()))))))
    return emitOpError("failed to verify that all control inputs must appear after any non-control input");
  if (!(((((*this->getOperation()).getNumResults() > 0)) && (((*this->getOperation()).getNumOperands() > 0)) && (((*this->getOperation()).getResult(0).getType().isa<::mlir::ShapedType>())) && (((*this->getOperation()).getOperand(0).getType().isa<::mlir::ShapedType>()))) && ((mlir::TF::BroadcastCompatible((*this->getOperation()).getOperand(0).getType(), (*this->getOperation()).getResult(0).getType())))))
    return emitOpError("failed to verify that data operand must be broadcastable to result");
  return ::mlir::success();
}

} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::ExitOp definitions
//===----------------------------------------------------------------------===//

ExitOpAdaptor::ExitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ExitOpAdaptor::ExitOpAdaptor(ExitOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ExitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ExitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExitOpAdaptor::data() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ExitOpAdaptor::controlInputs() {
  return getODSOperands(1);
}

::mlir::LogicalResult ExitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ExitOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "output");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "control");
}

::llvm::StringRef ExitOp::getOperationName() {
  return "tf_executor.Exit";
}

std::pair<unsigned, unsigned> ExitOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ExitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExitOp::data() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ExitOp::controlInputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ExitOp::dataMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ExitOp::controlInputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ExitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExitOp::output() {
  return *getODSResults(0).begin();
}

::mlir::Value ExitOp::control() {
  return *getODSResults(1).begin();
}

void ExitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type control, ::mlir::Value data, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(data);
  odsState.addOperands(controlInputs);
  odsState.addTypes(output);
  odsState.addTypes(control);
}

void ExitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(data);
  odsState.addOperands(controlInputs);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ExitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ParseExitOp(parser, result);
}

void ExitOp::print(::mlir::OpAsmPrinter &p) {
  return ::mlir::tf_executor::Print(*this, p);
}

::mlir::LogicalResult ExitOp::verify() {
  if (failed(ExitOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("operand #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("result #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((((*this->getOperation()).getNumResults() > 0)) && (((*this->getOperation()).getNumOperands() > 0)) && (((*this->getOperation()).getResult(0).getType().isa<::mlir::ShapedType>())) && (((*this->getOperation()).getOperand(0).getType().isa<::mlir::ShapedType>()))) && ((mlir::TF::BroadcastCompatible((*this->getOperation()).getOperand(0).getType(), (*this->getOperation()).getResult(0).getType())))))
    return emitOpError("failed to verify that data operand must be broadcastable to result");
  return ::mlir::success();
}

} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::FetchOp definitions
//===----------------------------------------------------------------------===//

FetchOpAdaptor::FetchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

FetchOpAdaptor::FetchOpAdaptor(FetchOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> FetchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange FetchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange FetchOpAdaptor::fetches() {
  return getODSOperands(0);
}

::mlir::LogicalResult FetchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef FetchOp::getOperationName() {
  return "tf_executor.fetch";
}

std::pair<unsigned, unsigned> FetchOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range FetchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FetchOp::fetches() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange FetchOp::fetchesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FetchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FetchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FetchOp::build(OpBuilder &builder, OperationState &result) {
      build(builder, result, {});
    
}

void FetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange fetches) {
  odsState.addOperands(fetches);
}

void FetchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FetchOp::verify() {
  if (failed(FetchOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  if (!((succeeded(VerifyControlOperandsAfterAllData(&(*this->getOperation()))))))
    return emitOpError("failed to verify that all control inputs must appear after any non-control input");
  return ::mlir::success();
}

::mlir::ParseResult FetchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> fetchesOperands;
  ::llvm::SMLoc fetchesOperandsLoc;
  (void)fetchesOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> fetchesTypes;

  fetchesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(fetchesOperands))
    return ::mlir::failure();
  if (!fetchesOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(fetchesTypes))
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(fetchesOperands, fetchesTypes, fetchesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FetchOp::print(::mlir::OpAsmPrinter &p) {
  p << "tf_executor.fetch";
  if (!fetches().empty()) {
  p << " ";
  p << fetches();
  p << " " << ":";
  p << " ";
  p << fetches().getTypes();
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::GraphOp definitions
//===----------------------------------------------------------------------===//

GraphOpAdaptor::GraphOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

GraphOpAdaptor::GraphOpAdaptor(GraphOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> GraphOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GraphOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::LogicalResult GraphOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef GraphOp::getOperationName() {
  return "tf_executor.graph";
}

std::pair<unsigned, unsigned> GraphOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GraphOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GraphOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range GraphOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range GraphOp::results() {
  return getODSResults(0);
}

::mlir::Region &GraphOp::body() {
  return this->getOperation()->getRegion(0);
}

void GraphOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results) {
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void GraphOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult GraphOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ParseGraphOp(parser, result);
}

void GraphOp::print(::mlir::OpAsmPrinter &p) {
  return ::mlir::tf_executor::Print(*this, p);
}

::mlir::LogicalResult GraphOp::verify() {
  if (failed(GraphOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::mlir::tf_executor::Verify(*this);
}



} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::IslandOp definitions
//===----------------------------------------------------------------------===//

IslandOpAdaptor::IslandOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

IslandOpAdaptor::IslandOpAdaptor(IslandOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> IslandOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange IslandOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange IslandOpAdaptor::controlInputs() {
  return getODSOperands(0);
}

::mlir::LogicalResult IslandOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void IslandOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "outputs");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "control");
}

::llvm::StringRef IslandOp::getOperationName() {
  return "tf_executor.island";
}

std::pair<unsigned, unsigned> IslandOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range IslandOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range IslandOp::controlInputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange IslandOp::controlInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> IslandOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range IslandOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range IslandOp::outputs() {
  return getODSResults(0);
}

::mlir::Value IslandOp::control() {
  return *getODSResults(1).begin();
}

::mlir::Region &IslandOp::body() {
  return this->getOperation()->getRegion(0);
}

void IslandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Type control, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(controlInputs);
  (void)odsState.addRegion();
  odsState.addTypes(outputs);
  odsState.addTypes(control);
}

void IslandOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult IslandOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ParseIslandOp(parser, result);
}

void IslandOp::print(::mlir::OpAsmPrinter &p) {
  return ::mlir::tf_executor::Print(*this, p);
}

::mlir::LogicalResult IslandOp::verify() {
  if (failed(IslandOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("operand #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("result #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::mlir::tf_executor::Verify(*this);
}





} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::LoopCondOp definitions
//===----------------------------------------------------------------------===//

LoopCondOpAdaptor::LoopCondOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

LoopCondOpAdaptor::LoopCondOpAdaptor(LoopCondOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> LoopCondOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange LoopCondOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopCondOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange LoopCondOpAdaptor::controlInputs() {
  return getODSOperands(1);
}

::mlir::LogicalResult LoopCondOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void LoopCondOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "output");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "control");
}

::llvm::StringRef LoopCondOp::getOperationName() {
  return "tf_executor.LoopCond";
}

std::pair<unsigned, unsigned> LoopCondOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range LoopCondOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopCondOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range LoopCondOp::controlInputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange LoopCondOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange LoopCondOp::controlInputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LoopCondOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoopCondOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopCondOp::output() {
  return *getODSResults(0).begin();
}

::mlir::Value LoopCondOp::control() {
  return *getODSResults(1).begin();
}

void LoopCondOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type control, ::mlir::Value input, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(input);
  odsState.addOperands(controlInputs);
  odsState.addTypes(output);
  odsState.addTypes(control);
}

void LoopCondOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(input);
  odsState.addOperands(controlInputs);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopCondOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult LoopCondOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ParseLoopCondOp(parser, result);
}

void LoopCondOp::print(::mlir::OpAsmPrinter &p) {
  return ::mlir::tf_executor::Print(*this, p);
}

::mlir::LogicalResult LoopCondOp::verify() {
  if (failed(LoopCondOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))))) {
        return emitOpError("operand #") << index << " must be tensor of 1-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("operand #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))))) {
        return emitOpError("result #") << index << " must be tensor of 1-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("result #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}

} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::MergeOp definitions
//===----------------------------------------------------------------------===//

MergeOpAdaptor::MergeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

MergeOpAdaptor::MergeOpAdaptor(MergeOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> MergeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange MergeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MergeOpAdaptor::inputs_and_control() {
  return getODSOperands(0);
}

::mlir::LogicalResult MergeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void MergeOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "output");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "value_index");
  auto resultGroup2 = getODSResults(2);
  if (!llvm::empty(resultGroup2))
    setNameFn(*resultGroup2.begin(), "control");
}

::llvm::StringRef MergeOp::getOperationName() {
  return "tf_executor.Merge";
}

std::pair<unsigned, unsigned> MergeOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range MergeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MergeOp::inputs_and_control() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange MergeOp::inputs_and_controlMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MergeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MergeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MergeOp::output() {
  return *getODSResults(0).begin();
}

::mlir::Value MergeOp::value_index() {
  return *getODSResults(1).begin();
}

::mlir::Value MergeOp::control() {
  return *getODSResults(2).begin();
}

void MergeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type value_index, ::mlir::Type control, ::mlir::ValueRange inputs_and_control) {
  odsState.addOperands(inputs_and_control);
  odsState.addTypes(output);
  odsState.addTypes(value_index);
  odsState.addTypes(control);
}

void MergeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 3u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult MergeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ParseMergeOp(parser, result);
}

void MergeOp::print(::mlir::OpAsmPrinter &p) {
  return ::mlir::tf_executor::Print(*this, p);
}

::mlir::LogicalResult MergeOp::verify() {
  if (failed(MergeOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))))) {
        return emitOpError("result #") << index << " must be tensor of 32-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSResults(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("result #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((succeeded(VerifyControlOperandsAfterAllData(&(*this->getOperation()))))))
    return emitOpError("failed to verify that all control inputs must appear after any non-control input");
  return ::mlir::tf_executor::Verify(*this);
}

} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::NextIterationSinkOp definitions
//===----------------------------------------------------------------------===//

NextIterationSinkOpAdaptor::NextIterationSinkOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

NextIterationSinkOpAdaptor::NextIterationSinkOpAdaptor(NextIterationSinkOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> NextIterationSinkOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange NextIterationSinkOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NextIterationSinkOpAdaptor::token() {
  return *getODSOperands(0).begin();
}

::mlir::Value NextIterationSinkOpAdaptor::input() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange NextIterationSinkOpAdaptor::controlInputs() {
  return getODSOperands(2);
}

::mlir::LogicalResult NextIterationSinkOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef NextIterationSinkOp::getOperationName() {
  return "tf_executor.NextIteration.Sink";
}

std::pair<unsigned, unsigned> NextIterationSinkOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range NextIterationSinkOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NextIterationSinkOp::token() {
  return *getODSOperands(0).begin();
}

::mlir::Value NextIterationSinkOp::input() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range NextIterationSinkOp::controlInputs() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange NextIterationSinkOp::tokenMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange NextIterationSinkOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange NextIterationSinkOp::controlInputsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> NextIterationSinkOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NextIterationSinkOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void NextIterationSinkOp::build(OpBuilder &builder, OperationState &result, Value token, ArrayRef<Value> operands, ArrayRef<NamedAttribute> attributes ) {
      assert(operands.size() >= 1 && "tf_executor.NextIteration.Sink builder "
             "expects at least one operand");
      result.operands.push_back(token);
      result.operands.insert(result.operands.end(), operands.begin(),
                              operands.end());
      result.attributes.append(attributes.begin(), attributes.end());
    
}

void NextIterationSinkOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token, ::mlir::Value input, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(token);
  odsState.addOperands(input);
  odsState.addOperands(controlInputs);
}

void NextIterationSinkOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value input, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(token);
  odsState.addOperands(input);
  odsState.addOperands(controlInputs);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NextIterationSinkOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult NextIterationSinkOp::verify() {
  if (failed(NextIterationSinkOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<TokenType>()))) {
        return emitOpError("operand #") << index << " must be token, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("operand #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::tf_executor::Verify(*this);
}

::mlir::ParseResult NextIterationSinkOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType tokenRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> tokenOperands(tokenRawOperands);  ::llvm::SMLoc tokenOperandsLoc;
  (void)tokenOperandsLoc;
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> controlInputsOperands;
  ::llvm::SMLoc controlInputsOperandsLoc;
  (void)controlInputsOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  if (parser.parseLSquare())
    return ::mlir::failure();

  tokenOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tokenRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalComma())) {

  controlInputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(controlInputsOperands))
    return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(inputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<TokenType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<ControlType>();
  if (parser.resolveOperands(tokenOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(controlInputsOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void NextIterationSinkOp::print(::mlir::OpAsmPrinter &p) {
  p << "tf_executor.NextIteration.Sink";
  p << "[";
  p << token();
  p << "]";
  p << " ";
  p << input();
  if (!controlInputs().empty()) {
  p << ",";
  p << " ";
  p << controlInputs();
  }
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(input().getType());
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::NextIterationSourceOp definitions
//===----------------------------------------------------------------------===//

NextIterationSourceOpAdaptor::NextIterationSourceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

NextIterationSourceOpAdaptor::NextIterationSourceOpAdaptor(NextIterationSourceOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> NextIterationSourceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange NextIterationSourceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::LogicalResult NextIterationSourceOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void NextIterationSourceOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "output");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "token");
  auto resultGroup2 = getODSResults(2);
  if (!llvm::empty(resultGroup2))
    setNameFn(*resultGroup2.begin(), "control");
}

::llvm::StringRef NextIterationSourceOp::getOperationName() {
  return "tf_executor.NextIteration.Source";
}

std::pair<unsigned, unsigned> NextIterationSourceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NextIterationSourceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> NextIterationSourceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NextIterationSourceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NextIterationSourceOp::output() {
  return *getODSResults(0).begin();
}

::mlir::Value NextIterationSourceOp::token() {
  return *getODSResults(1).begin();
}

::mlir::Value NextIterationSourceOp::control() {
  return *getODSResults(2).begin();
}

void NextIterationSourceOp::build(OpBuilder &builder, OperationState &result, Type result_type, ArrayRef<NamedAttribute> attributes ) {
      Type token_type = TokenType::get(builder.getContext());
      Type control_type = ControlType::get(builder.getContext());
      result.types = { result_type, token_type, control_type };
      result.attributes.append(attributes.begin(), attributes.end());
    
}

void NextIterationSourceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type token, ::mlir::Type control) {
  odsState.addTypes(output);
  odsState.addTypes(token);
  odsState.addTypes(control);
}

void NextIterationSourceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 3u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NextIterationSourceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 3u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult NextIterationSourceOp::verify() {
  if (failed(NextIterationSourceOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<TokenType>()))) {
        return emitOpError("result #") << index << " must be token, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSResults(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("result #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::tf_executor::Verify(*this);
}

::mlir::ParseResult NextIterationSourceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(outputRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<TokenType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<ControlType>();
  result.addTypes(outputTypes);
  result.addTypes(odsBuildableType0);
  result.addTypes(odsBuildableType1);
  return ::mlir::success();
}

void NextIterationSourceOp::print(::mlir::OpAsmPrinter &p) {
  p << "tf_executor.NextIteration.Source";
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(output().getType());
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::SwitchNOp definitions
//===----------------------------------------------------------------------===//

SwitchNOpAdaptor::SwitchNOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SwitchNOpAdaptor::SwitchNOpAdaptor(SwitchNOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SwitchNOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange SwitchNOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchNOpAdaptor::data() {
  return *getODSOperands(0).begin();
}

::mlir::Value SwitchNOpAdaptor::index() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange SwitchNOpAdaptor::controlInputs() {
  return getODSOperands(2);
}

::mlir::IntegerAttr SwitchNOpAdaptor::num_outs() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("num_outs").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult SwitchNOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_num_outs = odsAttrs.get("num_outs");
  if (!tblgen_num_outs) return emitError(loc, "'tf_executor._SwitchN' op ""requires attribute 'num_outs'");
    if (!(((tblgen_num_outs.isa<::mlir::IntegerAttr>())) && ((tblgen_num_outs.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) return emitError(loc, "'tf_executor._SwitchN' op ""attribute 'num_outs' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  return ::mlir::success();
}

void SwitchNOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "outputs");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "control");
}

::llvm::StringRef SwitchNOp::getOperationName() {
  return "tf_executor._SwitchN";
}

std::pair<unsigned, unsigned> SwitchNOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range SwitchNOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchNOp::data() {
  return *getODSOperands(0).begin();
}

::mlir::Value SwitchNOp::index() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range SwitchNOp::controlInputs() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange SwitchNOp::dataMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SwitchNOp::indexMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SwitchNOp::controlInputsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SwitchNOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range SwitchNOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range SwitchNOp::outputs() {
  return getODSResults(0);
}

::mlir::Value SwitchNOp::control() {
  return *getODSResults(1).begin();
}

::mlir::IntegerAttr SwitchNOp::num_outsAttr() {
  return this->getAttr("num_outs").cast<::mlir::IntegerAttr>();
}

uint64_t SwitchNOp::num_outs() {
  auto attr = num_outsAttr();
  return attr.getValue().getZExtValue();
}

void SwitchNOp::num_outsAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("num_outs", attr);
}

void SwitchNOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Type control, ::mlir::Value data, ::mlir::Value index, ::mlir::ValueRange controlInputs, ::mlir::IntegerAttr num_outs) {
  odsState.addOperands(data);
  odsState.addOperands(index);
  odsState.addOperands(controlInputs);
  odsState.addAttribute("num_outs", num_outs);
  odsState.addTypes(outputs);
  odsState.addTypes(control);
}

void SwitchNOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value index, ::mlir::ValueRange controlInputs, ::mlir::IntegerAttr num_outs) {
  odsState.addOperands(data);
  odsState.addOperands(index);
  odsState.addOperands(controlInputs);
  odsState.addAttribute("num_outs", num_outs);
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchNOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Type control, ::mlir::Value data, ::mlir::Value index, ::mlir::ValueRange controlInputs, uint64_t num_outs) {
  odsState.addOperands(data);
  odsState.addOperands(index);
  odsState.addOperands(controlInputs);
  odsState.addAttribute("num_outs", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_outs));
  odsState.addTypes(outputs);
  odsState.addTypes(control);
}

void SwitchNOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value index, ::mlir::ValueRange controlInputs, uint64_t num_outs) {
  odsState.addOperands(data);
  odsState.addOperands(index);
  odsState.addOperands(controlInputs);
  odsState.addAttribute("num_outs", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_outs));
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchNOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult SwitchNOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ParseSwitchNOp(parser, result);
}

void SwitchNOp::print(::mlir::OpAsmPrinter &p) {
  return ::mlir::tf_executor::Print(*this, p);
}

::mlir::LogicalResult SwitchNOp::verify() {
  if (failed(SwitchNOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))))) {
        return emitOpError("operand #") << index << " must be tensor of 32-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("operand #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("result #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((succeeded(VerifyControlOperandsAfterAllData(&(*this->getOperation()))))))
    return emitOpError("failed to verify that all control inputs must appear after any non-control input");
  return ::mlir::tf_executor::Verify(*this);
}

} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::SwitchOp definitions
//===----------------------------------------------------------------------===//

SwitchOpAdaptor::SwitchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SwitchOpAdaptor::SwitchOpAdaptor(SwitchOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SwitchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange SwitchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOpAdaptor::data() {
  return *getODSOperands(0).begin();
}

::mlir::Value SwitchOpAdaptor::predicate() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange SwitchOpAdaptor::controlInputs() {
  return getODSOperands(2);
}

::mlir::LogicalResult SwitchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void SwitchOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "falseOutput");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "trueOutput");
  auto resultGroup2 = getODSResults(2);
  if (!llvm::empty(resultGroup2))
    setNameFn(*resultGroup2.begin(), "control");
}

::llvm::StringRef SwitchOp::getOperationName() {
  return "tf_executor.Switch";
}

std::pair<unsigned, unsigned> SwitchOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range SwitchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOp::data() {
  return *getODSOperands(0).begin();
}

::mlir::Value SwitchOp::predicate() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range SwitchOp::controlInputs() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange SwitchOp::dataMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SwitchOp::predicateMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SwitchOp::controlInputsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SwitchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOp::falseOutput() {
  return *getODSResults(0).begin();
}

::mlir::Value SwitchOp::trueOutput() {
  return *getODSResults(1).begin();
}

::mlir::Value SwitchOp::control() {
  return *getODSResults(2).begin();
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type falseOutput, ::mlir::Type trueOutput, ::mlir::Type control, ::mlir::Value data, ::mlir::Value predicate, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(data);
  odsState.addOperands(predicate);
  odsState.addOperands(controlInputs);
  odsState.addTypes(falseOutput);
  odsState.addTypes(trueOutput);
  odsState.addTypes(control);
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value predicate, ::mlir::ValueRange controlInputs) {
  odsState.addOperands(data);
  odsState.addOperands(predicate);
  odsState.addOperands(controlInputs);
  assert(resultTypes.size() == 3u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 3u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult SwitchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ParseSwitchOp(parser, result);
}

void SwitchOp::print(::mlir::OpAsmPrinter &p) {
  return ::mlir::tf_executor::Print(*this, p);
}

::mlir::LogicalResult SwitchOp::verify() {
  if (failed(SwitchOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))))) {
        return emitOpError("operand #") << index << " must be tensor of 1-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("operand #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSResults(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<ControlType>()))) {
        return emitOpError("result #") << index << " must be control, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((succeeded(VerifyControlOperandsAfterAllData(&(*this->getOperation()))))))
    return emitOpError("failed to verify that all control inputs must appear after any non-control input");
  if (!(((((*this->getOperation()).getNumResults() > 0)) && (((*this->getOperation()).getNumOperands() > 0)) && (((*this->getOperation()).getResult(0).getType().isa<::mlir::ShapedType>())) && (((*this->getOperation()).getOperand(0).getType().isa<::mlir::ShapedType>()))) && ((mlir::TF::BroadcastCompatible((*this->getOperation()).getOperand(0).getType(), (*this->getOperation()).getResult(0).getType())))))
    return emitOpError("failed to verify that data operand must be broadcastable to true result");
  if (!(((((*this->getOperation()).getNumResults() > 0)) && (((*this->getOperation()).getNumOperands() > 1)) && (((*this->getOperation()).getResult(0).getType().isa<::mlir::ShapedType>())) && (((*this->getOperation()).getOperand(1).getType().isa<::mlir::ShapedType>()))) && ((mlir::TF::BroadcastCompatible((*this->getOperation()).getOperand(0).getType(), (*this->getOperation()).getResult(1).getType())))))
    return emitOpError("failed to verify that data operand must be broadcastable to false result");
  return ::mlir::success();
}

} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

YieldOpAdaptor::YieldOpAdaptor(YieldOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> YieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange YieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange YieldOpAdaptor::fetches() {
  return getODSOperands(0);
}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef YieldOp::getOperationName() {
  return "tf_executor.yield";
}

std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range YieldOp::fetches() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::fetchesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> YieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range YieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void YieldOp::build(OpBuilder &builder, OperationState &result) {
      build(builder, result, {});
    
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange fetches) {
  odsState.addOperands(fetches);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult YieldOp::verify() {
  if (failed(YieldOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  if (!((succeeded(VerifyControlOperandsAfterAllData(&(*this->getOperation()))))))
    return emitOpError("failed to verify that all control inputs must appear after any non-control input");
  return ::mlir::success();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> fetchesOperands;
  ::llvm::SMLoc fetchesOperandsLoc;
  (void)fetchesOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> fetchesTypes;

  fetchesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(fetchesOperands))
    return ::mlir::failure();
  if (!fetchesOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(fetchesTypes))
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(fetchesOperands, fetchesTypes, fetchesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &p) {
  p << "tf_executor.yield";
  if (!fetches().empty()) {
  p << " ";
  p << fetches();
  p << " " << ":";
  p << " ";
  p << fetches().getTypes();
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

} // namespace tf_executor
} // namespace mlir

#endif  // GET_OP_CLASSES

