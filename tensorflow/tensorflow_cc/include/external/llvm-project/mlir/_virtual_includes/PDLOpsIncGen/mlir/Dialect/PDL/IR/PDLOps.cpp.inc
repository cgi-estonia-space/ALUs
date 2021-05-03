/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::pdl::ApplyConstraintOp,
::mlir::pdl::AttributeOp,
::mlir::pdl::CreateNativeOp,
::mlir::pdl::EraseOp,
::mlir::pdl::InputOp,
::mlir::pdl::OperationOp,
::mlir::pdl::PatternOp,
::mlir::pdl::ReplaceOp,
::mlir::pdl::RewriteEndOp,
::mlir::pdl::RewriteOp,
::mlir::pdl::TypeOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ApplyConstraintOp definitions
//===----------------------------------------------------------------------===//

ApplyConstraintOpAdaptor::ApplyConstraintOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ApplyConstraintOpAdaptor::ApplyConstraintOpAdaptor(ApplyConstraintOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ApplyConstraintOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ApplyConstraintOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ApplyConstraintOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::StringAttr ApplyConstraintOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr ApplyConstraintOpAdaptor::constParams() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("constParams").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult ApplyConstraintOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (!tblgen_name) return emitError(loc, "'pdl.apply_constraint' op ""requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.apply_constraint' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_constParams = odsAttrs.get("constParams");
  if (tblgen_constParams) {
    if (!((tblgen_constParams.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'pdl.apply_constraint' op ""attribute 'constParams' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef ApplyConstraintOp::getOperationName() {
  return "pdl.apply_constraint";
}

std::pair<unsigned, unsigned> ApplyConstraintOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ApplyConstraintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyConstraintOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyConstraintOp::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ApplyConstraintOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ApplyConstraintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr ApplyConstraintOp::nameAttr() {
  return this->getAttr("name").cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyConstraintOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr ApplyConstraintOp::constParamsAttr() {
  return this->getAttr("constParams").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ApplyConstraintOp::constParams() {
  auto attr = constParamsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void ApplyConstraintOp::nameAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("name", attr);
}

void ApplyConstraintOp::constParamsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("constParams", attr);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, ValueRange args ,  ArrayRef<Attribute> params ) {
      build(odsBuilder, odsState, odsBuilder.getStringAttr(name), args,
            params.empty() ? ArrayAttr() : odsBuilder.getArrayAttr(params));
    
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", name);
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", name);
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyConstraintOp::verify() {
  if (failed(ApplyConstraintOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::pdl::AttributeType>())) || ((v.getType().isa<mlir::pdl::OperationType>())) || ((v.getType().isa<mlir::pdl::TypeType>())) || ((v.getType().isa<mlir::pdl::ValueType>())))) {
        return emitOpError("operand #") << index << " must be Positional Value, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult ApplyConstraintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::ArrayAttr constParamsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> argsTypes;

  if (parser.parseAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes))
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(constParamsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "constParams", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (constParamsAttr) {
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyConstraintOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.apply_constraint";
  p << " ";
  p.printAttributeWithoutType(nameAttr());
  if (getAttr("constParams")) {
  p << " ";
  p.printAttributeWithoutType(constParamsAttr());
  }
  p << "(";
  p << args();
  p << " " << ":";
  p << " ";
  p << args().getTypes();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"name", "constParams"});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::AttributeOp definitions
//===----------------------------------------------------------------------===//

AttributeOpAdaptor::AttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AttributeOpAdaptor::AttributeOpAdaptor(AttributeOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange AttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AttributeOpAdaptor::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? Value() : *operands.begin();
}

::mlir::Attribute AttributeOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("value").dyn_cast_or_null<::mlir::Attribute>();
  return attr;
}

::mlir::LogicalResult AttributeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (tblgen_value) {
    if (!((true))) return emitError(loc, "'pdl.attribute' op ""attribute 'value' failed to satisfy constraint: any attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef AttributeOp::getOperationName() {
  return "pdl.attribute";
}

std::pair<unsigned, unsigned> AttributeOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range AttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AttributeOp::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? Value() : *operands.begin();
}

::mlir::MutableOperandRange AttributeOp::typeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AttributeOp::attr() {
  return *getODSResults(0).begin();
}

::mlir::Attribute AttributeOp::valueAttr() {
  return this->getAttr("value").dyn_cast_or_null<::mlir::Attribute>();
}

::llvm::Optional<::mlir::Attribute> AttributeOp::value() {
  auto attr = valueAttr();
  return attr ? ::llvm::Optional<::mlir::Attribute>(attr) : (::llvm::None);
}

void AttributeOp::valueAttr(::mlir::Attribute attr) {
  this->getOperation()->setAttr("value", attr);
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value type ) {
      build(odsBuilder, odsState, odsBuilder.getType<AttributeType>(), type,
            Attribute());
    
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute attr) {
      build(odsBuilder, odsState, odsBuilder.getType<AttributeType>(), Value(), attr);
    
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attr, /*optional*/::mlir::Value type, /*optional*/::mlir::Attribute value) {
  if (type)
    odsState.addOperands(type);
  if (value) {
  odsState.addAttribute("value", value);
  }
  odsState.addTypes(attr);
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value type, /*optional*/::mlir::Attribute value) {
  if (type)
    odsState.addOperands(type);
  if (value) {
  odsState.addAttribute("value", value);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AttributeOp::verify() {
  if (failed(AttributeOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::TypeType>()))) {
        return emitOpError("operand #") << index << " must be mlir::pdl::TypeType, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::AttributeType>()))) {
        return emitOpError("result #") << index << " must be mlir::pdl::AttributeType, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult AttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> typeOperands;
  ::llvm::SMLoc typeOperandsLoc;
  (void)typeOperandsLoc;
  ::mlir::Attribute valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalColon())) {

  {
    typeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      typeOperands.push_back(operand);
    }
  }
  }

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(valueAttr, "value", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (valueAttr) {
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<mlir::pdl::AttributeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(typeOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AttributeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.attribute";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"value"});
  if (type()) {
  p << " " << ":";
  p << " ";
  if (::mlir::Value value = type())
    p << value;
  }
  if (getAttr("value")) {
  p << " ";
  p.printAttribute(valueAttr());
  }
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::CreateNativeOp definitions
//===----------------------------------------------------------------------===//

CreateNativeOpAdaptor::CreateNativeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CreateNativeOpAdaptor::CreateNativeOpAdaptor(CreateNativeOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CreateNativeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange CreateNativeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange CreateNativeOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::StringAttr CreateNativeOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr CreateNativeOpAdaptor::constParams() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("constParams").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult CreateNativeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (!tblgen_name) return emitError(loc, "'pdl.create_native' op ""requires attribute 'name'");
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.create_native' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_constParams = odsAttrs.get("constParams");
  if (tblgen_constParams) {
    if (!((tblgen_constParams.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'pdl.create_native' op ""attribute 'constParams' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef CreateNativeOp::getOperationName() {
  return "pdl.create_native";
}

std::pair<unsigned, unsigned> CreateNativeOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CreateNativeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CreateNativeOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange CreateNativeOp::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CreateNativeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateNativeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateNativeOp::result() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr CreateNativeOp::nameAttr() {
  return this->getAttr("name").cast<::mlir::StringAttr>();
}

::llvm::StringRef CreateNativeOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CreateNativeOp::constParamsAttr() {
  return this->getAttr("constParams").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CreateNativeOp::constParams() {
  auto attr = constParamsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void CreateNativeOp::nameAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("name", attr);
}

void CreateNativeOp::constParamsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("constParams", attr);
}

void CreateNativeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", name);
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  odsState.addTypes(result);
}

void CreateNativeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", name);
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateNativeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  odsState.addTypes(result);
}

void CreateNativeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange args, /*optional*/::mlir::ArrayAttr constParams) {
  odsState.addOperands(args);
  odsState.addAttribute("name", odsBuilder.getStringAttr(name));
  if (constParams) {
  odsState.addAttribute("constParams", constParams);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateNativeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateNativeOp::verify() {
  if (failed(CreateNativeOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::pdl::AttributeType>())) || ((v.getType().isa<mlir::pdl::OperationType>())) || ((v.getType().isa<mlir::pdl::TypeType>())) || ((v.getType().isa<mlir::pdl::ValueType>())))) {
        return emitOpError("operand #") << index << " must be Positional Value, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<mlir::pdl::AttributeType>())) || ((v.getType().isa<mlir::pdl::OperationType>())) || ((v.getType().isa<mlir::pdl::TypeType>())) || ((v.getType().isa<mlir::pdl::ValueType>())))) {
        return emitOpError("result #") << index << " must be Positional Value, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}

::mlir::ParseResult CreateNativeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::ArrayAttr constParamsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> argsTypes;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  if (parser.parseAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes))
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(constParamsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "constParams", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (constParamsAttr) {
  }
  if (succeeded(parser.parseOptionalLParen())) {

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateNativeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.create_native";
  p << " ";
  p.printAttributeWithoutType(nameAttr());
  if (getAttr("constParams")) {
  p << " ";
  p.printAttributeWithoutType(constParamsAttr());
  }
  if (!args().empty()) {
  p << "(";
  p << args();
  p << " " << ":";
  p << " ";
  p << args().getTypes();
  p << ")";
  }
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"name", "constParams"});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::EraseOp definitions
//===----------------------------------------------------------------------===//

EraseOpAdaptor::EraseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

EraseOpAdaptor::EraseOpAdaptor(EraseOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> EraseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange EraseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EraseOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult EraseOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef EraseOp::getOperationName() {
  return "pdl.erase";
}

std::pair<unsigned, unsigned> EraseOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range EraseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EraseOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange EraseOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> EraseOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EraseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation) {
  odsState.addOperands(operation);
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation) {
  odsState.addOperands(operation);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EraseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EraseOp::verify() {
  if (failed(EraseOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::OperationType>()))) {
        return emitOpError("operand #") << index << " must be mlir::pdl::OperationType, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult EraseOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EraseOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.erase";
  p << " ";
  p << operation();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::InputOp definitions
//===----------------------------------------------------------------------===//

InputOpAdaptor::InputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

InputOpAdaptor::InputOpAdaptor(InputOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> InputOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange InputOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InputOpAdaptor::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? Value() : *operands.begin();
}

::mlir::LogicalResult InputOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef InputOp::getOperationName() {
  return "pdl.input";
}

std::pair<unsigned, unsigned> InputOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range InputOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InputOp::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? Value() : *operands.begin();
}

::mlir::MutableOperandRange InputOp::typeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> InputOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InputOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InputOp::val() {
  return *getODSResults(0).begin();
}

void InputOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
      build(odsBuilder, odsState, odsBuilder.getType<ValueType>(), Value());
    
}

void InputOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, /*optional*/::mlir::Value type) {
  if (type)
    odsState.addOperands(type);
  odsState.addTypes(val);
}

void InputOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value type) {
  if (type)
    odsState.addOperands(type);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InputOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InputOp::verify() {
  if (failed(InputOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    if (valueGroup0.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup0.size();
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::TypeType>()))) {
        return emitOpError("operand #") << index << " must be mlir::pdl::TypeType, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::ValueType>()))) {
        return emitOpError("result #") << index << " must be mlir::pdl::ValueType, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult InputOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> typeOperands;
  ::llvm::SMLoc typeOperandsLoc;
  (void)typeOperandsLoc;
  if (succeeded(parser.parseOptionalColon())) {

  {
    typeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      typeOperands.push_back(operand);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(typeOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InputOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.input";
  if (type()) {
  p << " " << ":";
  p << " ";
  if (::mlir::Value value = type())
    p << value;
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::OperationOp definitions
//===----------------------------------------------------------------------===//

OperationOpAdaptor::OperationOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

OperationOpAdaptor::OperationOpAdaptor(OperationOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> OperationOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange OperationOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange OperationOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::ValueRange OperationOpAdaptor::attributes() {
  return getODSOperands(1);
}

::mlir::ValueRange OperationOpAdaptor::types() {
  return getODSOperands(2);
}

::mlir::StringAttr OperationOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr OperationOpAdaptor::attributeNames() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("attributeNames").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult OperationOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    {
  auto tblgen_name = odsAttrs.get("name");
  if (tblgen_name) {
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.operation' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_attributeNames = odsAttrs.get("attributeNames");
  if (!tblgen_attributeNames) return emitError(loc, "'pdl.operation' op ""requires attribute 'attributeNames'");
    if (!(((tblgen_attributeNames.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_attributeNames.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return (attr.isa<::mlir::StringAttr>()); })))) return emitError(loc, "'pdl.operation' op ""attribute 'attributeNames' failed to satisfy constraint: string array attribute");
  }
  return ::mlir::success();
}

void OperationOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "op");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "results");
}

::llvm::StringRef OperationOp::getOperationName() {
  return "pdl.operation";
}

std::pair<unsigned, unsigned> OperationOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range OperationOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range OperationOp::operands() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range OperationOp::attributes() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range OperationOp::types() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange OperationOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange OperationOp::attributesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange OperationOp::typesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> OperationOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
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

::mlir::Operation::result_range OperationOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperationOp::op() {
  return *getODSResults(0).begin();
}

::mlir::Operation::result_range OperationOp::results() {
  return getODSResults(1);
}

::mlir::StringAttr OperationOp::nameAttr() {
  return this->getAttr("name").dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > OperationOp::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr OperationOp::attributeNamesAttr() {
  return this->getAttr("attributeNames").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr OperationOp::attributeNames() {
  auto attr = attributeNamesAttr();
  return attr;
}

void OperationOp::nameAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("name", attr);
}

void OperationOp::attributeNamesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("attributeNames", attr);
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Optional<StringRef> name  ,  ValueRange operandValues ,  ArrayRef<StringRef> attrNames ,  ValueRange attrValues  ,  ValueRange resultTypes ) {
      auto nameAttr = name ? StringAttr() : odsBuilder.getStringAttr(*name);
      build(odsBuilder, odsState, odsBuilder.getType<OperationType>(), {}, nameAttr,
            operandValues, attrValues, odsBuilder.getStrArrayAttr(attrNames),
            resultTypes);
      odsState.types.append(resultTypes.size(), odsBuilder.getType<ValueType>());
    
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type op, ::mlir::TypeRange results, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange operands, ::mlir::ValueRange attributes, ::mlir::ArrayAttr attributeNames, ::mlir::ValueRange types) {
  odsState.addOperands(operands);
  odsState.addOperands(attributes);
  odsState.addOperands(types);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(attributes.size()), static_cast<int32_t>(types.size())}));
  if (name) {
  odsState.addAttribute("name", name);
  }
  odsState.addAttribute("attributeNames", attributeNames);
  odsState.addTypes(op);
  odsState.addTypes(results);
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange operands, ::mlir::ValueRange attributes, ::mlir::ArrayAttr attributeNames, ::mlir::ValueRange types) {
  odsState.addOperands(operands);
  odsState.addOperands(attributes);
  odsState.addOperands(types);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(attributes.size()), static_cast<int32_t>(types.size())}));
  if (name) {
  odsState.addAttribute("name", name);
  }
  odsState.addAttribute("attributeNames", attributeNames);
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OperationOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult OperationOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseOperationOp(parser, result);
}

void OperationOp::print(::mlir::OpAsmPrinter &p) {
  ::print(p, *this);
}

::mlir::LogicalResult OperationOp::verify() {
  if (failed(OperationOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::ValueType>()))) {
        return emitOpError("operand #") << index << " must be mlir::pdl::ValueType, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::AttributeType>()))) {
        return emitOpError("operand #") << index << " must be mlir::pdl::AttributeType, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::TypeType>()))) {
        return emitOpError("operand #") << index << " must be mlir::pdl::TypeType, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::OperationType>()))) {
        return emitOpError("result #") << index << " must be mlir::pdl::OperationType, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::ValueType>()))) {
        return emitOpError("result #") << index << " must be mlir::pdl::ValueType, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

void OperationOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::PatternOp definitions
//===----------------------------------------------------------------------===//

PatternOpAdaptor::PatternOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

PatternOpAdaptor::PatternOpAdaptor(PatternOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> PatternOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PatternOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr PatternOpAdaptor::rootKind() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("rootKind").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::IntegerAttr PatternOpAdaptor::benefit() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("benefit").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::StringAttr PatternOpAdaptor::sym_name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult PatternOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_rootKind = odsAttrs.get("rootKind");
  if (tblgen_rootKind) {
    if (!((tblgen_rootKind.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.pattern' op ""attribute 'rootKind' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_benefit = odsAttrs.get("benefit");
  if (!tblgen_benefit) return emitError(loc, "'pdl.pattern' op ""requires attribute 'benefit'");
    if (!((((tblgen_benefit.isa<::mlir::IntegerAttr>())) && ((tblgen_benefit.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(16)))) && ((!tblgen_benefit.cast<::mlir::IntegerAttr>().getValue().isNegative())))) return emitError(loc, "'pdl.pattern' op ""attribute 'benefit' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative");
  }
  {
  auto tblgen_sym_name = odsAttrs.get("sym_name");
  if (tblgen_sym_name) {
    if (!((tblgen_sym_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.pattern' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef PatternOp::getOperationName() {
  return "pdl.pattern";
}

std::pair<unsigned, unsigned> PatternOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PatternOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> PatternOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PatternOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &PatternOp::body() {
  return this->getOperation()->getRegion(0);
}

::mlir::StringAttr PatternOp::rootKindAttr() {
  return this->getAttr("rootKind").dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > PatternOp::rootKind() {
  auto attr = rootKindAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr PatternOp::benefitAttr() {
  return this->getAttr("benefit").cast<::mlir::IntegerAttr>();
}

uint16_t PatternOp::benefit() {
  auto attr = benefitAttr();
  return attr.getValue().getZExtValue();
}

::mlir::StringAttr PatternOp::sym_nameAttr() {
  return this->getAttr("sym_name").dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > PatternOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void PatternOp::rootKindAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("rootKind", attr);
}

void PatternOp::benefitAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("benefit", attr);
}

void PatternOp::sym_nameAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("sym_name", attr);
}



void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::StringAttr rootKind, ::mlir::IntegerAttr benefit, /*optional*/::mlir::StringAttr sym_name) {
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  odsState.addAttribute("benefit", benefit);
  if (sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  }
  (void)odsState.addRegion();
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::StringAttr rootKind, ::mlir::IntegerAttr benefit, /*optional*/::mlir::StringAttr sym_name) {
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  odsState.addAttribute("benefit", benefit);
  if (sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::StringAttr rootKind, uint16_t benefit, /*optional*/::mlir::StringAttr sym_name) {
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  odsState.addAttribute("benefit", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  if (sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  }
  (void)odsState.addRegion();
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::StringAttr rootKind, uint16_t benefit, /*optional*/::mlir::StringAttr sym_name) {
  if (rootKind) {
  odsState.addAttribute("rootKind", rootKind);
  }
  odsState.addAttribute("benefit", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  if (sym_name) {
  odsState.addAttribute("sym_name", sym_name);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PatternOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PatternOp::verify() {
  if (failed(PatternOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
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
  return ::verify(*this);
}

::mlir::ParseResult PatternOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::IntegerAttr benefitAttr;
  ::mlir::StringAttr rootKindAttr;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();

  // Parsing an optional symbol name doesn't fail, so no need to check the
  // result.
  (void)parser.parseOptionalSymbolName(sym_nameAttr, "sym_name", result.attributes);
  if (sym_nameAttr) {
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseKeyword("benefit"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseAttribute(benefitAttr, parser.getBuilder().getIntegerType(16), "benefit", result.attributes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalComma())) {
  if (parser.parseKeyword("root"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(rootKindAttr, parser.getBuilder().getType<::mlir::NoneType>(), "rootKind", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegion));
  return ::mlir::success();
}

void PatternOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.pattern";
  if (getAttr("sym_name")) {
  p << " ";
  p.printSymbolName(sym_nameAttr().getValue());
  }
  p << " " << ":";
  p << " " << "benefit";
  p << "(";
  p.printAttributeWithoutType(benefitAttr());
  p << ")";
  if (getAttr("rootKind")) {
  p << ",";
  p << " " << "root";
  p << "(";
  p.printAttributeWithoutType(rootKindAttr());
  p << ")";
  }
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{"sym_name", "benefit", "rootKind"});
  p << " ";
  p.printRegion(body());
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ReplaceOp definitions
//===----------------------------------------------------------------------===//

ReplaceOpAdaptor::ReplaceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ReplaceOpAdaptor::ReplaceOpAdaptor(ReplaceOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ReplaceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ReplaceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReplaceOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReplaceOpAdaptor::replOperation() {
  auto operands = getODSOperands(1);
  return operands.empty() ? Value() : *operands.begin();
}

::mlir::ValueRange ReplaceOpAdaptor::replValues() {
  return getODSOperands(2);
}

::mlir::LogicalResult ReplaceOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef ReplaceOp::getOperationName() {
  return "pdl.replace";
}

std::pair<unsigned, unsigned> ReplaceOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ReplaceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReplaceOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReplaceOp::replOperation() {
  auto operands = getODSOperands(1);
  return operands.empty() ? Value() : *operands.begin();
}

::mlir::Operation::operand_range ReplaceOp::replValues() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ReplaceOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ReplaceOp::replOperationMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ReplaceOp::replValuesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ReplaceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReplaceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, /*optional*/::mlir::Value replOperation, ::mlir::ValueRange replValues) {
  odsState.addOperands(operation);
  if (replOperation)
    odsState.addOperands(replOperation);
  odsState.addOperands(replValues);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, (replOperation ? 1 : 0), static_cast<int32_t>(replValues.size())}));
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, /*optional*/::mlir::Value replOperation, ::mlir::ValueRange replValues) {
  odsState.addOperands(operation);
  if (replOperation)
    odsState.addOperands(replOperation);
  odsState.addOperands(replValues);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, (replOperation ? 1 : 0), static_cast<int32_t>(replValues.size())}));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReplaceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReplaceOp::verify() {
  if (failed(ReplaceOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::OperationType>()))) {
        return emitOpError("operand #") << index << " must be mlir::pdl::OperationType, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    if (valueGroup1.size() > 1)
      return emitOpError("operand group starting at #") << index << " requires 0 or 1 element, but found " << valueGroup1.size();
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::OperationType>()))) {
        return emitOpError("operand #") << index << " must be mlir::pdl::OperationType, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::ValueType>()))) {
        return emitOpError("operand #") << index << " must be mlir::pdl::ValueType, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult ReplaceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> replValuesOperands;
  ::llvm::SMLoc replValuesOperandsLoc;
  (void)replValuesOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> replOperationOperands;
  ::llvm::SMLoc replOperationOperandsLoc;
  (void)replOperationOperandsLoc;

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("with"))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  replValuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(replValuesOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  {
    replOperationOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      replOperationOperands.push_back(operand);
    }
  }
  if (!replOperationOperands.empty()) {
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<mlir::pdl::OperationType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<mlir::pdl::ValueType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replOperationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replValuesOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(replOperationOperands.size()), static_cast<int32_t>(replValuesOperands.size())}));
  return ::mlir::success();
}

void ReplaceOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.replace";
  p << " ";
  p << operation();
  p << " " << "with";
  if (!replValues().empty()) {
  p << "(";
  p << replValues();
  p << ")";
  }
  if (replOperation()) {
  p << " ";
  if (::mlir::Value value = replOperation())
    p << value;
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::RewriteEndOp definitions
//===----------------------------------------------------------------------===//

RewriteEndOpAdaptor::RewriteEndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

RewriteEndOpAdaptor::RewriteEndOpAdaptor(RewriteEndOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> RewriteEndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange RewriteEndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::LogicalResult RewriteEndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef RewriteEndOp::getOperationName() {
  return "pdl.rewrite_end";
}

std::pair<unsigned, unsigned> RewriteEndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RewriteEndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> RewriteEndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RewriteEndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void RewriteEndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {

}

void RewriteEndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RewriteEndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RewriteEndOp::verify() {
  if (failed(RewriteEndOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::ParseResult RewriteEndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void RewriteEndOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.rewrite_end";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::RewriteOp definitions
//===----------------------------------------------------------------------===//

RewriteOpAdaptor::RewriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

RewriteOpAdaptor::RewriteOpAdaptor(RewriteOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> RewriteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange RewriteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RewriteOpAdaptor::root() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange RewriteOpAdaptor::externalArgs() {
  return getODSOperands(1);
}

::mlir::StringAttr RewriteOpAdaptor::name() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::mlir::ArrayAttr RewriteOpAdaptor::externalConstParams() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("externalConstParams").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult RewriteOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_name = odsAttrs.get("name");
  if (tblgen_name) {
    if (!((tblgen_name.isa<::mlir::StringAttr>()))) return emitError(loc, "'pdl.rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  }
  {
  auto tblgen_externalConstParams = odsAttrs.get("externalConstParams");
  if (tblgen_externalConstParams) {
    if (!((tblgen_externalConstParams.isa<::mlir::ArrayAttr>()))) return emitError(loc, "'pdl.rewrite' op ""attribute 'externalConstParams' failed to satisfy constraint: array attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef RewriteOp::getOperationName() {
  return "pdl.rewrite";
}

std::pair<unsigned, unsigned> RewriteOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range RewriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RewriteOp::root() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range RewriteOp::externalArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange RewriteOp::rootMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange RewriteOp::externalArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> RewriteOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RewriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &RewriteOp::body() {
  return this->getOperation()->getRegion(0);
}

::mlir::StringAttr RewriteOp::nameAttr() {
  return this->getAttr("name").dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > RewriteOp::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr RewriteOp::externalConstParamsAttr() {
  return this->getAttr("externalConstParams").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > RewriteOp::externalConstParams() {
  auto attr = externalConstParamsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void RewriteOp::nameAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("name", attr);
}

void RewriteOp::externalConstParamsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("externalConstParams", attr);
}

void RewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value root, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange externalArgs, /*optional*/::mlir::ArrayAttr externalConstParams) {
  odsState.addOperands(root);
  odsState.addOperands(externalArgs);
  if (name) {
  odsState.addAttribute("name", name);
  }
  if (externalConstParams) {
  odsState.addAttribute("externalConstParams", externalConstParams);
  }
  (void)odsState.addRegion();
}

void RewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value root, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange externalArgs, /*optional*/::mlir::ArrayAttr externalConstParams) {
  odsState.addOperands(root);
  odsState.addOperands(externalArgs);
  if (name) {
  odsState.addAttribute("name", name);
  }
  if (externalConstParams) {
  odsState.addAttribute("externalConstParams", externalConstParams);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RewriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RewriteOp::verify() {
  if (failed(RewriteOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::OperationType>()))) {
        return emitOpError("operand #") << index << " must be mlir::pdl::OperationType, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<mlir::pdl::AttributeType>())) || ((v.getType().isa<mlir::pdl::OperationType>())) || ((v.getType().isa<mlir::pdl::TypeType>())) || ((v.getType().isa<mlir::pdl::ValueType>())))) {
        return emitOpError("operand #") << index << " must be Positional Value, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
    for (::mlir::Region &region : ::mlir::MutableArrayRef<::mlir::Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult RewriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType rootRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rootOperands(rootRawOperands);  ::llvm::SMLoc rootOperandsLoc;
  (void)rootOperandsLoc;
  ::mlir::StringAttr nameAttr;
  ::mlir::ArrayAttr externalConstParamsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> externalArgsOperands;
  ::llvm::SMLoc externalArgsOperandsLoc;
  (void)externalArgsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> externalArgsTypes;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();

  rootOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rootRawOperands[0]))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("with"))) {

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(externalConstParamsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "externalConstParams", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (externalConstParamsAttr) {
  }
  if (succeeded(parser.parseOptionalLParen())) {

  externalArgsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(externalArgsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(externalArgsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  }

  if (parser.parseOptionalRegion(*bodyRegion))
    return ::mlir::failure();
  if (!bodyRegion->empty()) {
  
  ensureTerminator(*bodyRegion, parser.getBuilder(), result.location);
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<mlir::pdl::OperationType>();
  if (parser.resolveOperands(rootOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(externalArgsOperands, externalArgsTypes, externalArgsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegion));
  return ::mlir::success();
}

void RewriteOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.rewrite";
  p << " ";
  p << root();
  if (getAttr("name")) {
  p << " " << "with";
  p << " ";
  p.printAttributeWithoutType(nameAttr());
  if (getAttr("externalConstParams")) {
  p << " ";
  p.printAttributeWithoutType(externalConstParamsAttr());
  }
  if (!externalArgs().empty()) {
  p << "(";
  p << externalArgs();
  p << " " << ":";
  p << " ";
  p << externalArgs().getTypes();
  p << ")";
  }
  }
  if (!body().empty()) {
  p << " ";

  {
    bool printTerminator = true;
    if (auto *term = body().empty() ? nullptr : body().begin()->getTerminator()) {
      printTerminator = !term->getMutableAttrDict().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    p.printRegion(body(), /*printEntryBlockArgs=*/true,
                  /*printBlockTerminators=*/printTerminator);
  }
  }
  p.printOptionalAttrDictWithKeyword(getAttrs(), /*elidedAttrs=*/{"name", "externalConstParams"});
}

} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::TypeOp definitions
//===----------------------------------------------------------------------===//

TypeOpAdaptor::TypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TypeOpAdaptor::TypeOpAdaptor(TypeOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::TypeAttr TypeOpAdaptor::type() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("type").dyn_cast_or_null<::mlir::TypeAttr>();
  return attr;
}

::mlir::LogicalResult TypeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_type = odsAttrs.get("type");
  if (tblgen_type) {
    if (!(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) return emitError(loc, "'pdl.type' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef TypeOp::getOperationName() {
  return "pdl.type";
}

std::pair<unsigned, unsigned> TypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TypeOp::result() {
  return *getODSResults(0).begin();
}

::mlir::TypeAttr TypeOp::typeAttr() {
  return this->getAttr("type").dyn_cast_or_null<::mlir::TypeAttr>();
}

::llvm::Optional<::mlir::Type> TypeOp::type() {
  auto attr = typeAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

void TypeOp::typeAttr(::mlir::TypeAttr attr) {
  this->getOperation()->setAttr("type", attr);
}

void TypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type ty ) {
      build(odsBuilder, odsState, odsBuilder.getType<AttributeType>(),
            ty ? TypeAttr::get(ty) : TypeAttr());
    
}

void TypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::TypeAttr type) {
  if (type) {
  odsState.addAttribute("type", type);
  }
  odsState.addTypes(result);
}

void TypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::TypeAttr type) {
  if (type) {
  odsState.addAttribute("type", type);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TypeOp::verify() {
  if (failed(TypeOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<mlir::pdl::TypeType>()))) {
        return emitOpError("result #") << index << " must be mlir::pdl::TypeType, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

::mlir::ParseResult TypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::TypeAttr typeAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalColon())) {

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "type", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void TypeOp::print(::mlir::OpAsmPrinter &p) {
  p << "pdl.type";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"type"});
  if (getAttr("type")) {
  p << " " << ":";
  p << " ";
  p.printAttributeWithoutType(typeAttr());
  }
}

} // namespace pdl
} // namespace mlir

#endif  // GET_OP_CLASSES

