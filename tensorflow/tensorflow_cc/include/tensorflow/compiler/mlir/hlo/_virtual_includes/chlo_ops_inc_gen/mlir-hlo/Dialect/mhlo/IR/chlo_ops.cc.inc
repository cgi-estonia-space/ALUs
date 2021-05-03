/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::chlo::AcosOp,
::mlir::chlo::AtanOp,
::mlir::chlo::BroadcastAddOp,
::mlir::chlo::BroadcastAndOp,
::mlir::chlo::BroadcastAtan2Op,
::mlir::chlo::BroadcastCompareOp,
::mlir::chlo::BroadcastComplexOp,
::mlir::chlo::BroadcastDivOp,
::mlir::chlo::BroadcastMaxOp,
::mlir::chlo::BroadcastMinOp,
::mlir::chlo::BroadcastMulOp,
::mlir::chlo::BroadcastOrOp,
::mlir::chlo::BroadcastPowOp,
::mlir::chlo::BroadcastRemOp,
::mlir::chlo::BroadcastShiftLeftOp,
::mlir::chlo::BroadcastShiftRightArithmeticOp,
::mlir::chlo::BroadcastShiftRightLogicalOp,
::mlir::chlo::BroadcastSubOp,
::mlir::chlo::BroadcastXorOp,
::mlir::chlo::ConstantLikeOp,
::mlir::chlo::SinhOp,
::mlir::chlo::TanOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::AcosOp definitions
//===----------------------------------------------------------------------===//

AcosOpAdaptor::AcosOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AcosOpAdaptor::AcosOpAdaptor(AcosOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AcosOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AcosOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AcosOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult AcosOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef AcosOp::getOperationName() {
  return "chlo.acos";
}

std::pair<unsigned, unsigned> AcosOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AcosOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AcosOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AcosOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AcosOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AcosOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AcosOp::result() {
  return *getODSResults(0).begin();
}

void AcosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void AcosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AcosOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AcosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void AcosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult AcosOp::verify() {
  if (failed(AcosOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}

::mlir::ParseResult AcosOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(operandRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(operandTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AcosOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.acos";
  p << " ";
  p << operand();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(operand().getType());
}

void AcosOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::AtanOp definitions
//===----------------------------------------------------------------------===//

AtanOpAdaptor::AtanOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AtanOpAdaptor::AtanOpAdaptor(AtanOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AtanOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtanOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtanOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult AtanOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef AtanOp::getOperationName() {
  return "chlo.atan";
}

std::pair<unsigned, unsigned> AtanOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtanOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtanOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AtanOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AtanOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtanOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtanOp::result() {
  return *getODSResults(0).begin();
}

void AtanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void AtanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtanOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AtanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void AtanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult AtanOp::verify() {
  if (failed(AtanOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}

::mlir::ParseResult AtanOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(operandRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(operandTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtanOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.atan";
  p << " ";
  p << operand();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(operand().getType());
}

void AtanOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastAddOp definitions
//===----------------------------------------------------------------------===//

BroadcastAddOpAdaptor::BroadcastAddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastAddOpAdaptor::BroadcastAddOpAdaptor(BroadcastAddOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastAddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastAddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastAddOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAddOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastAddOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastAddOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_add' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastAddOp::getOperationName() {
  return "chlo.broadcast_add";
}

std::pair<unsigned, unsigned> BroadcastAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastAddOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAddOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastAddOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastAddOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastAddOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAddOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastAddOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastAddOp::verify() {
  if (failed(BroadcastAddOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastAddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastAddOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_add";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastAddOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastAndOp definitions
//===----------------------------------------------------------------------===//

BroadcastAndOpAdaptor::BroadcastAndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastAndOpAdaptor::BroadcastAndOpAdaptor(BroadcastAndOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastAndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastAndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastAndOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAndOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastAndOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastAndOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_and' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastAndOp::getOperationName() {
  return "chlo.broadcast_and";
}

std::pair<unsigned, unsigned> BroadcastAndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastAndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastAndOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAndOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastAndOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastAndOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastAndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastAndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastAndOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAndOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastAndOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastAndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastAndOp::verify() {
  if (failed(BroadcastAndOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64)))))))) {
        return emitOpError("operand #") << index << " must be tensor of pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64)))))))) {
        return emitOpError("operand #") << index << " must be tensor of pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastAndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastAndOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_and";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastAndOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastAtan2Op definitions
//===----------------------------------------------------------------------===//

BroadcastAtan2OpAdaptor::BroadcastAtan2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastAtan2OpAdaptor::BroadcastAtan2OpAdaptor(BroadcastAtan2Op&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastAtan2OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastAtan2OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastAtan2OpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAtan2OpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastAtan2OpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastAtan2OpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_atan2' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastAtan2Op::getOperationName() {
  return "chlo.broadcast_atan2";
}

std::pair<unsigned, unsigned> BroadcastAtan2Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastAtan2Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastAtan2Op::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastAtan2Op::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastAtan2Op::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastAtan2Op::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastAtan2Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastAtan2Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastAtan2Op::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastAtan2Op::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastAtan2Op::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastAtan2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastAtan2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastAtan2Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastAtan2Op::verify() {
  if (failed(BroadcastAtan2OpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastAtan2Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastAtan2Op::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_atan2";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastAtan2Op::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastCompareOp definitions
//===----------------------------------------------------------------------===//

BroadcastCompareOpAdaptor::BroadcastCompareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastCompareOpAdaptor::BroadcastCompareOpAdaptor(BroadcastCompareOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastCompareOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastCompareOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastCompareOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastCompareOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastCompareOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::StringAttr BroadcastCompareOpAdaptor::comparison_direction() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("comparison_direction").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastCompareOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_compare' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  {
  auto tblgen_comparison_direction = odsAttrs.get("comparison_direction");
  if (!tblgen_comparison_direction) return emitError(loc, "'chlo.broadcast_compare' op ""requires attribute 'comparison_direction'");
    if (!(((tblgen_comparison_direction.isa<::mlir::StringAttr>())) && (((tblgen_comparison_direction.cast<::mlir::StringAttr>().getValue() == "EQ")) || ((tblgen_comparison_direction.cast<::mlir::StringAttr>().getValue() == "NE")) || ((tblgen_comparison_direction.cast<::mlir::StringAttr>().getValue() == "GE")) || ((tblgen_comparison_direction.cast<::mlir::StringAttr>().getValue() == "GT")) || ((tblgen_comparison_direction.cast<::mlir::StringAttr>().getValue() == "LE")) || ((tblgen_comparison_direction.cast<::mlir::StringAttr>().getValue() == "LT"))))) return emitError(loc, "'chlo.broadcast_compare' op ""attribute 'comparison_direction' failed to satisfy constraint: Which comparison operation to perform.");
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastCompareOp::getOperationName() {
  return "chlo.broadcast_compare";
}

std::pair<unsigned, unsigned> BroadcastCompareOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastCompareOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastCompareOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastCompareOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastCompareOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastCompareOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastCompareOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastCompareOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastCompareOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastCompareOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::StringAttr BroadcastCompareOp::comparison_directionAttr() {
  return this->getAttr("comparison_direction").cast<::mlir::StringAttr>();
}

::llvm::StringRef BroadcastCompareOp::comparison_direction() {
  auto attr = comparison_directionAttr();
  return attr.getValue();
}

void BroadcastCompareOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}

void BroadcastCompareOp::comparison_directionAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("comparison_direction", attr);
}



void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions, ::mlir::StringAttr comparison_direction) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addAttribute("comparison_direction", comparison_direction);
  odsState.addTypes(resultType0);
}

void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions, ::mlir::StringAttr comparison_direction) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addAttribute("comparison_direction", comparison_direction);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions, ::llvm::StringRef comparison_direction) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addAttribute("comparison_direction", odsBuilder.getStringAttr(comparison_direction));
  odsState.addTypes(resultType0);
}

void BroadcastCompareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions, ::llvm::StringRef comparison_direction) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addAttribute("comparison_direction", odsBuilder.getStringAttr(comparison_direction));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastCompareOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastCompareOp::verify() {
  if (failed(BroadcastCompareOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
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
        return emitOpError("result #") << index << " must be tensor of pred (AKA boolean or 1-bit integer) values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastCompareOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastCompareOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_compare";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastCompareOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastComplexOp definitions
//===----------------------------------------------------------------------===//

BroadcastComplexOpAdaptor::BroadcastComplexOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastComplexOpAdaptor::BroadcastComplexOpAdaptor(BroadcastComplexOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastComplexOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastComplexOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastComplexOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastComplexOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastComplexOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastComplexOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_complex' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastComplexOp::getOperationName() {
  return "chlo.broadcast_complex";
}

std::pair<unsigned, unsigned> BroadcastComplexOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastComplexOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastComplexOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastComplexOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastComplexOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastComplexOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastComplexOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastComplexOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastComplexOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastComplexOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastComplexOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastComplexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastComplexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastComplexOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastComplexOp::verify() {
  if (failed(BroadcastComplexOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64())))))) {
        return emitOpError("result #") << index << " must be tensor of complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastComplexOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastComplexOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_complex";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastComplexOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastDivOp definitions
//===----------------------------------------------------------------------===//

BroadcastDivOpAdaptor::BroadcastDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastDivOpAdaptor::BroadcastDivOpAdaptor(BroadcastDivOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastDivOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastDivOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastDivOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastDivOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_divide' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastDivOp::getOperationName() {
  return "chlo.broadcast_divide";
}

std::pair<unsigned, unsigned> BroadcastDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastDivOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastDivOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastDivOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastDivOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastDivOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastDivOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastDivOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastDivOp::verify() {
  if (failed(BroadcastDivOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastDivOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_divide";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastDivOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastMaxOp definitions
//===----------------------------------------------------------------------===//

BroadcastMaxOpAdaptor::BroadcastMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastMaxOpAdaptor::BroadcastMaxOpAdaptor(BroadcastMaxOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastMaxOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMaxOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastMaxOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastMaxOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_maximum' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastMaxOp::getOperationName() {
  return "chlo.broadcast_maximum";
}

std::pair<unsigned, unsigned> BroadcastMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastMaxOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMaxOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastMaxOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastMaxOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastMaxOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMaxOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastMaxOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastMaxOp::verify() {
  if (failed(BroadcastMaxOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastMaxOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastMaxOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_maximum";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastMaxOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastMinOp definitions
//===----------------------------------------------------------------------===//

BroadcastMinOpAdaptor::BroadcastMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastMinOpAdaptor::BroadcastMinOpAdaptor(BroadcastMinOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastMinOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMinOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastMinOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastMinOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_minimum' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastMinOp::getOperationName() {
  return "chlo.broadcast_minimum";
}

std::pair<unsigned, unsigned> BroadcastMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastMinOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMinOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastMinOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastMinOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastMinOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMinOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastMinOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastMinOp::verify() {
  if (failed(BroadcastMinOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastMinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastMinOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_minimum";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastMinOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastMulOp definitions
//===----------------------------------------------------------------------===//

BroadcastMulOpAdaptor::BroadcastMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastMulOpAdaptor::BroadcastMulOpAdaptor(BroadcastMulOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastMulOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMulOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastMulOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastMulOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_multiply' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastMulOp::getOperationName() {
  return "chlo.broadcast_multiply";
}

std::pair<unsigned, unsigned> BroadcastMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastMulOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastMulOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastMulOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastMulOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastMulOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastMulOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastMulOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastMulOp::verify() {
  if (failed(BroadcastMulOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastMulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastMulOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_multiply";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastMulOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastOrOp definitions
//===----------------------------------------------------------------------===//

BroadcastOrOpAdaptor::BroadcastOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastOrOpAdaptor::BroadcastOrOpAdaptor(BroadcastOrOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastOrOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastOrOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastOrOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastOrOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_or' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastOrOp::getOperationName() {
  return "chlo.broadcast_or";
}

std::pair<unsigned, unsigned> BroadcastOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastOrOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastOrOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastOrOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastOrOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastOrOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastOrOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastOrOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastOrOp::verify() {
  if (failed(BroadcastOrOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64)))))))) {
        return emitOpError("operand #") << index << " must be tensor of pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64)))))))) {
        return emitOpError("operand #") << index << " must be tensor of pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastOrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastOrOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_or";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastOrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastPowOp definitions
//===----------------------------------------------------------------------===//

BroadcastPowOpAdaptor::BroadcastPowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastPowOpAdaptor::BroadcastPowOpAdaptor(BroadcastPowOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastPowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastPowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastPowOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastPowOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastPowOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastPowOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_power' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastPowOp::getOperationName() {
  return "chlo.broadcast_power";
}

std::pair<unsigned, unsigned> BroadcastPowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastPowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastPowOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastPowOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastPowOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastPowOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastPowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastPowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastPowOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastPowOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastPowOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastPowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastPowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastPowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastPowOp::verify() {
  if (failed(BroadcastPowOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastPowOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastPowOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_power";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastPowOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastRemOp definitions
//===----------------------------------------------------------------------===//

BroadcastRemOpAdaptor::BroadcastRemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastRemOpAdaptor::BroadcastRemOpAdaptor(BroadcastRemOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastRemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastRemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastRemOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastRemOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastRemOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastRemOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_remainder' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastRemOp::getOperationName() {
  return "chlo.broadcast_remainder";
}

std::pair<unsigned, unsigned> BroadcastRemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastRemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastRemOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastRemOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastRemOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastRemOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastRemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastRemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastRemOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastRemOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastRemOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastRemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastRemOp::verify() {
  if (failed(BroadcastRemOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastRemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastRemOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_remainder";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastRemOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastShiftLeftOp definitions
//===----------------------------------------------------------------------===//

BroadcastShiftLeftOpAdaptor::BroadcastShiftLeftOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastShiftLeftOpAdaptor::BroadcastShiftLeftOpAdaptor(BroadcastShiftLeftOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastShiftLeftOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastShiftLeftOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastShiftLeftOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftLeftOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastShiftLeftOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastShiftLeftOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_shift_left' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastShiftLeftOp::getOperationName() {
  return "chlo.broadcast_shift_left";
}

std::pair<unsigned, unsigned> BroadcastShiftLeftOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastShiftLeftOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastShiftLeftOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftLeftOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastShiftLeftOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastShiftLeftOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastShiftLeftOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastShiftLeftOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastShiftLeftOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftLeftOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastShiftLeftOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastShiftLeftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastShiftLeftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastShiftLeftOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastShiftLeftOp::verify() {
  if (failed(BroadcastShiftLeftOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastShiftLeftOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastShiftLeftOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_shift_left";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastShiftLeftOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastShiftRightArithmeticOp definitions
//===----------------------------------------------------------------------===//

BroadcastShiftRightArithmeticOpAdaptor::BroadcastShiftRightArithmeticOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastShiftRightArithmeticOpAdaptor::BroadcastShiftRightArithmeticOpAdaptor(BroadcastShiftRightArithmeticOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastShiftRightArithmeticOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastShiftRightArithmeticOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastShiftRightArithmeticOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftRightArithmeticOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastShiftRightArithmeticOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastShiftRightArithmeticOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_shift_right_arithmetic' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastShiftRightArithmeticOp::getOperationName() {
  return "chlo.broadcast_shift_right_arithmetic";
}

std::pair<unsigned, unsigned> BroadcastShiftRightArithmeticOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastShiftRightArithmeticOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastShiftRightArithmeticOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftRightArithmeticOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastShiftRightArithmeticOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastShiftRightArithmeticOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastShiftRightArithmeticOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastShiftRightArithmeticOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastShiftRightArithmeticOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftRightArithmeticOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastShiftRightArithmeticOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastShiftRightArithmeticOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastShiftRightArithmeticOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastShiftRightArithmeticOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastShiftRightArithmeticOp::verify() {
  if (failed(BroadcastShiftRightArithmeticOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastShiftRightArithmeticOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastShiftRightArithmeticOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_shift_right_arithmetic";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastShiftRightArithmeticOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastShiftRightLogicalOp definitions
//===----------------------------------------------------------------------===//

BroadcastShiftRightLogicalOpAdaptor::BroadcastShiftRightLogicalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastShiftRightLogicalOpAdaptor::BroadcastShiftRightLogicalOpAdaptor(BroadcastShiftRightLogicalOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastShiftRightLogicalOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastShiftRightLogicalOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastShiftRightLogicalOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftRightLogicalOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastShiftRightLogicalOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastShiftRightLogicalOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_shift_right_logical' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastShiftRightLogicalOp::getOperationName() {
  return "chlo.broadcast_shift_right_logical";
}

std::pair<unsigned, unsigned> BroadcastShiftRightLogicalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastShiftRightLogicalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastShiftRightLogicalOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastShiftRightLogicalOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastShiftRightLogicalOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastShiftRightLogicalOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastShiftRightLogicalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastShiftRightLogicalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastShiftRightLogicalOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastShiftRightLogicalOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastShiftRightLogicalOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastShiftRightLogicalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastShiftRightLogicalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastShiftRightLogicalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastShiftRightLogicalOp::verify() {
  if (failed(BroadcastShiftRightLogicalOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastShiftRightLogicalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastShiftRightLogicalOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_shift_right_logical";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastShiftRightLogicalOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastSubOp definitions
//===----------------------------------------------------------------------===//

BroadcastSubOpAdaptor::BroadcastSubOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastSubOpAdaptor::BroadcastSubOpAdaptor(BroadcastSubOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastSubOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastSubOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastSubOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastSubOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastSubOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastSubOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_subtract' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastSubOp::getOperationName() {
  return "chlo.broadcast_subtract";
}

std::pair<unsigned, unsigned> BroadcastSubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastSubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastSubOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastSubOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastSubOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastSubOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastSubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastSubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastSubOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastSubOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastSubOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastSubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastSubOp::verify() {
  if (failed(BroadcastSubOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastSubOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastSubOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_subtract";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastSubOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::BroadcastXorOp definitions
//===----------------------------------------------------------------------===//

BroadcastXorOpAdaptor::BroadcastXorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BroadcastXorOpAdaptor::BroadcastXorOpAdaptor(BroadcastXorOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BroadcastXorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastXorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastXorOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastXorOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DenseIntElementsAttr BroadcastXorOpAdaptor::broadcast_dimensions() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::LogicalResult BroadcastXorOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_broadcast_dimensions = odsAttrs.get("broadcast_dimensions");
  if (tblgen_broadcast_dimensions) {
    if (!(((tblgen_broadcast_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_broadcast_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64))))) return emitError(loc, "'chlo.broadcast_xor' op ""attribute 'broadcast_dimensions' failed to satisfy constraint: 64-bit signless integer elements attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef BroadcastXorOp::getOperationName() {
  return "chlo.broadcast_xor";
}

std::pair<unsigned, unsigned> BroadcastXorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastXorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastXorOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastXorOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange BroadcastXorOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange BroadcastXorOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BroadcastXorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastXorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseIntElementsAttr BroadcastXorOp::broadcast_dimensionsAttr() {
  return this->getAttr("broadcast_dimensions").dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > BroadcastXorOp::broadcast_dimensions() {
  auto attr = broadcast_dimensionsAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

void BroadcastXorOp::broadcast_dimensionsAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("broadcast_dimensions", attr);
}



void BroadcastXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  odsState.addTypes(resultType0);
}

void BroadcastXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, /*optional*/::mlir::DenseIntElementsAttr broadcast_dimensions) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (broadcast_dimensions) {
  odsState.addAttribute("broadcast_dimensions", broadcast_dimensions);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastXorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastXorOp::verify() {
  if (failed(BroadcastXorOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64)))))))) {
        return emitOpError("operand #") << index << " must be tensor of pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64)))))))) {
        return emitOpError("operand #") << index << " must be tensor of pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





::mlir::ParseResult BroadcastXorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

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
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(rhsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastXorOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.broadcast_xor";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(rhs().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << getOperation()->getResultTypes();
}

void BroadcastXorOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::ConstantLikeOp definitions
//===----------------------------------------------------------------------===//

ConstantLikeOpAdaptor::ConstantLikeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConstantLikeOpAdaptor::ConstantLikeOpAdaptor(ConstantLikeOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConstantLikeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstantLikeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstantLikeOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Attribute ConstantLikeOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("value").cast<::mlir::Attribute>();
  return attr;
}

::mlir::LogicalResult ConstantLikeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (!tblgen_value) return emitError(loc, "'chlo.constant_like' op ""requires attribute 'value'");
    if (!((true))) return emitError(loc, "'chlo.constant_like' op ""attribute 'value' failed to satisfy constraint: any attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef ConstantLikeOp::getOperationName() {
  return "chlo.constant_like";
}

std::pair<unsigned, unsigned> ConstantLikeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantLikeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstantLikeOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ConstantLikeOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ConstantLikeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantLikeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Attribute ConstantLikeOp::valueAttr() {
  return this->getAttr("value").cast<::mlir::Attribute>();
}

::mlir::Attribute ConstantLikeOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstantLikeOp::valueAttr(::mlir::Attribute attr) {
  this->getOperation()->setAttr("value", attr);
}

void ConstantLikeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Attribute value, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute("value", value);
  odsState.addTypes(resultType0);
}

void ConstantLikeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Attribute value, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute("value", value);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (succeeded(ConstantLikeOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConstantLikeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantLikeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConstantLikeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

    ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
    if (succeeded(ConstantLikeOp::inferReturnTypes(odsBuilder.getContext(),
                  odsState.location, operands,
                  odsState.attributes.getDictionary(odsState.getContext()),
                  /*regions=*/{}, inferredReturnTypes))) {  assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
      odsState.addTypes(inferredReturnTypes);
    } else
      ::llvm::report_fatal_error("Failed to infer result type(s).");
}

::mlir::LogicalResult ConstantLikeOp::verify() {
  if (failed(ConstantLikeOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1))) || ((((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(64)))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(8))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(16))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(32))) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isUnsignedInteger(64))))) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or pred (AKA boolean or 1-bit integer) or 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}





void ConstantLikeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::SinhOp definitions
//===----------------------------------------------------------------------===//

SinhOpAdaptor::SinhOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SinhOpAdaptor::SinhOpAdaptor(SinhOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SinhOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SinhOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinhOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult SinhOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SinhOp::getOperationName() {
  return "chlo.sinh";
}

std::pair<unsigned, unsigned> SinhOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SinhOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinhOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SinhOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SinhOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SinhOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinhOp::result() {
  return *getODSResults(0).begin();
}

void SinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void SinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SinhOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void SinhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult SinhOp::verify() {
  if (failed(SinhOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}

::mlir::ParseResult SinhOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(operandRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(operandTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SinhOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.sinh";
  p << " ";
  p << operand();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(operand().getType());
}

void SinhOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir
namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// ::mlir::chlo::TanOp definitions
//===----------------------------------------------------------------------===//

TanOpAdaptor::TanOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TanOpAdaptor::TanOpAdaptor(TanOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TanOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TanOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TanOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult TanOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef TanOp::getOperationName() {
  return "chlo.tan";
}

std::pair<unsigned, unsigned> TanOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TanOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TanOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TanOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TanOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TanOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TanOp::result() {
  return *getODSResults(0).begin();
}

void TanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(result);
}

void TanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TanOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void TanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::LogicalResult TanOp::verify() {
  if (failed(TanOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("operand #") << index << " must be tensor of floating-point or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>())) || (((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::ComplexType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().cast<::mlir::ComplexType>().getElementType().isF64()))))))) {
        return emitOpError("result #") << index << " must be tensor of floating-point or complex type with 32-bit float or 64-bit float elements values, but got " << v.getType();
      }
      ++index;
    }
  }
  return Verify(*this);
}

::mlir::ParseResult TanOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type operandRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> operandTypes(operandRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(operandRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(operandTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TanOp::print(::mlir::OpAsmPrinter &p) {
  p << "chlo.tan";
  p << " ";
  p << operand();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(operand().getType());
}

void TanOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace chlo
} // namespace mlir

#endif  // GET_OP_CLASSES

