/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::avx512::MaskRndScaleOp,
::mlir::avx512::MaskScaleFOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::MaskRndScaleOp definitions
//===----------------------------------------------------------------------===//

MaskRndScaleOpAdaptor::MaskRndScaleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

MaskRndScaleOpAdaptor::MaskRndScaleOpAdaptor(MaskRndScaleOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> MaskRndScaleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskRndScaleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScaleOpAdaptor::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskRndScaleOpAdaptor::k() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskRndScaleOpAdaptor::a() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskRndScaleOpAdaptor::imm() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskRndScaleOpAdaptor::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::LogicalResult MaskRndScaleOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef MaskRndScaleOp::getOperationName() {
  return "avx512.mask.rndscale";
}

std::pair<unsigned, unsigned> MaskRndScaleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskRndScaleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScaleOp::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskRndScaleOp::k() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskRndScaleOp::a() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskRndScaleOp::imm() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskRndScaleOp::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange MaskRndScaleOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScaleOp::kMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScaleOp::aMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScaleOp::immMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskRndScaleOp::roundingMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskRndScaleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskRndScaleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskRndScaleOp::dst() {
  return *getODSResults(0).begin();
}

void MaskRndScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src, ::mlir::Value k, ::mlir::Value a, ::mlir::Value imm, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(k);
  odsState.addOperands(a);
  odsState.addOperands(imm);
  odsState.addOperands(rounding);
  odsState.addTypes(dst);
}

void MaskRndScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value k, ::mlir::Value a, ::mlir::Value imm, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(k);
  odsState.addOperands(a);
  odsState.addOperands(imm);
  odsState.addOperands(rounding);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskRndScaleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskRndScaleOp::verify() {
  if (failed(MaskRndScaleOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) && (((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
        return emitOpError("operand #") << index << " must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isSignlessInteger(32)))) {
        return emitOpError("operand #") << index << " must be 32-bit signless integer, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) && (((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
        return emitOpError("operand #") << index << " must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!(((v.getType().isSignlessInteger(16))) || ((v.getType().isSignlessInteger(8))))) {
        return emitOpError("operand #") << index << " must be 16-bit signless integer or 8-bit signless integer, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      (void)v;
      if (!((v.getType().isSignlessInteger(32)))) {
        return emitOpError("operand #") << index << " must be 32-bit signless integer, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) && (((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
        return emitOpError("result #") << index << " must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src, a, dst} have same type");
  if (!((IntegerType::get(((*this->getODSResults(0).begin()).getType().cast<VectorType>().getShape()[0]),  (*this->getODSResults(0).begin()).getType().getContext()) == (*this->getODSOperands(3).begin()).getType())))
    return emitOpError("failed to verify that imm has the same number of bits as elements in dst");
  return ::mlir::success();
}

::mlir::ParseResult MaskRndScaleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::OpAsmParser::OperandType kRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> kOperands(kRawOperands);  ::llvm::SMLoc kOperandsLoc;
  (void)kOperandsLoc;
  ::mlir::OpAsmParser::OperandType aRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> aOperands(aRawOperands);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::OperandType immRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> immOperands(immRawOperands);  ::llvm::SMLoc immOperandsLoc;
  (void)immOperandsLoc;
  ::mlir::OpAsmParser::OperandType roundingRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> roundingOperands(roundingRawOperands);  ::llvm::SMLoc roundingOperandsLoc;
  (void)roundingOperandsLoc;
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  kOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(kRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  immOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(immRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  roundingOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(roundingRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : dstTypes) {
    (void)type;
    if (!((((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF64())))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
      return parser.emitError(parser.getNameLoc()) << "'dst' must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(dstTypes);
  if (parser.resolveOperands(srcOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(kOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(aOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(immOperands, IntegerType::get((dstTypes[0].cast<VectorType>().getShape()[0]),  dstTypes[0].getContext()), immOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(roundingOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskRndScaleOp::print(::mlir::OpAsmPrinter &p) {
  p << "avx512.mask.rndscale";
  p << " ";
  p << src();
  p << ",";
  p << " ";
  p << k();
  p << ",";
  p << " ";
  p << a();
  p << ",";
  p << " ";
  p << imm();
  p << ",";
  p << " ";
  p << rounding();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
}

void MaskRndScaleOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir
namespace mlir {
namespace avx512 {

//===----------------------------------------------------------------------===//
// ::mlir::avx512::MaskScaleFOp definitions
//===----------------------------------------------------------------------===//

MaskScaleFOpAdaptor::MaskScaleFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

MaskScaleFOpAdaptor::MaskScaleFOpAdaptor(MaskScaleFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> MaskScaleFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskScaleFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFOpAdaptor::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskScaleFOpAdaptor::a() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskScaleFOpAdaptor::b() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskScaleFOpAdaptor::k() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskScaleFOpAdaptor::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::LogicalResult MaskScaleFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef MaskScaleFOp::getOperationName() {
  return "avx512.mask.scalef";
}

std::pair<unsigned, unsigned> MaskScaleFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskScaleFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFOp::src() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskScaleFOp::a() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskScaleFOp::b() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskScaleFOp::k() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskScaleFOp::rounding() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange MaskScaleFOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFOp::aMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFOp::bMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFOp::kMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MaskScaleFOp::roundingMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MaskScaleFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskScaleFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskScaleFOp::dst() {
  return *getODSResults(0).begin();
}

void MaskScaleFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dst, ::mlir::Value src, ::mlir::Value a, ::mlir::Value b, ::mlir::Value k, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(k);
  odsState.addOperands(rounding);
  odsState.addTypes(dst);
}

void MaskScaleFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value a, ::mlir::Value b, ::mlir::Value k, ::mlir::Value rounding) {
  odsState.addOperands(src);
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(k);
  odsState.addOperands(rounding);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskScaleFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskScaleFOp::verify() {
  if (failed(MaskScaleFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) && (((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
        return emitOpError("operand #") << index << " must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) && (((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
        return emitOpError("operand #") << index << " must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) && (((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
        return emitOpError("operand #") << index << " must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!(((v.getType().isSignlessInteger(16))) || ((v.getType().isSignlessInteger(8))))) {
        return emitOpError("operand #") << index << " must be 16-bit signless integer or 8-bit signless integer, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
      (void)v;
      if (!((v.getType().isSignlessInteger(32)))) {
        return emitOpError("operand #") << index << " must be 32-bit signless integer, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isF64())))) && (((v.getType().isa<::mlir::VectorType>())) && (((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((v.getType().cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
        return emitOpError("result #") << index << " must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {src, a, b, dst} have same type");
  if (!((IntegerType::get(((*this->getODSResults(0).begin()).getType().cast<VectorType>().getShape()[0]),  (*this->getODSResults(0).begin()).getType().getContext()) == (*this->getODSOperands(3).begin()).getType())))
    return emitOpError("failed to verify that k has the same number of bits as elements in dst");
  return ::mlir::success();
}

::mlir::ParseResult MaskScaleFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::OpAsmParser::OperandType aRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> aOperands(aRawOperands);  ::llvm::SMLoc aOperandsLoc;
  (void)aOperandsLoc;
  ::mlir::OpAsmParser::OperandType bRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> bOperands(bRawOperands);  ::llvm::SMLoc bOperandsLoc;
  (void)bOperandsLoc;
  ::mlir::OpAsmParser::OperandType kRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> kOperands(kRawOperands);  ::llvm::SMLoc kOperandsLoc;
  (void)kOperandsLoc;
  ::mlir::OpAsmParser::OperandType roundingRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> roundingOperands(roundingRawOperands);  ::llvm::SMLoc roundingOperandsLoc;
  (void)roundingOperandsLoc;
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  aOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  bOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(bRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  kOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(kRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  roundingOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(roundingRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(dstRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : dstTypes) {
    (void)type;
    if (!((((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::ShapedType>().getElementType().isF32())) || ((type.cast<::mlir::ShapedType>().getElementType().isF64())))) && (((type.isa<::mlir::VectorType>())) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)))))) {
      return parser.emitError(parser.getNameLoc()) << "'dst' must be vector of 32-bit float or 64-bit float values of length 16/8, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(dstTypes);
  if (parser.resolveOperands(srcOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(aOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(bOperands, dstTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(kOperands, IntegerType::get((dstTypes[0].cast<VectorType>().getShape()[0]),  dstTypes[0].getContext()), kOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(roundingOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskScaleFOp::print(::mlir::OpAsmPrinter &p) {
  p << "avx512.mask.scalef";
  p << " ";
  p << src();
  p << ",";
  p << " ";
  p << a();
  p << ",";
  p << " ";
  p << b();
  p << ",";
  p << " ";
  p << k();
  p << ",";
  p << " ";
  p << rounding();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(dst().getType());
}

void MaskScaleFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace avx512
} // namespace mlir

#endif  // GET_OP_CLASSES

