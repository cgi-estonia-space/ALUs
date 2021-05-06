/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

AbsFOp,
AddCFOp,
AddFOp,
AddIOp,
AllocOp,
AllocaOp,
AndOp,
AssertOp,
AssumeAlignmentOp,
Atan2Op,
AtanOp,
AtomicRMWOp,
AtomicYieldOp,
BranchOp,
CallIndirectOp,
CallOp,
CeilFOp,
CmpFOp,
CmpIOp,
CondBranchOp,
ConstantOp,
CopySignOp,
CosOp,
CreateComplexOp,
DeallocOp,
DimOp,
DivFOp,
DynamicTensorFromElementsOp,
Exp2Op,
ExpOp,
ExtractElementOp,
FPExtOp,
FPToSIOp,
FPToUIOp,
FPTruncOp,
FloorFOp,
GenericAtomicRMWOp,
ImOp,
IndexCastOp,
LoadOp,
Log10Op,
Log2Op,
LogOp,
MemRefCastOp,
MulFOp,
MulIOp,
NegFOp,
OrOp,
PrefetchOp,
RankOp,
ReOp,
RemFOp,
ReturnOp,
RsqrtOp,
SIToFPOp,
SelectOp,
ShiftLeftOp,
SignExtendIOp,
SignedDivIOp,
SignedRemIOp,
SignedShiftRightOp,
SinOp,
SplatOp,
SqrtOp,
StoreOp,
SubCFOp,
SubFOp,
SubIOp,
SubTensorInsertOp,
SubTensorOp,
SubViewOp,
TanhOp,
TensorCastOp,
TensorFromElementsOp,
TensorLoadOp,
TensorStoreOp,
TensorToMemrefOp,
TransposeOp,
TruncateIOp,
UIToFPOp,
UnsignedDivIOp,
UnsignedRemIOp,
UnsignedShiftRightOp,
ViewOp,
XOrOp,
YieldOp,
ZeroExtendIOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// AbsFOp definitions
//===----------------------------------------------------------------------===//

AbsFOpAdaptor::AbsFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AbsFOpAdaptor::AbsFOpAdaptor(AbsFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AbsFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AbsFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AbsFOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult AbsFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef AbsFOp::getOperationName() {
  return "std.absf";
}

std::pair<unsigned, unsigned> AbsFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AbsFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AbsFOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AbsFOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AbsFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AbsFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AbsFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void AbsFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AbsFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AbsFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void AbsFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult AbsFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void AbsFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult AbsFOp::verify() {
  if (failed(AbsFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void AbsFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AddCFOp definitions
//===----------------------------------------------------------------------===//

AddCFOpAdaptor::AddCFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AddCFOpAdaptor::AddCFOpAdaptor(AddCFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AddCFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddCFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddCFOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddCFOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult AddCFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef AddCFOp::getOperationName() {
  return "std.addcf";
}

std::pair<unsigned, unsigned> AddCFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddCFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddCFOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddCFOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AddCFOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AddCFOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AddCFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddCFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AddCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AddCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddCFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AddCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AddCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult AddCFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void AddCFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult AddCFOp::verify() {
  if (failed(AddCFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isa<::mlir::FloatType>())))) {
        return emitOpError("operand #") << index << " must be complex type with floating-point elements, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isa<::mlir::FloatType>())))) {
        return emitOpError("operand #") << index << " must be complex type with floating-point elements, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void AddCFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AddFOp definitions
//===----------------------------------------------------------------------===//

AddFOpAdaptor::AddFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AddFOpAdaptor::AddFOpAdaptor(AddFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AddFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddFOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddFOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult AddFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef AddFOp::getOperationName() {
  return "std.addf";
}

std::pair<unsigned, unsigned> AddFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddFOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddFOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AddFOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AddFOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AddFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AddFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AddFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AddFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AddFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult AddFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void AddFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult AddFOp::verify() {
  if (failed(AddFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void AddFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AddIOp definitions
//===----------------------------------------------------------------------===//

AddIOpAdaptor::AddIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AddIOpAdaptor::AddIOpAdaptor(AddIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AddIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddIOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddIOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult AddIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef AddIOp::getOperationName() {
  return "std.addi";
}

std::pair<unsigned, unsigned> AddIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddIOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddIOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AddIOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AddIOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AddIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AddIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AddIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AddIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AddIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult AddIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void AddIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult AddIOp::verify() {
  if (failed(AddIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void AddIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AllocOp definitions
//===----------------------------------------------------------------------===//

AllocOpAdaptor::AllocOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AllocOpAdaptor::AllocOpAdaptor(AllocOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AllocOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange AllocOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange AllocOpAdaptor::value() {
  return getODSOperands(0);
}

::mlir::IntegerAttr AllocOpAdaptor::alignment() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("alignment").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult AllocOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_alignment = odsAttrs.get("alignment");
  if (tblgen_alignment) {
    if (!((((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getInt() >= 0)))) return emitError(loc, "'std.alloc' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef AllocOp::getOperationName() {
  return "std.alloc";
}

std::pair<unsigned, unsigned> AllocOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range AllocOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AllocOp::value() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AllocOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AllocOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AllocOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr AllocOp::alignmentAttr() {
  return this->getAttr("alignment").dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> AllocOp::alignment() {
  auto attr = alignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void AllocOp::alignmentAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("alignment", attr);
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType) {
       odsState.types.push_back(memrefType);
     
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange operands, IntegerAttr alignment ) {
       odsState.addOperands(operands);
       odsState.types.push_back(memrefType);
       if (alignment)
         odsState.addAttribute(getAlignmentAttrName(), alignment);
     
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::ValueRange value, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  odsState.addTypes(resultType0);
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange value, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult AllocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseAllocLikeOp(parser, result);
}

void AllocOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult AllocOp::verify() {
  if (failed(AllocOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
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
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}



void AllocOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(MemoryEffects::Allocate::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// AllocaOp definitions
//===----------------------------------------------------------------------===//

AllocaOpAdaptor::AllocaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AllocaOpAdaptor::AllocaOpAdaptor(AllocaOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AllocaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange AllocaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange AllocaOpAdaptor::value() {
  return getODSOperands(0);
}

::mlir::IntegerAttr AllocaOpAdaptor::alignment() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("alignment").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult AllocaOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_alignment = odsAttrs.get("alignment");
  if (tblgen_alignment) {
    if (!((((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getInt() >= 0)))) return emitError(loc, "'std.alloca' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef AllocaOp::getOperationName() {
  return "std.alloca";
}

std::pair<unsigned, unsigned> AllocaOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range AllocaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AllocaOp::value() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AllocaOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AllocaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AllocaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr AllocaOp::alignmentAttr() {
  return this->getAttr("alignment").dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> AllocaOp::alignment() {
  auto attr = alignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void AllocaOp::alignmentAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("alignment", attr);
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType) {
       odsState.types.push_back(memrefType);
     
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange operands, IntegerAttr alignment ) {
       odsState.addOperands(operands);
       odsState.types.push_back(memrefType);
       if (alignment)
         odsState.addAttribute(getAlignmentAttrName(), alignment);
     
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::ValueRange value, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  odsState.addTypes(resultType0);
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange value, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult AllocaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseAllocLikeOp(parser, result);
}

void AllocaOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult AllocaOp::verify() {
  if (failed(AllocaOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
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
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}



void AllocaOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(), ::mlir::SideEffects::AutomaticAllocationScopeResource::get());
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(MemoryEffects::Allocate::get(), value, ::mlir::SideEffects::AutomaticAllocationScopeResource::get());
}


//===----------------------------------------------------------------------===//
// AndOp definitions
//===----------------------------------------------------------------------===//

AndOpAdaptor::AndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AndOpAdaptor::AndOpAdaptor(AndOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AndOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AndOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult AndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef AndOp::getOperationName() {
  return "std.and";
}

std::pair<unsigned, unsigned> AndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AndOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AndOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AndOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AndOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult AndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void AndOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult AndOp::verify() {
  if (failed(AndOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void AndOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AssertOp definitions
//===----------------------------------------------------------------------===//

AssertOpAdaptor::AssertOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AssertOpAdaptor::AssertOpAdaptor(AssertOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AssertOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AssertOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AssertOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::StringAttr AssertOpAdaptor::msg() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("msg").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult AssertOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_msg = odsAttrs.get("msg");
  if (!tblgen_msg) return emitError(loc, "'std.assert' op ""requires attribute 'msg'");
    if (!((tblgen_msg.isa<::mlir::StringAttr>()))) return emitError(loc, "'std.assert' op ""attribute 'msg' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef AssertOp::getOperationName() {
  return "std.assert";
}

std::pair<unsigned, unsigned> AssertOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AssertOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AssertOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AssertOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AssertOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AssertOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr AssertOp::msgAttr() {
  return this->getAttr("msg").cast<::mlir::StringAttr>();
}

::llvm::StringRef AssertOp::msg() {
  auto attr = msgAttr();
  return attr.getValue();
}

void AssertOp::msgAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("msg", attr);
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::mlir::StringAttr msg) {
  odsState.addOperands(arg);
  odsState.addAttribute("msg", msg);
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::mlir::StringAttr msg) {
  odsState.addOperands(arg);
  odsState.addAttribute("msg", msg);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::llvm::StringRef msg) {
  odsState.addOperands(arg);
  odsState.addAttribute("msg", odsBuilder.getStringAttr(msg));
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::llvm::StringRef msg) {
  odsState.addOperands(arg);
  odsState.addAttribute("msg", odsBuilder.getStringAttr(msg));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssertOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AssertOp::verify() {
  if (failed(AssertOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isSignlessInteger(1)))) {
        return emitOpError("operand #") << index << " must be 1-bit signless integer, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}



::mlir::ParseResult AssertOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::StringAttr msgAttr;

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseAttribute(msgAttr, parser.getBuilder().getType<::mlir::NoneType>(), "msg", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(argOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AssertOp::print(::mlir::OpAsmPrinter &p) {
  p << "assert";
  p << " ";
  p << arg();
  p << ",";
  p << " ";
  p.printAttributeWithoutType(msgAttr());
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"msg"});
}


//===----------------------------------------------------------------------===//
// AssumeAlignmentOp definitions
//===----------------------------------------------------------------------===//

AssumeAlignmentOpAdaptor::AssumeAlignmentOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AssumeAlignmentOpAdaptor::AssumeAlignmentOpAdaptor(AssumeAlignmentOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AssumeAlignmentOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AssumeAlignmentOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AssumeAlignmentOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::IntegerAttr AssumeAlignmentOpAdaptor::alignment() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("alignment").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult AssumeAlignmentOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_alignment = odsAttrs.get("alignment");
  if (!tblgen_alignment) return emitError(loc, "'std.assume_alignment' op ""requires attribute 'alignment'");
    if (!((((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((tblgen_alignment.cast<IntegerAttr>().getValue().isStrictlyPositive())))) return emitError(loc, "'std.assume_alignment' op ""attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive");
  }
  return ::mlir::success();
}

::llvm::StringRef AssumeAlignmentOp::getOperationName() {
  return "std.assume_alignment";
}

std::pair<unsigned, unsigned> AssumeAlignmentOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AssumeAlignmentOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AssumeAlignmentOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AssumeAlignmentOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AssumeAlignmentOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AssumeAlignmentOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr AssumeAlignmentOp::alignmentAttr() {
  return this->getAttr("alignment").cast<::mlir::IntegerAttr>();
}

uint32_t AssumeAlignmentOp::alignment() {
  auto attr = alignmentAttr();
  return attr.getValue().getZExtValue();
}

void AssumeAlignmentOp::alignmentAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("alignment", attr);
}

void AssumeAlignmentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute("alignment", alignment);
}

void AssumeAlignmentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute("alignment", alignment);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssumeAlignmentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, uint32_t alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute("alignment", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
}

void AssumeAlignmentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, uint32_t alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute("alignment", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssumeAlignmentOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AssumeAlignmentOp::verify() {
  if (failed(AssumeAlignmentOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

::mlir::ParseResult AssumeAlignmentOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::IntegerAttr alignmentAttr;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseAttribute(alignmentAttr, parser.getBuilder().getIntegerType(32), "alignment", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AssumeAlignmentOp::print(::mlir::OpAsmPrinter &p) {
  p << "assume_alignment";
  p << " ";
  p << memref();
  p << ",";
  p << " ";
  p.printAttributeWithoutType(alignmentAttr());
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"alignment"});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(memref().getType());
}


//===----------------------------------------------------------------------===//
// Atan2Op definitions
//===----------------------------------------------------------------------===//

Atan2OpAdaptor::Atan2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

Atan2OpAdaptor::Atan2OpAdaptor(Atan2Op&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> Atan2OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Atan2OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Atan2OpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value Atan2OpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult Atan2OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef Atan2Op::getOperationName() {
  return "std.atan2";
}

std::pair<unsigned, unsigned> Atan2Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Atan2Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Atan2Op::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value Atan2Op::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange Atan2Op::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange Atan2Op::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Atan2Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Atan2Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void Atan2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void Atan2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Atan2Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Atan2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void Atan2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult Atan2Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void Atan2Op::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult Atan2Op::verify() {
  if (failed(Atan2OpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void Atan2Op::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AtanOp definitions
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
  return "std.atan";
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

void AtanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
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

::mlir::ParseResult AtanOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void AtanOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult AtanOp::verify() {
  if (failed(AtanOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void AtanOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AtomicRMWOp definitions
//===----------------------------------------------------------------------===//

AtomicRMWOpAdaptor::AtomicRMWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AtomicRMWOpAdaptor::AtomicRMWOpAdaptor(AtomicRMWOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AtomicRMWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange AtomicRMWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicRMWOpAdaptor::memref() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange AtomicRMWOpAdaptor::indices() {
  return getODSOperands(2);
}

::mlir::IntegerAttr AtomicRMWOpAdaptor::kind() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("kind").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult AtomicRMWOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_kind = odsAttrs.get("kind");
  if (!tblgen_kind) return emitError(loc, "'std.atomic_rmw' op ""requires attribute 'kind'");
    if (!((((tblgen_kind.isa<::mlir::IntegerAttr>())) && ((tblgen_kind.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((tblgen_kind.cast<::mlir::IntegerAttr>().getInt() == 10))))) return emitError(loc, "'std.atomic_rmw' op ""attribute 'kind' failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10");
  }
  return ::mlir::success();
}

::llvm::StringRef AtomicRMWOp::getOperationName() {
  return "std.atomic_rmw";
}

std::pair<unsigned, unsigned> AtomicRMWOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range AtomicRMWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicRMWOp::memref() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range AtomicRMWOp::indices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange AtomicRMWOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AtomicRMWOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange AtomicRMWOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AtomicRMWOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicRMWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOp::result() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr AtomicRMWOp::kindAttr() {
  return this->getAttr("kind").cast<::mlir::IntegerAttr>();
}

::mlir::AtomicRMWKind AtomicRMWOp::kind() {
  auto attr = kindAttr();
  return static_cast<::mlir::AtomicRMWKind>(attr.getInt());
}

void AtomicRMWOp::kindAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("kind", attr);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("kind", kind);
  odsState.addTypes(result);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("kind", kind);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::AtomicRMWKind kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("kind", odsBuilder.getI64IntegerAttr(static_cast<int64_t>(kind)));
  odsState.addTypes(result);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AtomicRMWKind kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("kind", odsBuilder.getI64IntegerAttr(static_cast<int64_t>(kind)));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicRMWOp::verify() {
  if (failed(AtomicRMWOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::FloatType>())))) {
        return emitOpError("operand #") << index << " must be signless integer or floating-point, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be memref of signless integer or floating-point values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
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
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::FloatType>())))) {
        return emitOpError("result #") << index << " must be signless integer or floating-point, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {value, result} have same type");
  if (!(((*this->getODSOperands(1).begin()).getType().cast<MemRefType>().getElementType() == (*this->getODSOperands(0).begin()).getType())))
    return emitOpError("failed to verify that value type matches element type of memref");
  return ::verify(*this);
}

::mlir::ParseResult AtomicRMWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr kindAttr;
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::OperandType memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  {
    ::mlir::StringAttr attrVal;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                              "kind", attrStorage))
      return ::mlir::failure();

    auto attrOptional = ::mlir::symbolizeAtomicRMWKind(attrVal.getValue());
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "kind attribute specification: " << attrVal;

    kindAttr = parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue()));
    result.addAttribute("kind", kindAttr);
  }

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(memrefRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicRMWOp::print(::mlir::OpAsmPrinter &p) {
  p << "atomic_rmw";
  p << " ";
  p << "\"" << stringifyAtomicRMWKind(kind()) << "\"";
  p << " ";
  p << value();
  p << ",";
  p << " ";
  p << memref();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"kind"});
  p << " " << ":";
  p << " " << "(";
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
  p << ",";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(memref().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
}


//===----------------------------------------------------------------------===//
// AtomicYieldOp definitions
//===----------------------------------------------------------------------===//

AtomicYieldOpAdaptor::AtomicYieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

AtomicYieldOpAdaptor::AtomicYieldOpAdaptor(AtomicYieldOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> AtomicYieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtomicYieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicYieldOpAdaptor::result() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult AtomicYieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef AtomicYieldOp::getOperationName() {
  return "std.atomic_yield";
}

std::pair<unsigned, unsigned> AtomicYieldOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicYieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicYieldOp::result() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AtomicYieldOp::resultMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> AtomicYieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicYieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AtomicYieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value result) {
  odsState.addOperands(result);
}

void AtomicYieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value result) {
  odsState.addOperands(result);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicYieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicYieldOp::verify() {
  if (failed(AtomicYieldOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
  return ::verify(*this);
}

::mlir::ParseResult AtomicYieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType resultRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> resultOperands(resultRawOperands);  ::llvm::SMLoc resultOperandsLoc;
  (void)resultOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  resultOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(resultRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(resultOperands, resultTypes, resultOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicYieldOp::print(::mlir::OpAsmPrinter &p) {
  p << "atomic_yield";
  p << " ";
  p << result();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
}

void AtomicYieldOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// BranchOp definitions
//===----------------------------------------------------------------------===//

BranchOpAdaptor::BranchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

BranchOpAdaptor::BranchOpAdaptor(BranchOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> BranchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange BranchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BranchOpAdaptor::destOperands() {
  return getODSOperands(0);
}

::mlir::LogicalResult BranchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef BranchOp::getOperationName() {
  return "std.br";
}

std::pair<unsigned, unsigned> BranchOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range BranchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BranchOp::destOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange BranchOp::destOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> BranchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BranchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *BranchOp::dest() {
  return this->getOperation()->getSuccessor(0);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Block *dest, ValueRange destOperands ) {
    odsState.addSuccessors(dest);
    odsState.addOperands(destOperands);
  
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange destOperands, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange destOperands, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BranchOp::verify() {
  if (failed(BranchOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}







::mlir::ParseResult BranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Block *destSuccessor = nullptr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> destOperandsOperands;
  ::llvm::SMLoc destOperandsOperandsLoc;
  (void)destOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> destOperandsTypes;

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  destOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(destOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(destOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperandsOperands, destOperandsTypes, destOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
  return ::mlir::success();
}

void BranchOp::print(::mlir::OpAsmPrinter &p) {
  p << "br";
  p << " ";
  p << dest();
  if (!destOperands().empty()) {
  p << "(";
  p << destOperands();
  p << " " << ":";
  p << " ";
  p << destOperands().getTypes();
  p << ")";
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

void BranchOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CallIndirectOp definitions
//===----------------------------------------------------------------------===//

CallIndirectOpAdaptor::CallIndirectOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CallIndirectOpAdaptor::CallIndirectOpAdaptor(CallIndirectOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CallIndirectOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange CallIndirectOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CallIndirectOpAdaptor::callee() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange CallIndirectOpAdaptor::operands() {
  return getODSOperands(1);
}

::mlir::LogicalResult CallIndirectOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef CallIndirectOp::getOperationName() {
  return "std.call_indirect";
}

std::pair<unsigned, unsigned> CallIndirectOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CallIndirectOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CallIndirectOp::callee() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range CallIndirectOp::operands() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange CallIndirectOp::calleeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CallIndirectOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CallIndirectOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range CallIndirectOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range CallIndirectOp::results() {
  return getODSResults(0);
}

void CallIndirectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value callee, ValueRange operands ) {
      odsState.operands.push_back(callee);
      odsState.addOperands(operands);
      odsState.addTypes(callee.getType().cast<FunctionType>().getResults());
  
}

void CallIndirectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value callee, ::mlir::ValueRange operands) {
  odsState.addOperands(callee);
  odsState.addOperands(operands);
  odsState.addTypes(results);
}

void CallIndirectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CallIndirectOp::verify() {
  if (failed(CallIndirectOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::FunctionType>()))) {
        return emitOpError("operand #") << index << " must be function type, but got " << v.getType();
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
  if (!(((*this->getODSOperands(0).begin()).getType().cast<FunctionType>().getInputs() == this->getODSOperands(1).getType())))
    return emitOpError("failed to verify that callee input types match argument types");
  if (!(((*this->getODSOperands(0).begin()).getType().cast<FunctionType>().getResults() == this->getODSResults(0).getType())))
    return emitOpError("failed to verify that callee result types match result types");
  return ::mlir::success();
}



::mlir::ParseResult CallIndirectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType calleeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> calleeOperands(calleeRawOperands);  ::llvm::SMLoc calleeOperandsLoc;
  (void)calleeOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::mlir::Type calleeRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> calleeTypes(calleeRawTypes);

  calleeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(calleeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(calleeRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : calleeTypes) {
    (void)type;
    if (!((type.isa<::mlir::FunctionType>()))) {
      return parser.emitError(parser.getNameLoc()) << "'callee' must be function type, but got " << type;
    }
  }
  result.addTypes(calleeTypes[0].cast<FunctionType>().getResults());
  if (parser.resolveOperands(calleeOperands, calleeTypes, calleeOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(operandsOperands, calleeTypes[0].cast<FunctionType>().getInputs(), operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CallIndirectOp::print(::mlir::OpAsmPrinter &p) {
  p << "call_indirect";
  p << " ";
  p << callee();
  p << "(";
  p << operands();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(callee().getType());
}


//===----------------------------------------------------------------------===//
// CallOp definitions
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

::mlir::ValueRange CallOpAdaptor::operands() {
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
  if (!tblgen_callee) return emitError(loc, "'std.call' op ""requires attribute 'callee'");
    if (!((tblgen_callee.isa<::mlir::FlatSymbolRefAttr>()))) return emitError(loc, "'std.call' op ""attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef CallOp::getOperationName() {
  return "std.call";
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

::mlir::Operation::operand_range CallOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange CallOp::operandsMutable() {
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

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, FuncOp callee, ValueRange operands ) {
      odsState.addOperands(operands);
      odsState.addAttribute("callee",odsBuilder.getSymbolRefAttr(callee));
      odsState.addTypes(callee.getType().getResults());
  
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, SymbolRefAttr callee, TypeRange results, ValueRange operands ) {
      odsState.addOperands(operands);
      odsState.addAttribute("callee", callee);
      odsState.addTypes(results);
  
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef callee, TypeRange results, ValueRange operands ) {
      build(odsBuilder, odsState, odsBuilder.getSymbolRefAttr(callee), results,
            operands);
  
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("callee", callee);
  odsState.addTypes(resultType0);
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::llvm::StringRef callee, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("callee", odsBuilder.getSymbolRefAttr(callee));
  odsState.addTypes(resultType0);
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



::mlir::ParseResult CallOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr calleeAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> operandsTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  if (parser.parseAttribute(calleeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "callee", result.attributes))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType operands__allResult_functionType;
  if (parser.parseType(operands__allResult_functionType))
    return ::mlir::failure();
  operandsTypes = operands__allResult_functionType.getInputs();
  allResultTypes = operands__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CallOp::print(::mlir::OpAsmPrinter &p) {
  p << "call";
  p << " ";
  p.printAttributeWithoutType(calleeAttr());
  p << "(";
  p << operands();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " " << ":";
  p << " ";
  p.printFunctionalType(operands().getTypes(), getOperation()->getResultTypes());
}


//===----------------------------------------------------------------------===//
// CeilFOp definitions
//===----------------------------------------------------------------------===//

CeilFOpAdaptor::CeilFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CeilFOpAdaptor::CeilFOpAdaptor(CeilFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CeilFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CeilFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CeilFOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult CeilFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef CeilFOp::getOperationName() {
  return "std.ceilf";
}

std::pair<unsigned, unsigned> CeilFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CeilFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CeilFOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CeilFOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CeilFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CeilFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void CeilFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void CeilFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CeilFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CeilFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void CeilFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult CeilFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void CeilFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult CeilFOp::verify() {
  if (failed(CeilFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void CeilFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CmpFOp definitions
//===----------------------------------------------------------------------===//

CmpFOpAdaptor::CmpFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CmpFOpAdaptor::CmpFOpAdaptor(CmpFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CmpFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CmpFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CmpFOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value CmpFOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::IntegerAttr CmpFOpAdaptor::predicate() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("predicate").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult CmpFOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_predicate = odsAttrs.get("predicate");
  if (!tblgen_predicate) return emitError(loc, "'std.cmpf' op ""requires attribute 'predicate'");
    if (!((((tblgen_predicate.isa<::mlir::IntegerAttr>())) && ((tblgen_predicate.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 9)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 10)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 11)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 12)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 13)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 14)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 15))))) return emitError(loc, "'std.cmpf' op ""attribute 'predicate' failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15");
  }
  return ::mlir::success();
}

::llvm::StringRef CmpFOp::getOperationName() {
  return "std.cmpf";
}

std::pair<unsigned, unsigned> CmpFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CmpFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CmpFOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value CmpFOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CmpFOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CmpFOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CmpFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CmpFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CmpFOp::result() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr CmpFOp::predicateAttr() {
  return this->getAttr("predicate").cast<::mlir::IntegerAttr>();
}

::mlir::CmpFPredicate CmpFOp::predicate() {
  auto attr = predicateAttr();
  return static_cast<::mlir::CmpFPredicate>(attr.getInt());
}

void CmpFOp::predicateAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("predicate", attr);
}

void CmpFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, CmpFPredicate predicate, Value lhs, Value rhs) {
      ::buildCmpFOp(odsBuilder, odsState, predicate, lhs, rhs);
  
}

void CmpFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  odsState.addTypes(result);
}

void CmpFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CmpFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::CmpFPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", odsBuilder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
  odsState.addTypes(result);
}

void CmpFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::CmpFPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", odsBuilder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CmpFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CmpFOp::verify() {
  if (failed(CmpFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  if (!((getI1SameShape((*this->getODSOperands(0).begin()).getType()) == (*this->getODSResults(0).begin()).getType())))
    return emitOpError("failed to verify that result type has i1 element type and same shape as operands");
  return success();
}



::mlir::ParseResult CmpFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr predicateAttr;
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);

  {
    ::mlir::StringAttr attrVal;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                              "predicate", attrStorage))
      return ::mlir::failure();

    auto attrOptional = ::mlir::symbolizeCmpFPredicate(attrVal.getValue());
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "predicate attribute specification: " << attrVal;

    predicateAttr = parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue()));
    result.addAttribute("predicate", predicateAttr);
  }
  if (parser.parseComma())
    return ::mlir::failure();

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

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : lhsTypes) {
    (void)type;
    if (!(((type.isa<::mlir::FloatType>())) || (((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
      return parser.emitError(parser.getNameLoc()) << "'lhs' must be floating-point-like, but got " << type;
    }
  }
  result.addTypes(getI1SameShape(lhsTypes[0]));
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, lhsTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CmpFOp::print(::mlir::OpAsmPrinter &p) {
  p << "cmpf";
  p << " ";
  p << "\"" << stringifyCmpFPredicate(predicate()) << "\"";
  p << ",";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"predicate"});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
}

void CmpFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CmpIOp definitions
//===----------------------------------------------------------------------===//

CmpIOpAdaptor::CmpIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CmpIOpAdaptor::CmpIOpAdaptor(CmpIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CmpIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CmpIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CmpIOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value CmpIOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::IntegerAttr CmpIOpAdaptor::predicate() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("predicate").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::LogicalResult CmpIOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_predicate = odsAttrs.get("predicate");
  if (!tblgen_predicate) return emitError(loc, "'std.cmpi' op ""requires attribute 'predicate'");
    if (!((((tblgen_predicate.isa<::mlir::IntegerAttr>())) && ((tblgen_predicate.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && (((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 0)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 1)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 2)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 3)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 4)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 5)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 6)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 7)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 8)) || ((tblgen_predicate.cast<::mlir::IntegerAttr>().getInt() == 9))))) return emitError(loc, "'std.cmpi' op ""attribute 'predicate' failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9");
  }
  return ::mlir::success();
}

::llvm::StringRef CmpIOp::getOperationName() {
  return "std.cmpi";
}

std::pair<unsigned, unsigned> CmpIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CmpIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CmpIOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value CmpIOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CmpIOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CmpIOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CmpIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CmpIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CmpIOp::result() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr CmpIOp::predicateAttr() {
  return this->getAttr("predicate").cast<::mlir::IntegerAttr>();
}

::mlir::CmpIPredicate CmpIOp::predicate() {
  auto attr = predicateAttr();
  return static_cast<::mlir::CmpIPredicate>(attr.getInt());
}

void CmpIOp::predicateAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("predicate", attr);
}

void CmpIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, CmpIPredicate predicate, Value lhs, Value rhs) {
      ::buildCmpIOp(odsBuilder, odsState, predicate, lhs, rhs);
  
}

void CmpIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::IntegerAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  odsState.addTypes(result);
}

void CmpIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CmpIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::CmpIPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", odsBuilder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
  odsState.addTypes(result);
}

void CmpIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::CmpIPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", odsBuilder.getI64IntegerAttr(static_cast<int64_t>(predicate)));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CmpIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CmpIOp::verify() {
  if (failed(CmpIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  if (!((getI1SameShape((*this->getODSOperands(0).begin()).getType()) == (*this->getODSResults(0).begin()).getType())))
    return emitOpError("failed to verify that result type has i1 element type and same shape as operands");
  return success();
}



::mlir::ParseResult CmpIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr predicateAttr;
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);

  {
    ::mlir::StringAttr attrVal;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                              "predicate", attrStorage))
      return ::mlir::failure();

    auto attrOptional = ::mlir::symbolizeCmpIPredicate(attrVal.getValue());
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "predicate attribute specification: " << attrVal;

    predicateAttr = parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue()));
    result.addAttribute("predicate", predicateAttr);
  }
  if (parser.parseComma())
    return ::mlir::failure();

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

  if (parser.parseType(lhsRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : lhsTypes) {
    (void)type;
    if (!(((type.isSignlessInteger())) || ((type.isa<::mlir::IndexType>())) || (((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
      return parser.emitError(parser.getNameLoc()) << "'lhs' must be signless-integer-like, but got " << type;
    }
  }
  result.addTypes(getI1SameShape(lhsTypes[0]));
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, lhsTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CmpIOp::print(::mlir::OpAsmPrinter &p) {
  p << "cmpi";
  p << " ";
  p << "\"" << stringifyCmpIPredicate(predicate()) << "\"";
  p << ",";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"predicate"});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(lhs().getType());
}

void CmpIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CondBranchOp definitions
//===----------------------------------------------------------------------===//

CondBranchOpAdaptor::CondBranchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CondBranchOpAdaptor::CondBranchOpAdaptor(CondBranchOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CondBranchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange CondBranchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CondBranchOpAdaptor::condition() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange CondBranchOpAdaptor::trueDestOperands() {
  return getODSOperands(1);
}

::mlir::ValueRange CondBranchOpAdaptor::falseDestOperands() {
  return getODSOperands(2);
}

::mlir::LogicalResult CondBranchOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    return ::mlir::success();
}

::llvm::StringRef CondBranchOp::getOperationName() {
  return "std.cond_br";
}

std::pair<unsigned, unsigned> CondBranchOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range CondBranchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CondBranchOp::condition() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range CondBranchOp::trueDestOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CondBranchOp::falseDestOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CondBranchOp::conditionMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange CondBranchOp::trueDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange CondBranchOp::falseDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> CondBranchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CondBranchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CondBranchOp::trueDest() {
  return this->getOperation()->getSuccessor(0);
}

::mlir::Block *CondBranchOp::falseDest() {
  return this->getOperation()->getSuccessor(1);
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block *trueDest, ValueRange trueOperands,Block *falseDest, ValueRange falseOperands) {
      build(odsBuilder, odsState, condition, trueOperands, falseOperands, trueDest,
            falseDest);
  
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block *trueDest, Block *falseDest, ValueRange falseOperands ) {
      build(odsBuilder, odsState, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
  
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CondBranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CondBranchOp::verify() {
  if (failed(CondBranchOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isSignlessInteger(1)))) {
        return emitOpError("operand #") << index << " must be 1-bit signless integer, but got " << v.getType();
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
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}







::mlir::ParseResult CondBranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType conditionRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> conditionOperands(conditionRawOperands);  ::llvm::SMLoc conditionOperandsLoc;
  (void)conditionOperandsLoc;
  ::mlir::Block *trueDestSuccessor = nullptr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> trueDestOperandsOperands;
  ::llvm::SMLoc trueDestOperandsOperandsLoc;
  (void)trueDestOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> trueDestOperandsTypes;
  ::mlir::Block *falseDestSuccessor = nullptr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> falseDestOperandsOperands;
  ::llvm::SMLoc falseDestOperandsOperandsLoc;
  (void)falseDestOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> falseDestOperandsTypes;

  conditionOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(conditionRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(trueDestSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  trueDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(trueDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(trueDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(falseDestSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  falseDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(falseDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(falseDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(conditionOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(trueDestOperandsOperands, trueDestOperandsTypes, trueDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(falseDestOperandsOperands, falseDestOperandsTypes, falseDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  result.addSuccessors(trueDestSuccessor);
  result.addSuccessors(falseDestSuccessor);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(trueDestOperandsOperands.size()), static_cast<int32_t>(falseDestOperandsOperands.size())}));
  return ::mlir::success();
}

void CondBranchOp::print(::mlir::OpAsmPrinter &p) {
  p << "cond_br";
  p << " ";
  p << condition();
  p << ",";
  p << " ";
  p << trueDest();
  if (!trueDestOperands().empty()) {
  p << "(";
  p << trueDestOperands();
  p << " " << ":";
  p << " ";
  p << trueDestOperands().getTypes();
  p << ")";
  }
  p << ",";
  p << " ";
  p << falseDest();
  if (!falseDestOperands().empty()) {
  p << "(";
  p << falseDestOperands();
  p << " " << ":";
  p << " ";
  p << falseDestOperands().getTypes();
  p << ")";
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

void CondBranchOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ConstantOp definitions
//===----------------------------------------------------------------------===//

ConstantOpAdaptor::ConstantOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ConstantOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstantOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Attribute ConstantOpAdaptor::value() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("value").cast<::mlir::Attribute>();
  return attr;
}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_value = odsAttrs.get("value");
  if (!tblgen_value) return emitError(loc, "'std.constant' op ""requires attribute 'value'");
    if (!((true))) return emitError(loc, "'std.constant' op ""attribute 'value' failed to satisfy constraint: any attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef ConstantOp::getOperationName() {
  return "std.constant";
}

std::pair<unsigned, unsigned> ConstantOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Attribute ConstantOp::valueAttr() {
  return this->getAttr("value").cast<::mlir::Attribute>();
}

::mlir::Attribute ConstantOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstantOp::valueAttr(::mlir::Attribute attr) {
  this->getOperation()->setAttr("value", attr);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute value) {
 build(odsBuilder, odsState, value.getType(), value); 
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Attribute value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(resultType0);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseConstantOp(parser, result);
}

void ConstantOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult ConstantOp::verify() {
  if (failed(ConstantOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
  return ::verify(*this);
}





void ConstantOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CopySignOp definitions
//===----------------------------------------------------------------------===//

CopySignOpAdaptor::CopySignOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CopySignOpAdaptor::CopySignOpAdaptor(CopySignOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CopySignOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CopySignOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopySignOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopySignOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult CopySignOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef CopySignOp::getOperationName() {
  return "std.copysign";
}

std::pair<unsigned, unsigned> CopySignOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CopySignOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopySignOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopySignOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CopySignOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CopySignOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CopySignOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CopySignOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CopySignOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult CopySignOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void CopySignOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult CopySignOp::verify() {
  if (failed(CopySignOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void CopySignOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CosOp definitions
//===----------------------------------------------------------------------===//

CosOpAdaptor::CosOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CosOpAdaptor::CosOpAdaptor(CosOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CosOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CosOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CosOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult CosOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef CosOp::getOperationName() {
  return "std.cos";
}

std::pair<unsigned, unsigned> CosOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CosOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CosOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CosOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CosOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CosOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CosOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult CosOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void CosOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult CosOp::verify() {
  if (failed(CosOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void CosOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CreateComplexOp definitions
//===----------------------------------------------------------------------===//

CreateComplexOpAdaptor::CreateComplexOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

CreateComplexOpAdaptor::CreateComplexOpAdaptor(CreateComplexOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> CreateComplexOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CreateComplexOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateComplexOpAdaptor::real() {
  return *getODSOperands(0).begin();
}

::mlir::Value CreateComplexOpAdaptor::imaginary() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult CreateComplexOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef CreateComplexOp::getOperationName() {
  return "std.create_complex";
}

std::pair<unsigned, unsigned> CreateComplexOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CreateComplexOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateComplexOp::real() {
  return *getODSOperands(0).begin();
}

::mlir::Value CreateComplexOp::imaginary() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CreateComplexOp::realMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange CreateComplexOp::imaginaryMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> CreateComplexOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateComplexOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateComplexOp::complex() {
  return *getODSResults(0).begin();
}

void CreateComplexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type complex, ::mlir::Value real, ::mlir::Value imaginary) {
  odsState.addOperands(real);
  odsState.addOperands(imaginary);
  odsState.addTypes(complex);
}

void CreateComplexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value real, ::mlir::Value imaginary) {
  odsState.addOperands(real);
  odsState.addOperands(imaginary);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateComplexOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateComplexOp::verify() {
  if (failed(CreateComplexOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::FloatType>()))) {
        return emitOpError("operand #") << index << " must be floating-point, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<::mlir::FloatType>()))) {
        return emitOpError("operand #") << index << " must be floating-point, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isa<::mlir::FloatType>())))) {
        return emitOpError("result #") << index << " must be complex type with floating-point elements, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType()})))))
    return emitOpError("failed to verify that all of {real, imaginary} have same type");
  if (!(((*this->getODSResults(0).begin()).getType().cast<ComplexType>().getElementType() == (*this->getODSOperands(0).begin()).getType())))
    return emitOpError("failed to verify that complex element type matches real operand type");
  if (!(((*this->getODSResults(0).begin()).getType().cast<ComplexType>().getElementType() == (*this->getODSOperands(1).begin()).getType())))
    return emitOpError("failed to verify that complex element type matches imaginary operand type");
  return ::mlir::success();
}

::mlir::ParseResult CreateComplexOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType realRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> realOperands(realRawOperands);  ::llvm::SMLoc realOperandsLoc;
  (void)realOperandsLoc;
  ::mlir::OpAsmParser::OperandType imaginaryRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> imaginaryOperands(imaginaryRawOperands);  ::llvm::SMLoc imaginaryOperandsLoc;
  (void)imaginaryOperandsLoc;
  ::mlir::Type complexRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> complexTypes(complexRawTypes);

  realOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(realRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  imaginaryOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(imaginaryRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(complexRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : complexTypes) {
    (void)type;
    if (!(((type.isa<::mlir::ComplexType>())) && ((type.cast<::mlir::ComplexType>().getElementType().isa<::mlir::FloatType>())))) {
      return parser.emitError(parser.getNameLoc()) << "'complex' must be complex type with floating-point elements, but got " << type;
    }
  }
  result.addTypes(complexTypes);
  if (parser.resolveOperands(realOperands, complexTypes[0].cast<ComplexType>().getElementType(), realOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(imaginaryOperands, complexTypes[0].cast<ComplexType>().getElementType(), imaginaryOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateComplexOp::print(::mlir::OpAsmPrinter &p) {
  p << "create_complex";
  p << " ";
  p << real();
  p << ",";
  p << " ";
  p << imaginary();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(complex().getType());
}

void CreateComplexOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// DeallocOp definitions
//===----------------------------------------------------------------------===//

DeallocOpAdaptor::DeallocOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

DeallocOpAdaptor::DeallocOpAdaptor(DeallocOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> DeallocOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DeallocOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DeallocOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult DeallocOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef DeallocOp::getOperationName() {
  return "std.dealloc";
}

std::pair<unsigned, unsigned> DeallocOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DeallocOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DeallocOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange DeallocOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> DeallocOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DeallocOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void DeallocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref) {
  odsState.addOperands(memref);
}

void DeallocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref) {
  odsState.addOperands(memref);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeallocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DeallocOp::verify() {
  if (failed(DeallocOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





::mlir::ParseResult DeallocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DeallocOp::print(::mlir::OpAsmPrinter &p) {
  p << "dealloc";
  p << " ";
  p << memref();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(memref().getType());
}

void DeallocOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Free::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Free::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// DimOp definitions
//===----------------------------------------------------------------------===//

DimOpAdaptor::DimOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

DimOpAdaptor::DimOpAdaptor(DimOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> DimOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DimOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DimOpAdaptor::memrefOrTensor() {
  return *getODSOperands(0).begin();
}

::mlir::Value DimOpAdaptor::index() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult DimOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef DimOp::getOperationName() {
  return "std.dim";
}

std::pair<unsigned, unsigned> DimOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DimOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DimOp::memrefOrTensor() {
  return *getODSOperands(0).begin();
}

::mlir::Value DimOp::index() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange DimOp::memrefOrTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange DimOp::indexMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> DimOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DimOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DimOp::result() {
  return *getODSResults(0).begin();
}





void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memrefOrTensor, ::mlir::Value index) {
  odsState.addOperands(memrefOrTensor);
  odsState.addOperands(index);
  odsState.addTypes(result);
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memrefOrTensor, ::mlir::Value index) {
  odsState.addOperands(memrefOrTensor);
  odsState.addOperands(index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DimOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DimOp::verify() {
  if (failed(DimOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((((v.getType().isa<::mlir::UnrankedMemRefType>())) && ((true))) || (((v.getType().isa<::mlir::MemRefType>())) && ((true)))) || (((v.getType().isa<::mlir::TensorType>())) && ((true))))) {
        return emitOpError("operand #") << index << " must be any tensor or memref type, but got " << v.getType();
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
      if (!((v.getType().isa<::mlir::IndexType>()))) {
        return emitOpError("result #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}



::mlir::ParseResult DimOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType memrefOrTensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOrTensorOperands(memrefOrTensorRawOperands);  ::llvm::SMLoc memrefOrTensorOperandsLoc;
  (void)memrefOrTensorOperandsLoc;
  ::mlir::OpAsmParser::OperandType indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> indexOperands(indexRawOperands);  ::llvm::SMLoc indexOperandsLoc;
  (void)indexOperandsLoc;
  ::mlir::Type memrefOrTensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefOrTensorTypes(memrefOrTensorRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  memrefOrTensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefOrTensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(indexRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefOrTensorRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(memrefOrTensorOperands, memrefOrTensorTypes, memrefOrTensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indexOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DimOp::print(::mlir::OpAsmPrinter &p) {
  p << "dim";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " ";
  p << memrefOrTensor();
  p << ",";
  p << " ";
  p << index();
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(memrefOrTensor().getType());
}

void DimOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// DivFOp definitions
//===----------------------------------------------------------------------===//

DivFOpAdaptor::DivFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

DivFOpAdaptor::DivFOpAdaptor(DivFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> DivFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DivFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DivFOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value DivFOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult DivFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef DivFOp::getOperationName() {
  return "std.divf";
}

std::pair<unsigned, unsigned> DivFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DivFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DivFOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value DivFOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange DivFOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange DivFOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> DivFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DivFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void DivFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void DivFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DivFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void DivFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void DivFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult DivFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void DivFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult DivFOp::verify() {
  if (failed(DivFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void DivFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// DynamicTensorFromElementsOp definitions
//===----------------------------------------------------------------------===//

DynamicTensorFromElementsOpAdaptor::DynamicTensorFromElementsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

DynamicTensorFromElementsOpAdaptor::DynamicTensorFromElementsOpAdaptor(DynamicTensorFromElementsOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> DynamicTensorFromElementsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DynamicTensorFromElementsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DynamicTensorFromElementsOpAdaptor::dynamicExtents() {
  return getODSOperands(0);
}

::mlir::LogicalResult DynamicTensorFromElementsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef DynamicTensorFromElementsOp::getOperationName() {
  return "std.dynamic_tensor_from_elements";
}

std::pair<unsigned, unsigned> DynamicTensorFromElementsOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range DynamicTensorFromElementsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DynamicTensorFromElementsOp::dynamicExtents() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange DynamicTensorFromElementsOp::dynamicExtentsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> DynamicTensorFromElementsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DynamicTensorFromElementsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DynamicTensorFromElementsOp::result() {
  return *getODSResults(0).begin();
}

::mlir::Region &DynamicTensorFromElementsOp::body() {
  return this->getOperation()->getRegion(0);
}



void DynamicTensorFromElementsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange dynamicExtents) {
  odsState.addOperands(dynamicExtents);
  (void)odsState.addRegion();
  odsState.addTypes(result);
}

void DynamicTensorFromElementsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult DynamicTensorFromElementsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseDynamicTensorFromElementsOp(parser, result);
}

void DynamicTensorFromElementsOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult DynamicTensorFromElementsOp::verify() {
  if (failed(DynamicTensorFromElementsOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
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
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return ::verify(*this);
}




//===----------------------------------------------------------------------===//
// Exp2Op definitions
//===----------------------------------------------------------------------===//

Exp2OpAdaptor::Exp2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

Exp2OpAdaptor::Exp2OpAdaptor(Exp2Op&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> Exp2OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Exp2OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Exp2OpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult Exp2OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef Exp2Op::getOperationName() {
  return "std.exp2";
}

std::pair<unsigned, unsigned> Exp2Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Exp2Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Exp2Op::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange Exp2Op::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Exp2Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Exp2Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Exp2Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult Exp2Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void Exp2Op::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult Exp2Op::verify() {
  if (failed(Exp2OpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void Exp2Op::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ExpOp definitions
//===----------------------------------------------------------------------===//

ExpOpAdaptor::ExpOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ExpOpAdaptor::ExpOpAdaptor(ExpOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ExpOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExpOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult ExpOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ExpOp::getOperationName() {
  return "std.exp";
}

std::pair<unsigned, unsigned> ExpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ExpOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ExpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult ExpOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void ExpOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult ExpOp::verify() {
  if (failed(ExpOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void ExpOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ExtractElementOp definitions
//===----------------------------------------------------------------------===//

ExtractElementOpAdaptor::ExtractElementOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ExtractElementOpAdaptor::ExtractElementOpAdaptor(ExtractElementOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ExtractElementOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ExtractElementOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOpAdaptor::aggregate() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ExtractElementOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::LogicalResult ExtractElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ExtractElementOp::getOperationName() {
  return "std.extract_element";
}

std::pair<unsigned, unsigned> ExtractElementOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ExtractElementOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOp::aggregate() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ExtractElementOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ExtractElementOp::aggregateMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ExtractElementOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ExtractElementOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractElementOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOp::result() {
  return *getODSResults(0).begin();
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value aggregate, ValueRange indices ) {
      auto resType = aggregate.getType().cast<ShapedType>()
                                         .getElementType();
      build(odsBuilder, odsState, resType, aggregate, indices);
    
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value aggregate, ::mlir::ValueRange indices) {
  odsState.addOperands(aggregate);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value aggregate, ::mlir::ValueRange indices) {
  odsState.addOperands(aggregate);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractElementOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractElementOp::verify() {
  if (failed(ExtractElementOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::VectorType>())) && ((true))) || (((v.getType().isa<::mlir::TensorType>())) && ((true))))) {
        return emitOpError("operand #") << index << " must be vector of any type values or tensor of any type values, but got " << v.getType();
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSOperands(0).begin()).getType().cast<ShapedType>().getElementType() == (*this->getODSResults(0).begin()).getType())))
    return emitOpError("failed to verify that result type matches element type of aggregate");
  return ::verify(*this);
}



::mlir::ParseResult ExtractElementOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType aggregateRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> aggregateOperands(aggregateRawOperands);  ::llvm::SMLoc aggregateOperandsLoc;
  (void)aggregateOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type aggregateRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> aggregateTypes(aggregateRawTypes);

  aggregateOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(aggregateRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(aggregateRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : aggregateTypes) {
    (void)type;
    if (!((((type.isa<::mlir::VectorType>())) && ((true))) || (((type.isa<::mlir::TensorType>())) && ((true))))) {
      return parser.emitError(parser.getNameLoc()) << "'aggregate' must be vector of any type values or tensor of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(aggregateTypes[0].cast<ShapedType>().getElementType());
  if (parser.resolveOperands(aggregateOperands, aggregateTypes, aggregateOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractElementOp::print(::mlir::OpAsmPrinter &p) {
  p << "extract_element";
  p << " ";
  p << aggregate();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(aggregate().getType());
}

void ExtractElementOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// FPExtOp definitions
//===----------------------------------------------------------------------===//

FPExtOpAdaptor::FPExtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

FPExtOpAdaptor::FPExtOpAdaptor(FPExtOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> FPExtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPExtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPExtOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult FPExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef FPExtOp::getOperationName() {
  return "std.fpext";
}

std::pair<unsigned, unsigned> FPExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPExtOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPExtOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FPExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Type destType) {
       impl::buildCastOp(odsBuilder, odsState, source, destType);
  
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FPExtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void FPExtOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult FPExtOp::verify() {
  if (failed(FPExtOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}

void FPExtOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// FPToSIOp definitions
//===----------------------------------------------------------------------===//

FPToSIOpAdaptor::FPToSIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

FPToSIOpAdaptor::FPToSIOpAdaptor(FPToSIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> FPToSIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPToSIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToSIOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult FPToSIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef FPToSIOp::getOperationName() {
  return "std.fptosi";
}

std::pair<unsigned, unsigned> FPToSIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPToSIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToSIOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPToSIOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FPToSIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPToSIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Type destType) {
       impl::buildCastOp(odsBuilder, odsState, source, destType);
  
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPToSIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FPToSIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void FPToSIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult FPToSIOp::verify() {
  if (failed(FPToSIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}

void FPToSIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// FPToUIOp definitions
//===----------------------------------------------------------------------===//

FPToUIOpAdaptor::FPToUIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

FPToUIOpAdaptor::FPToUIOpAdaptor(FPToUIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> FPToUIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPToUIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToUIOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult FPToUIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef FPToUIOp::getOperationName() {
  return "std.fptoui";
}

std::pair<unsigned, unsigned> FPToUIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPToUIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToUIOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPToUIOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FPToUIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPToUIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Type destType) {
       impl::buildCastOp(odsBuilder, odsState, source, destType);
  
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPToUIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FPToUIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void FPToUIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult FPToUIOp::verify() {
  if (failed(FPToUIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}

void FPToUIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// FPTruncOp definitions
//===----------------------------------------------------------------------===//

FPTruncOpAdaptor::FPTruncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

FPTruncOpAdaptor::FPTruncOpAdaptor(FPTruncOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> FPTruncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPTruncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPTruncOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult FPTruncOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef FPTruncOp::getOperationName() {
  return "std.fptrunc";
}

std::pair<unsigned, unsigned> FPTruncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPTruncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPTruncOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPTruncOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FPTruncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPTruncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Type destType) {
       impl::buildCastOp(odsBuilder, odsState, source, destType);
  
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPTruncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult FPTruncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void FPTruncOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult FPTruncOp::verify() {
  if (failed(FPTruncOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}

void FPTruncOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// FloorFOp definitions
//===----------------------------------------------------------------------===//

FloorFOpAdaptor::FloorFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

FloorFOpAdaptor::FloorFOpAdaptor(FloorFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> FloorFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FloorFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FloorFOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult FloorFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef FloorFOp::getOperationName() {
  return "std.floorf";
}

std::pair<unsigned, unsigned> FloorFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FloorFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FloorFOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FloorFOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> FloorFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FloorFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FloorFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void FloorFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FloorFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FloorFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void FloorFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult FloorFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void FloorFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult FloorFOp::verify() {
  if (failed(FloorFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void FloorFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// GenericAtomicRMWOp definitions
//===----------------------------------------------------------------------===//

GenericAtomicRMWOpAdaptor::GenericAtomicRMWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

GenericAtomicRMWOpAdaptor::GenericAtomicRMWOpAdaptor(GenericAtomicRMWOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> GenericAtomicRMWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange GenericAtomicRMWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GenericAtomicRMWOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange GenericAtomicRMWOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::LogicalResult GenericAtomicRMWOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef GenericAtomicRMWOp::getOperationName() {
  return "std.generic_atomic_rmw";
}

std::pair<unsigned, unsigned> GenericAtomicRMWOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range GenericAtomicRMWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GenericAtomicRMWOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range GenericAtomicRMWOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GenericAtomicRMWOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange GenericAtomicRMWOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> GenericAtomicRMWOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GenericAtomicRMWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GenericAtomicRMWOp::result() {
  return *getODSResults(0).begin();
}

::mlir::Region &GenericAtomicRMWOp::body() {
  return this->getOperation()->getRegion(0);
}



::mlir::ParseResult GenericAtomicRMWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseGenericAtomicRMWOp(parser, result);
}

void GenericAtomicRMWOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult GenericAtomicRMWOp::verify() {
  if (failed(GenericAtomicRMWOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && (((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be memref of signless integer or floating-point values, but got " << v.getType();
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
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::FloatType>())))) {
        return emitOpError("result #") << index << " must be signless integer or floating-point, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSOperands(0).begin()).getType().cast<MemRefType>().getElementType() == (*this->getODSResults(0).begin()).getType())))
    return emitOpError("failed to verify that result type matches element type of memref");
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


//===----------------------------------------------------------------------===//
// ImOp definitions
//===----------------------------------------------------------------------===//

ImOpAdaptor::ImOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ImOpAdaptor::ImOpAdaptor(ImOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ImOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ImOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ImOpAdaptor::complex() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult ImOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ImOp::getOperationName() {
  return "std.im";
}

std::pair<unsigned, unsigned> ImOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ImOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ImOp::complex() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ImOp::complexMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ImOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ImOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ImOp::imaginary() {
  return *getODSResults(0).begin();
}

void ImOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type imaginary, ::mlir::Value complex) {
  odsState.addOperands(complex);
  odsState.addTypes(imaginary);
}

void ImOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value complex) {
  odsState.addOperands(complex);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ImOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ImOp::verify() {
  if (failed(ImOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isa<::mlir::FloatType>())))) {
        return emitOpError("operand #") << index << " must be complex type with floating-point elements, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::FloatType>()))) {
        return emitOpError("result #") << index << " must be floating-point, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSOperands(0).begin()).getType().cast<ComplexType>().getElementType() == (*this->getODSResults(0).begin()).getType())))
    return emitOpError("failed to verify that complex element type matches result type");
  return ::mlir::success();
}

::mlir::ParseResult ImOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType complexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> complexOperands(complexRawOperands);  ::llvm::SMLoc complexOperandsLoc;
  (void)complexOperandsLoc;
  ::mlir::Type complexRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> complexTypes(complexRawTypes);

  complexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(complexRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(complexRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : complexTypes) {
    (void)type;
    if (!(((type.isa<::mlir::ComplexType>())) && ((type.cast<::mlir::ComplexType>().getElementType().isa<::mlir::FloatType>())))) {
      return parser.emitError(parser.getNameLoc()) << "'complex' must be complex type with floating-point elements, but got " << type;
    }
  }
  result.addTypes(complexTypes[0].cast<ComplexType>().getElementType());
  if (parser.resolveOperands(complexOperands, complexTypes, complexOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ImOp::print(::mlir::OpAsmPrinter &p) {
  p << "im";
  p << " ";
  p << complex();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(complex().getType());
}

void ImOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// IndexCastOp definitions
//===----------------------------------------------------------------------===//

IndexCastOpAdaptor::IndexCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

IndexCastOpAdaptor::IndexCastOpAdaptor(IndexCastOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> IndexCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IndexCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IndexCastOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult IndexCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef IndexCastOp::getOperationName() {
  return "std.index_cast";
}

std::pair<unsigned, unsigned> IndexCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IndexCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IndexCastOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange IndexCastOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> IndexCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IndexCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void IndexCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Type destType) {
       impl::buildCastOp(odsBuilder, odsState, source, destType);
  
}

void IndexCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void IndexCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IndexCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult IndexCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void IndexCastOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult IndexCastOp::verify() {
  if (failed(IndexCastOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}



void IndexCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// LoadOp definitions
//===----------------------------------------------------------------------===//

LoadOpAdaptor::LoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

LoadOpAdaptor::LoadOpAdaptor(LoadOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> LoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange LoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange LoadOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::LogicalResult LoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef LoadOp::getOperationName() {
  return "std.load";
}

std::pair<unsigned, unsigned> LoadOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range LoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range LoadOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange LoadOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange LoadOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOp::result() {
  return *getODSResults(0).begin();
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, ValueRange indices ) {
      auto memrefType = memref.getType().cast<MemRefType>();
      odsState.addOperands(memref);
      odsState.addOperands(indices);
      odsState.types.push_back(memrefType.getElementType());
  
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LoadOp::verify() {
  if (failed(LoadOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSOperands(0).begin()).getType().cast<MemRefType>().getElementType() == (*this->getODSResults(0).begin()).getType())))
    return emitOpError("failed to verify that result type matches element type of 'memref'");
  return ::verify(*this);
}



::mlir::ParseResult LoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : memrefTypes) {
    (void)type;
    if (!(((type.isa<::mlir::MemRefType>())) && ((true)))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be memref of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(memrefTypes[0].cast<MemRefType>().getElementType());
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadOp::print(::mlir::OpAsmPrinter &p) {
  p << "load";
  p << " ";
  p << memref();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(memref().getType());
}

void LoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// Log10Op definitions
//===----------------------------------------------------------------------===//

Log10OpAdaptor::Log10OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

Log10OpAdaptor::Log10OpAdaptor(Log10Op&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> Log10OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Log10OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log10OpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult Log10OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef Log10Op::getOperationName() {
  return "std.log10";
}

std::pair<unsigned, unsigned> Log10Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Log10Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log10Op::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange Log10Op::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Log10Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Log10Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Log10Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult Log10Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void Log10Op::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult Log10Op::verify() {
  if (failed(Log10OpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void Log10Op::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// Log2Op definitions
//===----------------------------------------------------------------------===//

Log2OpAdaptor::Log2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

Log2OpAdaptor::Log2OpAdaptor(Log2Op&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> Log2OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Log2OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log2OpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult Log2OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef Log2Op::getOperationName() {
  return "std.log2";
}

std::pair<unsigned, unsigned> Log2Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Log2Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log2Op::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange Log2Op::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> Log2Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Log2Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Log2Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult Log2Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void Log2Op::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult Log2Op::verify() {
  if (failed(Log2OpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void Log2Op::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// LogOp definitions
//===----------------------------------------------------------------------===//

LogOpAdaptor::LogOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

LogOpAdaptor::LogOpAdaptor(LogOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> LogOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult LogOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef LogOp::getOperationName() {
  return "std.log";
}

std::pair<unsigned, unsigned> LogOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LogOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LogOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult LogOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void LogOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult LogOp::verify() {
  if (failed(LogOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void LogOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// MemRefCastOp definitions
//===----------------------------------------------------------------------===//

MemRefCastOpAdaptor::MemRefCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

MemRefCastOpAdaptor::MemRefCastOpAdaptor(MemRefCastOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> MemRefCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MemRefCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemRefCastOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult MemRefCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef MemRefCastOp::getOperationName() {
  return "std.memref_cast";
}

std::pair<unsigned, unsigned> MemRefCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MemRefCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemRefCastOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MemRefCastOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MemRefCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MemRefCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MemRefCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Type destType) {
       impl::buildCastOp(odsBuilder, odsState, source, destType);
  
}

void MemRefCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(resultType0);
}

void MemRefCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemRefCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult MemRefCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void MemRefCastOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult MemRefCastOp::verify() {
  if (failed(MemRefCastOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::UnrankedMemRefType>())) && ((true))) || (((v.getType().isa<::mlir::MemRefType>())) && ((true))))) {
        return emitOpError("operand #") << index << " must be unranked.memref of any type values or memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::UnrankedMemRefType>())) && ((true))) || (((v.getType().isa<::mlir::MemRefType>())) && ((true))))) {
        return emitOpError("result #") << index << " must be unranked.memref of any type values or memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}





void MemRefCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// MulFOp definitions
//===----------------------------------------------------------------------===//

MulFOpAdaptor::MulFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

MulFOpAdaptor::MulFOpAdaptor(MulFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> MulFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MulFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulFOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulFOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult MulFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef MulFOp::getOperationName() {
  return "std.mulf";
}

std::pair<unsigned, unsigned> MulFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MulFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulFOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulFOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MulFOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MulFOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MulFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MulFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MulFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void MulFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MulFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void MulFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult MulFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void MulFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult MulFOp::verify() {
  if (failed(MulFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void MulFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// MulIOp definitions
//===----------------------------------------------------------------------===//

MulIOpAdaptor::MulIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

MulIOpAdaptor::MulIOpAdaptor(MulIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> MulIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MulIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulIOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulIOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult MulIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef MulIOp::getOperationName() {
  return "std.muli";
}

std::pair<unsigned, unsigned> MulIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MulIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulIOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulIOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MulIOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange MulIOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> MulIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MulIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MulIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void MulIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MulIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void MulIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult MulIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void MulIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult MulIOp::verify() {
  if (failed(MulIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void MulIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// NegFOp definitions
//===----------------------------------------------------------------------===//

NegFOpAdaptor::NegFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

NegFOpAdaptor::NegFOpAdaptor(NegFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> NegFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange NegFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NegFOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult NegFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef NegFOp::getOperationName() {
  return "std.negf";
}

std::pair<unsigned, unsigned> NegFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NegFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NegFOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange NegFOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> NegFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NegFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void NegFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void NegFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NegFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void NegFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void NegFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult NegFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void NegFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult NegFOp::verify() {
  if (failed(NegFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void NegFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// OrOp definitions
//===----------------------------------------------------------------------===//

OrOpAdaptor::OrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

OrOpAdaptor::OrOpAdaptor(OrOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> OrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange OrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OrOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value OrOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult OrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef OrOp::getOperationName() {
  return "std.or";
}

std::pair<unsigned, unsigned> OrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range OrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OrOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value OrOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange OrOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange OrOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> OrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult OrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void OrOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult OrOp::verify() {
  if (failed(OrOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void OrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// PrefetchOp definitions
//===----------------------------------------------------------------------===//

PrefetchOpAdaptor::PrefetchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

PrefetchOpAdaptor::PrefetchOpAdaptor(PrefetchOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> PrefetchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange PrefetchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PrefetchOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange PrefetchOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::BoolAttr PrefetchOpAdaptor::isWrite() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("isWrite").cast<::mlir::BoolAttr>();
  return attr;
}

::mlir::IntegerAttr PrefetchOpAdaptor::localityHint() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("localityHint").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::BoolAttr PrefetchOpAdaptor::isDataCache() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("isDataCache").cast<::mlir::BoolAttr>();
  return attr;
}

::mlir::LogicalResult PrefetchOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_isWrite = odsAttrs.get("isWrite");
  if (!tblgen_isWrite) return emitError(loc, "'std.prefetch' op ""requires attribute 'isWrite'");
    if (!((tblgen_isWrite.isa<::mlir::BoolAttr>()))) return emitError(loc, "'std.prefetch' op ""attribute 'isWrite' failed to satisfy constraint: bool attribute");
  }
  {
  auto tblgen_localityHint = odsAttrs.get("localityHint");
  if (!tblgen_localityHint) return emitError(loc, "'std.prefetch' op ""requires attribute 'localityHint'");
    if (!((((tblgen_localityHint.isa<::mlir::IntegerAttr>())) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getInt() >= 0)) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getInt() <= 3)))) return emitError(loc, "'std.prefetch' op ""attribute 'localityHint' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3");
  }
  {
  auto tblgen_isDataCache = odsAttrs.get("isDataCache");
  if (!tblgen_isDataCache) return emitError(loc, "'std.prefetch' op ""requires attribute 'isDataCache'");
    if (!((tblgen_isDataCache.isa<::mlir::BoolAttr>()))) return emitError(loc, "'std.prefetch' op ""attribute 'isDataCache' failed to satisfy constraint: bool attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef PrefetchOp::getOperationName() {
  return "std.prefetch";
}

std::pair<unsigned, unsigned> PrefetchOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range PrefetchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PrefetchOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range PrefetchOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PrefetchOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange PrefetchOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> PrefetchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PrefetchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::BoolAttr PrefetchOp::isWriteAttr() {
  return this->getAttr("isWrite").cast<::mlir::BoolAttr>();
}

bool PrefetchOp::isWrite() {
  auto attr = isWriteAttr();
  return attr.getValue();
}

::mlir::IntegerAttr PrefetchOp::localityHintAttr() {
  return this->getAttr("localityHint").cast<::mlir::IntegerAttr>();
}

uint32_t PrefetchOp::localityHint() {
  auto attr = localityHintAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr PrefetchOp::isDataCacheAttr() {
  return this->getAttr("isDataCache").cast<::mlir::BoolAttr>();
}

bool PrefetchOp::isDataCache() {
  auto attr = isDataCacheAttr();
  return attr.getValue();
}

void PrefetchOp::isWriteAttr(::mlir::BoolAttr attr) {
  this->getOperation()->setAttr("isWrite", attr);
}

void PrefetchOp::localityHintAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("localityHint", attr);
}

void PrefetchOp::isDataCacheAttr(::mlir::BoolAttr attr) {
  this->getOperation()->setAttr("isDataCache", attr);
}

void PrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", isWrite);
  odsState.addAttribute("localityHint", localityHint);
  odsState.addAttribute("isDataCache", isDataCache);
}

void PrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", isWrite);
  odsState.addAttribute("localityHint", localityHint);
  odsState.addAttribute("isDataCache", isDataCache);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", odsBuilder.getBoolAttr(isWrite));
  odsState.addAttribute("localityHint", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), localityHint));
  odsState.addAttribute("isDataCache", odsBuilder.getBoolAttr(isDataCache));
}

void PrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", odsBuilder.getBoolAttr(isWrite));
  odsState.addAttribute("localityHint", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), localityHint));
  odsState.addAttribute("isDataCache", odsBuilder.getBoolAttr(isDataCache));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrefetchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult PrefetchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parsePrefetchOp(parser, result);
}

void PrefetchOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult PrefetchOp::verify() {
  if (failed(PrefetchOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
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
  }
  return ::verify(*this);
}




//===----------------------------------------------------------------------===//
// RankOp definitions
//===----------------------------------------------------------------------===//

RankOpAdaptor::RankOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

RankOpAdaptor::RankOpAdaptor(RankOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> RankOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange RankOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RankOpAdaptor::memrefOrTensor() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult RankOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef RankOp::getOperationName() {
  return "std.rank";
}

std::pair<unsigned, unsigned> RankOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RankOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RankOp::memrefOrTensor() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange RankOp::memrefOrTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> RankOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RankOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value tensor) {
      auto indexType = odsBuilder.getIndexType();
      build(odsBuilder, odsState, indexType, tensor);
    
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value memrefOrTensor) {
  odsState.addOperands(memrefOrTensor);
  odsState.addTypes(resultType0);
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memrefOrTensor) {
  odsState.addOperands(memrefOrTensor);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RankOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RankOp::verify() {
  if (failed(RankOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((((v.getType().isa<::mlir::UnrankedMemRefType>())) && ((true))) || (((v.getType().isa<::mlir::MemRefType>())) && ((true)))) || (((v.getType().isa<::mlir::TensorType>())) && ((true))))) {
        return emitOpError("operand #") << index << " must be any tensor or memref type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::IndexType>()))) {
        return emitOpError("result #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}



::mlir::ParseResult RankOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType memrefOrTensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOrTensorOperands(memrefOrTensorRawOperands);  ::llvm::SMLoc memrefOrTensorOperandsLoc;
  (void)memrefOrTensorOperandsLoc;
  ::mlir::Type memrefOrTensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefOrTensorTypes(memrefOrTensorRawTypes);

  memrefOrTensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefOrTensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefOrTensorRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(memrefOrTensorOperands, memrefOrTensorTypes, memrefOrTensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RankOp::print(::mlir::OpAsmPrinter &p) {
  p << "rank";
  p << " ";
  p << memrefOrTensor();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(memrefOrTensor().getType());
}

void RankOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ReOp definitions
//===----------------------------------------------------------------------===//

ReOpAdaptor::ReOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ReOpAdaptor::ReOpAdaptor(ReOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ReOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReOpAdaptor::complex() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult ReOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ReOp::getOperationName() {
  return "std.re";
}

std::pair<unsigned, unsigned> ReOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReOp::complex() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ReOp::complexMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReOp::real() {
  return *getODSResults(0).begin();
}

void ReOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type real, ::mlir::Value complex) {
  odsState.addOperands(complex);
  odsState.addTypes(real);
}

void ReOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value complex) {
  odsState.addOperands(complex);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReOp::verify() {
  if (failed(ReOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isa<::mlir::FloatType>())))) {
        return emitOpError("operand #") << index << " must be complex type with floating-point elements, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::FloatType>()))) {
        return emitOpError("result #") << index << " must be floating-point, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSOperands(0).begin()).getType().cast<ComplexType>().getElementType() == (*this->getODSResults(0).begin()).getType())))
    return emitOpError("failed to verify that complex element type matches result type");
  return ::mlir::success();
}

::mlir::ParseResult ReOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType complexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> complexOperands(complexRawOperands);  ::llvm::SMLoc complexOperandsLoc;
  (void)complexOperandsLoc;
  ::mlir::Type complexRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> complexTypes(complexRawTypes);

  complexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(complexRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(complexRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : complexTypes) {
    (void)type;
    if (!(((type.isa<::mlir::ComplexType>())) && ((type.cast<::mlir::ComplexType>().getElementType().isa<::mlir::FloatType>())))) {
      return parser.emitError(parser.getNameLoc()) << "'complex' must be complex type with floating-point elements, but got " << type;
    }
  }
  result.addTypes(complexTypes[0].cast<ComplexType>().getElementType());
  if (parser.resolveOperands(complexOperands, complexTypes, complexOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReOp::print(::mlir::OpAsmPrinter &p) {
  p << "re";
  p << " ";
  p << complex();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(complex().getType());
}

void ReOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// RemFOp definitions
//===----------------------------------------------------------------------===//

RemFOpAdaptor::RemFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

RemFOpAdaptor::RemFOpAdaptor(RemFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> RemFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange RemFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RemFOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value RemFOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult RemFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef RemFOp::getOperationName() {
  return "std.remf";
}

std::pair<unsigned, unsigned> RemFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RemFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RemFOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value RemFOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange RemFOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange RemFOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> RemFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RemFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void RemFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void RemFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RemFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void RemFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void RemFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult RemFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void RemFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult RemFOp::verify() {
  if (failed(RemFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void RemFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpAdaptor::ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ReturnOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ReturnOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ReturnOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ReturnOp::getOperationName() {
  return "std.return";
}

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReturnOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 build(odsBuilder, odsState, llvm::None); 
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verify() {
  if (failed(ReturnOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
  return ::verify(*this);
}

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void ReturnOp::print(::mlir::OpAsmPrinter &p) {
  p << "return";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
  p << " ";
  p << operands();
  p << " " << ":";
  p << " ";
  p << operands().getTypes();
  }
}

void ReturnOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// RsqrtOp definitions
//===----------------------------------------------------------------------===//

RsqrtOpAdaptor::RsqrtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

RsqrtOpAdaptor::RsqrtOpAdaptor(RsqrtOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> RsqrtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange RsqrtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RsqrtOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult RsqrtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef RsqrtOp::getOperationName() {
  return "std.rsqrt";
}

std::pair<unsigned, unsigned> RsqrtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RsqrtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RsqrtOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange RsqrtOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> RsqrtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RsqrtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void RsqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void RsqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RsqrtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void RsqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void RsqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult RsqrtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void RsqrtOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult RsqrtOp::verify() {
  if (failed(RsqrtOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void RsqrtOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SIToFPOp definitions
//===----------------------------------------------------------------------===//

SIToFPOpAdaptor::SIToFPOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SIToFPOpAdaptor::SIToFPOpAdaptor(SIToFPOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SIToFPOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SIToFPOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SIToFPOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult SIToFPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SIToFPOp::getOperationName() {
  return "std.sitofp";
}

std::pair<unsigned, unsigned> SIToFPOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SIToFPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SIToFPOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SIToFPOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SIToFPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SIToFPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Type destType) {
       impl::buildCastOp(odsBuilder, odsState, source, destType);
  
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SIToFPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult SIToFPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void SIToFPOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult SIToFPOp::verify() {
  if (failed(SIToFPOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}

void SIToFPOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SelectOp definitions
//===----------------------------------------------------------------------===//

SelectOpAdaptor::SelectOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SelectOpAdaptor::SelectOpAdaptor(SelectOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SelectOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SelectOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOpAdaptor::condition() {
  return *getODSOperands(0).begin();
}

::mlir::Value SelectOpAdaptor::true_value() {
  return *getODSOperands(1).begin();
}

::mlir::Value SelectOpAdaptor::false_value() {
  return *getODSOperands(2).begin();
}

::mlir::LogicalResult SelectOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SelectOp::getOperationName() {
  return "std.select";
}

std::pair<unsigned, unsigned> SelectOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SelectOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOp::condition() {
  return *getODSOperands(0).begin();
}

::mlir::Value SelectOp::true_value() {
  return *getODSOperands(1).begin();
}

::mlir::Value SelectOp::false_value() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange SelectOp::conditionMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SelectOp::true_valueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SelectOp::false_valueMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SelectOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SelectOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOp::result() {
  return *getODSResults(0).begin();
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Value trueValue, Value falseValue) {
      odsState.addOperands({condition, trueValue, falseValue});
      odsState.addTypes(trueValue.getType());
  
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value condition, ::mlir::Value true_value, ::mlir::Value false_value) {
  odsState.addOperands(condition);
  odsState.addOperands(true_value);
  odsState.addOperands(false_value);
  odsState.addTypes(result);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::Value true_value, ::mlir::Value false_value) {
  odsState.addOperands(condition);
  odsState.addOperands(true_value);
  odsState.addOperands(false_value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult SelectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseSelectOp(parser, result);
}

void SelectOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult SelectOp::verify() {
  if (failed(SelectOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger(1))) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1)))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(1)))))) {
        return emitOpError("operand #") << index << " must be bool-like, but got " << v.getType();
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {true_value, false_value, result} have same type");
  return ::verify(*this);
}



void SelectOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ShiftLeftOp definitions
//===----------------------------------------------------------------------===//

ShiftLeftOpAdaptor::ShiftLeftOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ShiftLeftOpAdaptor::ShiftLeftOpAdaptor(ShiftLeftOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ShiftLeftOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ShiftLeftOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShiftLeftOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShiftLeftOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult ShiftLeftOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ShiftLeftOp::getOperationName() {
  return "std.shift_left";
}

std::pair<unsigned, unsigned> ShiftLeftOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShiftLeftOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShiftLeftOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShiftLeftOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ShiftLeftOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ShiftLeftOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ShiftLeftOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShiftLeftOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ShiftLeftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void ShiftLeftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShiftLeftOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ShiftLeftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void ShiftLeftOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult ShiftLeftOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void ShiftLeftOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult ShiftLeftOp::verify() {
  if (failed(ShiftLeftOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void ShiftLeftOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SignExtendIOp definitions
//===----------------------------------------------------------------------===//

SignExtendIOpAdaptor::SignExtendIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SignExtendIOpAdaptor::SignExtendIOpAdaptor(SignExtendIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SignExtendIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SignExtendIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SignExtendIOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult SignExtendIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SignExtendIOp::getOperationName() {
  return "std.sexti";
}

std::pair<unsigned, unsigned> SignExtendIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SignExtendIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SignExtendIOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SignExtendIOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SignExtendIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SignExtendIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SignExtendIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value, Type destType) {
      odsState.addOperands(value);
      odsState.addTypes(destType);
  
}

void SignExtendIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(resultType0);
}

void SignExtendIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SignExtendIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult SignExtendIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void SignExtendIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult SignExtendIOp::verify() {
  if (failed(SignExtendIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("result #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

void SignExtendIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SignedDivIOp definitions
//===----------------------------------------------------------------------===//

SignedDivIOpAdaptor::SignedDivIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SignedDivIOpAdaptor::SignedDivIOpAdaptor(SignedDivIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SignedDivIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SignedDivIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SignedDivIOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SignedDivIOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult SignedDivIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SignedDivIOp::getOperationName() {
  return "std.divi_signed";
}

std::pair<unsigned, unsigned> SignedDivIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SignedDivIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SignedDivIOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SignedDivIOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SignedDivIOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SignedDivIOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SignedDivIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SignedDivIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SignedDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SignedDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SignedDivIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SignedDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SignedDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult SignedDivIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SignedDivIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult SignedDivIOp::verify() {
  if (failed(SignedDivIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void SignedDivIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SignedRemIOp definitions
//===----------------------------------------------------------------------===//

SignedRemIOpAdaptor::SignedRemIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SignedRemIOpAdaptor::SignedRemIOpAdaptor(SignedRemIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SignedRemIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SignedRemIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SignedRemIOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SignedRemIOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult SignedRemIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SignedRemIOp::getOperationName() {
  return "std.remi_signed";
}

std::pair<unsigned, unsigned> SignedRemIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SignedRemIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SignedRemIOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SignedRemIOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SignedRemIOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SignedRemIOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SignedRemIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SignedRemIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SignedRemIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SignedRemIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SignedRemIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SignedRemIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SignedRemIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult SignedRemIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SignedRemIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult SignedRemIOp::verify() {
  if (failed(SignedRemIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void SignedRemIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SignedShiftRightOp definitions
//===----------------------------------------------------------------------===//

SignedShiftRightOpAdaptor::SignedShiftRightOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SignedShiftRightOpAdaptor::SignedShiftRightOpAdaptor(SignedShiftRightOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SignedShiftRightOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SignedShiftRightOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SignedShiftRightOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SignedShiftRightOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult SignedShiftRightOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SignedShiftRightOp::getOperationName() {
  return "std.shift_right_signed";
}

std::pair<unsigned, unsigned> SignedShiftRightOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SignedShiftRightOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SignedShiftRightOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SignedShiftRightOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SignedShiftRightOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SignedShiftRightOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SignedShiftRightOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SignedShiftRightOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SignedShiftRightOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SignedShiftRightOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SignedShiftRightOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SignedShiftRightOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SignedShiftRightOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult SignedShiftRightOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SignedShiftRightOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult SignedShiftRightOp::verify() {
  if (failed(SignedShiftRightOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void SignedShiftRightOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SinOp definitions
//===----------------------------------------------------------------------===//

SinOpAdaptor::SinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SinOpAdaptor::SinOpAdaptor(SinOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult SinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SinOp::getOperationName() {
  return "std.sin";
}

std::pair<unsigned, unsigned> SinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SinOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult SinOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SinOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult SinOp::verify() {
  if (failed(SinOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void SinOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SplatOp definitions
//===----------------------------------------------------------------------===//

SplatOpAdaptor::SplatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SplatOpAdaptor::SplatOpAdaptor(SplatOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SplatOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SplatOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SplatOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult SplatOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SplatOp::getOperationName() {
  return "std.splat";
}

std::pair<unsigned, unsigned> SplatOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SplatOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SplatOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SplatOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SplatOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SplatOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SplatOp::aggregate() {
  return *getODSResults(0).begin();
}

void SplatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value element, Type aggregateType) {
 build(odsBuilder, odsState, aggregateType, element); 
}

void SplatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type aggregate, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(aggregate);
}

void SplatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SplatOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SplatOp::verify() {
  if (failed(SplatOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::FloatType>())))) {
        return emitOpError("operand #") << index << " must be integer or float type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::VectorType>())) && ((true))) || ((((v.getType().isa<::mlir::TensorType>())) && ((true))) && ((v.getType().cast<::mlir::ShapedType>().hasStaticShape()))))) {
        return emitOpError("result #") << index << " must be vector of any type values or statically shaped tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSResults(0).begin()).getType().cast<ShapedType>().getElementType() == (*this->getODSOperands(0).begin()).getType())))
    return emitOpError("failed to verify that operand type matches element type of result");
  return ::verify(*this);
}



::mlir::ParseResult SplatOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type aggregateRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> aggregateTypes(aggregateRawTypes);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(aggregateRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : aggregateTypes) {
    (void)type;
    if (!((((type.isa<::mlir::VectorType>())) && ((true))) || ((((type.isa<::mlir::TensorType>())) && ((true))) && ((type.cast<::mlir::ShapedType>().hasStaticShape()))))) {
      return parser.emitError(parser.getNameLoc()) << "'aggregate' must be vector of any type values or statically shaped tensor of any type values, but got " << type;
    }
  }
  result.addTypes(aggregateTypes);
  if (parser.resolveOperands(inputOperands, aggregateTypes[0].cast<ShapedType>().getElementType(), inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SplatOp::print(::mlir::OpAsmPrinter &p) {
  p << "splat";
  p << " ";
  p << input();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(aggregate().getType());
}

void SplatOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SqrtOp definitions
//===----------------------------------------------------------------------===//

SqrtOpAdaptor::SqrtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SqrtOpAdaptor::SqrtOpAdaptor(SqrtOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SqrtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SqrtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SqrtOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult SqrtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SqrtOp::getOperationName() {
  return "std.sqrt";
}

std::pair<unsigned, unsigned> SqrtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SqrtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SqrtOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SqrtOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SqrtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SqrtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SqrtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult SqrtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SqrtOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult SqrtOp::verify() {
  if (failed(SqrtOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void SqrtOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// StoreOp definitions
//===----------------------------------------------------------------------===//

StoreOpAdaptor::StoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

StoreOpAdaptor::StoreOpAdaptor(StoreOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> StoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange StoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOpAdaptor::memref() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange StoreOpAdaptor::indices() {
  return getODSOperands(2);
}

::mlir::LogicalResult StoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef StoreOp::getOperationName() {
  return "std.store";
}

std::pair<unsigned, unsigned> StoreOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range StoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOp::memref() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range StoreOp::indices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange StoreOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange StoreOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange StoreOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> StoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref) {
      odsState.addOperands(valueToStore);
      odsState.addOperands(memref);
  
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StoreOp::verify() {
  if (failed(StoreOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<::mlir::IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  if (!(((*this->getODSOperands(1).begin()).getType().cast<MemRefType>().getElementType() == (*this->getODSOperands(0).begin()).getType())))
    return emitOpError("failed to verify that type of 'value' matches element type of 'memref'");
  return ::verify(*this);
}



::mlir::ParseResult StoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::OperandType memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : memrefTypes) {
    (void)type;
    if (!(((type.isa<::mlir::MemRefType>())) && ((true)))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be memref of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(valueOperands, memrefTypes[0].cast<MemRefType>().getElementType(), valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StoreOp::print(::mlir::OpAsmPrinter &p) {
  p << "store";
  p << " ";
  p << value();
  p << ",";
  p << " ";
  p << memref();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(memref().getType());
}

void StoreOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// SubCFOp definitions
//===----------------------------------------------------------------------===//

SubCFOpAdaptor::SubCFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SubCFOpAdaptor::SubCFOpAdaptor(SubCFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SubCFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubCFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubCFOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubCFOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult SubCFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SubCFOp::getOperationName() {
  return "std.subcf";
}

std::pair<unsigned, unsigned> SubCFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubCFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubCFOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubCFOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SubCFOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SubCFOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SubCFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubCFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SubCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SubCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubCFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SubCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SubCFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult SubCFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SubCFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult SubCFOp::verify() {
  if (failed(SubCFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isa<::mlir::FloatType>())))) {
        return emitOpError("operand #") << index << " must be complex type with floating-point elements, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::ComplexType>())) && ((v.getType().cast<::mlir::ComplexType>().getElementType().isa<::mlir::FloatType>())))) {
        return emitOpError("operand #") << index << " must be complex type with floating-point elements, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void SubCFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SubFOp definitions
//===----------------------------------------------------------------------===//

SubFOpAdaptor::SubFOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SubFOpAdaptor::SubFOpAdaptor(SubFOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SubFOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubFOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubFOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubFOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult SubFOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SubFOp::getOperationName() {
  return "std.subf";
}

std::pair<unsigned, unsigned> SubFOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubFOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubFOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubFOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SubFOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SubFOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SubFOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubFOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SubFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SubFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubFOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SubFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SubFOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult SubFOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SubFOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult SubFOp::verify() {
  if (failed(SubFOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void SubFOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SubIOp definitions
//===----------------------------------------------------------------------===//

SubIOpAdaptor::SubIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SubIOpAdaptor::SubIOpAdaptor(SubIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SubIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubIOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubIOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult SubIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef SubIOp::getOperationName() {
  return "std.subi";
}

std::pair<unsigned, unsigned> SubIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubIOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubIOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SubIOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange SubIOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> SubIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void SubIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SubIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SubIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SubIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult SubIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SubIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult SubIOp::verify() {
  if (failed(SubIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void SubIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SubTensorInsertOp definitions
//===----------------------------------------------------------------------===//

SubTensorInsertOpAdaptor::SubTensorInsertOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SubTensorInsertOpAdaptor::SubTensorInsertOpAdaptor(SubTensorInsertOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SubTensorInsertOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange SubTensorInsertOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubTensorInsertOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubTensorInsertOpAdaptor::dest() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange SubTensorInsertOpAdaptor::offsets() {
  return getODSOperands(2);
}

::mlir::ValueRange SubTensorInsertOpAdaptor::sizes() {
  return getODSOperands(3);
}

::mlir::ValueRange SubTensorInsertOpAdaptor::strides() {
  return getODSOperands(4);
}

::mlir::ArrayAttr SubTensorInsertOpAdaptor::static_offsets() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_offsets").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SubTensorInsertOpAdaptor::static_sizes() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SubTensorInsertOpAdaptor::static_strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_strides").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult SubTensorInsertOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 5)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 5 elements");
  }
    {
  auto tblgen_static_offsets = odsAttrs.get("static_offsets");
  if (!tblgen_static_offsets) return emitError(loc, "'std.subtensor_insert' op ""requires attribute 'static_offsets'");
    if (!(((tblgen_static_offsets.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_offsets.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'std.subtensor_insert' op ""attribute 'static_offsets' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_static_sizes = odsAttrs.get("static_sizes");
  if (!tblgen_static_sizes) return emitError(loc, "'std.subtensor_insert' op ""requires attribute 'static_sizes'");
    if (!(((tblgen_static_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_sizes.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'std.subtensor_insert' op ""attribute 'static_sizes' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_static_strides = odsAttrs.get("static_strides");
  if (!tblgen_static_strides) return emitError(loc, "'std.subtensor_insert' op ""requires attribute 'static_strides'");
    if (!(((tblgen_static_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_strides.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'std.subtensor_insert' op ""attribute 'static_strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef SubTensorInsertOp::getOperationName() {
  return "std.subtensor_insert";
}

std::pair<unsigned, unsigned> SubTensorInsertOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range SubTensorInsertOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubTensorInsertOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubTensorInsertOp::dest() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range SubTensorInsertOp::offsets() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range SubTensorInsertOp::sizes() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range SubTensorInsertOp::strides() {
  return getODSOperands(4);
}

::mlir::MutableOperandRange SubTensorInsertOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SubTensorInsertOp::destMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SubTensorInsertOp::offsetsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SubTensorInsertOp::sizesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SubTensorInsertOp::stridesMutable() {
  auto range = getODSOperandIndexAndLength(4);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> SubTensorInsertOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubTensorInsertOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubTensorInsertOp::result() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr SubTensorInsertOp::static_offsetsAttr() {
  return this->getAttr("static_offsets").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SubTensorInsertOp::static_offsets() {
  auto attr = static_offsetsAttr();
  return attr;
}

::mlir::ArrayAttr SubTensorInsertOp::static_sizesAttr() {
  return this->getAttr("static_sizes").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SubTensorInsertOp::static_sizes() {
  auto attr = static_sizesAttr();
  return attr;
}

::mlir::ArrayAttr SubTensorInsertOp::static_stridesAttr() {
  return this->getAttr("static_strides").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SubTensorInsertOp::static_strides() {
  auto attr = static_stridesAttr();
  return attr;
}

void SubTensorInsertOp::static_offsetsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("static_offsets", attr);
}

void SubTensorInsertOp::static_sizesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("static_sizes", attr);
}

void SubTensorInsertOp::static_stridesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("static_strides", attr);
}





void SubTensorInsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute("static_offsets", static_offsets);
  odsState.addAttribute("static_sizes", static_sizes);
  odsState.addAttribute("static_strides", static_strides);
  odsState.addTypes(result);
}

void SubTensorInsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute("static_offsets", static_offsets);
  odsState.addAttribute("static_sizes", static_sizes);
  odsState.addAttribute("static_strides", static_strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubTensorInsertOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult SubTensorInsertOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseSubTensorInsertOp(parser, result);
}

void SubTensorInsertOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult SubTensorInsertOp::verify() {
  if (failed(SubTensorInsertOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<::mlir::IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
      (void)v;
      if (!((v.getType().isa<::mlir::IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup4 = getODSOperands(4);
    for (::mlir::Value v : valueGroup4) {
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
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

void SubTensorInsertOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SubTensorOp definitions
//===----------------------------------------------------------------------===//

SubTensorOpAdaptor::SubTensorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SubTensorOpAdaptor::SubTensorOpAdaptor(SubTensorOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SubTensorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange SubTensorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubTensorOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange SubTensorOpAdaptor::offsets() {
  return getODSOperands(1);
}

::mlir::ValueRange SubTensorOpAdaptor::sizes() {
  return getODSOperands(2);
}

::mlir::ValueRange SubTensorOpAdaptor::strides() {
  return getODSOperands(3);
}

::mlir::ArrayAttr SubTensorOpAdaptor::static_offsets() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_offsets").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SubTensorOpAdaptor::static_sizes() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SubTensorOpAdaptor::static_strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_strides").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult SubTensorOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 4 elements");
  }
    {
  auto tblgen_static_offsets = odsAttrs.get("static_offsets");
  if (!tblgen_static_offsets) return emitError(loc, "'std.subtensor' op ""requires attribute 'static_offsets'");
    if (!(((tblgen_static_offsets.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_offsets.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'std.subtensor' op ""attribute 'static_offsets' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_static_sizes = odsAttrs.get("static_sizes");
  if (!tblgen_static_sizes) return emitError(loc, "'std.subtensor' op ""requires attribute 'static_sizes'");
    if (!(((tblgen_static_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_sizes.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'std.subtensor' op ""attribute 'static_sizes' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_static_strides = odsAttrs.get("static_strides");
  if (!tblgen_static_strides) return emitError(loc, "'std.subtensor' op ""requires attribute 'static_strides'");
    if (!(((tblgen_static_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_strides.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'std.subtensor' op ""attribute 'static_strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef SubTensorOp::getOperationName() {
  return "std.subtensor";
}

std::pair<unsigned, unsigned> SubTensorOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range SubTensorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubTensorOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range SubTensorOp::offsets() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range SubTensorOp::sizes() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range SubTensorOp::strides() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange SubTensorOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SubTensorOp::offsetsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SubTensorOp::sizesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SubTensorOp::stridesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> SubTensorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubTensorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubTensorOp::result() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr SubTensorOp::static_offsetsAttr() {
  return this->getAttr("static_offsets").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SubTensorOp::static_offsets() {
  auto attr = static_offsetsAttr();
  return attr;
}

::mlir::ArrayAttr SubTensorOp::static_sizesAttr() {
  return this->getAttr("static_sizes").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SubTensorOp::static_sizes() {
  auto attr = static_sizesAttr();
  return attr;
}

::mlir::ArrayAttr SubTensorOp::static_stridesAttr() {
  return this->getAttr("static_strides").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SubTensorOp::static_strides() {
  auto attr = static_stridesAttr();
  return attr;
}

void SubTensorOp::static_offsetsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("static_offsets", attr);
}

void SubTensorOp::static_sizesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("static_sizes", attr);
}

void SubTensorOp::static_stridesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("static_strides", attr);
}





void SubTensorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute("static_offsets", static_offsets);
  odsState.addAttribute("static_sizes", static_sizes);
  odsState.addAttribute("static_strides", static_strides);
  odsState.addTypes(result);
}

void SubTensorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute("static_offsets", static_offsets);
  odsState.addAttribute("static_sizes", static_sizes);
  odsState.addAttribute("static_strides", static_strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubTensorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult SubTensorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseSubTensorOp(parser, result);
}

void SubTensorOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult SubTensorOp::verify() {
  if (failed(SubTensorOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("operand #") << index << " must be ranked tensor of any type values, but got " << v.getType();
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
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<::mlir::IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
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
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().hasRank()))) && ((true)))) {
        return emitOpError("result #") << index << " must be ranked tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}



void SubTensorOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SubViewOp definitions
//===----------------------------------------------------------------------===//

SubViewOpAdaptor::SubViewOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

SubViewOpAdaptor::SubViewOpAdaptor(SubViewOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> SubViewOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange SubViewOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubViewOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange SubViewOpAdaptor::offsets() {
  return getODSOperands(1);
}

::mlir::ValueRange SubViewOpAdaptor::sizes() {
  return getODSOperands(2);
}

::mlir::ValueRange SubViewOpAdaptor::strides() {
  return getODSOperands(3);
}

::mlir::ArrayAttr SubViewOpAdaptor::static_offsets() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_offsets").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SubViewOpAdaptor::static_sizes() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SubViewOpAdaptor::static_strides() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("static_strides").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::LogicalResult SubViewOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 4 elements");
  }
    {
  auto tblgen_static_offsets = odsAttrs.get("static_offsets");
  if (!tblgen_static_offsets) return emitError(loc, "'std.subview' op ""requires attribute 'static_offsets'");
    if (!(((tblgen_static_offsets.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_offsets.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'std.subview' op ""attribute 'static_offsets' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_static_sizes = odsAttrs.get("static_sizes");
  if (!tblgen_static_sizes) return emitError(loc, "'std.subview' op ""requires attribute 'static_sizes'");
    if (!(((tblgen_static_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_sizes.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'std.subview' op ""attribute 'static_sizes' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
  auto tblgen_static_strides = odsAttrs.get("static_strides");
  if (!tblgen_static_strides) return emitError(loc, "'std.subview' op ""requires attribute 'static_strides'");
    if (!(((tblgen_static_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_strides.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) return emitError(loc, "'std.subview' op ""attribute 'static_strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef SubViewOp::getOperationName() {
  return "std.subview";
}

std::pair<unsigned, unsigned> SubViewOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range SubViewOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubViewOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range SubViewOp::offsets() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range SubViewOp::sizes() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range SubViewOp::strides() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange SubViewOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SubViewOp::offsetsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SubViewOp::sizesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange SubViewOp::stridesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> SubViewOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubViewOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubViewOp::result() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr SubViewOp::static_offsetsAttr() {
  return this->getAttr("static_offsets").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SubViewOp::static_offsets() {
  auto attr = static_offsetsAttr();
  return attr;
}

::mlir::ArrayAttr SubViewOp::static_sizesAttr() {
  return this->getAttr("static_sizes").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SubViewOp::static_sizes() {
  auto attr = static_sizesAttr();
  return attr;
}

::mlir::ArrayAttr SubViewOp::static_stridesAttr() {
  return this->getAttr("static_strides").cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SubViewOp::static_strides() {
  auto attr = static_stridesAttr();
  return attr;
}

void SubViewOp::static_offsetsAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("static_offsets", attr);
}

void SubViewOp::static_sizesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("static_sizes", attr);
}

void SubViewOp::static_stridesAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("static_strides", attr);
}









void SubViewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute("static_offsets", static_offsets);
  odsState.addAttribute("static_sizes", static_sizes);
  odsState.addAttribute("static_strides", static_strides);
  odsState.addTypes(result);
}

void SubViewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute("static_offsets", static_offsets);
  odsState.addAttribute("static_sizes", static_sizes);
  odsState.addAttribute("static_strides", static_strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubViewOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult SubViewOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseSubViewOp(parser, result);
}

void SubViewOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult SubViewOp::verify() {
  if (failed(SubViewOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
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
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<::mlir::IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (::mlir::Value v : valueGroup3) {
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
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}







void SubViewOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// TanhOp definitions
//===----------------------------------------------------------------------===//

TanhOpAdaptor::TanhOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TanhOpAdaptor::TanhOpAdaptor(TanhOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TanhOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TanhOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TanhOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult TanhOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef TanhOp::getOperationName() {
  return "std.tanh";
}

std::pair<unsigned, unsigned> TanhOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TanhOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TanhOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TanhOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TanhOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TanhOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void TanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TanhOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void TanhOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult TanhOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void TanhOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

::mlir::LogicalResult TanhOp::verify() {
  if (failed(TanhOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::FloatType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isa<::mlir::FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void TanhOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// TensorCastOp definitions
//===----------------------------------------------------------------------===//

TensorCastOpAdaptor::TensorCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TensorCastOpAdaptor::TensorCastOpAdaptor(TensorCastOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TensorCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TensorCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorCastOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult TensorCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef TensorCastOp::getOperationName() {
  return "std.tensor_cast";
}

std::pair<unsigned, unsigned> TensorCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TensorCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorCastOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TensorCastOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TensorCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TensorCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TensorCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Type destType) {
       impl::buildCastOp(odsBuilder, odsState, source, destType);
  
}

void TensorCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(resultType0);
}

void TensorCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult TensorCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void TensorCastOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult TensorCastOp::verify() {
  if (failed(TensorCastOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be tensor of any type values, but got " << v.getType();
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
  }
  return ::verifyCastOp(*this);
}





void TensorCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// TensorFromElementsOp definitions
//===----------------------------------------------------------------------===//

TensorFromElementsOpAdaptor::TensorFromElementsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TensorFromElementsOpAdaptor::TensorFromElementsOpAdaptor(TensorFromElementsOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TensorFromElementsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange TensorFromElementsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange TensorFromElementsOpAdaptor::elements() {
  return getODSOperands(0);
}

::mlir::LogicalResult TensorFromElementsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef TensorFromElementsOp::getOperationName() {
  return "std.tensor_from_elements";
}

std::pair<unsigned, unsigned> TensorFromElementsOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range TensorFromElementsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range TensorFromElementsOp::elements() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange TensorFromElementsOp::elementsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TensorFromElementsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TensorFromElementsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorFromElementsOp::result() {
  return *getODSResults(0).begin();
}





::mlir::LogicalResult TensorFromElementsOp::verify() {
  if (failed(TensorFromElementsOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
      if (!((((v.getType().isa<::mlir::TensorType>())) && ((true))) && (((v.getType().cast<::mlir::ShapedType>().hasRank())) && ((v.getType().cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
        return emitOpError("result #") << index << " must be 1D tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((SmallVector<Type, 2>((*this->getODSResults(0).begin()).getType().cast<ShapedType>().getDimSize(0), (*this->getODSResults(0).begin()).getType().cast<ShapedType>().getElementType()) == this->getODSOperands(0).getType())))
    return emitOpError("failed to verify that operand types match result element type");
  return ::mlir::success();
}



::mlir::ParseResult TensorFromElementsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> elementsOperands;
  ::llvm::SMLoc elementsOperandsLoc;
  (void)elementsOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  elementsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(elementsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!((((type.isa<::mlir::TensorType>())) && ((true))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be 1D tensor of any type values, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(elementsOperands, SmallVector<Type, 2>(resultTypes[0].cast<ShapedType>().getDimSize(0), resultTypes[0].cast<ShapedType>().getElementType()), elementsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TensorFromElementsOp::print(::mlir::OpAsmPrinter &p) {
  p << "tensor_from_elements";
  p << " ";
  p << elements();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(result().getType());
}

void TensorFromElementsOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// TensorLoadOp definitions
//===----------------------------------------------------------------------===//

TensorLoadOpAdaptor::TensorLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TensorLoadOpAdaptor::TensorLoadOpAdaptor(TensorLoadOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TensorLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TensorLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorLoadOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult TensorLoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef TensorLoadOp::getOperationName() {
  return "std.tensor_load";
}

std::pair<unsigned, unsigned> TensorLoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TensorLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorLoadOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TensorLoadOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TensorLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TensorLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorLoadOp::result() {
  return *getODSResults(0).begin();
}

void TensorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref) {
      odsState.addOperands(memref);
      odsState.addTypes(getTensorTypeFromMemRefType(memref.getType()));
  
}

void TensorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref) {
  odsState.addOperands(memref);
  odsState.addTypes(result);
}

void TensorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref) {
  odsState.addOperands(memref);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TensorLoadOp::verify() {
  if (failed(TensorLoadOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::UnrankedMemRefType>())) && ((true))) || (((v.getType().isa<::mlir::MemRefType>())) && ((true))))) {
        return emitOpError("operand #") << index << " must be unranked.memref of any type values or memref of any type values, but got " << v.getType();
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
  }
  if (!((getTensorTypeFromMemRefType((*this->getODSOperands(0).begin()).getType()) == (*this->getODSResults(0).begin()).getType())))
    return emitOpError("failed to verify that result type matches tensor equivalent of 'memref'");
  return ::mlir::success();
}



::mlir::ParseResult TensorLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : memrefTypes) {
    (void)type;
    if (!((((type.isa<::mlir::UnrankedMemRefType>())) && ((true))) || (((type.isa<::mlir::MemRefType>())) && ((true))))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be unranked.memref of any type values or memref of any type values, but got " << type;
    }
  }
  result.addTypes(getTensorTypeFromMemRefType(memrefTypes[0]));
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TensorLoadOp::print(::mlir::OpAsmPrinter &p) {
  p << "tensor_load";
  p << " ";
  p << memref();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(memref().getType());
}

void TensorLoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// TensorStoreOp definitions
//===----------------------------------------------------------------------===//

TensorStoreOpAdaptor::TensorStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TensorStoreOpAdaptor::TensorStoreOpAdaptor(TensorStoreOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TensorStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TensorStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorStoreOpAdaptor::tensor() {
  return *getODSOperands(0).begin();
}

::mlir::Value TensorStoreOpAdaptor::memref() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult TensorStoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef TensorStoreOp::getOperationName() {
  return "std.tensor_store";
}

std::pair<unsigned, unsigned> TensorStoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TensorStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorStoreOp::tensor() {
  return *getODSOperands(0).begin();
}

::mlir::Value TensorStoreOp::memref() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange TensorStoreOp::tensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange TensorStoreOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TensorStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TensorStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TensorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::mlir::Value memref) {
  odsState.addOperands(tensor);
  odsState.addOperands(memref);
}

void TensorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::Value memref) {
  odsState.addOperands(tensor);
  odsState.addOperands(memref);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TensorStoreOp::verify() {
  if (failed(TensorStoreOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!((((v.getType().isa<::mlir::UnrankedMemRefType>())) && ((true))) || (((v.getType().isa<::mlir::MemRefType>())) && ((true))))) {
        return emitOpError("operand #") << index << " must be unranked.memref of any type values or memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  if (!((getTensorTypeFromMemRefType((*this->getODSOperands(1).begin()).getType()) == (*this->getODSOperands(0).begin()).getType())))
    return emitOpError("failed to verify that type of 'value' matches tensor equivalent of 'memref'");
  return ::mlir::success();
}

::mlir::ParseResult TensorStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::OpAsmParser::OperandType memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : memrefTypes) {
    (void)type;
    if (!((((type.isa<::mlir::UnrankedMemRefType>())) && ((true))) || (((type.isa<::mlir::MemRefType>())) && ((true))))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be unranked.memref of any type values or memref of any type values, but got " << type;
    }
  }
  if (parser.resolveOperands(tensorOperands, getTensorTypeFromMemRefType(memrefTypes[0]), tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TensorStoreOp::print(::mlir::OpAsmPrinter &p) {
  p << "tensor_store";
  p << " ";
  p << tensor();
  p << ",";
  p << " ";
  p << memref();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(memref().getType());
}

void TensorStoreOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// TensorToMemrefOp definitions
//===----------------------------------------------------------------------===//

TensorToMemrefOpAdaptor::TensorToMemrefOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TensorToMemrefOpAdaptor::TensorToMemrefOpAdaptor(TensorToMemrefOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TensorToMemrefOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TensorToMemrefOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorToMemrefOpAdaptor::tensor() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult TensorToMemrefOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef TensorToMemrefOp::getOperationName() {
  return "std.tensor_to_memref";
}

std::pair<unsigned, unsigned> TensorToMemrefOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TensorToMemrefOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorToMemrefOp::tensor() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TensorToMemrefOp::tensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TensorToMemrefOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TensorToMemrefOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TensorToMemrefOp::memref() {
  return *getODSResults(0).begin();
}

void TensorToMemrefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  odsState.addTypes(memref);
}

void TensorToMemrefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorToMemrefOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TensorToMemrefOp::verify() {
  if (failed(TensorToMemrefOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::UnrankedMemRefType>())) && ((true))) || (((v.getType().isa<::mlir::MemRefType>())) && ((true))))) {
        return emitOpError("result #") << index << " must be unranked.memref of any type values or memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((getTensorTypeFromMemRefType((*this->getODSResults(0).begin()).getType()) == (*this->getODSOperands(0).begin()).getType())))
    return emitOpError("failed to verify that type of 'tensor' is the tensor equivalent of 'memref'");
  return ::mlir::success();
}



::mlir::ParseResult TensorToMemrefOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(memrefRawTypes[0]))
    return ::mlir::failure();
  for (::mlir::Type type : memrefTypes) {
    (void)type;
    if (!((((type.isa<::mlir::UnrankedMemRefType>())) && ((true))) || (((type.isa<::mlir::MemRefType>())) && ((true))))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be unranked.memref of any type values or memref of any type values, but got " << type;
    }
  }
  result.addTypes(memrefTypes);
  if (parser.resolveOperands(tensorOperands, getTensorTypeFromMemRefType(memrefTypes[0]), tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TensorToMemrefOp::print(::mlir::OpAsmPrinter &p) {
  p << "tensor_to_memref";
  p << " ";
  p << tensor();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(memref().getType());
}

void TensorToMemrefOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(MemoryEffects::Allocate::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// TransposeOp definitions
//===----------------------------------------------------------------------===//

TransposeOpAdaptor::TransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TransposeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TransposeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::AffineMapAttr TransposeOpAdaptor::permutation() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("permutation").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_permutation = odsAttrs.get("permutation");
  if (!tblgen_permutation) return emitError(loc, "'std.transpose' op ""requires attribute 'permutation'");
    if (!((tblgen_permutation.isa<::mlir::AffineMapAttr>()))) return emitError(loc, "'std.transpose' op ""attribute 'permutation' failed to satisfy constraint: AffineMap attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef TransposeOp::getOperationName() {
  return "std.transpose";
}

std::pair<unsigned, unsigned> TransposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TransposeOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TransposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::AffineMapAttr TransposeOp::permutationAttr() {
  return this->getAttr("permutation").cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap TransposeOp::permutation() {
  auto attr = permutationAttr();
  return attr.getValue();
}

void TransposeOp::permutationAttr(::mlir::AffineMapAttr attr) {
  this->getOperation()->setAttr("permutation", attr);
}



void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in, ::mlir::AffineMapAttr permutation) {
  odsState.addOperands(in);
  odsState.addAttribute("permutation", permutation);
  odsState.addTypes(resultType0);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in, ::mlir::AffineMapAttr permutation) {
  odsState.addOperands(in);
  odsState.addAttribute("permutation", permutation);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in, ::mlir::AffineMap permutation) {
  odsState.addOperands(in);
  odsState.addAttribute("permutation", ::mlir::AffineMapAttr::get(permutation));
  odsState.addTypes(resultType0);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in, ::mlir::AffineMap permutation) {
  odsState.addOperands(in);
  odsState.addAttribute("permutation", ::mlir::AffineMapAttr::get(permutation));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult TransposeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseTransposeOp(parser, result);
}

void TransposeOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult TransposeOp::verify() {
  if (failed(TransposeOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((true))) && (( isStrided(v.getType().cast<::mlir::MemRefType>()) )))) {
        return emitOpError("result #") << index << " must be strided memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}



void TransposeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// TruncateIOp definitions
//===----------------------------------------------------------------------===//

TruncateIOpAdaptor::TruncateIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

TruncateIOpAdaptor::TruncateIOpAdaptor(TruncateIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> TruncateIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TruncateIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TruncateIOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult TruncateIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef TruncateIOp::getOperationName() {
  return "std.trunci";
}

std::pair<unsigned, unsigned> TruncateIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TruncateIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TruncateIOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TruncateIOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> TruncateIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TruncateIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TruncateIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value, Type destType) {
      odsState.addOperands(value);
      odsState.addTypes(destType);
  
}

void TruncateIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(resultType0);
}

void TruncateIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TruncateIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult TruncateIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void TruncateIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult TruncateIOp::verify() {
  if (failed(TruncateIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("result #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

void TruncateIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// UIToFPOp definitions
//===----------------------------------------------------------------------===//

UIToFPOpAdaptor::UIToFPOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

UIToFPOpAdaptor::UIToFPOpAdaptor(UIToFPOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> UIToFPOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UIToFPOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UIToFPOpAdaptor::in() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult UIToFPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef UIToFPOp::getOperationName() {
  return "std.uitofp";
}

std::pair<unsigned, unsigned> UIToFPOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UIToFPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UIToFPOp::in() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange UIToFPOp::inMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> UIToFPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UIToFPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Type destType) {
       impl::buildCastOp(odsBuilder, odsState, source, destType);
  
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UIToFPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult UIToFPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void UIToFPOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult UIToFPOp::verify() {
  if (failed(UIToFPOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}

void UIToFPOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// UnsignedDivIOp definitions
//===----------------------------------------------------------------------===//

UnsignedDivIOpAdaptor::UnsignedDivIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

UnsignedDivIOpAdaptor::UnsignedDivIOpAdaptor(UnsignedDivIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> UnsignedDivIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UnsignedDivIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UnsignedDivIOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value UnsignedDivIOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult UnsignedDivIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef UnsignedDivIOp::getOperationName() {
  return "std.divi_unsigned";
}

std::pair<unsigned, unsigned> UnsignedDivIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UnsignedDivIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UnsignedDivIOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value UnsignedDivIOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange UnsignedDivIOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange UnsignedDivIOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> UnsignedDivIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UnsignedDivIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void UnsignedDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void UnsignedDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnsignedDivIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UnsignedDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void UnsignedDivIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult UnsignedDivIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void UnsignedDivIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult UnsignedDivIOp::verify() {
  if (failed(UnsignedDivIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void UnsignedDivIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// UnsignedRemIOp definitions
//===----------------------------------------------------------------------===//

UnsignedRemIOpAdaptor::UnsignedRemIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

UnsignedRemIOpAdaptor::UnsignedRemIOpAdaptor(UnsignedRemIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> UnsignedRemIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UnsignedRemIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UnsignedRemIOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value UnsignedRemIOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult UnsignedRemIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef UnsignedRemIOp::getOperationName() {
  return "std.remi_unsigned";
}

std::pair<unsigned, unsigned> UnsignedRemIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UnsignedRemIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UnsignedRemIOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value UnsignedRemIOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange UnsignedRemIOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange UnsignedRemIOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> UnsignedRemIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UnsignedRemIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void UnsignedRemIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void UnsignedRemIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnsignedRemIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UnsignedRemIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void UnsignedRemIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult UnsignedRemIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void UnsignedRemIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult UnsignedRemIOp::verify() {
  if (failed(UnsignedRemIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void UnsignedRemIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// UnsignedShiftRightOp definitions
//===----------------------------------------------------------------------===//

UnsignedShiftRightOpAdaptor::UnsignedShiftRightOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

UnsignedShiftRightOpAdaptor::UnsignedShiftRightOpAdaptor(UnsignedShiftRightOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> UnsignedShiftRightOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UnsignedShiftRightOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UnsignedShiftRightOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value UnsignedShiftRightOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult UnsignedShiftRightOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef UnsignedShiftRightOp::getOperationName() {
  return "std.shift_right_unsigned";
}

std::pair<unsigned, unsigned> UnsignedShiftRightOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UnsignedShiftRightOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UnsignedShiftRightOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value UnsignedShiftRightOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange UnsignedShiftRightOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange UnsignedShiftRightOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> UnsignedShiftRightOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UnsignedShiftRightOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void UnsignedShiftRightOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void UnsignedShiftRightOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnsignedShiftRightOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UnsignedShiftRightOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void UnsignedShiftRightOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult UnsignedShiftRightOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void UnsignedShiftRightOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult UnsignedShiftRightOp::verify() {
  if (failed(UnsignedShiftRightOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}

void UnsignedShiftRightOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ViewOp definitions
//===----------------------------------------------------------------------===//

ViewOpAdaptor::ViewOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ViewOpAdaptor::ViewOpAdaptor(ViewOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ViewOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ViewOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ViewOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value ViewOpAdaptor::byte_shift() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange ViewOpAdaptor::sizes() {
  return getODSOperands(2);
}

::mlir::LogicalResult ViewOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ViewOp::getOperationName() {
  return "std.view";
}

std::pair<unsigned, unsigned> ViewOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ViewOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ViewOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value ViewOp::byte_shift() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range ViewOp::sizes() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ViewOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ViewOp::byte_shiftMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange ViewOp::sizesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ViewOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ViewOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ViewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value source, ::mlir::Value byte_shift, ::mlir::ValueRange sizes) {
  odsState.addOperands(source);
  odsState.addOperands(byte_shift);
  odsState.addOperands(sizes);
  odsState.addTypes(resultType0);
}

void ViewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value byte_shift, ::mlir::ValueRange sizes) {
  odsState.addOperands(source);
  odsState.addOperands(byte_shift);
  odsState.addOperands(sizes);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ViewOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ViewOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ::parseViewOp(parser, result);
}

void ViewOp::print(::mlir::OpAsmPrinter &p) {
  return ::print(p, *this);
}

::mlir::LogicalResult ViewOp::verify() {
  if (failed(ViewOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<::mlir::MemRefType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger(8)))) && (((v.getType().cast<::mlir::ShapedType>().hasRank())) && ((v.getType().cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
        return emitOpError("operand #") << index << " must be 1D memref of 8-bit signless integer values, but got " << v.getType();
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
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
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
      if (!(((v.getType().isa<::mlir::MemRefType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}





void ViewOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// XOrOp definitions
//===----------------------------------------------------------------------===//

XOrOpAdaptor::XOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

XOrOpAdaptor::XOrOpAdaptor(XOrOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> XOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange XOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value XOrOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value XOrOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::LogicalResult XOrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef XOrOp::getOperationName() {
  return "std.xor";
}

std::pair<unsigned, unsigned> XOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range XOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value XOrOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value XOrOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange XOrOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange XOrOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> XOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range XOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void XOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

::mlir::ParseResult XOrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void XOrOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

::mlir::LogicalResult XOrOp::verify() {
  if (failed(XOrOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
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
  }
  return ::mlir::success();
}



void XOrOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

YieldOpAdaptor::YieldOpAdaptor(YieldOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> YieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange YieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value YieldOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef YieldOp::getOperationName() {
  return "std.yield";
}

std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value YieldOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange YieldOp::valueMutable() {
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

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value) {
  odsState.addOperands(value);
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
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
  return ::mlir::success();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &p) {
  p << "yield";
  p << " ";
  p << value();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ::llvm::ArrayRef<::mlir::Type>(value().getType());
}

void YieldOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ZeroExtendIOp definitions
//===----------------------------------------------------------------------===//

ZeroExtendIOpAdaptor::ZeroExtendIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ZeroExtendIOpAdaptor::ZeroExtendIOpAdaptor(ZeroExtendIOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ZeroExtendIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ZeroExtendIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZeroExtendIOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::LogicalResult ZeroExtendIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ZeroExtendIOp::getOperationName() {
  return "std.zexti";
}

std::pair<unsigned, unsigned> ZeroExtendIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ZeroExtendIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZeroExtendIOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ZeroExtendIOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ZeroExtendIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ZeroExtendIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ZeroExtendIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value, Type destType) {
      odsState.addOperands(value);
      odsState.addTypes(destType);
  
}

void ZeroExtendIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(resultType0);
}

void ZeroExtendIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ZeroExtendIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ZeroExtendIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void ZeroExtendIOp::print(::mlir::OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

::mlir::LogicalResult ZeroExtendIOp::verify() {
  if (failed(ZeroExtendIOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<::mlir::IndexType>())) || (((v.getType().isa<::mlir::VectorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("result #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

void ZeroExtendIOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}


#endif  // GET_OP_CLASSES

