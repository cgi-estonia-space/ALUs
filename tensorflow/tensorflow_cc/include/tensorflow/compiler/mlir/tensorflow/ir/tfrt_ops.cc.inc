/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::TF::_JitFusedMatMulOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_JitFusedMatMulOp definitions
//===----------------------------------------------------------------------===//

_JitFusedMatMulOpAdaptor::_JitFusedMatMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

_JitFusedMatMulOpAdaptor::_JitFusedMatMulOpAdaptor(_JitFusedMatMulOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> _JitFusedMatMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange _JitFusedMatMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value _JitFusedMatMulOpAdaptor::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value _JitFusedMatMulOpAdaptor::b() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange _JitFusedMatMulOpAdaptor::additional_args() {
  return getODSOperands(2);
}

::mlir::BoolAttr _JitFusedMatMulOpAdaptor::transpose_a() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("transpose_a").dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

::mlir::BoolAttr _JitFusedMatMulOpAdaptor::transpose_b() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("transpose_b").dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

::mlir::ArrayAttr _JitFusedMatMulOpAdaptor::fusion() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("fusion").dyn_cast_or_null<::mlir::ArrayAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getStrArrayAttr({});
  return attr;
}

::mlir::LogicalResult _JitFusedMatMulOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_transpose_a = odsAttrs.get("transpose_a");
  if (tblgen_transpose_a) {
    if (!((tblgen_transpose_a.isa<::mlir::BoolAttr>()))) return emitError(loc, "'tf._JitFusedMatMul' op ""attribute 'transpose_a' failed to satisfy constraint: bool attribute");
  }
  }
  {
  auto tblgen_transpose_b = odsAttrs.get("transpose_b");
  if (tblgen_transpose_b) {
    if (!((tblgen_transpose_b.isa<::mlir::BoolAttr>()))) return emitError(loc, "'tf._JitFusedMatMul' op ""attribute 'transpose_b' failed to satisfy constraint: bool attribute");
  }
  }
  {
  auto tblgen_fusion = odsAttrs.get("fusion");
  if (tblgen_fusion) {
    if (!(((tblgen_fusion.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_fusion.cast<::mlir::ArrayAttr>(), [](::mlir::Attribute attr) { return (attr.isa<::mlir::StringAttr>()); })))) return emitError(loc, "'tf._JitFusedMatMul' op ""attribute 'fusion' failed to satisfy constraint: string array attribute");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef _JitFusedMatMulOp::getOperationName() {
  return "tf._JitFusedMatMul";
}

std::pair<unsigned, unsigned> _JitFusedMatMulOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range _JitFusedMatMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value _JitFusedMatMulOp::a() {
  return *getODSOperands(0).begin();
}

::mlir::Value _JitFusedMatMulOp::b() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range _JitFusedMatMulOp::additional_args() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange _JitFusedMatMulOp::aMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange _JitFusedMatMulOp::bMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

::mlir::MutableOperandRange _JitFusedMatMulOp::additional_argsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> _JitFusedMatMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range _JitFusedMatMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value _JitFusedMatMulOp::product() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr _JitFusedMatMulOp::transpose_aAttr() {
  return this->getAttr("transpose_a").dyn_cast_or_null<::mlir::BoolAttr>();
}

bool _JitFusedMatMulOp::transpose_a() {
  auto attr = transpose_aAttr();
    if (!attr)
      return ::mlir::Builder(this->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::BoolAttr _JitFusedMatMulOp::transpose_bAttr() {
  return this->getAttr("transpose_b").dyn_cast_or_null<::mlir::BoolAttr>();
}

bool _JitFusedMatMulOp::transpose_b() {
  auto attr = transpose_bAttr();
    if (!attr)
      return ::mlir::Builder(this->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::ArrayAttr _JitFusedMatMulOp::fusionAttr() {
  return this->getAttr("fusion").dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr _JitFusedMatMulOp::fusion() {
  auto attr = fusionAttr();
    if (!attr)
      return ::mlir::Builder(this->getContext()).getStrArrayAttr({});
  return attr;
}

Type _JitFusedMatMulOp::T() {
  return mlir::getElementTypeOrSelf(*getODSOperands(0).begin());
}

bool _JitFusedMatMulOp::isDerivedAttribute(::llvm::StringRef name) {
  if (name == "T") return true;
 return false;
}

::mlir::DictionaryAttr _JitFusedMatMulOp::materializeDerivedAttributes() {
  ::mlir::MLIRContext* ctx = getContext();
  ::mlir::Builder odsBuilder(ctx); (void)odsBuilder;
  return ::mlir::DictionaryAttr::get({
    {::mlir::Identifier::get("T", ctx),
::mlir::TypeAttr::get(T())}
    }, ctx);
}

void _JitFusedMatMulOp::transpose_aAttr(::mlir::BoolAttr attr) {
  this->getOperation()->setAttr("transpose_a", attr);
}

void _JitFusedMatMulOp::transpose_bAttr(::mlir::BoolAttr attr) {
  this->getOperation()->setAttr("transpose_b", attr);
}

void _JitFusedMatMulOp::fusionAttr(::mlir::ArrayAttr attr) {
  this->getOperation()->setAttr("fusion", attr);
}

void _JitFusedMatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type product, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange additional_args, ::mlir::BoolAttr transpose_a, ::mlir::BoolAttr transpose_b, ::mlir::ArrayAttr fusion) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(additional_args);
  odsState.addAttribute("transpose_a", transpose_a);
  odsState.addAttribute("transpose_b", transpose_b);
  odsState.addAttribute("fusion", fusion);
  odsState.addTypes(product);
}

void _JitFusedMatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange additional_args, ::mlir::BoolAttr transpose_a, ::mlir::BoolAttr transpose_b, ::mlir::ArrayAttr fusion) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(additional_args);
  odsState.addAttribute("transpose_a", transpose_a);
  odsState.addAttribute("transpose_b", transpose_b);
  odsState.addAttribute("fusion", fusion);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void _JitFusedMatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type product, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange additional_args, bool transpose_a, bool transpose_b, ::mlir::ArrayAttr fusion) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(additional_args);
  odsState.addAttribute("transpose_a", odsBuilder.getBoolAttr(transpose_a));
  odsState.addAttribute("transpose_b", odsBuilder.getBoolAttr(transpose_b));
  odsState.addAttribute("fusion", fusion);
  odsState.addTypes(product);
}

void _JitFusedMatMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange additional_args, bool transpose_a, bool transpose_b, ::mlir::ArrayAttr fusion) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(additional_args);
  odsState.addAttribute("transpose_a", odsBuilder.getBoolAttr(transpose_a));
  odsState.addAttribute("transpose_b", odsBuilder.getBoolAttr(transpose_b));
  odsState.addAttribute("fusion", fusion);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void _JitFusedMatMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult _JitFusedMatMulOp::verify() {
  if (failed(_JitFusedMatMulOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())))) {
        return emitOpError("operand #") << index << " must be tensor of 32-bit float values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (::mlir::Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())))) {
        return emitOpError("operand #") << index << " must be tensor of 32-bit float values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (::mlir::Value v : valueGroup2) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())))) {
        return emitOpError("operand #") << index << " must be tensor of 32-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (::mlir::Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::TensorType>())) && ((v.getType().cast<::mlir::ShapedType>().getElementType().isF32())))) {
        return emitOpError("result #") << index << " must be tensor of 32-bit float values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::mlir::success();
}

void _JitFusedMatMulOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {

}

} // namespace TF
} // namespace mlir

#endif  // GET_OP_CLASSES

