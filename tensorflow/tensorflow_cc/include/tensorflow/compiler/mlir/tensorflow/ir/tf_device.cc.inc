/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::tf_device::ClusterFuncOp,
::mlir::tf_device::ClusterOp,
::mlir::tf_device::LaunchFuncOp,
::mlir::tf_device::LaunchOp,
::mlir::tf_device::ParallelExecuteOp,
::mlir::tf_device::ReplicateOp,
::mlir::tf_device::ReturnOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ClusterFuncOp definitions
//===----------------------------------------------------------------------===//

ClusterFuncOpAdaptor::ClusterFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ClusterFuncOpAdaptor::ClusterFuncOpAdaptor(ClusterFuncOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ClusterFuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ClusterFuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ClusterFuncOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::FlatSymbolRefAttr ClusterFuncOpAdaptor::func() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FlatSymbolRefAttr attr = odsAttrs.get("func").cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::mlir::LogicalResult ClusterFuncOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_func = odsAttrs.get("func");
  if (!tblgen_func) return emitError(loc, "'tf_device.cluster_func' op ""requires attribute 'func'");
    if (!((tblgen_func.isa<::mlir::FlatSymbolRefAttr>()))) return emitError(loc, "'tf_device.cluster_func' op ""attribute 'func' failed to satisfy constraint: flat symbol reference attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef ClusterFuncOp::getOperationName() {
  return "tf_device.cluster_func";
}

std::pair<unsigned, unsigned> ClusterFuncOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ClusterFuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ClusterFuncOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ClusterFuncOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> ClusterFuncOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ClusterFuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ClusterFuncOp::results() {
  return getODSResults(0);
}

::mlir::FlatSymbolRefAttr ClusterFuncOp::funcAttr() {
  return this->getAttr("func").cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef ClusterFuncOp::func() {
  auto attr = funcAttr();
  return attr.getValue();
}

void ClusterFuncOp::funcAttr(::mlir::FlatSymbolRefAttr attr) {
  this->getOperation()->setAttr("func", attr);
}

void ClusterFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::FlatSymbolRefAttr func, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("func", func);
  odsState.addTypes(results);
}

void ClusterFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef func, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("func", odsBuilder.getSymbolRefAttr(func));
  odsState.addTypes(results);
}

void ClusterFuncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ClusterFuncOp::verify() {
  if (failed(ClusterFuncOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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

} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ClusterOp definitions
//===----------------------------------------------------------------------===//

ClusterOpAdaptor::ClusterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ClusterOpAdaptor::ClusterOpAdaptor(ClusterOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ClusterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ClusterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::LogicalResult ClusterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ClusterOp::getOperationName() {
  return "tf_device.cluster";
}

std::pair<unsigned, unsigned> ClusterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ClusterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ClusterOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ClusterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ClusterOp::results() {
  return getODSResults(0);
}

::mlir::Region &ClusterOp::body() {
  return this->getOperation()->getRegion(0);
}

void ClusterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results) {
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void ClusterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ClusterOp::verify() {
  if (failed(ClusterOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
  return ::mlir::success();
}

} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::LaunchFuncOp definitions
//===----------------------------------------------------------------------===//

LaunchFuncOpAdaptor::LaunchFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

LaunchFuncOpAdaptor::LaunchFuncOpAdaptor(LaunchFuncOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> LaunchFuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange LaunchFuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange LaunchFuncOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::StringAttr LaunchFuncOpAdaptor::device() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("device").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::FlatSymbolRefAttr LaunchFuncOpAdaptor::func() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::FlatSymbolRefAttr attr = odsAttrs.get("func").cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::mlir::LogicalResult LaunchFuncOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_device = odsAttrs.get("device");
  if (!tblgen_device) return emitError(loc, "'tf_device.launch_func' op ""requires attribute 'device'");
    if (!((tblgen_device.isa<::mlir::StringAttr>()))) return emitError(loc, "'tf_device.launch_func' op ""attribute 'device' failed to satisfy constraint: string attribute");
  }
  {
  auto tblgen_func = odsAttrs.get("func");
  if (!tblgen_func) return emitError(loc, "'tf_device.launch_func' op ""requires attribute 'func'");
    if (!((tblgen_func.isa<::mlir::FlatSymbolRefAttr>()))) return emitError(loc, "'tf_device.launch_func' op ""attribute 'func' failed to satisfy constraint: flat symbol reference attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef LaunchFuncOp::getOperationName() {
  return "tf_device.launch_func";
}

std::pair<unsigned, unsigned> LaunchFuncOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range LaunchFuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range LaunchFuncOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange LaunchFuncOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
}

std::pair<unsigned, unsigned> LaunchFuncOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range LaunchFuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range LaunchFuncOp::results() {
  return getODSResults(0);
}

::mlir::StringAttr LaunchFuncOp::deviceAttr() {
  return this->getAttr("device").cast<::mlir::StringAttr>();
}

::llvm::StringRef LaunchFuncOp::device() {
  auto attr = deviceAttr();
  return attr.getValue();
}

::mlir::FlatSymbolRefAttr LaunchFuncOp::funcAttr() {
  return this->getAttr("func").cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef LaunchFuncOp::func() {
  auto attr = funcAttr();
  return attr.getValue();
}

void LaunchFuncOp::deviceAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("device", attr);
}

void LaunchFuncOp::funcAttr(::mlir::FlatSymbolRefAttr attr) {
  this->getOperation()->setAttr("func", attr);
}

void LaunchFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr device, ::mlir::FlatSymbolRefAttr func, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("device", device);
  odsState.addAttribute("func", func);
  odsState.addTypes(results);
}

void LaunchFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef device, ::llvm::StringRef func, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("device", odsBuilder.getStringAttr(device));
  odsState.addAttribute("func", odsBuilder.getSymbolRefAttr(func));
  odsState.addTypes(results);
}

void LaunchFuncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LaunchFuncOp::verify() {
  if (failed(LaunchFuncOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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

} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::LaunchOp definitions
//===----------------------------------------------------------------------===//

LaunchOpAdaptor::LaunchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

LaunchOpAdaptor::LaunchOpAdaptor(LaunchOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> LaunchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LaunchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr LaunchOpAdaptor::device() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("device").cast<::mlir::StringAttr>();
  return attr;
}

::mlir::LogicalResult LaunchOpAdaptor::verify(::mlir::Location loc) {
  {
  auto tblgen_device = odsAttrs.get("device");
  if (!tblgen_device) return emitError(loc, "'tf_device.launch' op ""requires attribute 'device'");
    if (!((tblgen_device.isa<::mlir::StringAttr>()))) return emitError(loc, "'tf_device.launch' op ""attribute 'device' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}

::llvm::StringRef LaunchOp::getOperationName() {
  return "tf_device.launch";
}

std::pair<unsigned, unsigned> LaunchOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LaunchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LaunchOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range LaunchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range LaunchOp::results() {
  return getODSResults(0);
}

::mlir::Region &LaunchOp::body() {
  return this->getOperation()->getRegion(0);
}

::mlir::StringAttr LaunchOp::deviceAttr() {
  return this->getAttr("device").cast<::mlir::StringAttr>();
}

::llvm::StringRef LaunchOp::device() {
  auto attr = deviceAttr();
  return attr.getValue();
}

void LaunchOp::deviceAttr(::mlir::StringAttr attr) {
  this->getOperation()->setAttr("device", attr);
}

void LaunchOp::build(OpBuilder &builder, OperationState &result,
                StringAttr device, ArrayRef<Type> result_types) {
        result.addAttribute("device", device);
        result.addTypes(result_types);
        result.addRegion();
      
}

void LaunchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr device) {
  odsState.addAttribute("device", device);
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void LaunchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef device) {
  odsState.addAttribute("device", odsBuilder.getStringAttr(device));
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void LaunchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LaunchOp::verify() {
  if (failed(LaunchOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
  return ::mlir::success();
}



} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ParallelExecuteOp definitions
//===----------------------------------------------------------------------===//

ParallelExecuteOpAdaptor::ParallelExecuteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ParallelExecuteOpAdaptor::ParallelExecuteOpAdaptor(ParallelExecuteOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ParallelExecuteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ParallelExecuteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::LogicalResult ParallelExecuteOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ParallelExecuteOp::getOperationName() {
  return "tf_device.parallel_execute";
}

std::pair<unsigned, unsigned> ParallelExecuteOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ParallelExecuteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ParallelExecuteOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ParallelExecuteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ParallelExecuteOp::execute_outputs() {
  return getODSResults(0);
}

::mlir::MutableArrayRef<Region> ParallelExecuteOp::regions() {
  return this->getOperation()->getRegions().drop_front(0);
}



void ParallelExecuteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange execute_outputs, unsigned regionsCount) {
  for (unsigned i = 0; i < regionsCount; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(execute_outputs);
}

void ParallelExecuteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != numRegions; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ParallelExecuteOp::verify() {
  if (failed(ParallelExecuteOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
    for (::mlir::Region &region : regions()) {
      (void)region;
      if (!((::llvm::hasNItems(region, 1)))) {
        return emitOpError("region #") << index << " ('regions') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return Verify(*this);
}

} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ReplicateOp definitions
//===----------------------------------------------------------------------===//

ReplicateOpAdaptor::ReplicateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs)  : odsOperands(values), odsAttrs(attrs) {

}

ReplicateOpAdaptor::ReplicateOpAdaptor(ReplicateOp&op)  : odsOperands(op.getOperation()->getOperands()), odsAttrs(op.getOperation()->getAttrDictionary()) {

}

std::pair<unsigned, unsigned> ReplicateOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::ValueRange ReplicateOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ReplicateOpAdaptor::replicated_inputs() {
  return getODSOperands(0);
}

::mlir::ValueRange ReplicateOpAdaptor::packed_inputs() {
  return getODSOperands(1);
}

::mlir::DenseIntElementsAttr ReplicateOpAdaptor::operand_segment_sizes() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::IntegerAttr ReplicateOpAdaptor::n() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("n").cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::DictionaryAttr ReplicateOpAdaptor::devices() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DictionaryAttr attr = odsAttrs.get("devices").dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::mlir::LogicalResult ReplicateOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
    auto numElements = sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 2 elements");
  }
    {
  auto tblgen_operand_segment_sizes = odsAttrs.get("operand_segment_sizes");
  if (!tblgen_operand_segment_sizes) return emitError(loc, "'tf_device.replicate' op ""requires attribute 'operand_segment_sizes'");
    if (!(((tblgen_operand_segment_sizes.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32))))) return emitError(loc, "'tf_device.replicate' op ""attribute 'operand_segment_sizes' failed to satisfy constraint: 32-bit signless integer elements attribute");
  }
  {
  auto tblgen_n = odsAttrs.get("n");
  if (!tblgen_n) return emitError(loc, "'tf_device.replicate' op ""requires attribute 'n'");
    if (!((((tblgen_n.isa<::mlir::IntegerAttr>())) && ((tblgen_n.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((tblgen_n.cast<::mlir::IntegerAttr>().getInt() >= 2)))) return emitError(loc, "'tf_device.replicate' op ""attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 2");
  }
  {
  auto tblgen_devices = odsAttrs.get("devices");
  if (tblgen_devices) {
    if (!((tblgen_devices.isa<::mlir::DictionaryAttr>()))) return emitError(loc, "'tf_device.replicate' op ""attribute 'devices' failed to satisfy constraint: dictionary of named attribute values");
  }
  }
  return ::mlir::success();
}

::llvm::StringRef ReplicateOp::getOperationName() {
  return "tf_device.replicate";
}

std::pair<unsigned, unsigned> ReplicateOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getAttrOfType<::mlir::DenseIntElementsAttr>("operand_segment_sizes");

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned size = (*(sizeAttr.begin() + index)).getZExtValue();
  return {start, size};
}

::mlir::Operation::operand_range ReplicateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReplicateOp::replicated_inputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range ReplicateOp::packed_inputs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ReplicateOp::replicated_inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

::mlir::MutableOperandRange ReplicateOp::packed_inputsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  return ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getMutableAttrDict().getNamed("operand_segment_sizes")));
}

std::pair<unsigned, unsigned> ReplicateOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ReplicateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ReplicateOp::replicated_outputs() {
  return getODSResults(0);
}

::mlir::Region &ReplicateOp::body() {
  return this->getOperation()->getRegion(0);
}

::mlir::DenseIntElementsAttr ReplicateOp::operand_segment_sizesAttr() {
  return this->getAttr("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ReplicateOp::operand_segment_sizes() {
  auto attr = operand_segment_sizesAttr();
  return attr;
}

::mlir::IntegerAttr ReplicateOp::nAttr() {
  return this->getAttr("n").cast<::mlir::IntegerAttr>();
}

uint32_t ReplicateOp::n() {
  auto attr = nAttr();
  return attr.getValue().getZExtValue();
}

::mlir::DictionaryAttr ReplicateOp::devicesAttr() {
  return this->getAttr("devices").dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > ReplicateOp::devices() {
  auto attr = devicesAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

void ReplicateOp::operand_segment_sizesAttr(::mlir::DenseIntElementsAttr attr) {
  this->getOperation()->setAttr("operand_segment_sizes", attr);
}

void ReplicateOp::nAttr(::mlir::IntegerAttr attr) {
  this->getOperation()->setAttr("n", attr);
}

void ReplicateOp::devicesAttr(::mlir::DictionaryAttr attr) {
  this->getOperation()->setAttr("devices", attr);
}



void ReplicateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange replicated_outputs, ::mlir::ValueRange replicated_inputs, ::mlir::ValueRange packed_inputs, ::mlir::DenseIntElementsAttr operand_segment_sizes, ::mlir::IntegerAttr n, /*optional*/::mlir::DictionaryAttr devices) {
  odsState.addOperands(replicated_inputs);
  odsState.addOperands(packed_inputs);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(replicated_inputs.size()), static_cast<int32_t>(packed_inputs.size())}));
  odsState.addAttribute("operand_segment_sizes", operand_segment_sizes);
  odsState.addAttribute("n", n);
  if (devices) {
  odsState.addAttribute("devices", devices);
  }
  (void)odsState.addRegion();
  odsState.addTypes(replicated_outputs);
}

void ReplicateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange replicated_outputs, ::mlir::ValueRange replicated_inputs, ::mlir::ValueRange packed_inputs, ::mlir::DenseIntElementsAttr operand_segment_sizes, uint32_t n, /*optional*/::mlir::DictionaryAttr devices) {
  odsState.addOperands(replicated_inputs);
  odsState.addOperands(packed_inputs);
  odsState.addAttribute("operand_segment_sizes", odsBuilder.getI32VectorAttr({static_cast<int32_t>(replicated_inputs.size()), static_cast<int32_t>(packed_inputs.size())}));
  odsState.addAttribute("operand_segment_sizes", operand_segment_sizes);
  odsState.addAttribute("n", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
  if (devices) {
  odsState.addAttribute("devices", devices);
  }
  (void)odsState.addRegion();
  odsState.addTypes(replicated_outputs);
}

void ReplicateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::ParseResult ReplicateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  return ParseReplicateOp(&parser, &result);
}

void ReplicateOp::print(::mlir::OpAsmPrinter &p) {
  return Print(*this, &p);
}

::mlir::LogicalResult ReplicateOp::verify() {
  if (failed(ReplicateOpAdaptor(*this).verify(this->getLoc()))) return ::mlir::failure();
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
  return Verify(*this);
}

} // namespace tf_device
} // namespace mlir
namespace mlir {
namespace tf_device {

//===----------------------------------------------------------------------===//
// ::mlir::tf_device::ReturnOp definitions
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

::mlir::ValueRange ReturnOpAdaptor::results() {
  return getODSOperands(0);
}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

::llvm::StringRef ReturnOp::getOperationName() {
  return "tf_device.return";
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

::mlir::Operation::operand_range ReturnOp::results() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::resultsMutable() {
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

void ReturnOp::build(OpBuilder &builder, OperationState &result) {
      build(builder, result, {});
    
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results) {
  odsState.addOperands(results);
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
  return ::mlir::success();
}

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> resultsOperands;
  ::llvm::SMLoc resultsOperandsLoc;
  (void)resultsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> resultsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  resultsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(resultsOperands))
    return ::mlir::failure();
  if (!resultsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(resultsOperands, resultsTypes, resultsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::print(::mlir::OpAsmPrinter &p) {
  p << "tf_device.return";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  if (!results().empty()) {
  p << " ";
  p << results();
  p << " " << ":";
  p << " ";
  p << results().getTypes();
  }
}

} // namespace tf_device
} // namespace mlir

#endif  // GET_OP_CLASSES

