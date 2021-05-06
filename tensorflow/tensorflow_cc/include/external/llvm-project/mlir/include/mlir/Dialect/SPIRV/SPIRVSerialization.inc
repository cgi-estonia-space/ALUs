/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* SPIR-V Serialization Utilities/Functions                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_SERIALIZATION_FNS

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AccessChainOp>(::mlir::spirv::AccessChainOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAccessChain, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicAndOp>(::mlir::spirv::AtomicAndOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op.getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op.getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicAnd, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicCompareExchangeWeakOp>(::mlir::spirv::AtomicCompareExchangeWeakOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op.getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op.getAttr("equal_semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("equal_semantics");
  }
  {
    if (auto attr = op.getAttr("unequal_semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("unequal_semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(2)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #2 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicCompareExchangeWeak, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicIAddOp>(::mlir::spirv::AtomicIAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op.getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op.getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicIAdd, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicIDecrementOp>(::mlir::spirv::AtomicIDecrementOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op.getAttr("memory_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
  }
  elidedAttrs.push_back("memory_scope");
  if (auto attr = op.getAttr("semantics")) {
    operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
  }
  elidedAttrs.push_back("semantics");
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicIDecrement, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicIIncrementOp>(::mlir::spirv::AtomicIIncrementOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op.getAttr("memory_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
  }
  elidedAttrs.push_back("memory_scope");
  if (auto attr = op.getAttr("semantics")) {
    operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
  }
  elidedAttrs.push_back("semantics");
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicIIncrement, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicISubOp>(::mlir::spirv::AtomicISubOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op.getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op.getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicISub, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicOrOp>(::mlir::spirv::AtomicOrOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op.getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op.getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicOr, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicSMaxOp>(::mlir::spirv::AtomicSMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op.getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op.getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicSMax, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicSMinOp>(::mlir::spirv::AtomicSMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op.getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op.getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicSMin, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicUMaxOp>(::mlir::spirv::AtomicUMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op.getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op.getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicUMax, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicUMinOp>(::mlir::spirv::AtomicUMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op.getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op.getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicUMin, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::AtomicXorOp>(::mlir::spirv::AtomicXorOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    if (auto attr = op.getAttr("memory_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("memory_scope");
  }
  {
    if (auto attr = op.getAttr("semantics")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("semantics");
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpAtomicXor, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitCountOp>(::mlir::spirv::BitCountOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpBitCount, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitFieldInsertOp>(::mlir::spirv::BitFieldInsertOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpBitFieldInsert, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitFieldSExtractOp>(::mlir::spirv::BitFieldSExtractOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpBitFieldSExtract, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitFieldUExtractOp>(::mlir::spirv::BitFieldUExtractOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpBitFieldUExtract, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitReverseOp>(::mlir::spirv::BitReverseOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpBitReverse, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitcastOp>(::mlir::spirv::BitcastOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpBitcast, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitwiseAndOp>(::mlir::spirv::BitwiseAndOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpBitwiseAnd, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitwiseOrOp>(::mlir::spirv::BitwiseOrOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpBitwiseOr, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::BitwiseXorOp>(::mlir::spirv::BitwiseXorOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpBitwiseXor, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CompositeConstructOp>(::mlir::spirv::CompositeConstructOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpCompositeConstruct, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CompositeExtractOp>(::mlir::spirv::CompositeExtractOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op.getAttr("indices")) {
    for (auto attrElem : attr.cast<ArrayAttr>()) {
      operands.push_back(static_cast<uint32_t>(attrElem.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  elidedAttrs.push_back("indices");
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpCompositeExtract, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CompositeInsertOp>(::mlir::spirv::CompositeInsertOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op.getAttr("indices")) {
    for (auto attrElem : attr.cast<ArrayAttr>()) {
      operands.push_back(static_cast<uint32_t>(attrElem.cast<IntegerAttr>().getValue().getZExtValue()));
    }
  }
  elidedAttrs.push_back("indices");
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpCompositeInsert, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertFToSOp>(::mlir::spirv::ConvertFToSOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpConvertFToS, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertFToUOp>(::mlir::spirv::ConvertFToUOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpConvertFToU, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertSToFOp>(::mlir::spirv::ConvertSToFOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpConvertSToF, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ConvertUToFOp>(::mlir::spirv::ConvertUToFOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpConvertUToF, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CooperativeMatrixLengthNVOp>(::mlir::spirv::CooperativeMatrixLengthNVOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  if (auto attr = op.getAttr("type")) {
    operands.push_back(static_cast<uint32_t>(getTypeID(attr.cast<TypeAttr>().getValue())));
  }
  elidedAttrs.push_back("type");
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpCooperativeMatrixLengthNV, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CooperativeMatrixLoadNVOp>(::mlir::spirv::CooperativeMatrixLoadNVOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op.getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("memory_access");
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpCooperativeMatrixLoadNV, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CooperativeMatrixMulAddNVOp>(::mlir::spirv::CooperativeMatrixMulAddNVOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpCooperativeMatrixMulAddNV, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::CooperativeMatrixStoreNVOp>(::mlir::spirv::CooperativeMatrixStoreNVOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op.getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("memory_access");
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpCooperativeMatrixStoreNV, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FAddOp>(::mlir::spirv::FAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFAdd, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FConvertOp>(::mlir::spirv::FConvertOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFConvert, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FDivOp>(::mlir::spirv::FDivOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFDiv, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FModOp>(::mlir::spirv::FModOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFMod, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FMulOp>(::mlir::spirv::FMulOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFMul, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FNegateOp>(::mlir::spirv::FNegateOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFNegate, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdEqualOp>(::mlir::spirv::FOrdEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFOrdEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdGreaterThanEqualOp>(::mlir::spirv::FOrdGreaterThanEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFOrdGreaterThanEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdGreaterThanOp>(::mlir::spirv::FOrdGreaterThanOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFOrdGreaterThan, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdLessThanEqualOp>(::mlir::spirv::FOrdLessThanEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFOrdLessThanEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdLessThanOp>(::mlir::spirv::FOrdLessThanOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFOrdLessThan, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FOrdNotEqualOp>(::mlir::spirv::FOrdNotEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFOrdNotEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FRemOp>(::mlir::spirv::FRemOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFRem, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FSubOp>(::mlir::spirv::FSubOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFSub, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordEqualOp>(::mlir::spirv::FUnordEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFUnordEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordGreaterThanEqualOp>(::mlir::spirv::FUnordGreaterThanEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFUnordGreaterThanEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordGreaterThanOp>(::mlir::spirv::FUnordGreaterThanOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFUnordGreaterThan, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordLessThanEqualOp>(::mlir::spirv::FUnordLessThanEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFUnordLessThanEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordLessThanOp>(::mlir::spirv::FUnordLessThanOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFUnordLessThan, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::FUnordNotEqualOp>(::mlir::spirv::FUnordNotEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpFUnordNotEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLAcosOp>(::mlir::spirv::GLSLAcosOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 17, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLAsinOp>(::mlir::spirv::GLSLAsinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 16, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLAtanOp>(::mlir::spirv::GLSLAtanOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 18, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLCeilOp>(::mlir::spirv::GLSLCeilOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 9, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLCosOp>(::mlir::spirv::GLSLCosOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 14, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLCoshOp>(::mlir::spirv::GLSLCoshOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 20, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLExpOp>(::mlir::spirv::GLSLExpOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 27, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFAbsOp>(::mlir::spirv::GLSLFAbsOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 4, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFMaxOp>(::mlir::spirv::GLSLFMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 40, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFMinOp>(::mlir::spirv::GLSLFMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 37, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFSignOp>(::mlir::spirv::GLSLFSignOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 6, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLFloorOp>(::mlir::spirv::GLSLFloorOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 8, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLInverseSqrtOp>(::mlir::spirv::GLSLInverseSqrtOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 32, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLLogOp>(::mlir::spirv::GLSLLogOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 28, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLPowOp>(::mlir::spirv::GLSLPowOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 26, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLRoundOp>(::mlir::spirv::GLSLRoundOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 1, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSAbsOp>(::mlir::spirv::GLSLSAbsOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 5, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSMaxOp>(::mlir::spirv::GLSLSMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 42, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSMinOp>(::mlir::spirv::GLSLSMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 39, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSSignOp>(::mlir::spirv::GLSLSSignOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 7, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSinOp>(::mlir::spirv::GLSLSinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 13, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSinhOp>(::mlir::spirv::GLSLSinhOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 19, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLSqrtOp>(::mlir::spirv::GLSLSqrtOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 31, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLTanOp>(::mlir::spirv::GLSLTanOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 15, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GLSLTanhOp>(::mlir::spirv::GLSLTanhOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "GLSL.std.450", 21, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupBroadcastOp>(::mlir::spirv::GroupBroadcastOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupBroadcast, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformBallotOp>(::mlir::spirv::GroupNonUniformBallotOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformBallot, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformBroadcastOp>(::mlir::spirv::GroupNonUniformBroadcastOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformBroadcast, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformElectOp>(::mlir::spirv::GroupNonUniformElectOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  if (auto attr = op.getAttr("execution_scope")) {
    operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
  }
  elidedAttrs.push_back("execution_scope");
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformElect, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFAddOp>(::mlir::spirv::GroupNonUniformFAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op.getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFAdd, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFMaxOp>(::mlir::spirv::GroupNonUniformFMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op.getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFMax, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFMinOp>(::mlir::spirv::GroupNonUniformFMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op.getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFMin, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformFMulOp>(::mlir::spirv::GroupNonUniformFMulOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op.getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformFMul, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformIAddOp>(::mlir::spirv::GroupNonUniformIAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op.getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformIAdd, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformIMulOp>(::mlir::spirv::GroupNonUniformIMulOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op.getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformIMul, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformSMaxOp>(::mlir::spirv::GroupNonUniformSMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op.getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformSMax, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformSMinOp>(::mlir::spirv::GroupNonUniformSMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op.getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformSMin, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformUMaxOp>(::mlir::spirv::GroupNonUniformUMaxOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op.getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformUMax, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::GroupNonUniformUMinOp>(::mlir::spirv::GroupNonUniformUMinOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("execution_scope")) {
      operands.push_back(prepareConstantInt(op.getLoc(), attr.cast<IntegerAttr>()));
    }
    elidedAttrs.push_back("execution_scope");
  }
  {
    if (auto attr = op.getAttr("group_operation")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("group_operation");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  {
    for (auto arg : op.getODSOperands(1)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #1 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpGroupNonUniformUMin, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IAddOp>(::mlir::spirv::IAddOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpIAdd, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IEqualOp>(::mlir::spirv::IEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpIEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::IMulOp>(::mlir::spirv::IMulOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpIMul, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::INotEqualOp>(::mlir::spirv::INotEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpINotEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ISubOp>(::mlir::spirv::ISubOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpISub, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LoadOp>(::mlir::spirv::LoadOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op.getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("memory_access");
  if (auto attr = op.getAttr("alignment")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("alignment");
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpLoad, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalAndOp>(::mlir::spirv::LogicalAndOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpLogicalAnd, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalEqualOp>(::mlir::spirv::LogicalEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpLogicalEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalNotEqualOp>(::mlir::spirv::LogicalNotEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpLogicalNotEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalNotOp>(::mlir::spirv::LogicalNotOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpLogicalNot, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::LogicalOrOp>(::mlir::spirv::LogicalOrOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpLogicalOr, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::MatrixTimesMatrixOp>(::mlir::spirv::MatrixTimesMatrixOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpMatrixTimesMatrix, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::MatrixTimesScalarOp>(::mlir::spirv::MatrixTimesScalarOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpMatrixTimesScalar, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::NotOp>(::mlir::spirv::NotOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpNot, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::OCLExpOp>(::mlir::spirv::OCLExpOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "OpenCL.std", 19, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::OCLFAbsOp>(::mlir::spirv::OCLFAbsOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "OpenCL.std", 23, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::OCLSAbsOp>(::mlir::spirv::OCLSAbsOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  encodeExtensionInstruction(op, "OpenCL.std", 141, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ReturnOp>(::mlir::spirv::ReturnOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpReturn, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ReturnValueOp>(::mlir::spirv::ReturnValueOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpReturnValue, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SConvertOp>(::mlir::spirv::SConvertOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSConvert, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SDivOp>(::mlir::spirv::SDivOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSDiv, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SGreaterThanEqualOp>(::mlir::spirv::SGreaterThanEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSGreaterThanEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SGreaterThanOp>(::mlir::spirv::SGreaterThanOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSGreaterThan, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SLessThanEqualOp>(::mlir::spirv::SLessThanEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSLessThanEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SLessThanOp>(::mlir::spirv::SLessThanOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSLessThan, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SModOp>(::mlir::spirv::SModOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSMod, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SNegateOp>(::mlir::spirv::SNegateOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSNegate, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SRemOp>(::mlir::spirv::SRemOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSRem, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SelectOp>(::mlir::spirv::SelectOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSelect, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ShiftLeftLogicalOp>(::mlir::spirv::ShiftLeftLogicalOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpShiftLeftLogical, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ShiftRightArithmeticOp>(::mlir::spirv::ShiftRightArithmeticOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpShiftRightArithmetic, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ShiftRightLogicalOp>(::mlir::spirv::ShiftRightLogicalOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpShiftRightLogical, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::StoreOp>(::mlir::spirv::StoreOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  if (auto attr = op.getAttr("memory_access")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("memory_access");
  if (auto attr = op.getAttr("alignment")) {
    operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
  }
  elidedAttrs.push_back("alignment");
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpStore, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SubgroupBallotKHROp>(::mlir::spirv::SubgroupBallotKHROp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSubgroupBallotKHR, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SubgroupBlockReadINTELOp>(::mlir::spirv::SubgroupBlockReadINTELOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSubgroupBlockReadINTEL, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::SubgroupBlockWriteINTELOp>(::mlir::spirv::SubgroupBlockWriteINTELOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpSubgroupBlockWriteINTEL, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::TransposeOp>(::mlir::spirv::TransposeOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpTranspose, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UConvertOp>(::mlir::spirv::UConvertOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpUConvert, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UDivOp>(::mlir::spirv::UDivOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpUDiv, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UGreaterThanEqualOp>(::mlir::spirv::UGreaterThanEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpUGreaterThanEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UGreaterThanOp>(::mlir::spirv::UGreaterThanOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpUGreaterThan, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ULessThanEqualOp>(::mlir::spirv::ULessThanEqualOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpULessThanEqual, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::ULessThanOp>(::mlir::spirv::ULessThanOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpULessThan, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UModOp>(::mlir::spirv::UModOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  for (Value operand : op.getOperation()->getOperands()) {
    auto id = getValueID(operand);
    assert(id && "use before def!");
    operands.push_back(id);
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpUMod, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::UnreachableOp>(::mlir::spirv::UnreachableOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpUnreachable, operands);
  return success();
}

template <> LogicalResult
Serializer::processOp<::mlir::spirv::VariableOp>(::mlir::spirv::VariableOp op) {
  SmallVector<uint32_t, 4> operands;
  SmallVector<StringRef, 2> elidedAttrs;
  uint32_t resultID = 0;
  uint32_t resultTypeID = 0;
  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) {
    return failure();
  }
  operands.push_back(resultTypeID);
  resultID = getNextID();
  valueIDMap[op.getResult()] = resultID;
  operands.push_back(resultID);
  {
    if (auto attr = op.getAttr("storage_class")) {
      operands.push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue().getZExtValue()));
    }
    elidedAttrs.push_back("storage_class");
  }
  {
    for (auto arg : op.getODSOperands(0)) {
      auto argID = getValueID(arg);
      if (!argID) {
        return emitError(op.getLoc(), "operand #0 has a use before def");
      }
      operands.push_back(argID);
    }
  }
  emitDebugLine(functionBody, op.getLoc());
  encodeInstructionInto(functionBody, spirv::Opcode::OpVariable, operands);
  for (auto attr : op.getAttrs()) {
    if (llvm::any_of(elidedAttrs, [&](StringRef elided) {return attr.first == elided;})) {
      continue;
    }
    if (failed(processDecoration(op.getLoc(), resultID, attr))) {
      return failure();
    }
  }
  return success();
}

LogicalResult Serializer::dispatchToAutogenSerialization(Operation *op) {
  if (isa<::mlir::spirv::AccessChainOp>(op)) {
    return processOp(cast<::mlir::spirv::AccessChainOp>(op));
  }
  if (isa<::mlir::spirv::AtomicAndOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicAndOp>(op));
  }
  if (isa<::mlir::spirv::AtomicCompareExchangeWeakOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicCompareExchangeWeakOp>(op));
  }
  if (isa<::mlir::spirv::AtomicIAddOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicIAddOp>(op));
  }
  if (isa<::mlir::spirv::AtomicIDecrementOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicIDecrementOp>(op));
  }
  if (isa<::mlir::spirv::AtomicIIncrementOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicIIncrementOp>(op));
  }
  if (isa<::mlir::spirv::AtomicISubOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicISubOp>(op));
  }
  if (isa<::mlir::spirv::AtomicOrOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicOrOp>(op));
  }
  if (isa<::mlir::spirv::AtomicSMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicSMaxOp>(op));
  }
  if (isa<::mlir::spirv::AtomicSMinOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicSMinOp>(op));
  }
  if (isa<::mlir::spirv::AtomicUMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicUMaxOp>(op));
  }
  if (isa<::mlir::spirv::AtomicUMinOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicUMinOp>(op));
  }
  if (isa<::mlir::spirv::AtomicXorOp>(op)) {
    return processOp(cast<::mlir::spirv::AtomicXorOp>(op));
  }
  if (isa<::mlir::spirv::BitCountOp>(op)) {
    return processOp(cast<::mlir::spirv::BitCountOp>(op));
  }
  if (isa<::mlir::spirv::BitFieldInsertOp>(op)) {
    return processOp(cast<::mlir::spirv::BitFieldInsertOp>(op));
  }
  if (isa<::mlir::spirv::BitFieldSExtractOp>(op)) {
    return processOp(cast<::mlir::spirv::BitFieldSExtractOp>(op));
  }
  if (isa<::mlir::spirv::BitFieldUExtractOp>(op)) {
    return processOp(cast<::mlir::spirv::BitFieldUExtractOp>(op));
  }
  if (isa<::mlir::spirv::BitReverseOp>(op)) {
    return processOp(cast<::mlir::spirv::BitReverseOp>(op));
  }
  if (isa<::mlir::spirv::BitcastOp>(op)) {
    return processOp(cast<::mlir::spirv::BitcastOp>(op));
  }
  if (isa<::mlir::spirv::BitwiseAndOp>(op)) {
    return processOp(cast<::mlir::spirv::BitwiseAndOp>(op));
  }
  if (isa<::mlir::spirv::BitwiseOrOp>(op)) {
    return processOp(cast<::mlir::spirv::BitwiseOrOp>(op));
  }
  if (isa<::mlir::spirv::BitwiseXorOp>(op)) {
    return processOp(cast<::mlir::spirv::BitwiseXorOp>(op));
  }
  if (isa<::mlir::spirv::BranchConditionalOp>(op)) {
    return processOp(cast<::mlir::spirv::BranchConditionalOp>(op));
  }
  if (isa<::mlir::spirv::BranchOp>(op)) {
    return processOp(cast<::mlir::spirv::BranchOp>(op));
  }
  if (isa<::mlir::spirv::CompositeConstructOp>(op)) {
    return processOp(cast<::mlir::spirv::CompositeConstructOp>(op));
  }
  if (isa<::mlir::spirv::CompositeExtractOp>(op)) {
    return processOp(cast<::mlir::spirv::CompositeExtractOp>(op));
  }
  if (isa<::mlir::spirv::CompositeInsertOp>(op)) {
    return processOp(cast<::mlir::spirv::CompositeInsertOp>(op));
  }
  if (isa<::mlir::spirv::ControlBarrierOp>(op)) {
    return processOp(cast<::mlir::spirv::ControlBarrierOp>(op));
  }
  if (isa<::mlir::spirv::ConvertFToSOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertFToSOp>(op));
  }
  if (isa<::mlir::spirv::ConvertFToUOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertFToUOp>(op));
  }
  if (isa<::mlir::spirv::ConvertSToFOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertSToFOp>(op));
  }
  if (isa<::mlir::spirv::ConvertUToFOp>(op)) {
    return processOp(cast<::mlir::spirv::ConvertUToFOp>(op));
  }
  if (isa<::mlir::spirv::CooperativeMatrixLengthNVOp>(op)) {
    return processOp(cast<::mlir::spirv::CooperativeMatrixLengthNVOp>(op));
  }
  if (isa<::mlir::spirv::CooperativeMatrixLoadNVOp>(op)) {
    return processOp(cast<::mlir::spirv::CooperativeMatrixLoadNVOp>(op));
  }
  if (isa<::mlir::spirv::CooperativeMatrixMulAddNVOp>(op)) {
    return processOp(cast<::mlir::spirv::CooperativeMatrixMulAddNVOp>(op));
  }
  if (isa<::mlir::spirv::CooperativeMatrixStoreNVOp>(op)) {
    return processOp(cast<::mlir::spirv::CooperativeMatrixStoreNVOp>(op));
  }
  if (isa<::mlir::spirv::CopyMemoryOp>(op)) {
    return processOp(cast<::mlir::spirv::CopyMemoryOp>(op));
  }
  if (isa<::mlir::spirv::EntryPointOp>(op)) {
    return processOp(cast<::mlir::spirv::EntryPointOp>(op));
  }
  if (isa<::mlir::spirv::ExecutionModeOp>(op)) {
    return processOp(cast<::mlir::spirv::ExecutionModeOp>(op));
  }
  if (isa<::mlir::spirv::FAddOp>(op)) {
    return processOp(cast<::mlir::spirv::FAddOp>(op));
  }
  if (isa<::mlir::spirv::FConvertOp>(op)) {
    return processOp(cast<::mlir::spirv::FConvertOp>(op));
  }
  if (isa<::mlir::spirv::FDivOp>(op)) {
    return processOp(cast<::mlir::spirv::FDivOp>(op));
  }
  if (isa<::mlir::spirv::FModOp>(op)) {
    return processOp(cast<::mlir::spirv::FModOp>(op));
  }
  if (isa<::mlir::spirv::FMulOp>(op)) {
    return processOp(cast<::mlir::spirv::FMulOp>(op));
  }
  if (isa<::mlir::spirv::FNegateOp>(op)) {
    return processOp(cast<::mlir::spirv::FNegateOp>(op));
  }
  if (isa<::mlir::spirv::FOrdEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdEqualOp>(op));
  }
  if (isa<::mlir::spirv::FOrdGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FOrdGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::FOrdLessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdLessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FOrdLessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdLessThanOp>(op));
  }
  if (isa<::mlir::spirv::FOrdNotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FOrdNotEqualOp>(op));
  }
  if (isa<::mlir::spirv::FRemOp>(op)) {
    return processOp(cast<::mlir::spirv::FRemOp>(op));
  }
  if (isa<::mlir::spirv::FSubOp>(op)) {
    return processOp(cast<::mlir::spirv::FSubOp>(op));
  }
  if (isa<::mlir::spirv::FUnordEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordEqualOp>(op));
  }
  if (isa<::mlir::spirv::FUnordGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FUnordGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::FUnordLessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordLessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::FUnordLessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordLessThanOp>(op));
  }
  if (isa<::mlir::spirv::FUnordNotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::FUnordNotEqualOp>(op));
  }
  if (isa<::mlir::spirv::FunctionCallOp>(op)) {
    return processOp(cast<::mlir::spirv::FunctionCallOp>(op));
  }
  if (isa<::mlir::spirv::GLSLAcosOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLAcosOp>(op));
  }
  if (isa<::mlir::spirv::GLSLAsinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLAsinOp>(op));
  }
  if (isa<::mlir::spirv::GLSLAtanOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLAtanOp>(op));
  }
  if (isa<::mlir::spirv::GLSLCeilOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLCeilOp>(op));
  }
  if (isa<::mlir::spirv::GLSLCosOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLCosOp>(op));
  }
  if (isa<::mlir::spirv::GLSLCoshOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLCoshOp>(op));
  }
  if (isa<::mlir::spirv::GLSLExpOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLExpOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFAbsOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFMaxOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFMinOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFSignOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFSignOp>(op));
  }
  if (isa<::mlir::spirv::GLSLFloorOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLFloorOp>(op));
  }
  if (isa<::mlir::spirv::GLSLInverseSqrtOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLInverseSqrtOp>(op));
  }
  if (isa<::mlir::spirv::GLSLLogOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLLogOp>(op));
  }
  if (isa<::mlir::spirv::GLSLPowOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLPowOp>(op));
  }
  if (isa<::mlir::spirv::GLSLRoundOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLRoundOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSAbsOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSMaxOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSMinOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSSignOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSSignOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSinOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSinOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSinhOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSinhOp>(op));
  }
  if (isa<::mlir::spirv::GLSLSqrtOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLSqrtOp>(op));
  }
  if (isa<::mlir::spirv::GLSLTanOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLTanOp>(op));
  }
  if (isa<::mlir::spirv::GLSLTanhOp>(op)) {
    return processOp(cast<::mlir::spirv::GLSLTanhOp>(op));
  }
  if (isa<::mlir::spirv::GroupBroadcastOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupBroadcastOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformBallotOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformBallotOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformBroadcastOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformBroadcastOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformElectOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformElectOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFAddOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFAddOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFMaxOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFMinOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformFMulOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformFMulOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformIAddOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformIAddOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformIMulOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformIMulOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformSMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformSMaxOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformSMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformSMinOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformUMaxOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformUMaxOp>(op));
  }
  if (isa<::mlir::spirv::GroupNonUniformUMinOp>(op)) {
    return processOp(cast<::mlir::spirv::GroupNonUniformUMinOp>(op));
  }
  if (isa<::mlir::spirv::IAddOp>(op)) {
    return processOp(cast<::mlir::spirv::IAddOp>(op));
  }
  if (isa<::mlir::spirv::IEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::IEqualOp>(op));
  }
  if (isa<::mlir::spirv::IMulOp>(op)) {
    return processOp(cast<::mlir::spirv::IMulOp>(op));
  }
  if (isa<::mlir::spirv::INotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::INotEqualOp>(op));
  }
  if (isa<::mlir::spirv::ISubOp>(op)) {
    return processOp(cast<::mlir::spirv::ISubOp>(op));
  }
  if (isa<::mlir::spirv::LoadOp>(op)) {
    return processOp(cast<::mlir::spirv::LoadOp>(op));
  }
  if (isa<::mlir::spirv::LogicalAndOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalAndOp>(op));
  }
  if (isa<::mlir::spirv::LogicalEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalEqualOp>(op));
  }
  if (isa<::mlir::spirv::LogicalNotEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalNotEqualOp>(op));
  }
  if (isa<::mlir::spirv::LogicalNotOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalNotOp>(op));
  }
  if (isa<::mlir::spirv::LogicalOrOp>(op)) {
    return processOp(cast<::mlir::spirv::LogicalOrOp>(op));
  }
  if (isa<::mlir::spirv::MatrixTimesMatrixOp>(op)) {
    return processOp(cast<::mlir::spirv::MatrixTimesMatrixOp>(op));
  }
  if (isa<::mlir::spirv::MatrixTimesScalarOp>(op)) {
    return processOp(cast<::mlir::spirv::MatrixTimesScalarOp>(op));
  }
  if (isa<::mlir::spirv::MemoryBarrierOp>(op)) {
    return processOp(cast<::mlir::spirv::MemoryBarrierOp>(op));
  }
  if (isa<::mlir::spirv::NotOp>(op)) {
    return processOp(cast<::mlir::spirv::NotOp>(op));
  }
  if (isa<::mlir::spirv::OCLExpOp>(op)) {
    return processOp(cast<::mlir::spirv::OCLExpOp>(op));
  }
  if (isa<::mlir::spirv::OCLFAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::OCLFAbsOp>(op));
  }
  if (isa<::mlir::spirv::OCLSAbsOp>(op)) {
    return processOp(cast<::mlir::spirv::OCLSAbsOp>(op));
  }
  if (isa<::mlir::spirv::ReturnOp>(op)) {
    return processOp(cast<::mlir::spirv::ReturnOp>(op));
  }
  if (isa<::mlir::spirv::ReturnValueOp>(op)) {
    return processOp(cast<::mlir::spirv::ReturnValueOp>(op));
  }
  if (isa<::mlir::spirv::SConvertOp>(op)) {
    return processOp(cast<::mlir::spirv::SConvertOp>(op));
  }
  if (isa<::mlir::spirv::SDivOp>(op)) {
    return processOp(cast<::mlir::spirv::SDivOp>(op));
  }
  if (isa<::mlir::spirv::SGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::SGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::SGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::SGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::SLessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::SLessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::SLessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::SLessThanOp>(op));
  }
  if (isa<::mlir::spirv::SModOp>(op)) {
    return processOp(cast<::mlir::spirv::SModOp>(op));
  }
  if (isa<::mlir::spirv::SNegateOp>(op)) {
    return processOp(cast<::mlir::spirv::SNegateOp>(op));
  }
  if (isa<::mlir::spirv::SRemOp>(op)) {
    return processOp(cast<::mlir::spirv::SRemOp>(op));
  }
  if (isa<::mlir::spirv::SelectOp>(op)) {
    return processOp(cast<::mlir::spirv::SelectOp>(op));
  }
  if (isa<::mlir::spirv::ShiftLeftLogicalOp>(op)) {
    return processOp(cast<::mlir::spirv::ShiftLeftLogicalOp>(op));
  }
  if (isa<::mlir::spirv::ShiftRightArithmeticOp>(op)) {
    return processOp(cast<::mlir::spirv::ShiftRightArithmeticOp>(op));
  }
  if (isa<::mlir::spirv::ShiftRightLogicalOp>(op)) {
    return processOp(cast<::mlir::spirv::ShiftRightLogicalOp>(op));
  }
  if (isa<::mlir::spirv::StoreOp>(op)) {
    return processOp(cast<::mlir::spirv::StoreOp>(op));
  }
  if (isa<::mlir::spirv::SubgroupBallotKHROp>(op)) {
    return processOp(cast<::mlir::spirv::SubgroupBallotKHROp>(op));
  }
  if (isa<::mlir::spirv::SubgroupBlockReadINTELOp>(op)) {
    return processOp(cast<::mlir::spirv::SubgroupBlockReadINTELOp>(op));
  }
  if (isa<::mlir::spirv::SubgroupBlockWriteINTELOp>(op)) {
    return processOp(cast<::mlir::spirv::SubgroupBlockWriteINTELOp>(op));
  }
  if (isa<::mlir::spirv::TransposeOp>(op)) {
    return processOp(cast<::mlir::spirv::TransposeOp>(op));
  }
  if (isa<::mlir::spirv::UConvertOp>(op)) {
    return processOp(cast<::mlir::spirv::UConvertOp>(op));
  }
  if (isa<::mlir::spirv::UDivOp>(op)) {
    return processOp(cast<::mlir::spirv::UDivOp>(op));
  }
  if (isa<::mlir::spirv::UGreaterThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::UGreaterThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::UGreaterThanOp>(op)) {
    return processOp(cast<::mlir::spirv::UGreaterThanOp>(op));
  }
  if (isa<::mlir::spirv::ULessThanEqualOp>(op)) {
    return processOp(cast<::mlir::spirv::ULessThanEqualOp>(op));
  }
  if (isa<::mlir::spirv::ULessThanOp>(op)) {
    return processOp(cast<::mlir::spirv::ULessThanOp>(op));
  }
  if (isa<::mlir::spirv::UModOp>(op)) {
    return processOp(cast<::mlir::spirv::UModOp>(op));
  }
  if (isa<::mlir::spirv::UnreachableOp>(op)) {
    return processOp(cast<::mlir::spirv::UnreachableOp>(op));
  }
  if (isa<::mlir::spirv::VariableOp>(op)) {
    return processOp(cast<::mlir::spirv::VariableOp>(op));
  }
  return op->emitError("unhandled operation serialization");
}

#endif // GET_SERIALIZATION_FNS

#ifdef GET_DESERIALIZATION_FNS

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AccessChainOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AccessChainOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AccessChainOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AccessChainOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AccessChainOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicAndOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicAndOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicAndOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicAndOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicAndOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicCompareExchangeWeakOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicCompareExchangeWeakOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicCompareExchangeWeakOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("equal_semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("unequal_semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicCompareExchangeWeakOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicCompareExchangeWeakOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicIAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicIAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicIAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicIAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicIAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicIDecrementOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicIDecrementOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicIDecrementOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicIDecrementOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicIDecrementOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicIIncrementOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicIIncrementOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicIIncrementOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicIIncrementOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicIIncrementOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicISubOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicISubOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicISubOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicISubOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicISubOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicOrOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicOrOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicOrOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicOrOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicOrOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicSMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicSMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicSMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicSMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicSMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicSMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicSMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicSMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicSMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicSMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicUMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicUMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicUMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicUMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicUMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicUMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicUMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicUMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicUMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicUMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::AtomicXorOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::AtomicXorOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::AtomicXorOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("semantics", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::AtomicXorOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::AtomicXorOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitCountOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::BitCountOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::BitCountOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::BitCountOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::BitCountOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitFieldInsertOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::BitFieldInsertOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::BitFieldInsertOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::BitFieldInsertOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::BitFieldInsertOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitFieldSExtractOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::BitFieldSExtractOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::BitFieldSExtractOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::BitFieldSExtractOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::BitFieldSExtractOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitFieldUExtractOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::BitFieldUExtractOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::BitFieldUExtractOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::BitFieldUExtractOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::BitFieldUExtractOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitReverseOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::BitReverseOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::BitReverseOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::BitReverseOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::BitReverseOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitcastOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::BitcastOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::BitcastOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::BitcastOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::BitcastOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitwiseAndOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::BitwiseAndOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::BitwiseAndOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::BitwiseAndOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::BitwiseAndOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitwiseOrOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::BitwiseOrOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::BitwiseOrOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::BitwiseOrOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::BitwiseOrOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::BitwiseXorOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::BitwiseXorOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::BitwiseXorOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::BitwiseXorOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::BitwiseXorOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CompositeConstructOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CompositeConstructOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CompositeConstructOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CompositeConstructOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CompositeConstructOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CompositeExtractOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CompositeExtractOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CompositeExtractOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    SmallVector<Attribute, 4> attrListElems;
    while (wordIndex < words.size()) {
      attrListElems.push_back(opBuilder.getI32IntegerAttr(words[wordIndex++]));
    }
    attributes.push_back(opBuilder.getNamedAttr("indices", opBuilder.getArrayAttr(attrListElems)));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CompositeExtractOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CompositeExtractOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CompositeInsertOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CompositeInsertOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CompositeInsertOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    SmallVector<Attribute, 4> attrListElems;
    while (wordIndex < words.size()) {
      attrListElems.push_back(opBuilder.getI32IntegerAttr(words[wordIndex++]));
    }
    attributes.push_back(opBuilder.getNamedAttr("indices", opBuilder.getArrayAttr(attrListElems)));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CompositeInsertOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CompositeInsertOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertFToSOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ConvertFToSOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ConvertFToSOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ConvertFToSOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ConvertFToSOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertFToUOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ConvertFToUOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ConvertFToUOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ConvertFToUOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ConvertFToUOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertSToFOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ConvertSToFOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ConvertSToFOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ConvertSToFOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ConvertSToFOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ConvertUToFOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ConvertUToFOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ConvertUToFOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ConvertUToFOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ConvertUToFOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CooperativeMatrixLengthNVOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CooperativeMatrixLengthNVOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CooperativeMatrixLengthNVOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("type", TypeAttr::get(getType(words[wordIndex++]))));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CooperativeMatrixLengthNVOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CooperativeMatrixLengthNVOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CooperativeMatrixLoadNVOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CooperativeMatrixLoadNVOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CooperativeMatrixLoadNVOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CooperativeMatrixLoadNVOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CooperativeMatrixLoadNVOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CooperativeMatrixMulAddNVOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::CooperativeMatrixMulAddNVOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::CooperativeMatrixMulAddNVOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CooperativeMatrixMulAddNVOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CooperativeMatrixMulAddNVOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::CooperativeMatrixStoreNVOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::CooperativeMatrixStoreNVOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::CooperativeMatrixStoreNVOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FConvertOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FConvertOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FConvertOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FConvertOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FConvertOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FDivOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FDivOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FDivOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FDivOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FDivOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FModOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FModOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FModOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FModOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FModOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FMulOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FMulOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FMulOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FMulOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FMulOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FNegateOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FNegateOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FNegateOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FNegateOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FNegateOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FOrdEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FOrdEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FOrdEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FOrdEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FOrdGreaterThanEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FOrdGreaterThanEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FOrdGreaterThanEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FOrdGreaterThanEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdGreaterThanOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FOrdGreaterThanOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FOrdGreaterThanOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FOrdGreaterThanOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FOrdGreaterThanOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdLessThanEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FOrdLessThanEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FOrdLessThanEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FOrdLessThanEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FOrdLessThanEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdLessThanOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FOrdLessThanOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FOrdLessThanOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FOrdLessThanOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FOrdLessThanOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FOrdNotEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FOrdNotEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FOrdNotEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FOrdNotEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FOrdNotEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FRemOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FRemOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FRemOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FRemOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FRemOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FSubOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FSubOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FSubOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FSubOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FSubOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FUnordEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FUnordEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FUnordEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FUnordEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FUnordGreaterThanEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FUnordGreaterThanEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FUnordGreaterThanEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FUnordGreaterThanEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordGreaterThanOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FUnordGreaterThanOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FUnordGreaterThanOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FUnordGreaterThanOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FUnordGreaterThanOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordLessThanEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FUnordLessThanEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FUnordLessThanEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FUnordLessThanEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FUnordLessThanEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordLessThanOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FUnordLessThanOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FUnordLessThanOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FUnordLessThanOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FUnordLessThanOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::FUnordNotEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::FUnordNotEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::FUnordNotEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::FUnordNotEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::FUnordNotEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLAcosOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLAcosOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLAcosOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLAcosOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLAcosOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLAsinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLAsinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLAsinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLAsinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLAsinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLAtanOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLAtanOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLAtanOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLAtanOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLAtanOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLCeilOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLCeilOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLCeilOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLCeilOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLCeilOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLCosOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLCosOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLCosOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLCosOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLCosOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLCoshOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLCoshOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLCoshOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLCoshOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLCoshOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLExpOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLExpOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLExpOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLExpOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLExpOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFAbsOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLFAbsOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLFAbsOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLFAbsOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLFAbsOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLFMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLFMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLFMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLFMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLFMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLFMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLFMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLFMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFSignOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLFSignOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLFSignOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLFSignOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLFSignOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLFloorOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLFloorOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLFloorOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLFloorOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLFloorOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLInverseSqrtOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLInverseSqrtOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLInverseSqrtOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLInverseSqrtOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLInverseSqrtOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLLogOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLLogOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLLogOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLLogOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLLogOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLPowOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLPowOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLPowOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLPowOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLPowOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLRoundOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLRoundOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLRoundOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLRoundOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLRoundOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSAbsOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLSAbsOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLSAbsOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLSAbsOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLSAbsOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLSMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLSMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLSMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLSMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLSMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLSMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLSMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLSMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSSignOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLSSignOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLSSignOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLSSignOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLSSignOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLSinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLSinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLSinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLSinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSinhOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLSinhOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLSinhOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLSinhOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLSinhOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLSqrtOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLSqrtOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLSqrtOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLSqrtOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLSqrtOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLTanOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLTanOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLTanOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLTanOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLTanOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GLSLTanhOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GLSLTanhOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GLSLTanhOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GLSLTanhOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GLSLTanhOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupBroadcastOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupBroadcastOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupBroadcastOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupBroadcastOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupBroadcastOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformBallotOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformBallotOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformBallotOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformBallotOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformBallotOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformBroadcastOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformBroadcastOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformBroadcastOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformBroadcastOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformBroadcastOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformElectOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformElectOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformElectOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformElectOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformElectOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformFMulOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformFMulOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformFMulOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformFMulOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformFMulOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformIAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformIAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformIAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformIAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformIAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformIMulOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformIMulOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformIMulOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformIMulOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformIMulOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformSMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformSMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformSMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformSMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformSMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformSMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformSMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformSMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformSMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformSMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformUMaxOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformUMaxOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformUMaxOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformUMaxOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformUMaxOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::GroupNonUniformUMinOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::GroupNonUniformUMinOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::GroupNonUniformUMinOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("execution_scope", getConstantInt(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("group_operation", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::GroupNonUniformUMinOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::GroupNonUniformUMinOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IAddOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::IAddOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::IAddOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::IAddOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::IAddOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::IEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::IEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::IEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::IEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::IMulOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::IMulOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::IMulOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::IMulOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::IMulOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::INotEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::INotEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::INotEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::INotEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::INotEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ISubOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ISubOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ISubOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ISubOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ISubOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LoadOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::LoadOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::LoadOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("alignment", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::LoadOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::LoadOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalAndOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::LogicalAndOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::LogicalAndOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::LogicalAndOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::LogicalAndOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::LogicalEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::LogicalEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::LogicalEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::LogicalEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalNotEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::LogicalNotEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::LogicalNotEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::LogicalNotEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::LogicalNotEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalNotOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::LogicalNotOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::LogicalNotOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::LogicalNotOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::LogicalNotOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::LogicalOrOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::LogicalOrOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::LogicalOrOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::LogicalOrOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::LogicalOrOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::MatrixTimesMatrixOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::MatrixTimesMatrixOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::MatrixTimesMatrixOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::MatrixTimesMatrixOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::MatrixTimesMatrixOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::MatrixTimesScalarOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::MatrixTimesScalarOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::MatrixTimesScalarOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::MatrixTimesScalarOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::MatrixTimesScalarOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::NotOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::NotOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::NotOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::NotOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::NotOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::OCLExpOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::OCLExpOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::OCLExpOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::OCLExpOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::OCLExpOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::OCLFAbsOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::OCLFAbsOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::OCLFAbsOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::OCLFAbsOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::OCLFAbsOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::OCLSAbsOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::OCLSAbsOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::OCLSAbsOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::OCLSAbsOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::OCLSAbsOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ReturnOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ReturnOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ReturnOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ReturnValueOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ReturnValueOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ReturnValueOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SConvertOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SConvertOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SConvertOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SConvertOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SConvertOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SDivOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SDivOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SDivOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SDivOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SDivOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SGreaterThanEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SGreaterThanEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SGreaterThanEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SGreaterThanEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SGreaterThanOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SGreaterThanOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SGreaterThanOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SGreaterThanOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SGreaterThanOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SLessThanEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SLessThanEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SLessThanEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SLessThanEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SLessThanEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SLessThanOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SLessThanOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SLessThanOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SLessThanOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SLessThanOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SModOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SModOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SModOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SModOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SModOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SNegateOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SNegateOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SNegateOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SNegateOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SNegateOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SRemOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SRemOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SRemOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SRemOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SRemOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SelectOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SelectOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SelectOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SelectOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SelectOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ShiftLeftLogicalOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ShiftLeftLogicalOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ShiftLeftLogicalOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ShiftLeftLogicalOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ShiftLeftLogicalOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ShiftRightArithmeticOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ShiftRightArithmeticOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ShiftRightArithmeticOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ShiftRightArithmeticOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ShiftRightArithmeticOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ShiftRightLogicalOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ShiftRightLogicalOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ShiftRightLogicalOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ShiftRightLogicalOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ShiftRightLogicalOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::StoreOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("memory_access", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("alignment", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::StoreOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::StoreOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SubgroupBallotKHROp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SubgroupBallotKHROp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SubgroupBallotKHROp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SubgroupBallotKHROp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SubgroupBallotKHROp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SubgroupBlockReadINTELOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::SubgroupBlockReadINTELOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::SubgroupBlockReadINTELOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SubgroupBlockReadINTELOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SubgroupBlockReadINTELOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::SubgroupBlockWriteINTELOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::SubgroupBlockWriteINTELOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::SubgroupBlockWriteINTELOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::TransposeOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::TransposeOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::TransposeOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::TransposeOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::TransposeOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UConvertOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::UConvertOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::UConvertOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::UConvertOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::UConvertOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UDivOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::UDivOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::UDivOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::UDivOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::UDivOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UGreaterThanEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::UGreaterThanEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::UGreaterThanEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::UGreaterThanEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::UGreaterThanEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UGreaterThanOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::UGreaterThanOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::UGreaterThanOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::UGreaterThanOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::UGreaterThanOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ULessThanEqualOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ULessThanEqualOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ULessThanEqualOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ULessThanEqualOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ULessThanEqualOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::ULessThanOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::ULessThanOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::ULessThanOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::ULessThanOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::ULessThanOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UModOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::UModOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::UModOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex < words.size()) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
    wordIndex++;
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::UModOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::UModOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::UnreachableOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::UnreachableOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::UnreachableOp>(loc, resultTypes, operands, attributes); (void)op;
  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  return success();
}

template <> LogicalResult
Deserializer::processOp<::mlir::spirv::VariableOp>(ArrayRef<uint32_t> words) {
  SmallVector<Type, 1> resultTypes;
  size_t wordIndex = 0; (void)wordIndex;
  uint32_t valueID = 0; (void)valueID;
  {
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result type <id> while deserializing ::mlir::spirv::VariableOp");
    }
    auto ty = getType(words[wordIndex]);
    if (!ty) {
      return emitError(unknownLoc, "unknown type result <id> : ") << words[wordIndex];
    }
    resultTypes.push_back(ty);
    wordIndex++;
    if (wordIndex >= words.size()) {
      return emitError(unknownLoc, "expected result <id> while deserializing ::mlir::spirv::VariableOp");
    }
  }
  valueID = words[wordIndex++];
  SmallVector<Value, 4> operands;
  SmallVector<NamedAttribute, 4> attributes;
  if (wordIndex < words.size()) {
    attributes.push_back(opBuilder.getNamedAttr("storage_class", opBuilder.getI32IntegerAttr(words[wordIndex++])));
  }
  for (; wordIndex < words.size(); ++wordIndex) {
    auto arg = getValue(words[wordIndex]);
    if (!arg) {
      return emitError(unknownLoc, "unknown result <id> : ") << words[wordIndex];
    }
    operands.push_back(arg);
  }
  if (wordIndex != words.size()) {
    return emitError(unknownLoc, "found more operands than expected when deserializing ::mlir::spirv::VariableOp, only ") << wordIndex << " of " << words.size() << " processed";
  }

  Location loc = createFileLineColLoc(opBuilder);
  auto op = opBuilder.create<::mlir::spirv::VariableOp>(loc, resultTypes, operands, attributes); (void)op;
  valueMap[valueID] = op.getResult();

  if (op.hasTrait<OpTrait::IsTerminator>())
    clearDebugLine();
  if (decorations.count(valueID)) {
    auto attrs = decorations[valueID].getAttrs();
    attributes.append(attrs.begin(), attrs.end());
  }
  return success();
}

LogicalResult Deserializer::dispatchToAutogenDeserialization(spirv::Opcode opcode, ArrayRef<uint32_t> words) {
  switch (opcode) {
  case spirv::Opcode::OpAccessChain:
    return processOp<::mlir::spirv::AccessChainOp>(words);
  case spirv::Opcode::OpAtomicAnd:
    return processOp<::mlir::spirv::AtomicAndOp>(words);
  case spirv::Opcode::OpAtomicCompareExchangeWeak:
    return processOp<::mlir::spirv::AtomicCompareExchangeWeakOp>(words);
  case spirv::Opcode::OpAtomicIAdd:
    return processOp<::mlir::spirv::AtomicIAddOp>(words);
  case spirv::Opcode::OpAtomicIDecrement:
    return processOp<::mlir::spirv::AtomicIDecrementOp>(words);
  case spirv::Opcode::OpAtomicIIncrement:
    return processOp<::mlir::spirv::AtomicIIncrementOp>(words);
  case spirv::Opcode::OpAtomicISub:
    return processOp<::mlir::spirv::AtomicISubOp>(words);
  case spirv::Opcode::OpAtomicOr:
    return processOp<::mlir::spirv::AtomicOrOp>(words);
  case spirv::Opcode::OpAtomicSMax:
    return processOp<::mlir::spirv::AtomicSMaxOp>(words);
  case spirv::Opcode::OpAtomicSMin:
    return processOp<::mlir::spirv::AtomicSMinOp>(words);
  case spirv::Opcode::OpAtomicUMax:
    return processOp<::mlir::spirv::AtomicUMaxOp>(words);
  case spirv::Opcode::OpAtomicUMin:
    return processOp<::mlir::spirv::AtomicUMinOp>(words);
  case spirv::Opcode::OpAtomicXor:
    return processOp<::mlir::spirv::AtomicXorOp>(words);
  case spirv::Opcode::OpBitCount:
    return processOp<::mlir::spirv::BitCountOp>(words);
  case spirv::Opcode::OpBitFieldInsert:
    return processOp<::mlir::spirv::BitFieldInsertOp>(words);
  case spirv::Opcode::OpBitFieldSExtract:
    return processOp<::mlir::spirv::BitFieldSExtractOp>(words);
  case spirv::Opcode::OpBitFieldUExtract:
    return processOp<::mlir::spirv::BitFieldUExtractOp>(words);
  case spirv::Opcode::OpBitReverse:
    return processOp<::mlir::spirv::BitReverseOp>(words);
  case spirv::Opcode::OpBitcast:
    return processOp<::mlir::spirv::BitcastOp>(words);
  case spirv::Opcode::OpBitwiseAnd:
    return processOp<::mlir::spirv::BitwiseAndOp>(words);
  case spirv::Opcode::OpBitwiseOr:
    return processOp<::mlir::spirv::BitwiseOrOp>(words);
  case spirv::Opcode::OpBitwiseXor:
    return processOp<::mlir::spirv::BitwiseXorOp>(words);
  case spirv::Opcode::OpBranchConditional:
    return processOp<::mlir::spirv::BranchConditionalOp>(words);
  case spirv::Opcode::OpBranch:
    return processOp<::mlir::spirv::BranchOp>(words);
  case spirv::Opcode::OpCompositeConstruct:
    return processOp<::mlir::spirv::CompositeConstructOp>(words);
  case spirv::Opcode::OpCompositeExtract:
    return processOp<::mlir::spirv::CompositeExtractOp>(words);
  case spirv::Opcode::OpCompositeInsert:
    return processOp<::mlir::spirv::CompositeInsertOp>(words);
  case spirv::Opcode::OpControlBarrier:
    return processOp<::mlir::spirv::ControlBarrierOp>(words);
  case spirv::Opcode::OpConvertFToS:
    return processOp<::mlir::spirv::ConvertFToSOp>(words);
  case spirv::Opcode::OpConvertFToU:
    return processOp<::mlir::spirv::ConvertFToUOp>(words);
  case spirv::Opcode::OpConvertSToF:
    return processOp<::mlir::spirv::ConvertSToFOp>(words);
  case spirv::Opcode::OpConvertUToF:
    return processOp<::mlir::spirv::ConvertUToFOp>(words);
  case spirv::Opcode::OpCooperativeMatrixLengthNV:
    return processOp<::mlir::spirv::CooperativeMatrixLengthNVOp>(words);
  case spirv::Opcode::OpCooperativeMatrixLoadNV:
    return processOp<::mlir::spirv::CooperativeMatrixLoadNVOp>(words);
  case spirv::Opcode::OpCooperativeMatrixMulAddNV:
    return processOp<::mlir::spirv::CooperativeMatrixMulAddNVOp>(words);
  case spirv::Opcode::OpCooperativeMatrixStoreNV:
    return processOp<::mlir::spirv::CooperativeMatrixStoreNVOp>(words);
  case spirv::Opcode::OpCopyMemory:
    return processOp<::mlir::spirv::CopyMemoryOp>(words);
  case spirv::Opcode::OpEntryPoint:
    return processOp<::mlir::spirv::EntryPointOp>(words);
  case spirv::Opcode::OpExecutionMode:
    return processOp<::mlir::spirv::ExecutionModeOp>(words);
  case spirv::Opcode::OpFAdd:
    return processOp<::mlir::spirv::FAddOp>(words);
  case spirv::Opcode::OpFConvert:
    return processOp<::mlir::spirv::FConvertOp>(words);
  case spirv::Opcode::OpFDiv:
    return processOp<::mlir::spirv::FDivOp>(words);
  case spirv::Opcode::OpFMod:
    return processOp<::mlir::spirv::FModOp>(words);
  case spirv::Opcode::OpFMul:
    return processOp<::mlir::spirv::FMulOp>(words);
  case spirv::Opcode::OpFNegate:
    return processOp<::mlir::spirv::FNegateOp>(words);
  case spirv::Opcode::OpFOrdEqual:
    return processOp<::mlir::spirv::FOrdEqualOp>(words);
  case spirv::Opcode::OpFOrdGreaterThanEqual:
    return processOp<::mlir::spirv::FOrdGreaterThanEqualOp>(words);
  case spirv::Opcode::OpFOrdGreaterThan:
    return processOp<::mlir::spirv::FOrdGreaterThanOp>(words);
  case spirv::Opcode::OpFOrdLessThanEqual:
    return processOp<::mlir::spirv::FOrdLessThanEqualOp>(words);
  case spirv::Opcode::OpFOrdLessThan:
    return processOp<::mlir::spirv::FOrdLessThanOp>(words);
  case spirv::Opcode::OpFOrdNotEqual:
    return processOp<::mlir::spirv::FOrdNotEqualOp>(words);
  case spirv::Opcode::OpFRem:
    return processOp<::mlir::spirv::FRemOp>(words);
  case spirv::Opcode::OpFSub:
    return processOp<::mlir::spirv::FSubOp>(words);
  case spirv::Opcode::OpFUnordEqual:
    return processOp<::mlir::spirv::FUnordEqualOp>(words);
  case spirv::Opcode::OpFUnordGreaterThanEqual:
    return processOp<::mlir::spirv::FUnordGreaterThanEqualOp>(words);
  case spirv::Opcode::OpFUnordGreaterThan:
    return processOp<::mlir::spirv::FUnordGreaterThanOp>(words);
  case spirv::Opcode::OpFUnordLessThanEqual:
    return processOp<::mlir::spirv::FUnordLessThanEqualOp>(words);
  case spirv::Opcode::OpFUnordLessThan:
    return processOp<::mlir::spirv::FUnordLessThanOp>(words);
  case spirv::Opcode::OpFUnordNotEqual:
    return processOp<::mlir::spirv::FUnordNotEqualOp>(words);
  case spirv::Opcode::OpFunctionCall:
    return processOp<::mlir::spirv::FunctionCallOp>(words);
  case spirv::Opcode::OpGroupBroadcast:
    return processOp<::mlir::spirv::GroupBroadcastOp>(words);
  case spirv::Opcode::OpGroupNonUniformBallot:
    return processOp<::mlir::spirv::GroupNonUniformBallotOp>(words);
  case spirv::Opcode::OpGroupNonUniformBroadcast:
    return processOp<::mlir::spirv::GroupNonUniformBroadcastOp>(words);
  case spirv::Opcode::OpGroupNonUniformElect:
    return processOp<::mlir::spirv::GroupNonUniformElectOp>(words);
  case spirv::Opcode::OpGroupNonUniformFAdd:
    return processOp<::mlir::spirv::GroupNonUniformFAddOp>(words);
  case spirv::Opcode::OpGroupNonUniformFMax:
    return processOp<::mlir::spirv::GroupNonUniformFMaxOp>(words);
  case spirv::Opcode::OpGroupNonUniformFMin:
    return processOp<::mlir::spirv::GroupNonUniformFMinOp>(words);
  case spirv::Opcode::OpGroupNonUniformFMul:
    return processOp<::mlir::spirv::GroupNonUniformFMulOp>(words);
  case spirv::Opcode::OpGroupNonUniformIAdd:
    return processOp<::mlir::spirv::GroupNonUniformIAddOp>(words);
  case spirv::Opcode::OpGroupNonUniformIMul:
    return processOp<::mlir::spirv::GroupNonUniformIMulOp>(words);
  case spirv::Opcode::OpGroupNonUniformSMax:
    return processOp<::mlir::spirv::GroupNonUniformSMaxOp>(words);
  case spirv::Opcode::OpGroupNonUniformSMin:
    return processOp<::mlir::spirv::GroupNonUniformSMinOp>(words);
  case spirv::Opcode::OpGroupNonUniformUMax:
    return processOp<::mlir::spirv::GroupNonUniformUMaxOp>(words);
  case spirv::Opcode::OpGroupNonUniformUMin:
    return processOp<::mlir::spirv::GroupNonUniformUMinOp>(words);
  case spirv::Opcode::OpIAdd:
    return processOp<::mlir::spirv::IAddOp>(words);
  case spirv::Opcode::OpIEqual:
    return processOp<::mlir::spirv::IEqualOp>(words);
  case spirv::Opcode::OpIMul:
    return processOp<::mlir::spirv::IMulOp>(words);
  case spirv::Opcode::OpINotEqual:
    return processOp<::mlir::spirv::INotEqualOp>(words);
  case spirv::Opcode::OpISub:
    return processOp<::mlir::spirv::ISubOp>(words);
  case spirv::Opcode::OpLoad:
    return processOp<::mlir::spirv::LoadOp>(words);
  case spirv::Opcode::OpLogicalAnd:
    return processOp<::mlir::spirv::LogicalAndOp>(words);
  case spirv::Opcode::OpLogicalEqual:
    return processOp<::mlir::spirv::LogicalEqualOp>(words);
  case spirv::Opcode::OpLogicalNotEqual:
    return processOp<::mlir::spirv::LogicalNotEqualOp>(words);
  case spirv::Opcode::OpLogicalNot:
    return processOp<::mlir::spirv::LogicalNotOp>(words);
  case spirv::Opcode::OpLogicalOr:
    return processOp<::mlir::spirv::LogicalOrOp>(words);
  case spirv::Opcode::OpMatrixTimesMatrix:
    return processOp<::mlir::spirv::MatrixTimesMatrixOp>(words);
  case spirv::Opcode::OpMatrixTimesScalar:
    return processOp<::mlir::spirv::MatrixTimesScalarOp>(words);
  case spirv::Opcode::OpMemoryBarrier:
    return processOp<::mlir::spirv::MemoryBarrierOp>(words);
  case spirv::Opcode::OpNot:
    return processOp<::mlir::spirv::NotOp>(words);
  case spirv::Opcode::OpReturn:
    return processOp<::mlir::spirv::ReturnOp>(words);
  case spirv::Opcode::OpReturnValue:
    return processOp<::mlir::spirv::ReturnValueOp>(words);
  case spirv::Opcode::OpSConvert:
    return processOp<::mlir::spirv::SConvertOp>(words);
  case spirv::Opcode::OpSDiv:
    return processOp<::mlir::spirv::SDivOp>(words);
  case spirv::Opcode::OpSGreaterThanEqual:
    return processOp<::mlir::spirv::SGreaterThanEqualOp>(words);
  case spirv::Opcode::OpSGreaterThan:
    return processOp<::mlir::spirv::SGreaterThanOp>(words);
  case spirv::Opcode::OpSLessThanEqual:
    return processOp<::mlir::spirv::SLessThanEqualOp>(words);
  case spirv::Opcode::OpSLessThan:
    return processOp<::mlir::spirv::SLessThanOp>(words);
  case spirv::Opcode::OpSMod:
    return processOp<::mlir::spirv::SModOp>(words);
  case spirv::Opcode::OpSNegate:
    return processOp<::mlir::spirv::SNegateOp>(words);
  case spirv::Opcode::OpSRem:
    return processOp<::mlir::spirv::SRemOp>(words);
  case spirv::Opcode::OpSelect:
    return processOp<::mlir::spirv::SelectOp>(words);
  case spirv::Opcode::OpShiftLeftLogical:
    return processOp<::mlir::spirv::ShiftLeftLogicalOp>(words);
  case spirv::Opcode::OpShiftRightArithmetic:
    return processOp<::mlir::spirv::ShiftRightArithmeticOp>(words);
  case spirv::Opcode::OpShiftRightLogical:
    return processOp<::mlir::spirv::ShiftRightLogicalOp>(words);
  case spirv::Opcode::OpStore:
    return processOp<::mlir::spirv::StoreOp>(words);
  case spirv::Opcode::OpSubgroupBallotKHR:
    return processOp<::mlir::spirv::SubgroupBallotKHROp>(words);
  case spirv::Opcode::OpSubgroupBlockReadINTEL:
    return processOp<::mlir::spirv::SubgroupBlockReadINTELOp>(words);
  case spirv::Opcode::OpSubgroupBlockWriteINTEL:
    return processOp<::mlir::spirv::SubgroupBlockWriteINTELOp>(words);
  case spirv::Opcode::OpTranspose:
    return processOp<::mlir::spirv::TransposeOp>(words);
  case spirv::Opcode::OpUConvert:
    return processOp<::mlir::spirv::UConvertOp>(words);
  case spirv::Opcode::OpUDiv:
    return processOp<::mlir::spirv::UDivOp>(words);
  case spirv::Opcode::OpUGreaterThanEqual:
    return processOp<::mlir::spirv::UGreaterThanEqualOp>(words);
  case spirv::Opcode::OpUGreaterThan:
    return processOp<::mlir::spirv::UGreaterThanOp>(words);
  case spirv::Opcode::OpULessThanEqual:
    return processOp<::mlir::spirv::ULessThanEqualOp>(words);
  case spirv::Opcode::OpULessThan:
    return processOp<::mlir::spirv::ULessThanOp>(words);
  case spirv::Opcode::OpUMod:
    return processOp<::mlir::spirv::UModOp>(words);
  case spirv::Opcode::OpUnreachable:
    return processOp<::mlir::spirv::UnreachableOp>(words);
  case spirv::Opcode::OpVariable:
    return processOp<::mlir::spirv::VariableOp>(words);
  default:
    ;
  }
  auto opcodeString = spirv::stringifyOpcode(opcode);
  if (!opcodeString.empty()) {
    return emitError(unknownLoc, "unhandled deserialization of ") << opcodeString;
  } else {
   return emitError(unknownLoc, "unhandled opcode ") << static_cast<uint32_t>(opcode);
  }
}
LogicalResult Deserializer::dispatchToExtensionSetAutogenDeserialization(StringRef extensionSetName, uint32_t instructionID, ArrayRef<uint32_t> words) {
  if (extensionSetName == "GLSL.std.450") {
    switch (instructionID) {
    case 17:
      return processOp<::mlir::spirv::GLSLAcosOp>(words);
    case 16:
      return processOp<::mlir::spirv::GLSLAsinOp>(words);
    case 18:
      return processOp<::mlir::spirv::GLSLAtanOp>(words);
    case 9:
      return processOp<::mlir::spirv::GLSLCeilOp>(words);
    case 14:
      return processOp<::mlir::spirv::GLSLCosOp>(words);
    case 20:
      return processOp<::mlir::spirv::GLSLCoshOp>(words);
    case 27:
      return processOp<::mlir::spirv::GLSLExpOp>(words);
    case 4:
      return processOp<::mlir::spirv::GLSLFAbsOp>(words);
    case 40:
      return processOp<::mlir::spirv::GLSLFMaxOp>(words);
    case 37:
      return processOp<::mlir::spirv::GLSLFMinOp>(words);
    case 6:
      return processOp<::mlir::spirv::GLSLFSignOp>(words);
    case 8:
      return processOp<::mlir::spirv::GLSLFloorOp>(words);
    case 32:
      return processOp<::mlir::spirv::GLSLInverseSqrtOp>(words);
    case 28:
      return processOp<::mlir::spirv::GLSLLogOp>(words);
    case 26:
      return processOp<::mlir::spirv::GLSLPowOp>(words);
    case 1:
      return processOp<::mlir::spirv::GLSLRoundOp>(words);
    case 5:
      return processOp<::mlir::spirv::GLSLSAbsOp>(words);
    case 42:
      return processOp<::mlir::spirv::GLSLSMaxOp>(words);
    case 39:
      return processOp<::mlir::spirv::GLSLSMinOp>(words);
    case 7:
      return processOp<::mlir::spirv::GLSLSSignOp>(words);
    case 13:
      return processOp<::mlir::spirv::GLSLSinOp>(words);
    case 19:
      return processOp<::mlir::spirv::GLSLSinhOp>(words);
    case 31:
      return processOp<::mlir::spirv::GLSLSqrtOp>(words);
    case 15:
      return processOp<::mlir::spirv::GLSLTanOp>(words);
    case 21:
      return processOp<::mlir::spirv::GLSLTanhOp>(words);
    default:
      return emitError(unknownLoc, "unhandled deserializations of ") << instructionID << " from extension set " << extensionSetName;
    }
  }
  if (extensionSetName == "OpenCL.std") {
    switch (instructionID) {
    case 19:
      return processOp<::mlir::spirv::OCLExpOp>(words);
    case 23:
      return processOp<::mlir::spirv::OCLFAbsOp>(words);
    case 141:
      return processOp<::mlir::spirv::OCLSAbsOp>(words);
    default:
      return emitError(unknownLoc, "unhandled deserializations of ") << instructionID << " from extension set " << extensionSetName;
    }
  }
  return emitError(unknownLoc, "unhandled deserialization of extended instruction set extensionSetName");
}
#endif // GET_DESERIALIZATION_FNS

