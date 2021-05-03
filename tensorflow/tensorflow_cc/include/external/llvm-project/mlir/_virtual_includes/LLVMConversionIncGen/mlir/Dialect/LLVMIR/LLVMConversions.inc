if (auto op = dyn_cast<::mlir::LLVM::AShrOp>(opInst)) {
valueMapping[op.res()] = builder.CreateAShr(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AddOp>(opInst)) {
valueMapping[op.res()] = builder.CreateAdd(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AddrSpaceCastOp>(opInst)) {
valueMapping[op.res()] = builder.CreateAddrSpaceCast(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AllocaOp>(opInst)) {

    auto *inst = builder.CreateAlloca(
      convertType(op.getResult().getType().cast<LLVM::LLVMType>())->getPointerElementType(), valueMapping.lookup(op.arraySize()));
    
    if (op.alignment().hasValue()) {
      auto align = op.alignment().getValue();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    valueMapping[op.res()] = inst;
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AndOp>(opInst)) {
valueMapping[op.res()] = builder.CreateAnd(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AssumeOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn =
        llvm::Intrinsic::getDeclaration(module, llvm::Intrinsic::assume, {});
    builder.CreateCall(fn, {valueMapping.lookup(op.cond())});
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AtomicCmpXchgOp>(opInst)) {

    valueMapping[op.res()] = builder.CreateAtomicCmpXchg(valueMapping.lookup(op.ptr()), valueMapping.lookup(op.cmp()), valueMapping.lookup(op.val()),
                   getLLVMAtomicOrdering(op.success_ordering()),
                   getLLVMAtomicOrdering(op.failure_ordering()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AtomicRMWOp>(opInst)) {

    valueMapping[op.res()] = builder.CreateAtomicRMW(getLLVMAtomicBinOp(op.bin_op()), valueMapping.lookup(op.ptr()), valueMapping.lookup(op.val()),
                                   getLLVMAtomicOrdering(op.ordering()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::BitReverseOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::bitreverse,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::BitcastOp>(opInst)) {
valueMapping[op.res()] = builder.CreateBitCast(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ConstantOp>(opInst)) {
valueMapping[op.res()] = getLLVMConstant(convertType(op.getResult().getType().cast<LLVM::LLVMType>()), op.value(), opInst.getLoc());
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CopySignOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::copysign,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CosOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::cos,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::CtPopOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::ctpop,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Exp2Op>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::exp2,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExpOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::exp,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExtractElementOp>(opInst)) {

    valueMapping[op.res()] = builder.CreateExtractElement(valueMapping.lookup(op.vector()), valueMapping.lookup(op.position()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExtractValueOp>(opInst)) {

    valueMapping[op.res()] = builder.CreateExtractValue(valueMapping.lookup(op.container()), extractPosition(op.position()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FAbsOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::fabs,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FAddOp>(opInst)) {
valueMapping[op.res()] = builder.CreateFAdd(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FCeilOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::ceil,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FCmpOp>(opInst)) {

    valueMapping[op.res()] = builder.CreateFCmp(getLLVMCmpPredicate(op.predicate()), valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FDivOp>(opInst)) {
valueMapping[op.res()] = builder.CreateFDiv(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FFloorOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::floor,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMAOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::fma,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMulAddOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::fmuladd,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMulOp>(opInst)) {
valueMapping[op.res()] = builder.CreateFMul(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FNegOp>(opInst)) {
valueMapping[op.res()] = builder.CreateFNeg(valueMapping.lookup(op.operand()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPExtOp>(opInst)) {
valueMapping[op.res()] = builder.CreateFPExt(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPToSIOp>(opInst)) {
valueMapping[op.res()] = builder.CreateFPToSI(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPToUIOp>(opInst)) {
valueMapping[op.res()] = builder.CreateFPToUI(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPTruncOp>(opInst)) {
valueMapping[op.res()] = builder.CreateFPTrunc(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FRemOp>(opInst)) {
valueMapping[op.res()] = builder.CreateFRem(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FSubOp>(opInst)) {
valueMapping[op.res()] = builder.CreateFSub(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FenceOp>(opInst)) {

    llvm::LLVMContext &llvmContext = builder.getContext();
    builder.CreateFence(getLLVMAtomicOrdering(op.ordering()),
      llvmContext.getOrInsertSyncScopeID(op.syncscope()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FreezeOp>(opInst)) {
builder.CreateFreeze(valueMapping.lookup(op.val()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::GEPOp>(opInst)) {
valueMapping[op.res()] = builder.CreateGEP(valueMapping.lookup(op.base()), lookupValues(op.indices()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::GetActiveLaneMaskOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::get_active_lane_mask,
        { convertType(opInst.getResult(0).getType().cast<LLVM::LLVMType>()), convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ICmpOp>(opInst)) {

    valueMapping[op.res()] = builder.CreateICmp(getLLVMCmpPredicate(op.predicate()), valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::InsertElementOp>(opInst)) {

    valueMapping[op.res()] = builder.CreateInsertElement(valueMapping.lookup(op.vector()), valueMapping.lookup(op.value()), valueMapping.lookup(op.position()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::InsertValueOp>(opInst)) {

    valueMapping[op.res()] = builder.CreateInsertValue(valueMapping.lookup(op.container()), valueMapping.lookup(op.value()),
                                     extractPosition(op.position()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::IntToPtrOp>(opInst)) {
valueMapping[op.res()] = builder.CreateIntToPtr(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LShrOp>(opInst)) {
valueMapping[op.res()] = builder.CreateLShr(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LoadOp>(opInst)) {

    auto *inst = builder.CreateLoad(valueMapping.lookup(op.addr()), op.volatile_());
  
    if (op.alignment().hasValue()) {
      auto align = op.alignment().getValue();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    if (op.nontemporal()) {
      llvm::Module *module = builder.GetInsertBlock()->getModule();
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(module->getMDKindID("nontemporal"), metadata);
    }
  
    valueMapping[op.res()] = inst;
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Log10Op>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::log10,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Log2Op>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::log2,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LogOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::log,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaskedLoadOp>(opInst)) {

    valueMapping[op.res()] = lookupValues(op.pass_thru()).empty() ? builder.CreateMaskedLoad(
      valueMapping.lookup(op.data()), llvm::Align(op.alignment()), valueMapping.lookup(op.mask())) :
      builder.CreateMaskedLoad(
        valueMapping.lookup(op.data()), llvm::Align(op.alignment()), valueMapping.lookup(op.mask()), lookupValues(op.pass_thru())[0]);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaskedStoreOp>(opInst)) {

    builder.CreateMaskedStore(
      valueMapping.lookup(op.value()), valueMapping.lookup(op.data()), llvm::Align(op.alignment()), valueMapping.lookup(op.mask()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixColumnMajorLoadOp>(opInst)) {

    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Align align = dl.getABITypeAlign(
      valueMapping.lookup(op.data())->getType()->getPointerElementType());
    valueMapping[op.res()] = mb.CreateColumnMajorLoad(
      valueMapping.lookup(op.data()), align, valueMapping.lookup(op.stride()), op.isVolatile(), op.rows(),
      op.columns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixColumnMajorStoreOp>(opInst)) {

    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    const llvm::DataLayout &dl =
      builder.GetInsertBlock()->getModule()->getDataLayout();
    llvm::Align align = dl.getABITypeAlign(
      valueMapping.lookup(op.data())->getType()->getPointerElementType());
    mb.CreateColumnMajorStore(
      valueMapping.lookup(op.matrix()), valueMapping.lookup(op.data()), align, valueMapping.lookup(op.stride()), op.isVolatile(),
      op.rows(), op.columns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixMultiplyOp>(opInst)) {

    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    valueMapping[op.res()] = mb.CreateMatrixMultiply(
      valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()), op.lhs_rows(), op.lhs_columns(),
      op.rhs_columns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MatrixTransposeOp>(opInst)) {

    llvm::MatrixBuilder<decltype(builder)> mb(builder);
    valueMapping[op.res()] = mb.CreateMatrixTranspose(
      valueMapping.lookup(op.matrix()), op.rows(), op.columns());
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MaxNumOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::maxnum,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemcpyInlineOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::memcpy_inline,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>()), convertType(opInst.getOperand(1).getType().cast<LLVM::LLVMType>()), convertType(opInst.getOperand(2).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MemcpyOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::memcpy,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>()), convertType(opInst.getOperand(1).getType().cast<LLVM::LLVMType>()), convertType(opInst.getOperand(2).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MinNumOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::minnum,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MulOp>(opInst)) {
valueMapping[op.res()] = builder.CreateMul(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::NullOp>(opInst)) {
valueMapping[op.res()] = llvm::ConstantPointerNull::get(    cast<llvm::PointerType>(convertType(op.getResult().getType().cast<LLVM::LLVMType>())));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::OrOp>(opInst)) {
valueMapping[op.res()] = builder.CreateOr(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PowOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::pow,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::Prefetch>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::prefetch,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PtrToIntOp>(opInst)) {
valueMapping[op.res()] = builder.CreatePtrToInt(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ResumeOp>(opInst)) {
 builder.CreateResume(valueMapping.lookup(op.value())); 
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ReturnOp>(opInst)) {

    if (opInst.getNumOperands() != 0)
      builder.CreateRet(lookupValues(op.args())[0]);
    else
      builder.CreateRetVoid();
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SDivOp>(opInst)) {
valueMapping[op.res()] = builder.CreateSDiv(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SExtOp>(opInst)) {
valueMapping[op.res()] = builder.CreateSExt(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SIToFPOp>(opInst)) {
valueMapping[op.res()] = builder.CreateSIToFP(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SMaxOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::smax,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SMinOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::smin,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SRemOp>(opInst)) {
valueMapping[op.res()] = builder.CreateSRem(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SelectOp>(opInst)) {
valueMapping[op.res()] = builder.CreateSelect(valueMapping.lookup(op.condition()), valueMapping.lookup(op.trueValue()), valueMapping.lookup(op.falseValue()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ShlOp>(opInst)) {
valueMapping[op.res()] = builder.CreateShl(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ShuffleVectorOp>(opInst)) {

      SmallVector<unsigned, 4> position = extractPosition(op.mask());
      SmallVector<int, 4> mask(position.begin(), position.end());
      valueMapping[op.res()] = builder.CreateShuffleVector(valueMapping.lookup(op.v1()), valueMapping.lookup(op.v2()), mask);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SinOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::sin,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SqrtOp>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::sqrt,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StoreOp>(opInst)) {

    auto *inst = builder.CreateStore(valueMapping.lookup(op.value()), valueMapping.lookup(op.addr()), op.volatile_());
  
    if (op.alignment().hasValue()) {
      auto align = op.alignment().getValue();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    if (op.nontemporal()) {
      llvm::Module *module = builder.GetInsertBlock()->getModule();
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(module->getMDKindID("nontemporal"), metadata);
    }
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SubOp>(opInst)) {
valueMapping[op.res()] = builder.CreateSub(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::TruncOp>(opInst)) {
valueMapping[op.res()] = builder.CreateTrunc(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UDivOp>(opInst)) {
valueMapping[op.res()] = builder.CreateUDiv(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UIToFPOp>(opInst)) {
valueMapping[op.res()] = builder.CreateUIToFP(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::URemOp>(opInst)) {
valueMapping[op.res()] = builder.CreateURem(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UndefOp>(opInst)) {
valueMapping[op.res()] = llvm::UndefValue::get(convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UnreachableOp>(opInst)) {
 builder.CreateUnreachable(); 
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::XOrOp>(opInst)) {
valueMapping[op.res()] = builder.CreateXor(valueMapping.lookup(op.lhs()), valueMapping.lookup(op.rhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ZExtOp>(opInst)) {
valueMapping[op.res()] = builder.CreateZExt(valueMapping.lookup(op.arg()), convertType(op.getResult().getType().cast<LLVM::LLVMType>()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_compressstore>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::masked_compressstore,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_expandload>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::masked_expandload,
        { convertType(opInst.getResult(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_gather>(opInst)) {

    valueMapping[op.res()] = lookupValues(op.pass_thru()).empty() ? builder.CreateMaskedGather(
      valueMapping.lookup(op.ptrs()), llvm::Align(op.alignment()), valueMapping.lookup(op.mask())) :
      builder.CreateMaskedGather(
        valueMapping.lookup(op.ptrs()), llvm::Align(op.alignment()), valueMapping.lookup(op.mask()), lookupValues(op.pass_thru())[0]);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::masked_scatter>(opInst)) {

    builder.CreateMaskedScatter(
      valueMapping.lookup(op.value()), valueMapping.lookup(op.ptrs()), llvm::Align(op.alignment()), valueMapping.lookup(op.mask()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_add>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_add,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_and>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_and,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fadd>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fadd,
        { convertType(opInst.getOperand(1).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    llvm::FastMathFlags origFM = builder.getFastMathFlags();
    llvm::FastMathFlags tempFM = origFM;
    tempFM.setAllowReassoc(op.reassoc());
    builder.setFastMathFlags(tempFM);  // set fastmath flag
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
    builder.setFastMathFlags(origFM);  // restore fastmath flag
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmax>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fmax,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmin>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fmin,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_fmul>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_fmul,
        { convertType(opInst.getOperand(1).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    llvm::FastMathFlags origFM = builder.getFastMathFlags();
    llvm::FastMathFlags tempFM = origFM;
    tempFM.setAllowReassoc(op.reassoc());
    builder.setFastMathFlags(tempFM);  // set fastmath flag
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
    builder.setFastMathFlags(origFM);  // restore fastmath flag
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_mul>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_mul,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_or>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_or,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_smax>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_smax,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_smin>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_smin,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_umax>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_umax,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_umin>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_umin,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::vector_reduce_xor>(opInst)) {

    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::vector_reduce_xor,
        { convertType(opInst.getOperand(0).getType().cast<LLVM::LLVMType>())
        });
    auto operands = lookupValues(opInst.getOperands());
    valueMapping[op.res()] = builder.CreateCall(fn, operands);
  
  return success();
}
