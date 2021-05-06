/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace acc {
class DataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class EnterDataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ExitDataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class InitOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class LoopOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ParallelOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ShutdownOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class TerminatorOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class UpdateOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class WaitOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class YieldOp;
} // namespace acc
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DataOp declarations
//===----------------------------------------------------------------------===//

class DataOpAdaptor {
public:
  DataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  DataOpAdaptor(DataOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::ValueRange copyOperands();
  ::mlir::ValueRange copyinOperands();
  ::mlir::ValueRange copyinReadonlyOperands();
  ::mlir::ValueRange copyoutOperands();
  ::mlir::ValueRange copyoutZeroOperands();
  ::mlir::ValueRange createOperands();
  ::mlir::ValueRange createZeroOperands();
  ::mlir::ValueRange noCreateOperands();
  ::mlir::ValueRange presentOperands();
  ::mlir::ValueRange deviceptrOperands();
  ::mlir::ValueRange attachOperands();
  ::mlir::StringAttr defaultAttr();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class DataOp : public ::mlir::Op<DataOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Adaptor = DataOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Operation::operand_range copyOperands();
  ::mlir::Operation::operand_range copyinOperands();
  ::mlir::Operation::operand_range copyinReadonlyOperands();
  ::mlir::Operation::operand_range copyoutOperands();
  ::mlir::Operation::operand_range copyoutZeroOperands();
  ::mlir::Operation::operand_range createOperands();
  ::mlir::Operation::operand_range createZeroOperands();
  ::mlir::Operation::operand_range noCreateOperands();
  ::mlir::Operation::operand_range presentOperands();
  ::mlir::Operation::operand_range deviceptrOperands();
  ::mlir::Operation::operand_range attachOperands();
  ::mlir::MutableOperandRange ifCondMutable();
  ::mlir::MutableOperandRange copyOperandsMutable();
  ::mlir::MutableOperandRange copyinOperandsMutable();
  ::mlir::MutableOperandRange copyinReadonlyOperandsMutable();
  ::mlir::MutableOperandRange copyoutOperandsMutable();
  ::mlir::MutableOperandRange copyoutZeroOperandsMutable();
  ::mlir::MutableOperandRange createOperandsMutable();
  ::mlir::MutableOperandRange createZeroOperandsMutable();
  ::mlir::MutableOperandRange noCreateOperandsMutable();
  ::mlir::MutableOperandRange presentOperandsMutable();
  ::mlir::MutableOperandRange deviceptrOperandsMutable();
  ::mlir::MutableOperandRange attachOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::StringAttr defaultAttrAttr();
  ::llvm::Optional< ::llvm::StringRef > defaultAttr();
  void defaultAttrAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, /*optional*/::mlir::StringAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, /*optional*/::mlir::StringAttr defaultAttr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::EnterDataOp declarations
//===----------------------------------------------------------------------===//

class EnterDataOpAdaptor {
public:
  EnterDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  EnterDataOpAdaptor(EnterDataOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::ValueRange waitOperands();
  ::mlir::ValueRange copyinOperands();
  ::mlir::ValueRange createOperands();
  ::mlir::ValueRange createZeroOperands();
  ::mlir::ValueRange attachOperands();
  ::mlir::UnitAttr async();
  ::mlir::UnitAttr wait();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class EnterDataOp : public ::mlir::Op<EnterDataOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Adaptor = EnterDataOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::Operation::operand_range waitOperands();
  ::mlir::Operation::operand_range copyinOperands();
  ::mlir::Operation::operand_range createOperands();
  ::mlir::Operation::operand_range createZeroOperands();
  ::mlir::Operation::operand_range attachOperands();
  ::mlir::MutableOperandRange ifCondMutable();
  ::mlir::MutableOperandRange asyncOperandMutable();
  ::mlir::MutableOperandRange waitDevnumMutable();
  ::mlir::MutableOperandRange waitOperandsMutable();
  ::mlir::MutableOperandRange copyinOperandsMutable();
  ::mlir::MutableOperandRange createOperandsMutable();
  ::mlir::MutableOperandRange createZeroOperandsMutable();
  ::mlir::MutableOperandRange attachOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr asyncAttr();
  bool async();
  ::mlir::UnitAttr waitAttr();
  bool wait();
  void asyncAttr(::mlir::UnitAttr attr);
  void waitAttr(::mlir::UnitAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ExitDataOp declarations
//===----------------------------------------------------------------------===//

class ExitDataOpAdaptor {
public:
  ExitDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ExitDataOpAdaptor(ExitDataOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::ValueRange waitOperands();
  ::mlir::ValueRange copyoutOperands();
  ::mlir::ValueRange deleteOperands();
  ::mlir::ValueRange detachOperands();
  ::mlir::UnitAttr async();
  ::mlir::UnitAttr wait();
  ::mlir::UnitAttr finalize();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ExitDataOp : public ::mlir::Op<ExitDataOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Adaptor = ExitDataOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::Operation::operand_range waitOperands();
  ::mlir::Operation::operand_range copyoutOperands();
  ::mlir::Operation::operand_range deleteOperands();
  ::mlir::Operation::operand_range detachOperands();
  ::mlir::MutableOperandRange ifCondMutable();
  ::mlir::MutableOperandRange asyncOperandMutable();
  ::mlir::MutableOperandRange waitDevnumMutable();
  ::mlir::MutableOperandRange waitOperandsMutable();
  ::mlir::MutableOperandRange copyoutOperandsMutable();
  ::mlir::MutableOperandRange deleteOperandsMutable();
  ::mlir::MutableOperandRange detachOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr asyncAttr();
  bool async();
  ::mlir::UnitAttr waitAttr();
  bool wait();
  ::mlir::UnitAttr finalizeAttr();
  bool finalize();
  void asyncAttr(::mlir::UnitAttr attr);
  void waitAttr(::mlir::UnitAttr attr);
  void finalizeAttr(::mlir::UnitAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/::mlir::UnitAttr finalize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/::mlir::UnitAttr finalize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/bool finalize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/bool finalize);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::InitOp declarations
//===----------------------------------------------------------------------===//

class InitOpAdaptor {
public:
  InitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  InitOpAdaptor(InitOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange deviceTypeOperands();
  ::mlir::Value deviceNumOperand();
  ::mlir::Value ifCond();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class InitOp : public ::mlir::Op<InitOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Adaptor = InitOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range deviceTypeOperands();
  ::mlir::Value deviceNumOperand();
  ::mlir::Value ifCond();
  ::mlir::MutableOperandRange deviceTypeOperandsMutable();
  ::mlir::MutableOperandRange deviceNumOperandMutable();
  ::mlir::MutableOperandRange ifCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::LoopOp declarations
//===----------------------------------------------------------------------===//

class LoopOpAdaptor {
public:
  LoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  LoopOpAdaptor(LoopOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value gangNum();
  ::mlir::Value gangStatic();
  ::mlir::Value workerNum();
  ::mlir::Value vectorLength();
  ::mlir::ValueRange tileOperands();
  ::mlir::ValueRange privateOperands();
  ::mlir::ValueRange reductionOperands();
  ::mlir::IntegerAttr collapse();
  ::mlir::UnitAttr seq();
  ::mlir::UnitAttr independent();
  ::mlir::UnitAttr auto_();
  ::mlir::StringAttr reductionOp();
  ::mlir::IntegerAttr exec_mapping();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class LoopOp : public ::mlir::Op<LoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<acc::YieldOp>::Impl> {
public:
  using Op::Op;
  using Adaptor = LoopOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value gangNum();
  ::mlir::Value gangStatic();
  ::mlir::Value workerNum();
  ::mlir::Value vectorLength();
  ::mlir::Operation::operand_range tileOperands();
  ::mlir::Operation::operand_range privateOperands();
  ::mlir::Operation::operand_range reductionOperands();
  ::mlir::MutableOperandRange gangNumMutable();
  ::mlir::MutableOperandRange gangStaticMutable();
  ::mlir::MutableOperandRange workerNumMutable();
  ::mlir::MutableOperandRange vectorLengthMutable();
  ::mlir::MutableOperandRange tileOperandsMutable();
  ::mlir::MutableOperandRange privateOperandsMutable();
  ::mlir::MutableOperandRange reductionOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range results();
  ::mlir::Region &region();
  ::mlir::IntegerAttr collapseAttr();
  ::llvm::Optional<uint64_t> collapse();
  ::mlir::UnitAttr seqAttr();
  bool seq();
  ::mlir::UnitAttr independentAttr();
  bool independent();
  ::mlir::UnitAttr auto_Attr();
  bool auto_();
  ::mlir::StringAttr reductionOpAttr();
  ::llvm::Optional< ::llvm::StringRef > reductionOp();
  ::mlir::IntegerAttr exec_mappingAttr();
  uint64_t exec_mapping();
  void collapseAttr(::mlir::IntegerAttr attr);
  void seqAttr(::mlir::UnitAttr attr);
  void independentAttr(::mlir::UnitAttr attr);
  void auto_Attr(::mlir::UnitAttr attr);
  void reductionOpAttr(::mlir::StringAttr attr);
  void exec_mappingAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::UnitAttr seq, /*optional*/::mlir::UnitAttr independent, /*optional*/::mlir::UnitAttr auto_, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::IntegerAttr exec_mapping);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/bool seq, /*optional*/bool independent, /*optional*/bool auto_, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, uint64_t exec_mapping = 0);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();

    static StringRef getCollapseAttrName() { return "collapse"; }
    static StringRef getSeqAttrName() { return "seq"; }
    static StringRef getIndependentAttrName() { return "independent"; }
    static StringRef getAutoAttrName() { return "auto"; }
    static StringRef getExecutionMappingAttrName() { return "exec_mapping"; }
    static StringRef getGangKeyword() { return "gang"; }
    static StringRef getGangNumKeyword() { return "num"; }
    static StringRef getGangStaticKeyword() { return "static"; }
    static StringRef getVectorKeyword() { return "vector"; }
    static StringRef getWorkerKeyword() { return "worker"; }
    static StringRef getTileKeyword() { return "tile"; }
    static StringRef getPrivateKeyword() { return "private"; }
    static StringRef getReductionKeyword() { return "reduction"; }
  
};
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ParallelOp declarations
//===----------------------------------------------------------------------===//

class ParallelOpAdaptor {
public:
  ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ParallelOpAdaptor(ParallelOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value async();
  ::mlir::ValueRange waitOperands();
  ::mlir::Value numGangs();
  ::mlir::Value numWorkers();
  ::mlir::Value vectorLength();
  ::mlir::Value ifCond();
  ::mlir::Value selfCond();
  ::mlir::ValueRange reductionOperands();
  ::mlir::ValueRange copyOperands();
  ::mlir::ValueRange copyinOperands();
  ::mlir::ValueRange copyinReadonlyOperands();
  ::mlir::ValueRange copyoutOperands();
  ::mlir::ValueRange copyoutZeroOperands();
  ::mlir::ValueRange createOperands();
  ::mlir::ValueRange createZeroOperands();
  ::mlir::ValueRange noCreateOperands();
  ::mlir::ValueRange presentOperands();
  ::mlir::ValueRange devicePtrOperands();
  ::mlir::ValueRange attachOperands();
  ::mlir::ValueRange gangPrivateOperands();
  ::mlir::ValueRange gangFirstPrivateOperands();
  ::mlir::UnitAttr asyncAttr();
  ::mlir::UnitAttr waitAttr();
  ::mlir::UnitAttr selfAttr();
  ::mlir::StringAttr reductionOp();
  ::mlir::StringAttr defaultAttr();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Adaptor = ParallelOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value async();
  ::mlir::Operation::operand_range waitOperands();
  ::mlir::Value numGangs();
  ::mlir::Value numWorkers();
  ::mlir::Value vectorLength();
  ::mlir::Value ifCond();
  ::mlir::Value selfCond();
  ::mlir::Operation::operand_range reductionOperands();
  ::mlir::Operation::operand_range copyOperands();
  ::mlir::Operation::operand_range copyinOperands();
  ::mlir::Operation::operand_range copyinReadonlyOperands();
  ::mlir::Operation::operand_range copyoutOperands();
  ::mlir::Operation::operand_range copyoutZeroOperands();
  ::mlir::Operation::operand_range createOperands();
  ::mlir::Operation::operand_range createZeroOperands();
  ::mlir::Operation::operand_range noCreateOperands();
  ::mlir::Operation::operand_range presentOperands();
  ::mlir::Operation::operand_range devicePtrOperands();
  ::mlir::Operation::operand_range attachOperands();
  ::mlir::Operation::operand_range gangPrivateOperands();
  ::mlir::Operation::operand_range gangFirstPrivateOperands();
  ::mlir::MutableOperandRange asyncMutable();
  ::mlir::MutableOperandRange waitOperandsMutable();
  ::mlir::MutableOperandRange numGangsMutable();
  ::mlir::MutableOperandRange numWorkersMutable();
  ::mlir::MutableOperandRange vectorLengthMutable();
  ::mlir::MutableOperandRange ifCondMutable();
  ::mlir::MutableOperandRange selfCondMutable();
  ::mlir::MutableOperandRange reductionOperandsMutable();
  ::mlir::MutableOperandRange copyOperandsMutable();
  ::mlir::MutableOperandRange copyinOperandsMutable();
  ::mlir::MutableOperandRange copyinReadonlyOperandsMutable();
  ::mlir::MutableOperandRange copyoutOperandsMutable();
  ::mlir::MutableOperandRange copyoutZeroOperandsMutable();
  ::mlir::MutableOperandRange createOperandsMutable();
  ::mlir::MutableOperandRange createZeroOperandsMutable();
  ::mlir::MutableOperandRange noCreateOperandsMutable();
  ::mlir::MutableOperandRange presentOperandsMutable();
  ::mlir::MutableOperandRange devicePtrOperandsMutable();
  ::mlir::MutableOperandRange attachOperandsMutable();
  ::mlir::MutableOperandRange gangPrivateOperandsMutable();
  ::mlir::MutableOperandRange gangFirstPrivateOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::UnitAttr asyncAttrAttr();
  bool asyncAttr();
  ::mlir::UnitAttr waitAttrAttr();
  bool waitAttr();
  ::mlir::UnitAttr selfAttrAttr();
  bool selfAttr();
  ::mlir::StringAttr reductionOpAttr();
  ::llvm::Optional< ::llvm::StringRef > reductionOp();
  ::mlir::StringAttr defaultAttrAttr();
  ::llvm::Optional< ::llvm::StringRef > defaultAttr();
  void asyncAttrAttr(::mlir::UnitAttr attr);
  void waitAttrAttr(::mlir::UnitAttr attr);
  void selfAttrAttr(::mlir::UnitAttr attr);
  void reductionOpAttr(::mlir::StringAttr attr);
  void defaultAttrAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::StringAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::StringAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::StringAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::StringAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::StringAttr defaultAttr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();

    static StringRef getAsyncKeyword() { return "async"; }
    static StringRef getAsyncAttrName() { return "asyncAttr"; }
    static StringRef getWaitKeyword() { return "wait"; }
    static StringRef getWaitAttrName() { return "waitAttr"; }
    static StringRef getNumGangsKeyword() { return "num_gangs"; }
    static StringRef getNumWorkersKeyword() { return "num_workers"; }
    static StringRef getVectorLengthKeyword() { return "vector_length"; }
    static StringRef getIfKeyword() { return "if"; }
    static StringRef getSelfKeyword() { return "self"; }
    static StringRef getSelfAttrName() { return "selfAttr"; }
    static StringRef getReductionKeyword() { return "reduction"; }
    static StringRef getCopyKeyword() { return "copy"; }
    static StringRef getCopyinKeyword() { return "copyin"; }
    static StringRef getCopyinReadonlyKeyword() { return "copyin_readonly"; }
    static StringRef getCopyoutKeyword() { return "copyout"; }
    static StringRef getCopyoutZeroKeyword() { return "copyout_zero"; }
    static StringRef getCreateKeyword() { return "create"; }
    static StringRef getCreateZeroKeyword() { return "create_zero"; }
    static StringRef getNoCreateKeyword() { return "no_create"; }
    static StringRef getPresentKeyword() { return "present"; }
    static StringRef getDevicePtrKeyword() { return "deviceptr"; }
    static StringRef getAttachKeyword() { return "attach"; }
    static StringRef getPrivateKeyword() { return "private"; }
    static StringRef getFirstPrivateKeyword() { return "firstprivate"; }
  
};
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ShutdownOp declarations
//===----------------------------------------------------------------------===//

class ShutdownOpAdaptor {
public:
  ShutdownOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ShutdownOpAdaptor(ShutdownOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange deviceTypeOperands();
  ::mlir::Value deviceNumOperand();
  ::mlir::Value ifCond();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ShutdownOp : public ::mlir::Op<ShutdownOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Adaptor = ShutdownOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range deviceTypeOperands();
  ::mlir::Value deviceNumOperand();
  ::mlir::Value ifCond();
  ::mlir::MutableOperandRange deviceTypeOperandsMutable();
  ::mlir::MutableOperandRange deviceNumOperandMutable();
  ::mlir::MutableOperandRange ifCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::TerminatorOp declarations
//===----------------------------------------------------------------------===//

class TerminatorOpAdaptor {
public:
  TerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  TerminatorOpAdaptor(TerminatorOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class TerminatorOp : public ::mlir::Op<TerminatorOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Adaptor = TerminatorOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::UpdateOp declarations
//===----------------------------------------------------------------------===//

class UpdateOpAdaptor {
public:
  UpdateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  UpdateOpAdaptor(UpdateOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::ValueRange waitOperands();
  ::mlir::ValueRange hostOperands();
  ::mlir::ValueRange deviceOperands();
  ::mlir::UnitAttr async();
  ::mlir::UnitAttr wait();
  ::mlir::UnitAttr ifPresent();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class UpdateOp : public ::mlir::Op<UpdateOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Adaptor = UpdateOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::Operation::operand_range waitOperands();
  ::mlir::Operation::operand_range hostOperands();
  ::mlir::Operation::operand_range deviceOperands();
  ::mlir::MutableOperandRange asyncOperandMutable();
  ::mlir::MutableOperandRange waitDevnumMutable();
  ::mlir::MutableOperandRange waitOperandsMutable();
  ::mlir::MutableOperandRange hostOperandsMutable();
  ::mlir::MutableOperandRange deviceOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr asyncAttr();
  bool async();
  ::mlir::UnitAttr waitAttr();
  bool wait();
  ::mlir::UnitAttr ifPresentAttr();
  bool ifPresent();
  void asyncAttr(::mlir::UnitAttr attr);
  void waitAttr(::mlir::UnitAttr attr);
  void ifPresentAttr(::mlir::UnitAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::WaitOp declarations
//===----------------------------------------------------------------------===//

class WaitOpAdaptor {
public:
  WaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  WaitOpAdaptor(WaitOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange waitOperands();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::Value ifCond();
  ::mlir::UnitAttr async();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class WaitOp : public ::mlir::Op<WaitOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments> {
public:
  using Op::Op;
  using Adaptor = WaitOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range waitOperands();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::Value ifCond();
  ::mlir::MutableOperandRange waitOperandsMutable();
  ::mlir::MutableOperandRange asyncOperandMutable();
  ::mlir::MutableOperandRange waitDevnumMutable();
  ::mlir::MutableOperandRange ifCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr asyncAttr();
  bool async();
  void asyncAttr(::mlir::UnitAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  YieldOpAdaptor(YieldOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::IsTerminator, ::mlir::OpTrait::HasParent<ParallelOp, LoopOp>::Impl> {
public:
  using Op::Op;
  using Adaptor = YieldOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace acc
} // namespace mlir

#endif  // GET_OP_CLASSES

