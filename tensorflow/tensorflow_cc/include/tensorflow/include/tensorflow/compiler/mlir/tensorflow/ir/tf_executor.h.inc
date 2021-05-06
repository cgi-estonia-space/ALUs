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
namespace tf_executor {
class ControlTriggerOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class EnterOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class ExitOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class FetchOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class GraphOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class IslandOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class LoopCondOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class MergeOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class NextIterationSinkOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class NextIterationSourceOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class SwitchNOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class SwitchOp;
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {
class YieldOp;
} // namespace tf_executor
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::ControlTriggerOp declarations
//===----------------------------------------------------------------------===//

class ControlTriggerOpAdaptor {
public:
  ControlTriggerOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  ControlTriggerOpAdaptor(ControlTriggerOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange controlInputs();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ControlTriggerOp : public ::mlir::Op<ControlTriggerOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<GraphOp>::Impl> {
public:
  using Op::Op;
  using Adaptor = ControlTriggerOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range controlInputs();
  ::mlir::MutableOperandRange controlInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value control();
  static void build(OpBuilder &builder, OperationState &result, ArrayRef<Value> operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type control, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::EnterOp declarations
//===----------------------------------------------------------------------===//

class EnterOpAdaptor {
public:
  EnterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  EnterOpAdaptor(EnterOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value data();
  ::mlir::ValueRange controlInputs();
  ::mlir::StringAttr frame_name();
  ::mlir::BoolAttr is_constant();
  ::mlir::IntegerAttr parallel_iterations();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class EnterOp : public ::mlir::Op<EnterOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::NResults<2>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::HasParent<GraphOp>::Impl, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = EnterOpAdaptor;
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value data();
  ::mlir::Operation::operand_range controlInputs();
  ::mlir::MutableOperandRange dataMutable();
  ::mlir::MutableOperandRange controlInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  ::mlir::Value control();
  ::mlir::StringAttr frame_nameAttr();
  ::llvm::StringRef frame_name();
  ::mlir::BoolAttr is_constantAttr();
  bool is_constant();
  ::mlir::IntegerAttr parallel_iterationsAttr();
  uint64_t parallel_iterations();
  void frame_nameAttr(::mlir::StringAttr attr);
  void is_constantAttr(::mlir::BoolAttr attr);
  void parallel_iterationsAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type control, ::mlir::Value data, ::mlir::StringAttr frame_name, ::mlir::BoolAttr is_constant, ::mlir::IntegerAttr parallel_iterations, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::StringAttr frame_name, ::mlir::BoolAttr is_constant, ::mlir::IntegerAttr parallel_iterations, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type control, ::mlir::Value data, ::llvm::StringRef frame_name, bool is_constant, uint64_t parallel_iterations, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::llvm::StringRef frame_name, bool is_constant, uint64_t parallel_iterations, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::ExitOp declarations
//===----------------------------------------------------------------------===//

class ExitOpAdaptor {
public:
  ExitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  ExitOpAdaptor(ExitOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value data();
  ::mlir::ValueRange controlInputs();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ExitOp : public ::mlir::Op<ExitOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::NResults<2>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::HasParent<GraphOp>::Impl, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = ExitOpAdaptor;
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value data();
  ::mlir::Operation::operand_range controlInputs();
  ::mlir::MutableOperandRange dataMutable();
  ::mlir::MutableOperandRange controlInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  ::mlir::Value control();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type control, ::mlir::Value data, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::FetchOp declarations
//===----------------------------------------------------------------------===//

class FetchOpAdaptor {
public:
  FetchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  FetchOpAdaptor(FetchOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange fetches();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class FetchOp : public ::mlir::Op<FetchOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::IsTerminator, ::mlir::OpTrait::HasParent<GraphOp>::Impl> {
public:
  using Op::Op;
  using Adaptor = FetchOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range fetches();
  ::mlir::MutableOperandRange fetchesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &builder, OperationState &result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange fetches);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::GraphOp declarations
//===----------------------------------------------------------------------===//

class GraphOpAdaptor {
public:
  GraphOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  GraphOpAdaptor(GraphOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class GraphOp : public ::mlir::Op<GraphOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<FetchOp>::Impl> {
public:
  using Op::Op;
  using Adaptor = GraphOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range results();
  ::mlir::Region &body();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);

    Block &GetBody() { return getOperation()->getRegion(0).front(); }
    FetchOp GetFetch();
  
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::IslandOp declarations
//===----------------------------------------------------------------------===//

class IslandOpAdaptor {
public:
  IslandOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  IslandOpAdaptor(IslandOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange controlInputs();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class IslandOp : public ::mlir::Op<IslandOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<GraphOp>::Impl, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = IslandOpAdaptor;
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range controlInputs();
  ::mlir::MutableOperandRange controlInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::Value control();
  ::mlir::Region &body();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Type control, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

    Block &GetBody() { return getOperation()->getRegion(0).front(); }
    YieldOp GetYield();
    bool WrapsSingleOp();
  
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::LoopCondOp declarations
//===----------------------------------------------------------------------===//

class LoopCondOpAdaptor {
public:
  LoopCondOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  LoopCondOpAdaptor(LoopCondOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::ValueRange controlInputs();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class LoopCondOp : public ::mlir::Op<LoopCondOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::NResults<2>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::HasParent<GraphOp>::Impl, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = LoopCondOpAdaptor;
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Operation::operand_range controlInputs();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange controlInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  ::mlir::Value control();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type control, ::mlir::Value input, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::MergeOp declarations
//===----------------------------------------------------------------------===//

class MergeOpAdaptor {
public:
  MergeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  MergeOpAdaptor(MergeOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs_and_control();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class MergeOp : public ::mlir::Op<MergeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::NResults<3>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<GraphOp>::Impl, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = MergeOpAdaptor;
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs_and_control();
  ::mlir::MutableOperandRange inputs_and_controlMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  ::mlir::Value value_index();
  ::mlir::Value control();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type value_index, ::mlir::Type control, ::mlir::ValueRange inputs_and_control);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::NextIterationSinkOp declarations
//===----------------------------------------------------------------------===//

class NextIterationSinkOpAdaptor {
public:
  NextIterationSinkOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  NextIterationSinkOpAdaptor(NextIterationSinkOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value token();
  ::mlir::Value input();
  ::mlir::ValueRange controlInputs();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class NextIterationSinkOp : public ::mlir::Op<NextIterationSinkOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::HasParent<GraphOp>::Impl> {
public:
  using Op::Op;
  using Adaptor = NextIterationSinkOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value token();
  ::mlir::Value input();
  ::mlir::Operation::operand_range controlInputs();
  ::mlir::MutableOperandRange tokenMutable();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange controlInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &builder, OperationState &result, Value token, ArrayRef<Value> operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value token, ::mlir::Value input, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value input, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::NextIterationSourceOp declarations
//===----------------------------------------------------------------------===//

class NextIterationSourceOpAdaptor {
public:
  NextIterationSourceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  NextIterationSourceOpAdaptor(NextIterationSourceOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class NextIterationSourceOp : public ::mlir::Op<NextIterationSourceOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::NResults<3>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::HasParent<GraphOp>::Impl, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = NextIterationSourceOpAdaptor;
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  ::mlir::Value token();
  ::mlir::Value control();
  static void build(OpBuilder &builder, OperationState &result, Type result_type, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Type token, ::mlir::Type control);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    NextIterationSinkOp GetSink() {
      return cast<NextIterationSinkOp>(*token().user_begin());
    }
  
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::SwitchNOp declarations
//===----------------------------------------------------------------------===//

class SwitchNOpAdaptor {
public:
  SwitchNOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  SwitchNOpAdaptor(SwitchNOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value data();
  ::mlir::Value index();
  ::mlir::ValueRange controlInputs();
  ::mlir::IntegerAttr num_outs();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class SwitchNOp : public ::mlir::Op<SwitchNOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::HasParent<GraphOp>::Impl, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = SwitchNOpAdaptor;
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value data();
  ::mlir::Value index();
  ::mlir::Operation::operand_range controlInputs();
  ::mlir::MutableOperandRange dataMutable();
  ::mlir::MutableOperandRange indexMutable();
  ::mlir::MutableOperandRange controlInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::Value control();
  ::mlir::IntegerAttr num_outsAttr();
  uint64_t num_outs();
  void num_outsAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Type control, ::mlir::Value data, ::mlir::Value index, ::mlir::ValueRange controlInputs, ::mlir::IntegerAttr num_outs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value index, ::mlir::ValueRange controlInputs, ::mlir::IntegerAttr num_outs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Type control, ::mlir::Value data, ::mlir::Value index, ::mlir::ValueRange controlInputs, uint64_t num_outs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value index, ::mlir::ValueRange controlInputs, uint64_t num_outs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::SwitchOp declarations
//===----------------------------------------------------------------------===//

class SwitchOpAdaptor {
public:
  SwitchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  SwitchOpAdaptor(SwitchOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value data();
  ::mlir::Value predicate();
  ::mlir::ValueRange controlInputs();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class SwitchOp : public ::mlir::Op<SwitchOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::NResults<3>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::HasParent<GraphOp>::Impl, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = SwitchOpAdaptor;
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value data();
  ::mlir::Value predicate();
  ::mlir::Operation::operand_range controlInputs();
  ::mlir::MutableOperandRange dataMutable();
  ::mlir::MutableOperandRange predicateMutable();
  ::mlir::MutableOperandRange controlInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value falseOutput();
  ::mlir::Value trueOutput();
  ::mlir::Value control();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type falseOutput, ::mlir::Type trueOutput, ::mlir::Type control, ::mlir::Value data, ::mlir::Value predicate, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value predicate, ::mlir::ValueRange controlInputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
};
} // namespace tf_executor
} // namespace mlir
namespace mlir {
namespace tf_executor {

//===----------------------------------------------------------------------===//
// ::mlir::tf_executor::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  YieldOpAdaptor(YieldOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange fetches();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::IsTerminator, ::mlir::OpTrait::HasParent<IslandOp>::Impl> {
public:
  using Op::Op;
  using Adaptor = YieldOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range fetches();
  ::mlir::MutableOperandRange fetchesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(OpBuilder &builder, OperationState &result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange fetches);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
};
} // namespace tf_executor
} // namespace mlir

#endif  // GET_OP_CLASSES

