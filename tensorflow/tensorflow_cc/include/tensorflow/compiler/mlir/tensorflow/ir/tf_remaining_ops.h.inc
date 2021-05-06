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
namespace TF {
class _FusedBatchNormExOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _FusedConv2DOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _FusedMatMulOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _RecvTPUEmbeddingActivationsOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _RecvTPUEmbeddingDeduplicationDataOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _SendTPUEmbeddingGradientsOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _TPUCompileMlirOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _TPUCompileMlirPlaceholderProgramKeyOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _UnaryOpsCompositionOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaHostComputeMlirOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaRecvAtHostOp;
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {
class _XlaSendFromHostOp;
} // namespace TF
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_FusedBatchNormExOp declarations
//===----------------------------------------------------------------------===//

class _FusedBatchNormExOpAdaptor {
public:
  _FusedBatchNormExOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _FusedBatchNormExOpAdaptor(_FusedBatchNormExOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value mean();
  ::mlir::Value variance();
  ::mlir::ValueRange side_input();
  ::mlir::FloatAttr epsilon();
  ::mlir::FloatAttr exponential_avg_factor();
  ::mlir::StringAttr activation_mode();
  ::mlir::StringAttr data_format();
  ::mlir::BoolAttr is_training();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _FusedBatchNormExOp : public ::mlir::Op<_FusedBatchNormExOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::NResults<6>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<5>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = _FusedBatchNormExOpAdaptor;
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::Value scale();
  ::mlir::Value offset();
  ::mlir::Value mean();
  ::mlir::Value variance();
  ::mlir::Operation::operand_range side_input();
  ::mlir::MutableOperandRange xMutable();
  ::mlir::MutableOperandRange scaleMutable();
  ::mlir::MutableOperandRange offsetMutable();
  ::mlir::MutableOperandRange meanMutable();
  ::mlir::MutableOperandRange varianceMutable();
  ::mlir::MutableOperandRange side_inputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value y();
  ::mlir::Value batch_mean();
  ::mlir::Value batch_variance();
  ::mlir::Value reserve_space_1();
  ::mlir::Value reserve_space_2();
  ::mlir::Value reserve_space_3();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr exponential_avg_factorAttr();
  ::llvm::APFloat exponential_avg_factor();
  ::mlir::StringAttr activation_modeAttr();
  ::llvm::StringRef activation_mode();
  ::mlir::StringAttr data_formatAttr();
  ::llvm::StringRef data_format();
  ::mlir::BoolAttr is_trainingAttr();
  bool is_training();
  Type T();
  Type U();
  size_t num_side_inputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void epsilonAttr(::mlir::FloatAttr attr);
  void exponential_avg_factorAttr(::mlir::FloatAttr attr);
  void activation_modeAttr(::mlir::StringAttr attr);
  void data_formatAttr(::mlir::StringAttr attr);
  void is_trainingAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type y, ::mlir::Type batch_mean, ::mlir::Type batch_variance, ::mlir::Type reserve_space_1, ::mlir::Type reserve_space_2, ::mlir::Type reserve_space_3, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, ::mlir::FloatAttr epsilon, ::mlir::FloatAttr exponential_avg_factor, ::mlir::StringAttr activation_mode, ::mlir::StringAttr data_format, ::mlir::BoolAttr is_training);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, ::mlir::FloatAttr epsilon, ::mlir::FloatAttr exponential_avg_factor, ::mlir::StringAttr activation_mode, ::mlir::StringAttr data_format, ::mlir::BoolAttr is_training);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type y, ::mlir::Type batch_mean, ::mlir::Type batch_variance, ::mlir::Type reserve_space_1, ::mlir::Type reserve_space_2, ::mlir::Type reserve_space_3, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, ::llvm::APFloat epsilon, ::llvm::APFloat exponential_avg_factor, ::llvm::StringRef activation_mode = "Identity", ::llvm::StringRef data_format = "NHWC", bool is_training = true);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value scale, ::mlir::Value offset, ::mlir::Value mean, ::mlir::Value variance, ::mlir::ValueRange side_input, ::llvm::APFloat epsilon, ::llvm::APFloat exponential_avg_factor, ::llvm::StringRef activation_mode = "Identity", ::llvm::StringRef data_format = "NHWC", bool is_training = true);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_FusedConv2DOp declarations
//===----------------------------------------------------------------------===//

class _FusedConv2DOpAdaptor {
public:
  _FusedConv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _FusedConv2DOpAdaptor(_FusedConv2DOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::ValueRange args();
  ::mlir::ArrayAttr strides();
  ::mlir::StringAttr padding();
  ::mlir::ArrayAttr explicit_paddings();
  ::mlir::StringAttr data_format();
  ::mlir::ArrayAttr dilations();
  ::mlir::BoolAttr use_cudnn_on_gpu();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilon();
  ::mlir::FloatAttr leakyrelu_alpha();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _FusedConv2DOp : public ::mlir::Op<_FusedConv2DOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = _FusedConv2DOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value filter();
  ::mlir::Operation::operand_range args();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange argsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  ::mlir::ArrayAttr stridesAttr();
  ::mlir::ArrayAttr strides();
  ::mlir::StringAttr paddingAttr();
  ::llvm::StringRef padding();
  ::mlir::ArrayAttr explicit_paddingsAttr();
  ::mlir::ArrayAttr explicit_paddings();
  ::mlir::StringAttr data_formatAttr();
  ::llvm::StringRef data_format();
  ::mlir::ArrayAttr dilationsAttr();
  ::mlir::ArrayAttr dilations();
  ::mlir::BoolAttr use_cudnn_on_gpuAttr();
  bool use_cudnn_on_gpu();
  ::mlir::ArrayAttr fused_opsAttr();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  ::mlir::FloatAttr leakyrelu_alphaAttr();
  ::llvm::APFloat leakyrelu_alpha();
  Type T();
  size_t num_args();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void stridesAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::StringAttr attr);
  void explicit_paddingsAttr(::mlir::ArrayAttr attr);
  void data_formatAttr(::mlir::StringAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void use_cudnn_on_gpuAttr(::mlir::BoolAttr attr);
  void fused_opsAttr(::mlir::ArrayAttr attr);
  void epsilonAttr(::mlir::FloatAttr attr);
  void leakyrelu_alphaAttr(::mlir::FloatAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ArrayAttr strides, ::mlir::StringAttr padding, ::mlir::ArrayAttr explicit_paddings, ::mlir::StringAttr data_format, ::mlir::ArrayAttr dilations, ::mlir::BoolAttr use_cudnn_on_gpu, ::mlir::ArrayAttr fused_ops, ::mlir::FloatAttr epsilon, ::mlir::FloatAttr leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ArrayAttr strides, ::mlir::StringAttr padding, ::mlir::ArrayAttr explicit_paddings, ::mlir::StringAttr data_format, ::mlir::ArrayAttr dilations, ::mlir::BoolAttr use_cudnn_on_gpu, ::mlir::ArrayAttr fused_ops, ::mlir::FloatAttr epsilon, ::mlir::FloatAttr leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ArrayAttr strides, ::llvm::StringRef padding, ::mlir::ArrayAttr explicit_paddings, ::llvm::StringRef data_format, ::mlir::ArrayAttr dilations, bool use_cudnn_on_gpu, ::mlir::ArrayAttr fused_ops, ::llvm::APFloat epsilon, ::llvm::APFloat leakyrelu_alpha);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value filter, ::mlir::ValueRange args, ::mlir::ArrayAttr strides, ::llvm::StringRef padding, ::mlir::ArrayAttr explicit_paddings, ::llvm::StringRef data_format, ::mlir::ArrayAttr dilations, bool use_cudnn_on_gpu, ::mlir::ArrayAttr fused_ops, ::llvm::APFloat epsilon, ::llvm::APFloat leakyrelu_alpha);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_FusedMatMulOp declarations
//===----------------------------------------------------------------------===//

class _FusedMatMulOpAdaptor {
public:
  _FusedMatMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _FusedMatMulOpAdaptor(_FusedMatMulOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value a();
  ::mlir::Value b();
  ::mlir::ValueRange args();
  ::mlir::BoolAttr transpose_a();
  ::mlir::BoolAttr transpose_b();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilon();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _FusedMatMulOp : public ::mlir::Op<_FusedMatMulOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultElementType, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = _FusedMatMulOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value a();
  ::mlir::Value b();
  ::mlir::Operation::operand_range args();
  ::mlir::MutableOperandRange aMutable();
  ::mlir::MutableOperandRange bMutable();
  ::mlir::MutableOperandRange argsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value product();
  ::mlir::BoolAttr transpose_aAttr();
  bool transpose_a();
  ::mlir::BoolAttr transpose_bAttr();
  bool transpose_b();
  ::mlir::ArrayAttr fused_opsAttr();
  ::mlir::ArrayAttr fused_ops();
  ::mlir::FloatAttr epsilonAttr();
  ::llvm::APFloat epsilon();
  Type T();
  size_t num_args();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void transpose_aAttr(::mlir::BoolAttr attr);
  void transpose_bAttr(::mlir::BoolAttr attr);
  void fused_opsAttr(::mlir::ArrayAttr attr);
  void epsilonAttr(::mlir::FloatAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type product, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, ::mlir::BoolAttr transpose_a, ::mlir::BoolAttr transpose_b, ::mlir::ArrayAttr fused_ops, ::mlir::FloatAttr epsilon);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, ::mlir::BoolAttr transpose_a, ::mlir::BoolAttr transpose_b, ::mlir::ArrayAttr fused_ops, ::mlir::FloatAttr epsilon);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type product, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, bool transpose_a, bool transpose_b, ::mlir::ArrayAttr fused_ops, ::llvm::APFloat epsilon);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::ValueRange args, bool transpose_a, bool transpose_b, ::mlir::ArrayAttr fused_ops, ::llvm::APFloat epsilon);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_RecvTPUEmbeddingActivationsOp declarations
//===----------------------------------------------------------------------===//

class _RecvTPUEmbeddingActivationsOpAdaptor {
public:
  _RecvTPUEmbeddingActivationsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _RecvTPUEmbeddingActivationsOpAdaptor(_RecvTPUEmbeddingActivationsOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value deduplication_data();
  ::mlir::StringAttr config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _RecvTPUEmbeddingActivationsOp : public ::mlir::Op<_RecvTPUEmbeddingActivationsOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = _RecvTPUEmbeddingActivationsOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value deduplication_data();
  ::mlir::MutableOperandRange deduplication_dataMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::StringAttr configAttr();
  ::llvm::StringRef config();
  size_t num_tables();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void configAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value deduplication_data, ::mlir::StringAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value deduplication_data, ::llvm::StringRef config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_RecvTPUEmbeddingDeduplicationDataOp declarations
//===----------------------------------------------------------------------===//

class _RecvTPUEmbeddingDeduplicationDataOpAdaptor {
public:
  _RecvTPUEmbeddingDeduplicationDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _RecvTPUEmbeddingDeduplicationDataOpAdaptor(_RecvTPUEmbeddingDeduplicationDataOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::StringAttr config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _RecvTPUEmbeddingDeduplicationDataOp : public ::mlir::Op<_RecvTPUEmbeddingDeduplicationDataOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Adaptor = _RecvTPUEmbeddingDeduplicationDataOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value output();
  ::mlir::StringAttr configAttr();
  ::llvm::StringRef config();
  void configAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::StringAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::llvm::StringRef config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_SendTPUEmbeddingGradientsOp declarations
//===----------------------------------------------------------------------===//

class _SendTPUEmbeddingGradientsOpAdaptor {
public:
  _SendTPUEmbeddingGradientsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  _SendTPUEmbeddingGradientsOpAdaptor(_SendTPUEmbeddingGradientsOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange gradients();
  ::mlir::ValueRange learning_rates();
  ::mlir::Value deduplication_data();
  ::mlir::StringAttr config();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _SendTPUEmbeddingGradientsOp : public ::mlir::Op<_SendTPUEmbeddingGradientsOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = _SendTPUEmbeddingGradientsOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range gradients();
  ::mlir::Operation::operand_range learning_rates();
  ::mlir::Value deduplication_data();
  ::mlir::MutableOperandRange gradientsMutable();
  ::mlir::MutableOperandRange learning_ratesMutable();
  ::mlir::MutableOperandRange deduplication_dataMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr configAttr();
  ::llvm::StringRef config();
  size_t NumTables();
  size_t NumLearningRateTags();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void configAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange gradients, ::mlir::ValueRange learning_rates, ::mlir::Value deduplication_data, ::mlir::StringAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange gradients, ::mlir::ValueRange learning_rates, ::mlir::Value deduplication_data, ::mlir::StringAttr config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange gradients, ::mlir::ValueRange learning_rates, ::mlir::Value deduplication_data, ::llvm::StringRef config);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange gradients, ::mlir::ValueRange learning_rates, ::mlir::Value deduplication_data, ::llvm::StringRef config);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_TPUCompileMlirOp declarations
//===----------------------------------------------------------------------===//

class _TPUCompileMlirOpAdaptor {
public:
  _TPUCompileMlirOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _TPUCompileMlirOpAdaptor(_TPUCompileMlirOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange dynamic_shapes();
  ::mlir::StringAttr mlir_module();
  ::mlir::StringAttr metadata();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _TPUCompileMlirOp : public ::mlir::Op<_TPUCompileMlirOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpAsmOpInterface::Trait, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = _TPUCompileMlirOpAdaptor;
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range dynamic_shapes();
  ::mlir::MutableOperandRange dynamic_shapesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value compilation_status();
  ::mlir::Operation::result_range program();
  ::mlir::StringAttr mlir_moduleAttr();
  ::llvm::StringRef mlir_module();
  ::mlir::StringAttr metadataAttr();
  ::llvm::StringRef metadata();
  size_t num_computations();
  size_t NumDynamicShapes();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void mlir_moduleAttr(::mlir::StringAttr attr);
  void metadataAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type compilation_status, ::mlir::TypeRange program, ::mlir::ValueRange dynamic_shapes, ::mlir::StringAttr mlir_module, ::mlir::StringAttr metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamic_shapes, ::mlir::StringAttr mlir_module, ::mlir::StringAttr metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type compilation_status, ::mlir::TypeRange program, ::mlir::ValueRange dynamic_shapes, ::llvm::StringRef mlir_module, ::llvm::StringRef metadata);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamic_shapes, ::llvm::StringRef mlir_module, ::llvm::StringRef metadata);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_TPUCompileMlirPlaceholderProgramKeyOp declarations
//===----------------------------------------------------------------------===//

class _TPUCompileMlirPlaceholderProgramKeyOpAdaptor {
public:
  _TPUCompileMlirPlaceholderProgramKeyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _TPUCompileMlirPlaceholderProgramKeyOpAdaptor(_TPUCompileMlirPlaceholderProgramKeyOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _TPUCompileMlirPlaceholderProgramKeyOp : public ::mlir::Op<_TPUCompileMlirPlaceholderProgramKeyOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Adaptor = _TPUCompileMlirPlaceholderProgramKeyOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value program();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type program);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_UnaryOpsCompositionOp declarations
//===----------------------------------------------------------------------===//

class _UnaryOpsCompositionOpAdaptor {
public:
  _UnaryOpsCompositionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _UnaryOpsCompositionOpAdaptor(_UnaryOpsCompositionOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::ArrayAttr op_names();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _UnaryOpsCompositionOp : public ::mlir::Op<_UnaryOpsCompositionOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = _UnaryOpsCompositionOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::MutableOperandRange xMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value y();
  ::mlir::ArrayAttr op_namesAttr();
  ::mlir::ArrayAttr op_names();
  Type T();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void op_namesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type y, ::mlir::Value x, ::mlir::ArrayAttr op_names);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::ArrayAttr op_names);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::ArrayAttr op_names);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaHostComputeMlirOp declarations
//===----------------------------------------------------------------------===//

class _XlaHostComputeMlirOpAdaptor {
public:
  _XlaHostComputeMlirOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _XlaHostComputeMlirOpAdaptor(_XlaHostComputeMlirOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::StringAttr send_key();
  ::mlir::StringAttr recv_key();
  ::mlir::IntegerAttr tpu_core();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _XlaHostComputeMlirOp : public ::mlir::Op<_XlaHostComputeMlirOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = _XlaHostComputeMlirOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::MutableOperandRange inputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::StringAttr send_keyAttr();
  ::llvm::StringRef send_key();
  ::mlir::StringAttr recv_keyAttr();
  ::llvm::StringRef recv_key();
  ::mlir::IntegerAttr tpu_coreAttr();
  uint64_t tpu_core();
  mlir::OperandElementTypeRange Tinputs();
  mlir::ResultElementTypeRange Toutputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void send_keyAttr(::mlir::StringAttr attr);
  void recv_keyAttr(::mlir::StringAttr attr);
  void tpu_coreAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::mlir::StringAttr send_key, ::mlir::StringAttr recv_key, ::mlir::IntegerAttr tpu_core);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::llvm::StringRef send_key, ::llvm::StringRef recv_key, uint64_t tpu_core = 0);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaRecvAtHostOp declarations
//===----------------------------------------------------------------------===//

class _XlaRecvAtHostOpAdaptor {
public:
  _XlaRecvAtHostOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _XlaRecvAtHostOpAdaptor(_XlaRecvAtHostOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value dynamic_key();
  ::mlir::StringAttr key();
  ::mlir::IntegerAttr device_ordinal();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _XlaRecvAtHostOp : public ::mlir::Op<_XlaRecvAtHostOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = _XlaRecvAtHostOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value dynamic_key();
  ::mlir::MutableOperandRange dynamic_keyMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range outputs();
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::IntegerAttr device_ordinalAttr();
  uint64_t device_ordinal();
  mlir::ResultElementTypeRange Toutputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void keyAttr(::mlir::StringAttr attr);
  void device_ordinalAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value dynamic_key, ::mlir::StringAttr key, ::mlir::IntegerAttr device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::Value dynamic_key, ::llvm::StringRef key, uint64_t device_ordinal);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace TF
} // namespace mlir
namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// ::mlir::TF::_XlaSendFromHostOp declarations
//===----------------------------------------------------------------------===//

class _XlaSendFromHostOpAdaptor {
public:
  _XlaSendFromHostOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  _XlaSendFromHostOpAdaptor(_XlaSendFromHostOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::Value dynamic_key();
  ::mlir::StringAttr key();
  ::mlir::IntegerAttr device_ordinal();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class _XlaSendFromHostOp : public ::mlir::Op<_XlaSendFromHostOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::DerivedAttributeOpInterface::Trait> {
public:
  using Op::Op;
  using Adaptor = _XlaSendFromHostOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Value dynamic_key();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange dynamic_keyMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr keyAttr();
  ::llvm::StringRef key();
  ::mlir::IntegerAttr device_ordinalAttr();
  uint64_t device_ordinal();
  mlir::OperandElementTypeRange Tinputs();
  static bool isDerivedAttribute(::llvm::StringRef name);
  ::mlir::DictionaryAttr materializeDerivedAttributes();
  void keyAttr(::mlir::StringAttr attr);
  void device_ordinalAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::StringAttr key, ::mlir::IntegerAttr device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::mlir::StringAttr key, ::mlir::IntegerAttr device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::llvm::StringRef key, uint64_t device_ordinal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::Value dynamic_key, ::llvm::StringRef key, uint64_t device_ordinal);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
};
} // namespace TF
} // namespace mlir

#endif  // GET_OP_CLASSES

