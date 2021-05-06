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
namespace linalg {
class BatchMatmulOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvDHWOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvHWOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvNCDHWOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvNCHWOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvNCWOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvNDHWCOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvNHWCOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvNWCOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvWOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class CopyOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DotOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class FillOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class GenericOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class IndexedGenericOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class MatmulOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class MatvecOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingMaxOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingMinOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingSumOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class VecmatOp;
} // namespace linalg
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulOp declarations
//===----------------------------------------------------------------------===//

class BatchMatmulOpAdaptor {
public:
  BatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  BatchMatmulOpAdaptor(BatchMatmulOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class BatchMatmulOp : public ::mlir::Op<BatchMatmulOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = BatchMatmulOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvDHWOp declarations
//===----------------------------------------------------------------------===//

class ConvDHWOpAdaptor {
public:
  ConvDHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ConvDHWOpAdaptor(ConvDHWOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ConvDHWOp : public ::mlir::Op<ConvDHWOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = ConvDHWOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvHWOp declarations
//===----------------------------------------------------------------------===//

class ConvHWOpAdaptor {
public:
  ConvHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ConvHWOpAdaptor(ConvHWOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ConvHWOp : public ::mlir::Op<ConvHWOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = ConvHWOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNCDHWOp declarations
//===----------------------------------------------------------------------===//

class ConvNCDHWOpAdaptor {
public:
  ConvNCDHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ConvNCDHWOpAdaptor(ConvNCDHWOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ConvNCDHWOp : public ::mlir::Op<ConvNCDHWOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = ConvNCDHWOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNCHWOp declarations
//===----------------------------------------------------------------------===//

class ConvNCHWOpAdaptor {
public:
  ConvNCHWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ConvNCHWOpAdaptor(ConvNCHWOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ConvNCHWOp : public ::mlir::Op<ConvNCHWOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = ConvNCHWOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNCWOp declarations
//===----------------------------------------------------------------------===//

class ConvNCWOpAdaptor {
public:
  ConvNCWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ConvNCWOpAdaptor(ConvNCWOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ConvNCWOp : public ::mlir::Op<ConvNCWOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = ConvNCWOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNDHWCOp declarations
//===----------------------------------------------------------------------===//

class ConvNDHWCOpAdaptor {
public:
  ConvNDHWCOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ConvNDHWCOpAdaptor(ConvNDHWCOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ConvNDHWCOp : public ::mlir::Op<ConvNDHWCOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = ConvNDHWCOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNHWCOp declarations
//===----------------------------------------------------------------------===//

class ConvNHWCOpAdaptor {
public:
  ConvNHWCOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ConvNHWCOpAdaptor(ConvNHWCOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ConvNHWCOp : public ::mlir::Op<ConvNHWCOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = ConvNHWCOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvNWCOp declarations
//===----------------------------------------------------------------------===//

class ConvNWCOpAdaptor {
public:
  ConvNWCOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ConvNWCOpAdaptor(ConvNWCOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ConvNWCOp : public ::mlir::Op<ConvNWCOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = ConvNWCOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvOp declarations
//===----------------------------------------------------------------------===//

class ConvOpAdaptor {
public:
  ConvOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  ConvOpAdaptor(ConvOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value filter();
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::ArrayAttr strides();
  ::mlir::ArrayAttr dilations();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ConvOp : public ::mlir::Op<ConvOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::linalg::NInputs<2>::Impl, ::mlir::OpTrait::linalg::ZeroInitTensors, ::mlir::OpTrait::linalg::NOutputs<1>::Impl, ::mlir::OpTrait::linalg::StructuredOpTraits, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = ConvOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value filter();
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::MutableOperandRange filterMutable();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr stridesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > strides();
  ::mlir::ArrayAttr dilationsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > dilations();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  void stridesAttr(::mlir::ArrayAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value filter, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value filter, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    int64_t getStride(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!strides().hasValue()) return 1;
      return strides()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getDilation(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!dilations().hasValue()) return 1;
      return dilations()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getLowPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 0});
    }

    int64_t getHighPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 1});
    }
  
    // TODO: extend to support more than 1 dimensions and potentially grouping
    // too.
    unsigned getNumBatchDimensions() { return 1; }

    unsigned getNumInputFeatureDimensions() { return 1; }

    unsigned getNumOutputFeatureDimensions() { return 1; }

    unsigned getNumSpatialDimensions() {
      return getOutputShapedType(0).getRank() - getNumBatchDimensions() -
             getNumOutputFeatureDimensions();
    }

    ArrayAttr iterator_types() {
      // Outer parallel loops are always the number of output dimensions; i.e.
      // [b, xs, q] in the TF notation above.
      unsigned nPar = getOutputShapedType(0).getRank();
      unsigned nRed = getNumInputFeatureDimensions();
      // Window loops are a special kind of reduction that is never tiled or
      // parallelized across; i.e. [zs] in the TF notation above whose number
      // match `xs` (i.e. 1 window loop per "image" dimension).
      // This may evolve in the future.
      unsigned nWin =
        nPar - getNumBatchDimensions() - getNumInputFeatureDimensions();
      SmallVector<StringRef, 8> iters;
      iters.reserve(nPar + nRed + nWin);
      iters.append(nPar - getNumOutputFeatureDimensions(),
                   getParallelIteratorTypeName());
      iters.append(nWin, getWindowIteratorTypeName());
      iters.append(nRed, getReductionIteratorTypeName());
      iters.append(getNumOutputFeatureDimensions(), getParallelIteratorTypeName());
      return Builder(getContext()).getStrArrayAttr(iters);
    }

    //   F(z0, ..., zN-1, q, k) *
    //     I(b, x0 + z0 - pad_low_0, ..., xN-1 + zN-1 - pad_low_N-1, q)
    //   ->  O(b, x0, ..., xN-1, k)
    // for N equal to `nWindow`. If there is no padding attribute, it will be
    // ignored.
    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto nWin = getNumWindowLoops();
      assert(nWin > 0 && "expected at least one window dimension");
      unsigned idx = 0;
      // In the following, AffineDimExprs are indexed in loop order:
      //   [ b, xs, k,           q,                     zs]
      //    parallels     non-window reductions     windows
      //
      // Parallel dims are exactly the dimensions indexing `output`:
      //     output[b, x[0], ..., x[N-1], k]; i.e.
      //  * batch dimensions (bs with #bs = 1 for now)
      //  * "image" dimensions (xs with #xs = #zs = output_rank - #bs - #ks)
      //  * output filter dimensions (ks with #ks = 1 for now)
      auto bs = makeAffineDimExprs(getNumBatchDimensions(), idx, context);
      auto xs = makeAffineDimExprs(nWin, idx, context);
      // Window reduction dims: sum_{z[0], ..., z[N-1], q}
      auto zs = makeAffineDimExprs(nWin, idx, context);
      // Non-window reduction dim: sum_{z[0], ..., z[N-1], q}
      auto qs = makeAffineDimExprs(
        getNumInputFeatureDimensions(), idx, context);
      auto ks = makeAffineDimExprs(
        getNumOutputFeatureDimensions(), idx, context);
      // Construct the weighedSum expression.
      auto ws = weightedPoolingInputIndex(*this, xs, zs);
      return Builder(getContext()).getAffineMapArrayAttr({
        // filter[z[0], ..., z[N-1], q, k]
        AffineMap::get(idx, 0, concat(concat(zs, qs), ks), context),
        // input[b,
        //       x[0]*s[0] + d[0]*z[0] - pad_low[0],
        //       ...
        //       x[N-1]*s[N-1] + d[N-1]*z[N-1] - pad_low[N-1],
        //       q]
        AffineMap::get(idx, 0, concat(concat(bs, ws), qs), context),
        // output[b, x[0], ..., x[N-1], k]
        AffineMap::get(idx, 0, concat(concat(bs, xs), ks), context)});
    }
  
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ConvWOp declarations
//===----------------------------------------------------------------------===//

class ConvWOpAdaptor {
public:
  ConvWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  ConvWOpAdaptor(ConvWOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class ConvWOp : public ::mlir::Op<ConvWOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = ConvWOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::CopyOp declarations
//===----------------------------------------------------------------------===//

class CopyOpAdaptor {
public:
  CopyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  CopyOpAdaptor(CopyOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::AffineMapAttr inputPermutation();
  ::mlir::AffineMapAttr outputPermutation();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class CopyOp : public ::mlir::Op<CopyOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::CopyOpInterface::Trait, ::mlir::OpTrait::linalg::NInputs<1>::Impl, ::mlir::OpTrait::linalg::ZeroInitTensors, ::mlir::OpTrait::linalg::NOutputs<1>::Impl, ::mlir::OpTrait::linalg::StructuredOpTraits, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = CopyOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value output();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::AffineMapAttr inputPermutationAttr();
  ::llvm::Optional< ::mlir::AffineMap > inputPermutation();
  ::mlir::AffineMapAttr outputPermutationAttr();
  ::llvm::Optional< ::mlir::AffineMap > outputPermutation();
  void inputPermutationAttr(::mlir::AffineMapAttr attr);
  void outputPermutationAttr(::mlir::AffineMapAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value input, Value output);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::AffineMapAttr inputPermutation, /*optional*/::mlir::AffineMapAttr outputPermutation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value output, /*optional*/::mlir::AffineMapAttr inputPermutation, /*optional*/::mlir::AffineMapAttr outputPermutation);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    // Rank-polymorphic.
    //   filling_value -> O(ivs) with parallel iterators.
    ArrayAttr iterator_types() {
      unsigned nPar = getInputShapedType(0).getRank();
      return Builder(getContext()).getStrArrayAttr(
        SmallVector<StringRef, 8>(nPar, getParallelIteratorTypeName()));
    }

    // I(input_perm(ivs)) -> O(output_perm(ivs))
    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto maybeInputMap = inputPermutation();
      auto maybeOutputMap = outputPermutation();
      unsigned inputRank = getInputShapedType(0).getRank();
      unsigned outputRank = getOutputShapedType(0).getRank();
      return Builder(getContext()).getAffineMapArrayAttr({
          extractOrIdentityMap(maybeInputMap, inputRank, context),
          extractOrIdentityMap(maybeOutputMap, outputRank, context)});
    }

    Value getSource() { return input();}
    Value getTarget() { return output(); }
  
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotOp declarations
//===----------------------------------------------------------------------===//

class DotOpAdaptor {
public:
  DotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  DotOpAdaptor(DotOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class DotOp : public ::mlir::Op<DotOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = DotOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillOp declarations
//===----------------------------------------------------------------------===//

class FillOpAdaptor {
public:
  FillOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  FillOpAdaptor(FillOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value output();
  ::mlir::Value value();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class FillOp : public ::mlir::Op<FillOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::linalg::NInputs<0>::Impl, ::mlir::OpTrait::linalg::ZeroInitTensors, ::mlir::OpTrait::linalg::NOutputs<1>::Impl, ::mlir::OpTrait::linalg::StructuredOpTraits, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = FillOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value output();
  ::mlir::Value value();
  ::mlir::MutableOperandRange outputMutable();
  ::mlir::MutableOperandRange valueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value output, ::mlir::Value value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value output, ::mlir::Value value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    // Rank-polymorphic.
    //   filling_value -> O(ivs) with parallel iterators.
    ArrayAttr iterator_types() {
      unsigned nPar = getOutputShapedType(0).getRank();
      return Builder(getContext()).getStrArrayAttr(
        SmallVector<StringRef, 8>(nPar, getParallelIteratorTypeName()));
    }

    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      // filling_value -> O(ivs)
      return Builder(getContext()).getAffineMapArrayAttr({
          extractOrIdentityMap(llvm::None, getNumParallelLoops(), context)});
    }
  
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::GenericOp declarations
//===----------------------------------------------------------------------===//

class GenericOpAdaptor {
public:
  GenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  GenericOpAdaptor(GenericOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::ArrayAttr indexing_maps();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::StringAttr doc();
  ::mlir::StringAttr library_call();
  ::mlir::IntegerAttr symbol_source();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class GenericOp : public ::mlir::Op<GenericOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = GenericOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::ArrayAttr indexing_mapsAttr();
  ::mlir::ArrayAttr indexing_maps();
  ::mlir::ArrayAttr iterator_typesAttr();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::StringAttr docAttr();
  ::llvm::Optional< ::llvm::StringRef > doc();
  ::mlir::StringAttr library_callAttr();
  ::llvm::Optional< ::llvm::StringRef > library_call();
  ::mlir::IntegerAttr symbol_sourceAttr();
  ::llvm::Optional<uint64_t> symbol_source();
  void indexing_mapsAttr(::mlir::ArrayAttr attr);
  void iterator_typesAttr(::mlir::ArrayAttr attr);
  void docAttr(::mlir::StringAttr attr);
  void library_callAttr(::mlir::StringAttr attr);
  void symbol_sourceAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes,ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall, IntegerAttr symbolSource, function_ref<void(OpBuilder &, Location, ValueRange)> = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall, IntegerAttr symbolSource, function_ref<void(OpBuilder &, Location, ValueRange)> = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes,ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, function_ref<void(OpBuilder &, Location, ValueRange)> = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, function_ref<void(OpBuilder &, Location, ValueRange)> = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange output_buffers, ::mlir::ValueRange init_tensors, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call, /*optional*/::mlir::IntegerAttr symbol_source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

    SmallVector<StringRef, 8> linalgTraitAttrNames() {
      return SmallVector<StringRef, 8>{
        getDocAttrName(),
        getIndexingMapsAttrName(), getLibraryCallAttrName(),
        getIteratorTypesAttrName(), getSymbolSourceAttrName()
      };
    }
    std::string getLibraryCallName() {
      return library_call().hasValue() ?
        library_call()->str() : "op_has_no_registered_library_name";
    }
    llvm::Optional<unsigned> getSymbolSource() {
      auto ss = symbol_source();
      return ss.hasValue() ?
        llvm::Optional<unsigned>(ss.getValue()) : llvm::None;
    }
  
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::IndexedGenericOp declarations
//===----------------------------------------------------------------------===//

class IndexedGenericOpAdaptor {
public:
  IndexedGenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  IndexedGenericOpAdaptor(IndexedGenericOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::ArrayAttr indexing_maps();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::StringAttr doc();
  ::mlir::StringAttr library_call();
  ::mlir::IntegerAttr symbol_source();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class IndexedGenericOp : public ::mlir::Op<IndexedGenericOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = IndexedGenericOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::ArrayAttr indexing_mapsAttr();
  ::mlir::ArrayAttr indexing_maps();
  ::mlir::ArrayAttr iterator_typesAttr();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::StringAttr docAttr();
  ::llvm::Optional< ::llvm::StringRef > doc();
  ::mlir::StringAttr library_callAttr();
  ::llvm::Optional< ::llvm::StringRef > library_call();
  ::mlir::IntegerAttr symbol_sourceAttr();
  ::llvm::Optional<uint64_t> symbol_source();
  void indexing_mapsAttr(::mlir::ArrayAttr attr);
  void iterator_typesAttr(::mlir::ArrayAttr attr);
  void docAttr(::mlir::StringAttr attr);
  void library_callAttr(::mlir::StringAttr attr);
  void symbol_sourceAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes,ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall, IntegerAttr symbolSource, function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)> = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall, IntegerAttr symbolSource, function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)> = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes,ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)> = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)> = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange output_buffers, ::mlir::ValueRange init_tensors, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call, /*optional*/::mlir::IntegerAttr symbol_source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

    SmallVector<StringRef, 8> linalgTraitAttrNames() {
      return SmallVector<StringRef, 8>{
        getDocAttrName(),
        getIndexingMapsAttrName(), getLibraryCallAttrName(),
        getIteratorTypesAttrName(), getSymbolSourceAttrName()
      };
    }
    std::string getLibraryCallName() {
      return library_call().hasValue() ?
        library_call()->str() : "op_has_no_registered_library_name";
    }
    llvm::Optional<unsigned> getSymbolSource() {
      auto ss = symbol_source();
      return ss.hasValue() ?
        llvm::Optional<unsigned>(ss.getValue()) : llvm::None;
    }
  
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulOp declarations
//===----------------------------------------------------------------------===//

class MatmulOpAdaptor {
public:
  MatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  MatmulOpAdaptor(MatmulOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class MatmulOp : public ::mlir::Op<MatmulOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = MatmulOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecOp declarations
//===----------------------------------------------------------------------===//

class MatvecOpAdaptor {
public:
  MatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  MatvecOpAdaptor(MatvecOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class MatvecOp : public ::mlir::Op<MatvecOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = MatvecOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingMaxOp declarations
//===----------------------------------------------------------------------===//

class PoolingMaxOpAdaptor {
public:
  PoolingMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  PoolingMaxOpAdaptor(PoolingMaxOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::ArrayAttr strides();
  ::mlir::ArrayAttr dilations();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class PoolingMaxOp : public ::mlir::Op<PoolingMaxOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::linalg::NInputs<2>::Impl, ::mlir::OpTrait::linalg::ZeroInitTensors, ::mlir::OpTrait::linalg::NOutputs<1>::Impl, ::mlir::OpTrait::linalg::StructuredOpTraits, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = PoolingMaxOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange windowDimsMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr stridesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > strides();
  ::mlir::ArrayAttr dilationsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > dilations();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  void stridesAttr(::mlir::ArrayAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    int64_t getStride(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!strides().hasValue()) return 1;
      return strides()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getDilation(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!dilations().hasValue()) return 1;
      return dilations()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getLowPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 0});
    }

    int64_t getHighPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 1});
    }
  
    ArrayAttr iterator_types() {
      // Outer parallel loops are always the number of output dimensions.
      unsigned nPar = getOutputShapedType(0).getRank();
      // The window loops has the same number loops with output dimensions.
      unsigned nWin = nPar;
      SmallVector<StringRef, 8> iters(nPar, getParallelIteratorTypeName());
      iters.reserve(nPar + nWin);
      iters.append(nWin, getWindowIteratorTypeName());
      return Builder(getContext()).getStrArrayAttr(iters);
    }

    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto nPar = getNumParallelLoops();
      auto nWin = getNumWindowLoops();
      assert(nWin > 0 && "expected at least one window dimension");
      unsigned idx = 0;
      auto outputDims = makeAffineDimExprs(nPar, idx, context);
      auto windowDims = makeAffineDimExprs(nWin, idx, context);
      // Construct the weighedSum expression.
      auto inputDims =
          weightedPoolingInputIndex(*this, outputDims, windowDims);
      return Builder(getContext()).getAffineMapArrayAttr({
        // input
        AffineMap::get(idx, 0, inputDims, context),
        // windowDims
        AffineMap::get(idx, 0, windowDims, context),
        // output
        AffineMap::get(idx, 0, outputDims, context)});
    }
  
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingMinOp declarations
//===----------------------------------------------------------------------===//

class PoolingMinOpAdaptor {
public:
  PoolingMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  PoolingMinOpAdaptor(PoolingMinOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::ArrayAttr strides();
  ::mlir::ArrayAttr dilations();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class PoolingMinOp : public ::mlir::Op<PoolingMinOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::linalg::NInputs<2>::Impl, ::mlir::OpTrait::linalg::ZeroInitTensors, ::mlir::OpTrait::linalg::NOutputs<1>::Impl, ::mlir::OpTrait::linalg::StructuredOpTraits, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = PoolingMinOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange windowDimsMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr stridesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > strides();
  ::mlir::ArrayAttr dilationsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > dilations();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  void stridesAttr(::mlir::ArrayAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    int64_t getStride(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!strides().hasValue()) return 1;
      return strides()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getDilation(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!dilations().hasValue()) return 1;
      return dilations()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getLowPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 0});
    }

    int64_t getHighPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 1});
    }
  
    ArrayAttr iterator_types() {
      // Outer parallel loops are always the number of output dimensions.
      unsigned nPar = getOutputShapedType(0).getRank();
      // The window loops has the same number loops with output dimensions.
      unsigned nWin = nPar;
      SmallVector<StringRef, 8> iters(nPar, getParallelIteratorTypeName());
      iters.reserve(nPar + nWin);
      iters.append(nWin, getWindowIteratorTypeName());
      return Builder(getContext()).getStrArrayAttr(iters);
    }

    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto nPar = getNumParallelLoops();
      auto nWin = getNumWindowLoops();
      assert(nWin > 0 && "expected at least one window dimension");
      unsigned idx = 0;
      auto outputDims = makeAffineDimExprs(nPar, idx, context);
      auto windowDims = makeAffineDimExprs(nWin, idx, context);
      // Construct the weighedSum expression.
      auto inputDims =
          weightedPoolingInputIndex(*this, outputDims, windowDims);
      return Builder(getContext()).getAffineMapArrayAttr({
        // input
        AffineMap::get(idx, 0, inputDims, context),
        // windowDims
        AffineMap::get(idx, 0, windowDims, context),
        // output
        AffineMap::get(idx, 0, outputDims, context)});
    }
  
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingSumOp declarations
//===----------------------------------------------------------------------===//

class PoolingSumOpAdaptor {
public:
  PoolingSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr);
  PoolingSumOpAdaptor(PoolingSumOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::ArrayAttr strides();
  ::mlir::ArrayAttr dilations();
  ::mlir::DenseIntElementsAttr padding();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class PoolingSumOp : public ::mlir::Op<PoolingSumOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::linalg::NInputs<2>::Impl, ::mlir::OpTrait::linalg::ZeroInitTensors, ::mlir::OpTrait::linalg::NOutputs<1>::Impl, ::mlir::OpTrait::linalg::StructuredOpTraits, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = PoolingSumOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value input();
  ::mlir::Value windowDims();
  ::mlir::Value output();
  ::mlir::MutableOperandRange inputMutable();
  ::mlir::MutableOperandRange windowDimsMutable();
  ::mlir::MutableOperandRange outputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr stridesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > strides();
  ::mlir::ArrayAttr dilationsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > dilations();
  ::mlir::DenseIntElementsAttr paddingAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > padding();
  void stridesAttr(::mlir::ArrayAttr attr);
  void dilationsAttr(::mlir::ArrayAttr attr);
  void paddingAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input, ::mlir::Value windowDims, ::mlir::Value output, /*optional*/::mlir::ArrayAttr strides, /*optional*/::mlir::ArrayAttr dilations, /*optional*/::mlir::DenseIntElementsAttr padding);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);

    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  
    int64_t getStride(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!strides().hasValue()) return 1;
      return strides()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getDilation(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!dilations().hasValue()) return 1;
      return dilations()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getLowPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 0});
    }

    int64_t getHighPad(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!padding().hasValue()) return 0;
      return padding().getValue().getValue<int64_t>({i, 1});
    }
  
    ArrayAttr iterator_types() {
      // Outer parallel loops are always the number of output dimensions.
      unsigned nPar = getOutputShapedType(0).getRank();
      // The window loops has the same number loops with output dimensions.
      unsigned nWin = nPar;
      SmallVector<StringRef, 8> iters(nPar, getParallelIteratorTypeName());
      iters.reserve(nPar + nWin);
      iters.append(nWin, getWindowIteratorTypeName());
      return Builder(getContext()).getStrArrayAttr(iters);
    }

    ArrayAttr indexing_maps() {
      MLIRContext *context = getContext();
      auto nPar = getNumParallelLoops();
      auto nWin = getNumWindowLoops();
      assert(nWin > 0 && "expected at least one window dimension");
      unsigned idx = 0;
      auto outputDims = makeAffineDimExprs(nPar, idx, context);
      auto windowDims = makeAffineDimExprs(nWin, idx, context);
      // Construct the weighedSum expression.
      auto inputDims =
          weightedPoolingInputIndex(*this, outputDims, windowDims);
      return Builder(getContext()).getAffineMapArrayAttr({
        // input
        AffineMap::get(idx, 0, inputDims, context),
        // windowDims
        AffineMap::get(idx, 0, windowDims, context),
        // output
        AffineMap::get(idx, 0, outputDims, context)});
    }
  
};
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatOp declarations
//===----------------------------------------------------------------------===//

class VecmatOpAdaptor {
public:
  VecmatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs);
  VecmatOpAdaptor(VecmatOp&op);
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange output_buffers();
  ::mlir::ValueRange init_tensors();
  ::mlir::LogicalResult verify(::mlir::Location loc);

private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
};
class VecmatOp : public ::mlir::Op<VecmatOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::linalg::NamedStructuredOpTrait, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::linalg::LinalgOp::Trait> {
public:
  using Op::Op;
  using Adaptor = VecmatOpAdaptor;
  static ::llvm::StringRef getOperationName();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range output_buffers();
  ::mlir::Operation::operand_range init_tensors();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange output_buffersMutable();
  ::mlir::MutableOperandRange init_tensorsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputBuffers, ValueRange initTensors);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::OwningRewritePatternList &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);

        // Auto-generated.
        ArrayAttr iterator_types();
        ArrayAttr indexing_maps();
        static void regionBuilder(Block &block);

        // Generic methods.
        static unsigned getNumRegionArgs() { return 3; }
        std::string getLibraryCallName() {
          return generateLibraryCallName(getOperation());
        }
      
};
} // namespace linalg
} // namespace mlir

#endif  // GET_OP_CLASSES

