/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/tf_data_optimization.td:23
*/
struct FuseMapAndBatch : public ::mlir::RewritePattern {
  FuseMapAndBatch(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.BatchDatasetV2", {"tf.Const", "tf.MapAndBatchDataset"}, 2, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::ArrayAttr batch_output_types;
    ::mlir::BoolAttr parallel_copy;
    ::mlir::Operation::operand_range drop_remainder(op0->getOperands());
    ::mlir::ArrayAttr batch_output_shapes;
    ::mlir::BoolAttr preserve_cardinality;
    ::mlir::Operation::operand_range batch_size(op0->getOperands());
    ::mlir::Operation::operand_range input_dataset(op0->getOperands());
    ::mlir::Operation::operand_range other_arguments(op0->getOperands());
    ::mlir::ArrayAttr output_shapes;
    ::mlir::SymbolRefAttr f;
    ::mlir::ArrayAttr output_types;
    ::mlir::BoolAttr use_inter_op_parallelism;
    ::mlir::Operation *tblgen_ops[2];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::TF::BatchDatasetV2Op>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::TF::MapDatasetOp>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      input_dataset = castedOp1.getODSOperands(0);
      other_arguments = castedOp1.getODSOperands(1);
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::SymbolRefAttr>("f");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'tf.MapDataset' to have attribute 'f' of type '::mlir::SymbolRefAttr'";
          });
        }
        f = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::ArrayAttr>("output_types");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'tf.MapDataset' to have attribute 'output_types' of type '::mlir::ArrayAttr'";
          });
        }
        output_types = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::ArrayAttr>("output_shapes");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'tf.MapDataset' to have attribute 'output_shapes' of type '::mlir::ArrayAttr'";
          });
        }
        output_shapes = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::BoolAttr>("use_inter_op_parallelism");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(true);
        use_inter_op_parallelism = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::BoolAttr>("preserve_cardinality");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
        preserve_cardinality = tblgen_attr;
      }
      tblgen_ops[1] = op1;
    }
    batch_size = castedOp0.getODSOperands(1);
    drop_remainder = castedOp0.getODSOperands(2);
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::BoolAttr>("parallel_copy");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(false);
      parallel_copy = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::ArrayAttr>("output_types");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'tf.BatchDatasetV2' to have attribute 'output_types' of type '::mlir::ArrayAttr'";
        });
      }
      batch_output_types = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::ArrayAttr>("output_shapes");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'tf.BatchDatasetV2' to have attribute 'output_shapes' of type '::mlir::ArrayAttr'";
        });
      }
      batch_output_shapes = tblgen_attr;
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ::mlir::TF::ConstOp tblgen_ConstOp_0;
    {
      tblgen_ConstOp_0 = rewriter.create<::mlir::TF::ConstOp>(odsLoc,
        /*value=*/DenseElementsAttr::get<int64_t>(RankedTensorType::get({}, rewriter.getIntegerType(64)), 1)
      );
    }
    ::mlir::TF::MapAndBatchDatasetOp tblgen_MapAndBatchDatasetOp_1;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input_dataset.begin()));
      for (auto v: other_arguments) {
        tblgen_values.push_back(v);
      }
      tblgen_values.push_back((*batch_size.begin()));
      tblgen_values.push_back((*tblgen_ConstOp_0.getODSResults(0).begin()));
      tblgen_values.push_back((*drop_remainder.begin()));
      if (auto tmpAttr = f) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("f"), tmpAttr);
      }
      if (auto tmpAttr = batch_output_types) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("output_types"), tmpAttr);
      }
      if (auto tmpAttr = batch_output_shapes) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("output_shapes"), tmpAttr);
      }
      if (auto tmpAttr = preserve_cardinality) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("preserve_cardinality"), tmpAttr);
      }
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_MapAndBatchDatasetOp_1 = rewriter.create<::mlir::TF::MapAndBatchDatasetOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_MapAndBatchDatasetOp_1.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::MLIRContext *context, ::mlir::OwningRewritePatternList &patterns) {
  patterns.insert<FuseMapAndBatch>(context);
}
