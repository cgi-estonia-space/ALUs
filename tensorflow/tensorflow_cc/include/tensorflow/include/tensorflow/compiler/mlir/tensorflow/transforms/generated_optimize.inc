/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/optimize.td:34
*/
struct FuseMulAndConv2D : public ::mlir::RewritePattern {
  FuseMulAndConv2D(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Mul", {"std.constant", "tf.Conv2D", "tf.Mul"}, 4, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ConstantOp multiplier;
    ::mlir::ArrayAttr dilations;
    ::mlir::StringAttr data_format;
    ::mlir::ArrayAttr explicit_padding;
    ::mlir::Attribute mul_value;
    ::mlir::StringAttr padding;
    ::mlir::BoolAttr use_cudnn;
    ::mlir::TF::Conv2DOp conv;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ConstantOp filter;
    ::mlir::Attribute filter_value;
    ::mlir::TF::MulOp mul;
    ::mlir::ArrayAttr strides;
    ::mlir::Operation *tblgen_ops[4];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::TF::MulOp>(op0); (void)castedOp0;
    mul = castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::TF::Conv2DOp>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      conv = castedOp1;
      input = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        auto castedOp2 = ::llvm::dyn_cast_or_null<ConstantOp>(op2); (void)castedOp2;
        if (!castedOp2) return failure();
        filter = castedOp2;
        {
          auto tblgen_attr = op2->getAttrOfType<::mlir::Attribute>("value");(void)tblgen_attr;
          if (!(tblgen_attr)){
            return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
              diag << "expected op 'std.constant' to have attribute 'value' of type '::mlir::Attribute'";
            });
          }
          if (!((tblgen_attr.cast<ElementsAttr>().getType().getElementType().isF32()))){
            return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
              diag << "op 'std.constant' attribute 'value' failed to satisfy constraint: float constant tensor";
            });
          }
          filter_value = tblgen_attr;
        }
        tblgen_ops[1] = op2;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::ArrayAttr>("strides");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'tf.Conv2D' to have attribute 'strides' of type '::mlir::ArrayAttr'";
          });
        }
        strides = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::BoolAttr>("use_cudnn_on_gpu");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getBoolAttr(true);
        use_cudnn = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::StringAttr>("padding");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'tf.Conv2D' to have attribute 'padding' of type '::mlir::StringAttr'";
          });
        }
        padding = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::ArrayAttr>("explicit_paddings");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getI64ArrayAttr({});
        explicit_padding = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::StringAttr>("data_format");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("NHWC");
        if (!((tblgen_attr == rewriter.getStringAttr("NHWC")))){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "op 'tf.Conv2D' attribute 'data_format' failed to satisfy constraint: constant attribute NHWC";
          });
        }
        data_format = tblgen_attr;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::ArrayAttr>("dilations");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getI64ArrayAttr({1, 1, 1, 1});
        dilations = tblgen_attr;
      }
      tblgen_ops[2] = op1;
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<ConstantOp>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      multiplier = castedOp1;
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::Attribute>("value");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'std.constant' to have attribute 'value' of type '::mlir::Attribute'";
          });
        }
        if (!((tblgen_attr.cast<ElementsAttr>().getType().getElementType().isF32()))){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "op 'std.constant' attribute 'value' failed to satisfy constraint: float constant tensor";
          });
        }
        mul_value = tblgen_attr;
      }
      tblgen_ops[3] = op1;
    }
    if (!((TFL::IsBroadcastableElementsAttrs(filter_value, mul_value) && TFL::IsDimensionsDegenerateExceptLastOne(mul_value)))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'filter_value, mul_value' failed to satisfy constraint: CanFuseMulAndConv2D";
      });
    }
    if (!(((*conv.getODSResults(0).begin()).hasOneUse()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'conv' failed to satisfy constraint: HasOneUse";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ConstantOp tblgen_ConstantOp_0;
    {
      tblgen_ConstantOp_0 = rewriter.create<ConstantOp>((*filter.getODSResults(0).begin()).getLoc(),
        /*value=*/filter_value
      );
    }
    ConstantOp tblgen_ConstantOp_1;
    {
      tblgen_ConstantOp_1 = rewriter.create<ConstantOp>((*multiplier.getODSResults(0).begin()).getLoc(),
        /*value=*/mul_value
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_2;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstantOp_0.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_1.getODSResults(0).begin());
      tblgen_MulOp_2 = rewriter.create<::mlir::TF::MulOp>((*mul.getODSResults(0).begin()).getLoc(),
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::Conv2DOp tblgen_Conv2DOp_3;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*input.begin()));
      tblgen_values.push_back((*tblgen_MulOp_2.getODSResults(0).begin()));
      if (auto tmpAttr = strides) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("strides"), tmpAttr);
      }
      if (auto tmpAttr = use_cudnn) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("use_cudnn_on_gpu"), tmpAttr);
      }
      if (auto tmpAttr = padding) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("padding"), tmpAttr);
      }
      if (auto tmpAttr = explicit_padding) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("explicit_paddings"), tmpAttr);
      }
      if (auto tmpAttr = data_format) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("data_format"), tmpAttr);
      }
      if (auto tmpAttr = dilations) {
        tblgen_attrs.emplace_back(rewriter.getIdentifier("dilations"), tmpAttr);
      }
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_Conv2DOp_3 = rewriter.create<::mlir::TF::Conv2DOp>((*conv.getODSResults(0).begin()).getLoc(), tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_Conv2DOp_3.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/optimize.td:80
*/
struct PassthroughMulAndAddV2 : public ::mlir::RewritePattern {
  PassthroughMulAndAddV2(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Mul", {"std.constant", "tf.AddV2", "tf.Mul"}, 4, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute value;
    ::mlir::Attribute bias;
    ::mlir::TF::AddV2Op output;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation *tblgen_ops[4];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::TF::MulOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::TF::AddV2Op>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      output = castedOp1;
      input = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        auto castedOp2 = ::llvm::dyn_cast_or_null<ConstantOp>(op2); (void)castedOp2;
        if (!castedOp2) return failure();
        {
          auto tblgen_attr = op2->getAttrOfType<::mlir::Attribute>("value");(void)tblgen_attr;
          if (!(tblgen_attr)){
            return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
              diag << "expected op 'std.constant' to have attribute 'value' of type '::mlir::Attribute'";
            });
          }
          if (!((tblgen_attr.cast<ElementsAttr>().getType().getElementType().isF32()))){
            return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
              diag << "op 'std.constant' attribute 'value' failed to satisfy constraint: float constant tensor";
            });
          }
          bias = tblgen_attr;
        }
        tblgen_ops[1] = op2;
      }
      tblgen_ops[2] = op1;
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<ConstantOp>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::Attribute>("value");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'std.constant' to have attribute 'value' of type '::mlir::Attribute'";
          });
        }
        if (!((tblgen_attr.cast<ElementsAttr>().getType().getElementType().isF32()))){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "op 'std.constant' attribute 'value' failed to satisfy constraint: float constant tensor";
          });
        }
        value = tblgen_attr;
      }
      tblgen_ops[3] = op1;
    }
    if (!((llvm::isa_and_nonnull<mlir::TF::Conv2DOp>((*input.begin()).getDefiningOp())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input' failed to satisfy constraint: DefinedByConv2D";
      });
    }
    if (!(((*output.getODSResults(0).begin()).hasOneUse()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'output' failed to satisfy constraint: HasOneUse";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ConstantOp tblgen_ConstantOp_0;
    {
      tblgen_ConstantOp_0 = rewriter.create<ConstantOp>(odsLoc,
        /*value=*/value
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_0.getODSResults(0).begin());
      tblgen_MulOp_1 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ConstantOp tblgen_ConstantOp_2;
    {
      tblgen_ConstantOp_2 = rewriter.create<ConstantOp>(odsLoc,
        /*value=*/bias
      );
    }
    ConstantOp tblgen_ConstantOp_3;
    {
      tblgen_ConstantOp_3 = rewriter.create<ConstantOp>(odsLoc,
        /*value=*/value
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstantOp_2.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_3.getODSResults(0).begin());
      tblgen_MulOp_4 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::AddV2Op tblgen_AddV2Op_5;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_MulOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_4.getODSResults(0).begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AddV2Op_5 = rewriter.create<::mlir::TF::AddV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddV2Op_5.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

/* Generated from:
    tensorflow/compiler/mlir/tensorflow/transforms/optimize.td:61
*/
struct PassthroughMulAndBiasAdd : public ::mlir::RewritePattern {
  PassthroughMulAndBiasAdd(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("tf.Mul", {"std.constant", "tf.AddV2", "tf.Mul"}, 4, context) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::Attribute value;
    ::mlir::StringAttr format;
    ::mlir::Attribute bias;
    ::mlir::TF::BiasAddOp output;
    ::mlir::Operation::operand_range input(op0->getOperands());
    ::mlir::Operation *tblgen_ops[4];

    // Match
    tblgen_ops[0] = op0;
    auto castedOp0 = ::llvm::dyn_cast_or_null<::mlir::TF::MulOp>(op0); (void)castedOp0;
    {
      auto *op1 = (*castedOp0.getODSOperands(0).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<::mlir::TF::BiasAddOp>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      output = castedOp1;
      input = castedOp1.getODSOperands(0);
      {
        auto *op2 = (*castedOp1.getODSOperands(1).begin()).getDefiningOp();
        auto castedOp2 = ::llvm::dyn_cast_or_null<ConstantOp>(op2); (void)castedOp2;
        if (!castedOp2) return failure();
        {
          auto tblgen_attr = op2->getAttrOfType<::mlir::Attribute>("value");(void)tblgen_attr;
          if (!(tblgen_attr)){
            return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
              diag << "expected op 'std.constant' to have attribute 'value' of type '::mlir::Attribute'";
            });
          }
          if (!((tblgen_attr.cast<ElementsAttr>().getType().getElementType().isF32()))){
            return rewriter.notifyMatchFailure(op2, [&](::mlir::Diagnostic &diag) {
              diag << "op 'std.constant' attribute 'value' failed to satisfy constraint: float constant tensor";
            });
          }
          bias = tblgen_attr;
        }
        tblgen_ops[1] = op2;
      }
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::StringAttr>("data_format");(void)tblgen_attr;
        if (!tblgen_attr) tblgen_attr = rewriter.getStringAttr("NHWC");
        if (!((tblgen_attr == rewriter.getStringAttr("NHWC")))){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "op 'tf.BiasAdd' attribute 'data_format' failed to satisfy constraint: constant attribute NHWC";
          });
        }
        format = tblgen_attr;
      }
      tblgen_ops[2] = op1;
    }
    {
      auto *op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
      auto castedOp1 = ::llvm::dyn_cast_or_null<ConstantOp>(op1); (void)castedOp1;
      if (!castedOp1) return failure();
      {
        auto tblgen_attr = op1->getAttrOfType<::mlir::Attribute>("value");(void)tblgen_attr;
        if (!(tblgen_attr)){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "expected op 'std.constant' to have attribute 'value' of type '::mlir::Attribute'";
          });
        }
        if (!((tblgen_attr.cast<ElementsAttr>().getType().getElementType().isF32()))){
          return rewriter.notifyMatchFailure(op1, [&](::mlir::Diagnostic &diag) {
            diag << "op 'std.constant' attribute 'value' failed to satisfy constraint: float constant tensor";
          });
        }
        value = tblgen_attr;
      }
      tblgen_ops[3] = op1;
    }
    if (!((llvm::isa_and_nonnull<mlir::TF::Conv2DOp>((*input.begin()).getDefiningOp())))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'input' failed to satisfy constraint: DefinedByConv2D";
      });
    }
    if (!(((*output.getODSResults(0).begin()).hasOneUse()))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'output' failed to satisfy constraint: HasOneUse";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc(), tblgen_ops[2]->getLoc(), tblgen_ops[3]->getLoc()}); (void)odsLoc;
    ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;
    ConstantOp tblgen_ConstantOp_0;
    {
      tblgen_ConstantOp_0 = rewriter.create<ConstantOp>(odsLoc,
        /*value=*/value
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_1;
    {
      ::mlir::Value tblgen_value_0 = (*input.begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_0.getODSResults(0).begin());
      tblgen_MulOp_1 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ConstantOp tblgen_ConstantOp_2;
    {
      tblgen_ConstantOp_2 = rewriter.create<ConstantOp>(odsLoc,
        /*value=*/bias
      );
    }
    ConstantOp tblgen_ConstantOp_3;
    {
      tblgen_ConstantOp_3 = rewriter.create<ConstantOp>(odsLoc,
        /*value=*/value
      );
    }
    ::mlir::TF::MulOp tblgen_MulOp_4;
    {
      ::mlir::Value tblgen_value_0 = (*tblgen_ConstantOp_2.getODSResults(0).begin());
      ::mlir::Value tblgen_value_1 = (*tblgen_ConstantOp_3.getODSResults(0).begin());
      tblgen_MulOp_4 = rewriter.create<::mlir::TF::MulOp>(odsLoc,
        /*x=*/tblgen_value_0,
        /*y=*/tblgen_value_1
      );
    }
    ::mlir::TF::AddV2Op tblgen_AddV2Op_5;
    {
      ::mlir::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::mlir::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      tblgen_values.push_back((*tblgen_MulOp_1.getODSResults(0).begin()));
      tblgen_values.push_back((*tblgen_MulOp_4.getODSResults(0).begin()));
      ::mlir::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      for (auto v: castedOp0.getODSResults(0)) {
        tblgen_types.push_back(v.getType());
      }
      tblgen_AddV2Op_5 = rewriter.create<::mlir::TF::AddV2Op>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }

    for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ tblgen_AddV2Op_5.getODSResults(0) }) {
      tblgen_repl_values.push_back(v);
    }

    rewriter.replaceOp(op0, tblgen_repl_values);
    return ::mlir::success();
  };
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::MLIRContext *context, ::mlir::OwningRewritePatternList &patterns) {
  patterns.insert<FuseMulAndConv2D>(context);
  patterns.insert<PassthroughMulAndAddV2>(context);
  patterns.insert<PassthroughMulAndBiasAdd>(context);
}
