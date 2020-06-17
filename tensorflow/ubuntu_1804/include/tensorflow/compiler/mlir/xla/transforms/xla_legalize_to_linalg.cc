/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include "absl/memory/memory.h"
#include "llvm/ADT/APInt.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // TF:llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/AffineExpr.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"

namespace mlir {
namespace {

ArrayAttr GetNParallelLoopsAttrs(unsigned nParallelLoops, Builder b) {
  auto parallelLoopTypeAttr = b.getStringAttr("parallel");
  SmallVector<Attribute, 3> iteratorTypes;
  for (int i = 0; i < nParallelLoops; ++i) {
    iteratorTypes.push_back(parallelLoopTypeAttr);
  }
  return b.getArrayAttr(iteratorTypes);
}

template <typename OpTy, bool isLHLO = true>
class PointwiseToLinalgConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      OpTy op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto argType =
        op.getOperation()->getOperand(0).getType().template cast<ShapedType>();
    if (!argType.hasRank()) {
      emitError(loc, "lhlo to linalg conversion expects ranked args");
      return ConversionPattern::matchFailure();
    }
    if (!argType.getElementType().isIntOrFloat()) {
      return ConversionPattern::matchFailure();
    }

    // Construct the indexing maps needed for linalg.generic ops.
    SmallVector<Attribute, 2> indexingMaps;
    SmallVector<Type, 4> bodyArgTypes, bodyResultTypes, opResultTypes;

    // This doesnt account for implicit broadcast, but the working assumption
    // here is that are broadcasts have been made explicit.
    unsigned nloops = argType.getRank();
    if (!nloops) {
      return ConversionPattern::matchFailure();
    }
    int operandCount = (isLHLO ? args.size() - 1 : args.size());
    auto verifyArgOrResultType = [&](Value val) -> ShapedType {
      auto shapedType = val.getType().dyn_cast<ShapedType>();
      if (!shapedType ||
          (!shapedType.isa<MemRefType>() &&
           !shapedType.isa<RankedTensorType>()) ||
          shapedType.getRank() != nloops)
        return nullptr;
      indexingMaps.emplace_back(
          AffineMapAttr::get(rewriter.getMultiDimIdentityMap(nloops)));
      return shapedType;
    };
    for (const auto& arg : llvm::enumerate(args)) {
      auto shapedType = verifyArgOrResultType(arg.value());
      if (!shapedType) return ConversionPattern::matchFailure();
      auto& result_or_body_arg =
          arg.index() < operandCount ? bodyArgTypes : bodyResultTypes;
      result_or_body_arg.emplace_back(shapedType.getElementType());
    }
    if (!isLHLO) {
      // HLO operations have return as tensor types.
      assert(bodyResultTypes.empty() &&
             "When lowering HLO ops result can't be part of arguments");
      Value result = op.getOperation()->getResult(0);
      auto shapedType = verifyArgOrResultType(result);
      if (!shapedType) return ConversionPattern::matchFailure();
      bodyResultTypes.push_back(shapedType.getElementType());
      opResultTypes.push_back(shapedType);
    }

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, opResultTypes, args,
        rewriter.getI64IntegerAttr(bodyArgTypes.size()),     // args_in
        rewriter.getI64IntegerAttr(bodyResultTypes.size()),  // args_out
        rewriter.getArrayAttr(indexingMaps),
        GetNParallelLoopsAttrs(nloops, rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    // Add a block to the region.
    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    block->addArguments(bodyArgTypes);
    if (isLHLO) block->addArguments(bodyResultTypes);

    SmallVector<Value, 4> bodyArgs;
    for (int i = 0, e = bodyArgTypes.size(); i < e; ++i) {
      bodyArgs.push_back(block->getArgument(i));
    }

    rewriter.setInsertionPointToEnd(block);
    // TODO(ravishankarm) : For now use the method in xla_lhlo namespace. That
    // method needs to be moved out of there.
    Value opResult = xla_lhlo::MapXlaOpToStdScalarOp<OpTy>(
        llvm::cast<OpTy>(op), bodyResultTypes, bodyArgs, &rewriter);
    if (!opResult) {
      return ConversionPattern::matchFailure();
    }
    rewriter.create<linalg::YieldOp>(loc, opResult);
    rewriter.replaceOp(op, linalgOp.getOperation()->getResults());
    return ConversionPattern::matchSuccess();
  }
};

template <typename LhloOp>
class ScalarPointwiseToStandardConverter : public OpConversionPattern<LhloOp> {
 public:
  using OpConversionPattern<LhloOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      LhloOp lhlo_op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = lhlo_op.getLoc();
    auto argType =
        lhlo_op.getOperand(0).getType().template dyn_cast<ShapedType>();
    if (!argType || !argType.getElementType().isIntOrFloat() ||
        (argType.getRank() != 0)) {
      return ConversionPattern::matchFailure();
    }

    // Create two loads from the input.
    auto lhs = rewriter.create<LoadOp>(loc, lhlo_op.lhs());
    auto rhs = rewriter.create<LoadOp>(loc, lhlo_op.rhs());
    // TODO(ravishankarm) : Move this method out of xla_lhlo namespace.
    Value opResult = xla_lhlo::MapXlaOpToStdScalarOp<LhloOp>(
        llvm::cast<LhloOp>(lhlo_op), argType.getElementType(),
        llvm::ArrayRef<Value>{lhs, rhs}, &rewriter);
    rewriter.create<StoreOp>(loc, opResult, lhlo_op.out());
    rewriter.eraseOp(lhlo_op);
    return ConversionPattern::matchSuccess();
  }
};

class BroadcastInDimConverter
    : public OpConversionPattern<xla_lhlo::BroadcastInDimOp> {
 public:
  using OpConversionPattern<xla_lhlo::BroadcastInDimOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_lhlo::BroadcastInDimOp broadcastOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto operandMemrefType =
        broadcastOp.operand().getType().dyn_cast<MemRefType>();
    auto resultMemrefType =
        broadcastOp.output().getType().dyn_cast<MemRefType>();
    if (!operandMemrefType || !resultMemrefType) return matchFailure();
    auto broadcastDims = broadcastOp.broadcast_dimensions();
    if (!broadcastDims.hasValue()) return matchFailure();

    return broadcastDims.getValue().getIntValues().empty()
               ? emitScalarBroadcast(broadcastOp, args, resultMemrefType,
                                     &rewriter)
               : emitNonScalarBroadcast(broadcastOp, args, operandMemrefType,
                                        resultMemrefType, &rewriter);
  }

 private:
  PatternMatchResult emitScalarBroadcast(
      xla_lhlo::BroadcastInDimOp broadcastOp, ArrayRef<Value> args,
      MemRefType resultMemrefType, ConversionPatternRewriter* rewriter) const {
    unsigned nloops = resultMemrefType.getRank();
    SmallVector<Attribute, 1> indexingMaps{
        AffineMapAttr::get(rewriter->getMultiDimIdentityMap(nloops))};
    auto loc = broadcastOp.getLoc();
    auto linalgOp = rewriter->create<linalg::GenericOp>(
        loc, ArrayRef<Type>{}, broadcastOp.output(),
        rewriter->getI64IntegerAttr(0),  // args_in
        rewriter->getI64IntegerAttr(1),  // args_out
        rewriter->getArrayAttr(indexingMaps),
        GetNParallelLoopsAttrs(nloops, *rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    // Add a block to the region.
    auto* region = &linalgOp.region();
    auto* block = rewriter->createBlock(region, region->end());
    block->addArguments(resultMemrefType.getElementType());

    rewriter->setInsertionPointToEnd(block);
    auto scalar =
        rewriter->create<LoadOp>(loc, broadcastOp.operand(), llvm::None);
    rewriter->create<linalg::YieldOp>(loc, scalar.getResult());
    rewriter->eraseOp(broadcastOp);
    return matchSuccess();
  }

  PatternMatchResult emitNonScalarBroadcast(
      xla_lhlo::BroadcastInDimOp broadcastOp, ArrayRef<Value> args,
      MemRefType operandMemrefType, MemRefType resultMemrefType,
      ConversionPatternRewriter* rewriter) const {
    SmallVector<Type, 4> bodyArgTypes{operandMemrefType.getElementType()};

    unsigned nloops = resultMemrefType.getRank();

    SmallVector<AffineExpr, 4> dimExprs;
    {
      dimExprs.reserve(nloops);

      auto operandShape = operandMemrefType.getShape();
      int index = 0;
      for (const auto& broadcastSize :
           broadcastOp.broadcast_dimensions().getValue().getIntValues()) {
        int size = broadcastSize.getSExtValue();
        dimExprs.push_back(
            operandShape[index++] == 1
                ? mlir::getAffineConstantExpr(0, broadcastOp.getContext())
                : mlir::getAffineDimExpr(size, broadcastOp.getContext()));
      }
    }

    // Construct the indexing maps needed for linalg.generic ops.
    SmallVector<Attribute, 2> indexingMaps{
        AffineMapAttr::get(AffineMap::get(nloops, /*symbolCount=*/0, dimExprs)),
        AffineMapAttr::get(rewriter->getMultiDimIdentityMap(nloops))};

    auto loc = broadcastOp.getLoc();
    auto linalgOp = rewriter->create<linalg::GenericOp>(
        loc, ArrayRef<Type>{}, args,
        rewriter->getI64IntegerAttr(bodyArgTypes.size()),  // args_in
        rewriter->getI64IntegerAttr(1),                    // args_out
        rewriter->getArrayAttr(indexingMaps),
        GetNParallelLoopsAttrs(nloops, *rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    // Add a block to the region.
    auto* region = &linalgOp.region();
    auto* block = rewriter->createBlock(region, region->end());
    block->addArguments(bodyArgTypes);
    block->addArguments(resultMemrefType.getElementType());

    rewriter->setInsertionPointToEnd(block);
    rewriter->create<linalg::YieldOp>(loc, block->getArgument(0));
    rewriter->eraseOp(broadcastOp);
    return matchSuccess();
  }
};

class IotaConverter : public OpConversionPattern<xla_lhlo::IotaOp> {
 public:
  using OpConversionPattern<xla_lhlo::IotaOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_lhlo::IotaOp iotaOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto resultMemrefType =
        iotaOp.getOperand().getType().dyn_cast<MemRefType>();
    if (!resultMemrefType) return matchFailure();

    auto resultElementType = resultMemrefType.getElementType();
    if (!resultElementType.isIntOrFloat()) return matchFailure();

    // Construct the indexing maps needed for linalg.generic ops.
    unsigned nloops = resultMemrefType.getRank();
    SmallVector<Attribute, 2> indexingMaps;
    indexingMaps.emplace_back(
        AffineMapAttr::get(rewriter.getMultiDimIdentityMap(nloops)));

    auto loc = iotaOp.getLoc();
    auto linalgOp = rewriter.create<linalg::IndexedGenericOp>(
        loc, ArrayRef<Type>{}, args,
        rewriter.getI64IntegerAttr(0),  // args_in
        rewriter.getI64IntegerAttr(1),  // args_out
        rewriter.getArrayAttr(indexingMaps),
        GetNParallelLoopsAttrs(nloops, rewriter),
        /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

    // Add a block to the region.
    auto* region = &linalgOp.region();
    auto* block = rewriter.createBlock(region, region->end());
    for (unsigned i = 0; i < nloops; ++i) {
      block->addArgument(rewriter.getIndexType());
    }
    block->addArguments(llvm::makeArrayRef(resultElementType));

    rewriter.setInsertionPointToEnd(block);
    Operation* castOp = rewriter.create<IndexCastOp>(
        loc, block->getArgument(iotaOp.iota_dimension().getZExtValue()),
        rewriter.getIntegerType(resultElementType.getIntOrFloatBitWidth()));
    if (resultElementType.isa<FloatType>()) {
      castOp = rewriter.create<SIToFPOp>(loc, castOp->getResult(0),
                                         resultElementType);
    }
    rewriter.create<linalg::YieldOp>(loc, castOp->getResult(0));
    rewriter.eraseOp(iotaOp);
    return matchSuccess();
  }
};

class ConstConverter : public OpConversionPattern<xla_lhlo::ConstOp> {
 public:
  using OpConversionPattern<xla_lhlo::ConstOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_lhlo::ConstOp constOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = constOp.getLoc();
    auto valueAttr = constOp.value().cast<DenseElementsAttr>();
    if (valueAttr.getType().getRank() != 0) return matchFailure();
    auto stdConstOp =
        rewriter.create<mlir::ConstantOp>(loc, valueAttr.getValue({}));
    rewriter.create<mlir::StoreOp>(loc, stdConstOp, constOp.getOperand());
    rewriter.eraseOp(constOp);
    return matchSuccess();
  }
};

class SliceConverter : public OpConversionPattern<xla_lhlo::SliceOp> {
 public:
  using OpConversionPattern<xla_lhlo::SliceOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(
      xla_lhlo::SliceOp sliceOp, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = sliceOp.getLoc();
    auto argType =
        sliceOp.getOperand(0).getType().template dyn_cast<ShapedType>();
    if (!argType || !argType.hasRank()) {
      emitError(loc, "lhlo to linalg conversion expects known-rank args");
      return ConversionPattern::matchFailure();
    }

    SmallVector<Value, 3> ranges;
    for (int i = 0, e = argType.getRank(); i < e; ++i) {
      Value start_index = rewriter.create<ConstantIndexOp>(
          loc, sliceOp.start_indices().getValue<int64_t>(i));
      Value limit_index = rewriter.create<ConstantIndexOp>(
          loc, sliceOp.limit_indices().getValue<int64_t>(i));
      Value stride = rewriter.create<ConstantIndexOp>(
          loc, sliceOp.strides().getValue<int64_t>(i));
      ranges.push_back(rewriter.create<linalg::RangeOp>(loc, start_index,
                                                        limit_index, stride));
    }
    auto linalg_slice =
        rewriter.create<linalg::SliceOp>(loc, sliceOp.getOperand(0), ranges);
    rewriter.create<linalg::CopyOp>(loc, linalg_slice, sliceOp.getOperand(1));
    rewriter.eraseOp(sliceOp);
    return matchSuccess();
  }
};

void populateLHLOToLinalgConversionPattern(MLIRContext* context,
                                           OwningRewritePatternList* patterns) {
  // clang-format off
  patterns->insert<BroadcastInDimConverter,
                   ConstConverter,
                   IotaConverter,
                   PointwiseToLinalgConverter<xla_lhlo::AbsOp>,
                   PointwiseToLinalgConverter<xla_lhlo::AddOp>,
                   PointwiseToLinalgConverter<xla_lhlo::AndOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CeilOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CompareOp>,
                   PointwiseToLinalgConverter<xla_lhlo::ConvertOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CopyOp>,
                   PointwiseToLinalgConverter<xla_lhlo::CosOp>,
                   PointwiseToLinalgConverter<xla_lhlo::DivOp>,
                   PointwiseToLinalgConverter<xla_lhlo::ExpOp>,
                   PointwiseToLinalgConverter<xla_lhlo::MaxOp>,
                   PointwiseToLinalgConverter<xla_lhlo::MinOp>,
                   PointwiseToLinalgConverter<xla_lhlo::MulOp>,
                   PointwiseToLinalgConverter<xla_lhlo::NegOp>,
                   PointwiseToLinalgConverter<xla_lhlo::RemOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SelectOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SignOp>,
                   PointwiseToLinalgConverter<xla_lhlo::SubOp>,
                   PointwiseToLinalgConverter<xla_lhlo::TanhOp>,
                   ScalarPointwiseToStandardConverter<xla_lhlo::AddOp>,
                   SliceConverter
                  >(context);
  // clang-format on
}

void populateHLOToLinalgConversionPattern(MLIRContext* context,
                                          OwningRewritePatternList* patterns) {
  patterns->insert<PointwiseToLinalgConverter<xla_hlo::AbsOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::AddOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::AndOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::CeilOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::CompareOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::CopyOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::ExpOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::MulOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::NegOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::RemOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::SelectOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::SubOp, false>,
                   PointwiseToLinalgConverter<xla_hlo::TanhOp, false>>(context);
}

// Converts LHLO ops to Linalg generic.
// Sample result for xla_lhlo::AddOp.
//
// "xla_lhlo.add"(%arg1, %arg2, %out) :
//      (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//
// will be converted to
//
// #map0 = (d0, d1) -> (d0, d1)
// "linalg.generic"(%arg1, %arg2, %out) ( {
//   ^bb0(%arg4: f32, %arg5: f32):
//     %0 = addf %arg4, %arg5 : f32
//     "linalg.yield"(%0) : (f32) -> ()
//   }) {
//     args_in = 2,
//     args_out = 1,
//     indexing_maps = [#map0, #map0, #map0],
//     iterator_types = ["parallel", "parallel"],
//   } : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
// }
struct LhloLegalizeToLinalg : public FunctionPass<LhloLegalizeToLinalg> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();

    auto func = getFunction();
    populateLHLOToLinalgConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

struct HloLegalizeToLinalg : public FunctionPass<HloLegalizeToLinalg> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();

    auto func = getFunction();
    populateHLOToLinalgConversionPattern(func.getContext(), &patterns);
    if (failed(applyPartialConversion(func, target, patterns, nullptr))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace xla_lhlo {
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeLhloToLinalgPass() {
  return absl::make_unique<LhloLegalizeToLinalg>();
}

static PassRegistration<LhloLegalizeToLinalg> legalize_lhlo_pass(
    "lhlo-legalize-to-linalg", "Legalize from LHLO dialect to Linalg dialect");
}  // namespace xla_lhlo

namespace xla_hlo {
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeHloToLinalgPass() {
  return absl::make_unique<HloLegalizeToLinalg>();
}

static PassRegistration<HloLegalizeToLinalg> legalize_hlo_pass(
    "hlo-legalize-to-linalg", "Legalize from HLO dialect to Linalg dialect");
}  // namespace xla_hlo
}  // namespace mlir
