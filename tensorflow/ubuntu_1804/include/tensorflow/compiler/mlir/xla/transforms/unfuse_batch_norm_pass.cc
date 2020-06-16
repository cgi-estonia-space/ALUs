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

#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla_hlo {

namespace {

struct TestUnfuseBatchNormPass : public FunctionPass<TestUnfuseBatchNormPass> {
  void runOnFunction() override {
    ConversionTarget conversionTarget(getContext());
    OwningRewritePatternList conversionPatterns;

    // Consider the xla_hlo dialect legal for tests.
    conversionTarget.addLegalDialect<XlaHloDialect>();
    conversionTarget.addIllegalOp<xla_hlo::BatchNormInferenceOp>();

    PopulateUnfuseBatchNormPatterns(&getContext(), &conversionPatterns);
    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace xla_hlo
}  // namespace mlir

static mlir::PassRegistration<mlir::xla_hlo::TestUnfuseBatchNormPass> pass(
    "test-xla-unfuse-batch-norm",
    "Test pass for materializing 'broadcast_dimensions' attributes");
