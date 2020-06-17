/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/SymbolTable.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "mlir/Transforms/RegionUtils.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace tf_executor {

namespace {
constexpr llvm::StringRef kNestedModule = "_tpu_v1_compat_outlined";
constexpr llvm::StringRef kOutlinedFuncPrefix = "_tpu_v1_compat_outlined_func";

// Extract the islands containing a TPU cluster computation into an outlined
// function in a nested module. This will allow to run the usual bridge on this
// nested module which exhibit a more friendly "V2-like" structure.
// This is only intended for V1 compatibility mode where the bridge runs without
// feed/fetches on session create/extend.
struct TPUBridgeExecutorIslandOutlining
    : public ModulePass<TPUBridgeExecutorIslandOutlining> {
  void runOnModule() override;
};

void TPUBridgeExecutorIslandOutlining::runOnModule() {
  MLIRContext *ctx = &getContext();

  SymbolTable symbol_table(getModule());
  if (Operation *nested_module = symbol_table.lookup(kNestedModule)) {
    nested_module->emitOpError("unexpected already present outlined module.");
    return signalPassFailure();
  }
  ModuleOp outlined_module = ModuleOp::create(getModule().getLoc());
  outlined_module.setAttrs(getModule().getAttrs());
  outlined_module.setAttr(SymbolTable::getSymbolAttrName(),
                          StringAttr::get(kNestedModule, ctx));
  symbol_table.insert(outlined_module);
  SymbolTable outlined_symbol_table(outlined_module);

  // Find every island that contains a TPUReplicateMetadata node and extract it
  // in a new module to run the V1 bridge there.
  SmallVector<IslandOp, 8> islands_to_outline;
  getModule().walk([&](TF::TPUReplicateMetadataOp replicate_op) {
    auto island_op = cast<IslandOp>(replicate_op.getParentOp());
    if (!island_op || island_op.WrapsSingleOp()) return;
    islands_to_outline.push_back(island_op);
  });
  int prefix_id = 0;
  for (IslandOp island_op : islands_to_outline) {
    // Build the function signature.

    // First the captured values in the island are function arguments
    llvm::SetVector<Value> operands;
    getUsedValuesDefinedAbove(island_op.body(), operands);

    SmallVector<Type, 16> func_operand_types;
    func_operand_types.reserve(operands.size());
    for (Value operand : operands)
      func_operand_types.push_back(operand.getType());

    // Function results are the yield operands
    SmallVector<Type, 16> func_result_types;
    for (Value operand : island_op.GetYield().getOperands())
      func_result_types.push_back(operand.getType());
    FunctionType func_type =
        FunctionType::get(func_operand_types, func_result_types, ctx);

    // Create the outlined function
    SmallString<32> name = kOutlinedFuncPrefix;
    name += llvm::Twine(prefix_id++).str();
    auto outlined_func = OpBuilder(ctx).create<FuncOp>(
        island_op.getLoc(), name, func_type, ArrayRef<NamedAttribute>());
    outlined_symbol_table.insert(outlined_func);

    // We will "steal" the body of the island and replace it with a call to the
    // new function later.
    {
      YieldOp yield_op = island_op.GetYield();
      outlined_func.getBody().takeBody(island_op.body());

      // Replace the yield with a return
      OpBuilder replacer(yield_op);
      island_op.body().push_back(new Block);
      replacer.create<ReturnOp>(yield_op.getLoc(), yield_op.getOperands());
      yield_op.erase();
    }

    // Remap the captured operands in the (former) island block with newly
    // created entry block arguments in the function body.
    {
      Block &entry_block = outlined_func.getBody().front();
      for (Value operand : operands) {
        BlockArgument newArg = entry_block.addArgument(operand.getType());
        replaceAllUsesInRegionWith(operand, newArg, outlined_func.getBody());
      }
    }

    // The function is in place in the nested module, create a call and yield in
    // the original island.
    OpBuilder builder(&island_op.GetBody());
    auto call_op = builder.create<mlir::TF::PartitionedCallOp>(
        island_op.getLoc(), func_result_types, operands.getArrayRef(),
        builder.getSymbolRefAttr(
            kNestedModule, builder.getSymbolRefAttr(outlined_func.getName())),
        /*config=*/builder.getStringAttr(""),
        /*config_proto=*/builder.getStringAttr(""),
        /*executor_type=*/builder.getStringAttr(""));
    SmallVector<Value, 16> yield_operands(call_op.getResults());
    builder.create<YieldOp>(island_op.getLoc(), yield_operands);

    // TODO(aminim): handle transitively referenced function and clone them in
    // the new module.
  }
}

PassRegistration<TPUBridgeExecutorIslandOutlining> tpu_pass(
    "tf-executor-tpu-v1-island-outlining",
    "Outline TPU clusters from island into a nested module, so it can be "
    "processed like a V2 module, intended for V1 compatibility mode");

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>>
CreateTFExecutorTPUV1IslandOutliningPass() {
  return std::make_unique<TPUBridgeExecutorIslandOutlining>();
}

}  // namespace tf_executor
}  // namespace mlir
