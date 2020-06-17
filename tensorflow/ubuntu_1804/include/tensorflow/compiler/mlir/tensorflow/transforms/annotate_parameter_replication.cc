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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Block.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFDevice {

namespace {

constexpr char kReplicationAttr[] = "tf_device.is_same_data_across_replicas";
constexpr char kMirroredVariableIndicesAttr[] = "_mirrored_variable_indices";

// Analyzes the inputs to LaunchFuncOps in the module, and annotates their
// invoked functions whether each input has the same data across replicas.
struct AnnotateParameterReplication
    : public ModulePass<AnnotateParameterReplication> {
  void runOnModule() override;
};

// Returns the first value in the chain of operands, which is not defined by a
// tf.IdentityOp or a tf.ReadVariableOp.
Value SkipIdentityAndReadVariable(Value v) {
  while (auto op = v.getDefiningOp()) {
    if (!(isa<TF::IdentityOp>(op) || isa<TF::ReadVariableOp>(op))) break;
    v = op->getOperand(0);
  }
  return v;
}

void AnnotateParameterReplication::runOnModule() {
  ModuleOp m = getModule();
  OpBuilder builder(m.getContext());
  m.walk([&](tf_device::LaunchFuncOp launch_func) {
    auto replicate = launch_func.getParentOfType<tf_device::ReplicateOp>();
    if (!replicate) return;
    auto mirrored_variable_indices_attr =
        replicate.getAttrOfType<ArrayAttr>(kMirroredVariableIndicesAttr);
    llvm::SmallDenseSet<int64_t, 8> mirrored_replicate_args;
    if (mirrored_variable_indices_attr) {
      for (const auto& mirrored_index : mirrored_variable_indices_attr) {
        mirrored_replicate_args.insert(
            mirrored_index.cast<IntegerAttr>().getInt());
      }
    }
    auto func = llvm::cast<FuncOp>(m.lookupSymbol(launch_func.func()));
    for (auto entry : llvm::enumerate(launch_func.getOperands())) {
      auto operand = SkipIdentityAndReadVariable(entry.value());
      auto block_arg = operand.dyn_cast<BlockArgument>();
      if (block_arg && block_arg.getOwner() == &replicate.GetBody()) {
        // Only mirrored args of ReplicateOp can be annotated.
        if (mirrored_replicate_args.count(block_arg.getArgNumber()) == 0) {
          continue;
        }
      } else if (!operand.getParentRegion()->isProperAncestor(
                     &replicate.body())) {
        // Not a replication-invariant operand.
        continue;
      }
      func.setArgAttr(entry.index(), kReplicationAttr,
                      builder.getBoolAttr(true));
    }
  });
}

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>> CreateAnnotateParameterReplicationPass() {
  return std::make_unique<AnnotateParameterReplication>();
}

static PassRegistration<AnnotateParameterReplication> pass(
    "tf-annotate-parameter-replication",
    "Annotate whether a LaunchFuncOp's parameters have the same data across "
    "replicas.");

}  // namespace TFDevice
}  // namespace mlir
