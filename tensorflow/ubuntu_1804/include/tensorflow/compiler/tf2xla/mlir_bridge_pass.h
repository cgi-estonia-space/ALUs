/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_MLIR_BRIDGE_PASS_H_
#define TENSORFLOW_COMPILER_TF2XLA_MLIR_BRIDGE_PASS_H_

#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// This pass uses MLIR to implement all the conversion steps to target XLA from
// a TensorFlow Function Graph. It is meant to expose a very limited set of
// functionalities during the bring-up of MLIR-based bridge.
class MlirBridgePass : public FunctionOptimizationPass {
 public:
  Status Run(const DeviceSet& device_set, const ConfigProto& config_proto,
             std::unique_ptr<Graph>* graph, FunctionLibraryDefinition* flib_def,
             std::vector<std::string>* control_ret_node_names,
             bool* control_rets_updated) override;
};

// This pass uses MLIR to implement all the conversion steps to target XLA from
// a TensorFlow V1 Graph. It is meant to expose a very limited set of
// functionalities during the bring-up of MLIR-based bridge.
class MlirBridgeV1CompatPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_MLIR_BRIDGE_PASS_H_
