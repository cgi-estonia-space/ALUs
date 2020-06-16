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

#include "tensorflow/compiler/tf2xla/mlir_bridge_pass.h"

#include <string>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_os_ostream.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_graphdef.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// Dumps the MLIR module to disk.
// This require the TF_DUMP_GRAPH_PREFIX to be set to a path that exist (or can
// be created).
static void DumpModule(mlir::ModuleOp module, llvm::StringRef file_prefix) {
  const char* prefix_env = GetDumpDirFromEnvVar();
  if (!prefix_env) {
    return;
  }
  std::string prefix = prefix_env;

  auto* env = tensorflow::Env::Default();
  auto status = env->RecursivelyCreateDir(prefix);
  if (!status.ok()) {
    LOG(WARNING) << "cannot create directory '" + prefix +
                        "': " + status.error_message();
    return;
  }
  prefix += "/" + file_prefix.str();
  if (!tensorflow::Env::Default()->CreateUniqueFileName(&prefix, ".mlir")) {
    LOG(WARNING) << "cannot create unique filename, won't dump MLIR module.";
    return;
  }

  std::unique_ptr<WritableFile> file_writer;
  status = env->NewWritableFile(prefix, &file_writer);
  if (!status.ok()) {
    LOG(WARNING) << "cannot open file '" + prefix +
                        "': " + status.error_message();
    return;
  }

  // Print the module to a string before writing to the file.
  std::string txt_module;
  {
    llvm::raw_string_ostream os(txt_module);
    module.print(os);
  }

  status = file_writer->Append(txt_module);
  if (!status.ok()) {
    LOG(WARNING) << "error writing to file '" + prefix +
                        "': " + status.error_message();
    return;
  }
  (void)file_writer->Close();
  VLOG(1) << "Dumped MLIR module to " << prefix;
}

// This runs the first phase of the "bridge", transforming the graph in a form
// that can be executed with delegation of some computations to an accelerator.
// This builds on the model of XLA where a subset of the graph is encapsulated
// and attached to a "compile" operation, whose result is fed to an "execute"
// operation. The kernel for these operations is responsible to lower the
// encapsulated graph to a particular device.
Status MlirBridgePass::Run(const DeviceSet& device_set,
                           const ConfigProto& config_proto,
                           std::unique_ptr<Graph>* graph,
                           FunctionLibraryDefinition* flib_def,
                           std::vector<std::string>* control_ret_node_names,
                           bool* control_rets_updated) {
  if (!config_proto.experimental().enable_mlir_bridge()) {
    VLOG(1) << "Skipping MLIR Bridge Pass, session flag not enabled";
    return Status::OK();
  }

  VLOG(1) << "Running MLIR Bridge Pass";

  GraphDebugInfo debug_info;
  mlir::MLIRContext context;
  GraphImportConfig import_config;
  import_config.graph_as_function = true;
  import_config.control_outputs = *control_ret_node_names;
  TF_ASSIGN_OR_RETURN(auto module_ref,
                      ConvertGraphToMlir(**graph, debug_info, *flib_def,
                                         import_config, &context));

  AddDevicesToOp(*module_ref, &device_set);

  if (VLOG_IS_ON(1)) DumpModule(*module_ref, "mlir_bridge_before_");

  // Run the bridge now
  TF_RETURN_IF_ERROR(
      mlir::TFTPU::TPUBridge(*module_ref, /*enable_logging=*/VLOG_IS_ON(1)));

  if (VLOG_IS_ON(1)) DumpModule(*module_ref, "mlir_bridge_after_");

  GraphExportConfig export_config;
  export_config.graph_as_function = true;
  absl::flat_hash_set<Node*> control_ret_nodes;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertMlirToGraph(*module_ref, export_config, graph, flib_def,
                         &control_ret_nodes),
      "Error converting MLIR module back to graph");

  control_ret_node_names->clear();
  control_ret_node_names->reserve(control_ret_nodes.size());
  for (const auto* node : control_ret_nodes)
    control_ret_node_names->push_back(node->name());

  *control_rets_updated = true;

  return Status::OK();
}

Status MlirBridgeV1CompatPass::Run(
    const GraphOptimizationPassOptions& options) {
  // Skip function graphs as MlirBridgePass will be used instead.
  if (options.is_function_graph) return Status::OK();

  if (!options.session_options->config.experimental().enable_mlir_bridge()) {
    VLOG(1) << "Skipping MLIR Bridge V1 Compat Pass, session flag not enabled";
    return Status::OK();
  }

  VLOG(1) << "Running MLIR Bridge V1 Compat Pass";

  GraphDebugInfo debug_info;
  mlir::MLIRContext context;
  GraphImportConfig import_config;
  import_config.upgrade_legacy = true;
  TF_ASSIGN_OR_RETURN(
      auto module_ref,
      ConvertGraphToMlir(**options.graph, debug_info, *options.flib_def,
                         import_config, &context));

  AddDevicesToOp(*module_ref, options.device_set);

  if (VLOG_IS_ON(1)) DumpModule(*module_ref, "mlir_bridge_v1_compat_before_");

  // Run the bridge now
  TF_RETURN_IF_ERROR(mlir::TFTPU::TPUBridgeV1Compat(
      *module_ref, /*enable_logging=*/VLOG_IS_ON(1)));

  if (VLOG_IS_ON(1)) DumpModule(*module_ref, "mlir_bridge_v1_compat_after_");

  GraphExportConfig export_config;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      ConvertMlirToGraph(*module_ref, export_config, options.graph,
                         options.flib_def),
      "Error converting MLIR module back to graph");

  return Status::OK();
}

}  // namespace tensorflow
