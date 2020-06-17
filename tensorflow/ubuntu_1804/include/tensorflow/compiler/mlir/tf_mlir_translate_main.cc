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

#include <unordered_set>

#include "absl/strings/str_split.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/Support/FileUtilities.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "mlir/Support/ToolUtilities.h"  // TF:llvm-project
#include "mlir/Support/TranslateClParser.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/core/platform/init_main.h"

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                                 llvm::cl::desc("<input file>"),
                                                 llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> output_filename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each chunk "
                   "independently"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> import_saved_model(
    "savedmodel-to-mlir",
    llvm::cl::desc("Import a saved model to its MLIR representation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> import_saved_model_v1(
    "savedmodel-v1-to-mlir",
    llvm::cl::desc("Import a saved model V1 to its MLIR representation"),
    llvm::cl::value_desc("dir"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> saved_model_tags(
    "tf-savedmodel-tags",
    llvm::cl::desc("Tags used to indicate which MetaGraphDef to import, "
                   "separated by ','"),
    llvm::cl::init("serve"));

// NOLINTNEXTLINE
static llvm::cl::opt<std::string> saved_model_exported_names(
    "tf-savedmodel-exported-names",
    llvm::cl::desc("Names to export from SavedModel, separated by ','. Empty "
                   "(the default) means export all."),
    llvm::cl::init(""));

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);

  // Add flags for all the registered translations.
  llvm::cl::opt<const mlir::TranslateFunction*, false, mlir::TranslationParser>
      requested_translation("", llvm::cl::desc("Translation to perform"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "TF MLIR translation driver\n");

  if (!import_saved_model && !import_saved_model_v1 && !requested_translation) {
    llvm::errs() << "error: need to specify one translation to perform\n";
    return 1;
  } else if (import_saved_model && import_saved_model_v1 &&
             requested_translation) {
    llvm::errs()
        << "error: cannot specify more than one translation to perform\n";
    return 1;
  }

  std::string error_message;
  auto output = mlir::openOutputFile(output_filename, &error_message);
  if (!output) {
    llvm::errs() << error_message << "\n";
    return 1;
  }

  if (import_saved_model) {
    std::unordered_set<std::string> tags =
        absl::StrSplit(saved_model_tags, ',');
    std::vector<std::string> exported_names =
        absl::StrSplit(saved_model_exported_names, ',', absl::SkipEmpty());
    mlir::MLIRContext context;

    auto module = tensorflow::SavedModelToMlirImport(
        input_filename, tags, absl::Span<std::string>(exported_names),
        &context);
    if (!module) return 1;

    module->print(output->os());
  } else if (import_saved_model_v1) {
    std::unordered_set<std::string> tags =
        absl::StrSplit(saved_model_tags, ',');
    mlir::MLIRContext context;

    auto module =
        tensorflow::SavedModelV1ToMlirImport(input_filename, tags, &context);
    if (!module) return 1;

    module->print(output->os());
  } else {
    auto input = mlir::openInputFile(input_filename, &error_message);

    if (!input) {
      llvm::errs() << error_message << "\n";
      return 1;
    }

    // Processes the memory buffer with a new MLIRContext.
    auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                             llvm::raw_ostream& os) {
      llvm::SourceMgr sourceMgr;
      sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());
      mlir::MLIRContext context;
      mlir::SourceMgrDiagnosticHandler diagnostic_handler(sourceMgr, &context);
      return (*requested_translation)(sourceMgr, os, &context);
    };

    if (splitInputFile) {
      if (failed(mlir::splitAndProcessBuffer(std::move(input), processBuffer,
                                             output->os())))
        return 1;
    } else {
      if (failed(processBuffer(std::move(input), output->os()))) return 1;
    }
  }

  output->keep();
  return 0;
}
