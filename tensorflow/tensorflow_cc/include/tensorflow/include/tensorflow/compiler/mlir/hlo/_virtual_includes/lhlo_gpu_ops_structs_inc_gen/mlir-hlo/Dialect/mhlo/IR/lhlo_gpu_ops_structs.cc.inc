/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Definitions                                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace lmhlo_gpu {
ConvolutionBackendConfig ConvolutionBackendConfig::get(
    ::mlir::IntegerAttr algorithm,
    ::mlir::BoolAttr tensor_ops_enabled,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 2> fields;

  assert(algorithm);
  auto algorithm_id = ::mlir::Identifier::get("algorithm", context);
  fields.emplace_back(algorithm_id, algorithm);

  assert(tensor_ops_enabled);
  auto tensor_ops_enabled_id = ::mlir::Identifier::get("tensor_ops_enabled", context);
  fields.emplace_back(tensor_ops_enabled_id, tensor_ops_enabled);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(fields, context);
  return dict.dyn_cast<ConvolutionBackendConfig>();
}

bool ConvolutionBackendConfig::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto algorithm = derived.get("algorithm");
  if (!algorithm || !(((algorithm.isa<::mlir::IntegerAttr>())) && ((algorithm.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  auto tensor_ops_enabled = derived.get("tensor_ops_enabled");
  if (!tensor_ops_enabled || !((tensor_ops_enabled.isa<::mlir::BoolAttr>())))
    return false;

  return derived.size() + num_absent_attrs == 2;
}

::mlir::IntegerAttr ConvolutionBackendConfig::algorithm() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto algorithm = derived.get("algorithm");
  assert(algorithm && "attribute not found.");
  assert(algorithm.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return algorithm.cast<::mlir::IntegerAttr>();
}

::mlir::BoolAttr ConvolutionBackendConfig::tensor_ops_enabled() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto tensor_ops_enabled = derived.get("tensor_ops_enabled");
  assert(tensor_ops_enabled && "attribute not found.");
  assert(tensor_ops_enabled.isa<::mlir::BoolAttr>() && "incorrect Attribute type found.");
  return tensor_ops_enabled.cast<::mlir::BoolAttr>();
}
} // namespace lmhlo_gpu
} // namespace mlir
