/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Definitions                                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace TF {
GpuDeviceMetadata GpuDeviceMetadata::get(
    ::mlir::IntegerAttr cc_major,
    ::mlir::IntegerAttr cc_minor,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 2> fields;

  assert(cc_major);
  auto cc_major_id = ::mlir::Identifier::get("cc_major", context);
  fields.emplace_back(cc_major_id, cc_major);

  assert(cc_minor);
  auto cc_minor_id = ::mlir::Identifier::get("cc_minor", context);
  fields.emplace_back(cc_minor_id, cc_minor);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(fields, context);
  return dict.dyn_cast<GpuDeviceMetadata>();
}

bool GpuDeviceMetadata::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto cc_major = derived.get("cc_major");
  if (!cc_major || !(((cc_major.isa<::mlir::IntegerAttr>())) && ((cc_major.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  auto cc_minor = derived.get("cc_minor");
  if (!cc_minor || !(((cc_minor.isa<::mlir::IntegerAttr>())) && ((cc_minor.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return false;

  return derived.size() + num_absent_attrs == 2;
}

::mlir::IntegerAttr GpuDeviceMetadata::cc_major() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto cc_major = derived.get("cc_major");
  assert(cc_major && "attribute not found.");
  assert(cc_major.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return cc_major.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr GpuDeviceMetadata::cc_minor() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto cc_minor = derived.get("cc_minor");
  assert(cc_minor && "attribute not found.");
  assert(cc_minor.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return cc_minor.cast<::mlir::IntegerAttr>();
}
} // namespace TF
} // namespace mlir
