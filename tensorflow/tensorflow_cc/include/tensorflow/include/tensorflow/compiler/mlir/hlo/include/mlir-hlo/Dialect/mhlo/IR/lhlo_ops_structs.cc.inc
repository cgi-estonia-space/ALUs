/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Struct Utility Definitions                                                 *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace mhlo {
ChannelHandle ChannelHandle::get(
    ::mlir::IntegerAttr handle,
    ::mlir::IntegerAttr type,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 2> fields;

  assert(handle);
  auto handle_id = ::mlir::Identifier::get("handle", context);
  fields.emplace_back(handle_id, handle);

  assert(type);
  auto type_id = ::mlir::Identifier::get("type", context);
  fields.emplace_back(type_id, type);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(fields, context);
  return dict.dyn_cast<ChannelHandle>();
}

bool ChannelHandle::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto handle = derived.get("handle");
  if (!handle || !(((handle.isa<::mlir::IntegerAttr>())) && ((handle.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  auto type = derived.get("type");
  if (!type || !(((type.isa<::mlir::IntegerAttr>())) && ((type.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  return derived.size() + num_absent_attrs == 2;
}

::mlir::IntegerAttr ChannelHandle::handle() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto handle = derived.get("handle");
  assert(handle && "attribute not found.");
  assert(handle.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return handle.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr ChannelHandle::type() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto type = derived.get("type");
  assert(type && "attribute not found.");
  assert(type.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return type.cast<::mlir::IntegerAttr>();
}
} // namespace mhlo
} // namespace mlir
namespace mlir {
namespace mhlo {
ConvDimensionNumbers ConvDimensionNumbers::get(
    ::mlir::IntegerAttr input_batch_dimension,
    ::mlir::IntegerAttr input_feature_dimension,
    ::mlir::DenseIntElementsAttr input_spatial_dimensions,
    ::mlir::IntegerAttr kernel_input_feature_dimension,
    ::mlir::IntegerAttr kernel_output_feature_dimension,
    ::mlir::DenseIntElementsAttr kernel_spatial_dimensions,
    ::mlir::IntegerAttr output_batch_dimension,
    ::mlir::IntegerAttr output_feature_dimension,
    ::mlir::DenseIntElementsAttr output_spatial_dimensions,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 9> fields;

  assert(input_batch_dimension);
  auto input_batch_dimension_id = ::mlir::Identifier::get("input_batch_dimension", context);
  fields.emplace_back(input_batch_dimension_id, input_batch_dimension);

  assert(input_feature_dimension);
  auto input_feature_dimension_id = ::mlir::Identifier::get("input_feature_dimension", context);
  fields.emplace_back(input_feature_dimension_id, input_feature_dimension);

  assert(input_spatial_dimensions);
  auto input_spatial_dimensions_id = ::mlir::Identifier::get("input_spatial_dimensions", context);
  fields.emplace_back(input_spatial_dimensions_id, input_spatial_dimensions);

  assert(kernel_input_feature_dimension);
  auto kernel_input_feature_dimension_id = ::mlir::Identifier::get("kernel_input_feature_dimension", context);
  fields.emplace_back(kernel_input_feature_dimension_id, kernel_input_feature_dimension);

  assert(kernel_output_feature_dimension);
  auto kernel_output_feature_dimension_id = ::mlir::Identifier::get("kernel_output_feature_dimension", context);
  fields.emplace_back(kernel_output_feature_dimension_id, kernel_output_feature_dimension);

  assert(kernel_spatial_dimensions);
  auto kernel_spatial_dimensions_id = ::mlir::Identifier::get("kernel_spatial_dimensions", context);
  fields.emplace_back(kernel_spatial_dimensions_id, kernel_spatial_dimensions);

  assert(output_batch_dimension);
  auto output_batch_dimension_id = ::mlir::Identifier::get("output_batch_dimension", context);
  fields.emplace_back(output_batch_dimension_id, output_batch_dimension);

  assert(output_feature_dimension);
  auto output_feature_dimension_id = ::mlir::Identifier::get("output_feature_dimension", context);
  fields.emplace_back(output_feature_dimension_id, output_feature_dimension);

  assert(output_spatial_dimensions);
  auto output_spatial_dimensions_id = ::mlir::Identifier::get("output_spatial_dimensions", context);
  fields.emplace_back(output_spatial_dimensions_id, output_spatial_dimensions);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(fields, context);
  return dict.dyn_cast<ConvDimensionNumbers>();
}

bool ConvDimensionNumbers::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto input_batch_dimension = derived.get("input_batch_dimension");
  if (!input_batch_dimension || !(((input_batch_dimension.isa<::mlir::IntegerAttr>())) && ((input_batch_dimension.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  auto input_feature_dimension = derived.get("input_feature_dimension");
  if (!input_feature_dimension || !(((input_feature_dimension.isa<::mlir::IntegerAttr>())) && ((input_feature_dimension.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  auto input_spatial_dimensions = derived.get("input_spatial_dimensions");
  if (!input_spatial_dimensions || !(((input_spatial_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((input_spatial_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto kernel_input_feature_dimension = derived.get("kernel_input_feature_dimension");
  if (!kernel_input_feature_dimension || !(((kernel_input_feature_dimension.isa<::mlir::IntegerAttr>())) && ((kernel_input_feature_dimension.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  auto kernel_output_feature_dimension = derived.get("kernel_output_feature_dimension");
  if (!kernel_output_feature_dimension || !(((kernel_output_feature_dimension.isa<::mlir::IntegerAttr>())) && ((kernel_output_feature_dimension.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  auto kernel_spatial_dimensions = derived.get("kernel_spatial_dimensions");
  if (!kernel_spatial_dimensions || !(((kernel_spatial_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((kernel_spatial_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto output_batch_dimension = derived.get("output_batch_dimension");
  if (!output_batch_dimension || !(((output_batch_dimension.isa<::mlir::IntegerAttr>())) && ((output_batch_dimension.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  auto output_feature_dimension = derived.get("output_feature_dimension");
  if (!output_feature_dimension || !(((output_feature_dimension.isa<::mlir::IntegerAttr>())) && ((output_feature_dimension.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  auto output_spatial_dimensions = derived.get("output_spatial_dimensions");
  if (!output_spatial_dimensions || !(((output_spatial_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((output_spatial_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  return derived.size() + num_absent_attrs == 9;
}

::mlir::IntegerAttr ConvDimensionNumbers::input_batch_dimension() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto input_batch_dimension = derived.get("input_batch_dimension");
  assert(input_batch_dimension && "attribute not found.");
  assert(input_batch_dimension.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return input_batch_dimension.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr ConvDimensionNumbers::input_feature_dimension() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto input_feature_dimension = derived.get("input_feature_dimension");
  assert(input_feature_dimension && "attribute not found.");
  assert(input_feature_dimension.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return input_feature_dimension.cast<::mlir::IntegerAttr>();
}

::mlir::DenseIntElementsAttr ConvDimensionNumbers::input_spatial_dimensions() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto input_spatial_dimensions = derived.get("input_spatial_dimensions");
  assert(input_spatial_dimensions && "attribute not found.");
  assert(input_spatial_dimensions.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return input_spatial_dimensions.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::IntegerAttr ConvDimensionNumbers::kernel_input_feature_dimension() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto kernel_input_feature_dimension = derived.get("kernel_input_feature_dimension");
  assert(kernel_input_feature_dimension && "attribute not found.");
  assert(kernel_input_feature_dimension.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return kernel_input_feature_dimension.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr ConvDimensionNumbers::kernel_output_feature_dimension() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto kernel_output_feature_dimension = derived.get("kernel_output_feature_dimension");
  assert(kernel_output_feature_dimension && "attribute not found.");
  assert(kernel_output_feature_dimension.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return kernel_output_feature_dimension.cast<::mlir::IntegerAttr>();
}

::mlir::DenseIntElementsAttr ConvDimensionNumbers::kernel_spatial_dimensions() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto kernel_spatial_dimensions = derived.get("kernel_spatial_dimensions");
  assert(kernel_spatial_dimensions && "attribute not found.");
  assert(kernel_spatial_dimensions.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return kernel_spatial_dimensions.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::IntegerAttr ConvDimensionNumbers::output_batch_dimension() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto output_batch_dimension = derived.get("output_batch_dimension");
  assert(output_batch_dimension && "attribute not found.");
  assert(output_batch_dimension.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return output_batch_dimension.cast<::mlir::IntegerAttr>();
}

::mlir::IntegerAttr ConvDimensionNumbers::output_feature_dimension() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto output_feature_dimension = derived.get("output_feature_dimension");
  assert(output_feature_dimension && "attribute not found.");
  assert(output_feature_dimension.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return output_feature_dimension.cast<::mlir::IntegerAttr>();
}

::mlir::DenseIntElementsAttr ConvDimensionNumbers::output_spatial_dimensions() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto output_spatial_dimensions = derived.get("output_spatial_dimensions");
  assert(output_spatial_dimensions && "attribute not found.");
  assert(output_spatial_dimensions.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return output_spatial_dimensions.cast<::mlir::DenseIntElementsAttr>();
}
} // namespace mhlo
} // namespace mlir
namespace mlir {
namespace mhlo {
DotDimensionNumbers DotDimensionNumbers::get(
    ::mlir::DenseIntElementsAttr lhs_batching_dimensions,
    ::mlir::DenseIntElementsAttr rhs_batching_dimensions,
    ::mlir::DenseIntElementsAttr lhs_contracting_dimensions,
    ::mlir::DenseIntElementsAttr rhs_contracting_dimensions,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 4> fields;

  assert(lhs_batching_dimensions);
  auto lhs_batching_dimensions_id = ::mlir::Identifier::get("lhs_batching_dimensions", context);
  fields.emplace_back(lhs_batching_dimensions_id, lhs_batching_dimensions);

  assert(rhs_batching_dimensions);
  auto rhs_batching_dimensions_id = ::mlir::Identifier::get("rhs_batching_dimensions", context);
  fields.emplace_back(rhs_batching_dimensions_id, rhs_batching_dimensions);

  assert(lhs_contracting_dimensions);
  auto lhs_contracting_dimensions_id = ::mlir::Identifier::get("lhs_contracting_dimensions", context);
  fields.emplace_back(lhs_contracting_dimensions_id, lhs_contracting_dimensions);

  assert(rhs_contracting_dimensions);
  auto rhs_contracting_dimensions_id = ::mlir::Identifier::get("rhs_contracting_dimensions", context);
  fields.emplace_back(rhs_contracting_dimensions_id, rhs_contracting_dimensions);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(fields, context);
  return dict.dyn_cast<DotDimensionNumbers>();
}

bool DotDimensionNumbers::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto lhs_batching_dimensions = derived.get("lhs_batching_dimensions");
  if (!lhs_batching_dimensions || !(((lhs_batching_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((lhs_batching_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto rhs_batching_dimensions = derived.get("rhs_batching_dimensions");
  if (!rhs_batching_dimensions || !(((rhs_batching_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((rhs_batching_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto lhs_contracting_dimensions = derived.get("lhs_contracting_dimensions");
  if (!lhs_contracting_dimensions || !(((lhs_contracting_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((lhs_contracting_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto rhs_contracting_dimensions = derived.get("rhs_contracting_dimensions");
  if (!rhs_contracting_dimensions || !(((rhs_contracting_dimensions.isa<::mlir::DenseIntElementsAttr>())) && ((rhs_contracting_dimensions.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  return derived.size() + num_absent_attrs == 4;
}

::mlir::DenseIntElementsAttr DotDimensionNumbers::lhs_batching_dimensions() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto lhs_batching_dimensions = derived.get("lhs_batching_dimensions");
  assert(lhs_batching_dimensions && "attribute not found.");
  assert(lhs_batching_dimensions.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return lhs_batching_dimensions.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DotDimensionNumbers::rhs_batching_dimensions() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto rhs_batching_dimensions = derived.get("rhs_batching_dimensions");
  assert(rhs_batching_dimensions && "attribute not found.");
  assert(rhs_batching_dimensions.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return rhs_batching_dimensions.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DotDimensionNumbers::lhs_contracting_dimensions() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto lhs_contracting_dimensions = derived.get("lhs_contracting_dimensions");
  assert(lhs_contracting_dimensions && "attribute not found.");
  assert(lhs_contracting_dimensions.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return lhs_contracting_dimensions.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr DotDimensionNumbers::rhs_contracting_dimensions() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto rhs_contracting_dimensions = derived.get("rhs_contracting_dimensions");
  assert(rhs_contracting_dimensions && "attribute not found.");
  assert(rhs_contracting_dimensions.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return rhs_contracting_dimensions.cast<::mlir::DenseIntElementsAttr>();
}
} // namespace mhlo
} // namespace mlir
namespace mlir {
namespace mhlo {
GatherDimensionNumbers GatherDimensionNumbers::get(
    ::mlir::DenseIntElementsAttr offset_dims,
    ::mlir::DenseIntElementsAttr collapsed_slice_dims,
    ::mlir::DenseIntElementsAttr start_index_map,
    ::mlir::IntegerAttr index_vector_dim,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 4> fields;

  assert(offset_dims);
  auto offset_dims_id = ::mlir::Identifier::get("offset_dims", context);
  fields.emplace_back(offset_dims_id, offset_dims);

  assert(collapsed_slice_dims);
  auto collapsed_slice_dims_id = ::mlir::Identifier::get("collapsed_slice_dims", context);
  fields.emplace_back(collapsed_slice_dims_id, collapsed_slice_dims);

  assert(start_index_map);
  auto start_index_map_id = ::mlir::Identifier::get("start_index_map", context);
  fields.emplace_back(start_index_map_id, start_index_map);

  assert(index_vector_dim);
  auto index_vector_dim_id = ::mlir::Identifier::get("index_vector_dim", context);
  fields.emplace_back(index_vector_dim_id, index_vector_dim);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(fields, context);
  return dict.dyn_cast<GatherDimensionNumbers>();
}

bool GatherDimensionNumbers::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto offset_dims = derived.get("offset_dims");
  if (!offset_dims || !(((offset_dims.isa<::mlir::DenseIntElementsAttr>())) && ((offset_dims.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto collapsed_slice_dims = derived.get("collapsed_slice_dims");
  if (!collapsed_slice_dims || !(((collapsed_slice_dims.isa<::mlir::DenseIntElementsAttr>())) && ((collapsed_slice_dims.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto start_index_map = derived.get("start_index_map");
  if (!start_index_map || !(((start_index_map.isa<::mlir::DenseIntElementsAttr>())) && ((start_index_map.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto index_vector_dim = derived.get("index_vector_dim");
  if (!index_vector_dim || !(((index_vector_dim.isa<::mlir::IntegerAttr>())) && ((index_vector_dim.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  return derived.size() + num_absent_attrs == 4;
}

::mlir::DenseIntElementsAttr GatherDimensionNumbers::offset_dims() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto offset_dims = derived.get("offset_dims");
  assert(offset_dims && "attribute not found.");
  assert(offset_dims.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return offset_dims.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr GatherDimensionNumbers::collapsed_slice_dims() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto collapsed_slice_dims = derived.get("collapsed_slice_dims");
  assert(collapsed_slice_dims && "attribute not found.");
  assert(collapsed_slice_dims.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return collapsed_slice_dims.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr GatherDimensionNumbers::start_index_map() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto start_index_map = derived.get("start_index_map");
  assert(start_index_map && "attribute not found.");
  assert(start_index_map.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return start_index_map.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::IntegerAttr GatherDimensionNumbers::index_vector_dim() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto index_vector_dim = derived.get("index_vector_dim");
  assert(index_vector_dim && "attribute not found.");
  assert(index_vector_dim.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return index_vector_dim.cast<::mlir::IntegerAttr>();
}
} // namespace mhlo
} // namespace mlir
namespace mlir {
namespace mhlo {
ScatterDimensionNumbers ScatterDimensionNumbers::get(
    ::mlir::DenseIntElementsAttr update_window_dims,
    ::mlir::DenseIntElementsAttr inserted_window_dims,
    ::mlir::DenseIntElementsAttr scatter_dims_to_operand_dims,
    ::mlir::IntegerAttr index_vector_dim,
    ::mlir::MLIRContext* context) {
  ::llvm::SmallVector<::mlir::NamedAttribute, 4> fields;

  assert(update_window_dims);
  auto update_window_dims_id = ::mlir::Identifier::get("update_window_dims", context);
  fields.emplace_back(update_window_dims_id, update_window_dims);

  assert(inserted_window_dims);
  auto inserted_window_dims_id = ::mlir::Identifier::get("inserted_window_dims", context);
  fields.emplace_back(inserted_window_dims_id, inserted_window_dims);

  assert(scatter_dims_to_operand_dims);
  auto scatter_dims_to_operand_dims_id = ::mlir::Identifier::get("scatter_dims_to_operand_dims", context);
  fields.emplace_back(scatter_dims_to_operand_dims_id, scatter_dims_to_operand_dims);

  assert(index_vector_dim);
  auto index_vector_dim_id = ::mlir::Identifier::get("index_vector_dim", context);
  fields.emplace_back(index_vector_dim_id, index_vector_dim);

  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(fields, context);
  return dict.dyn_cast<ScatterDimensionNumbers>();
}

bool ScatterDimensionNumbers::classof(::mlir::Attribute attr) {
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;

  auto update_window_dims = derived.get("update_window_dims");
  if (!update_window_dims || !(((update_window_dims.isa<::mlir::DenseIntElementsAttr>())) && ((update_window_dims.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto inserted_window_dims = derived.get("inserted_window_dims");
  if (!inserted_window_dims || !(((inserted_window_dims.isa<::mlir::DenseIntElementsAttr>())) && ((inserted_window_dims.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto scatter_dims_to_operand_dims = derived.get("scatter_dims_to_operand_dims");
  if (!scatter_dims_to_operand_dims || !(((scatter_dims_to_operand_dims.isa<::mlir::DenseIntElementsAttr>())) && ((scatter_dims_to_operand_dims.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(64)))))
    return false;

  auto index_vector_dim = derived.get("index_vector_dim");
  if (!index_vector_dim || !(((index_vector_dim.isa<::mlir::IntegerAttr>())) && ((index_vector_dim.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return false;

  return derived.size() + num_absent_attrs == 4;
}

::mlir::DenseIntElementsAttr ScatterDimensionNumbers::update_window_dims() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto update_window_dims = derived.get("update_window_dims");
  assert(update_window_dims && "attribute not found.");
  assert(update_window_dims.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return update_window_dims.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ScatterDimensionNumbers::inserted_window_dims() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto inserted_window_dims = derived.get("inserted_window_dims");
  assert(inserted_window_dims && "attribute not found.");
  assert(inserted_window_dims.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return inserted_window_dims.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr ScatterDimensionNumbers::scatter_dims_to_operand_dims() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto scatter_dims_to_operand_dims = derived.get("scatter_dims_to_operand_dims");
  assert(scatter_dims_to_operand_dims && "attribute not found.");
  assert(scatter_dims_to_operand_dims.isa<::mlir::DenseIntElementsAttr>() && "incorrect Attribute type found.");
  return scatter_dims_to_operand_dims.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::IntegerAttr ScatterDimensionNumbers::index_vector_dim() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto index_vector_dim = derived.get("index_vector_dim");
  assert(index_vector_dim && "attribute not found.");
  assert(index_vector_dim.isa<::mlir::IntegerAttr>() && "incorrect Attribute type found.");
  return index_vector_dim.cast<::mlir::IntegerAttr>();
}
} // namespace mhlo
} // namespace mlir
