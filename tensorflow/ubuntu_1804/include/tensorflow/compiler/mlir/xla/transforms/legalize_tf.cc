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

// This file implements logic for lowering TensorFlow dialect to XLA dialect.

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/Dialect/Traits.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Diagnostics.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Matchers.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/convert_op_folder.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_utils.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/kernels/conv_grad_shape_utils.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace xla_hlo {
namespace {

class LegalizeTF : public FunctionPass<LegalizeTF> {
 public:
  LegalizeTF() = default;
  LegalizeTF(const LegalizeTF &) {}
  explicit LegalizeTF(bool allow_partial_conversion) {
    allow_partial_conversion_ = allow_partial_conversion;
  }

  /// Performs the lowering to XLA dialect.
  void runOnFunction() override;

 private:
  Option<bool> allow_partial_conversion_{
      *this, "allow-partial-conversion",
      llvm::cl::desc("Allow operations that can't be legalized."),
      llvm::cl::init(false)};
};

/// Returns if the given TF data format string is the default format.
static bool IsDefaultDataFormat(StringRef format) { return format == "NHWC"; }

/// Returns the feature dimension for the given format and input type.
static size_t GetFeatureDimension(StringAttr format,
                                  RankedTensorType inputType) {
  return IsDefaultDataFormat(format.getValue()) ? inputType.getRank() - 1 : 1;
}

// Gets all integer values from the given attribute and push them to `values`.
void GetI64ArrayAttrValues(Attribute attr, SmallVectorImpl<int64_t> *values) {
  auto array_attr = attr.cast<ArrayAttr>();
  values->reserve(array_attr.getValue().size());
  for (Attribute val : array_attr.getValue())
    values->push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
}

// Returns 1D 64-bit dense elements attribute with the given values.
static DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                               Builder *builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

// Converts an ArrayAttr to a 1D 64-bit dense elements attribute.
static DenseIntElementsAttr GetI64ElementsAttr(ArrayAttr attr) {
  RankedTensorType ty =
      RankedTensorType::get(static_cast<int64_t>(attr.size()),
                            IntegerType::get(64, attr.getContext()));
  return DenseIntElementsAttr::get(ty, attr.getValue());
}

// Returns 1D 32-bit dense elements attribute with the given values.
static DenseIntElementsAttr GetI32ElementsAttr(ArrayRef<int32_t> values,
                                               Builder *builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int32_t>(values.size())}, builder->getIntegerType(32));
  return DenseIntElementsAttr::get(ty, values);
}

// Returns the corresponding type that should be used for performing sum
// accumulation over the given input type.
Type GetSumAccumulationType(Type input_type) {
  MLIRContext *ctx = input_type.getContext();
  if (input_type.isBF16() || input_type.isF16()) return FloatType::getF32(ctx);
  if (input_type.isInteger(8) || input_type.isInteger(16))
    return IntegerType::get(32, ctx);
  return input_type;
}

// Returns axis in HLO format from TF elements attr with exactly one element
// containing axis in the TensorFlow format. TensorFlow format supports negative
// indexing unlike HLO.
static IntegerAttr GetHLOAxisFromTFAxis(ElementsAttr attr, int64_t rank,
                                        Builder *b) {
  SmallVector<uint64_t, 1> index(attr.getType().getRank(), 0);
  int64_t axis = attr.getValue<IntegerAttr>(index).getInt();
  if (axis < 0) {
    axis += rank;
  }
  return b->getI64IntegerAttr(axis);
}

static IntegerAttr GetHLOAxisFromTFAxis(IntegerAttr attr, int64_t rank,
                                        Builder *b) {
  int64_t axis = attr.getInt();
  if (axis < 0) {
    axis += rank;
  }
  return b->getI64IntegerAttr(axis);
}

// If `value` is an IntegerAttr, returns the integer value for the HLO axis
// corresponding to the tensorflow axis. In particular, the tensorflow axis can
// be negative, in which case, the corresponding HLO axis is
// (axis + rank-of-the-tensor).
static llvm::Optional<int64_t> GetIntegerHLOAxisFromTFAxis(Value value,
                                                           int64_t rank) {
  DenseIntElementsAttr attrs;
  if (!matchPattern(value, m_Constant(&attrs)) ||
      attrs.getType().getRank() != 0) {
    return llvm::None;
  }
  int64_t axis = attrs.getValue<IntegerAttr>({}).getInt();
  return axis < 0 ? axis + rank : axis;
}

/// Returns a `ConvertOp` that casts the elements to a i64 type while retaining
/// the shape of the input value.
static ConvertOp CastValueToI64(Location loc, Value value,
                                PatternRewriter *rewriter) {
  return rewriter->create<ConvertOp>(loc, value, rewriter->getIntegerType(64));
}

// Returns size of dimension at the specified index, if ranked tensor.
// Otherwise, returns -1.
//
// Aborts if the type is ranked but doesn't have the dimension.
int64_t GetDimSize(Type ty, int64_t index) {
  RankedTensorType ranked_ty = ty.dyn_cast<RankedTensorType>();
  if (!ranked_ty) return -1;

  return ranked_ty.getDimSize(index);
}

template <typename T>
tensorflow::TensorShape ToTensorShape(llvm::ArrayRef<T> sizes) {
  return tensorflow::TensorShape(
      llvm::SmallVector<tensorflow::int64, 4>(sizes.begin(), sizes.end()));
}

// Returns minimal value for the given int or float element type.
static ConstOp GetMinValueForType(Type ty, Location loc,
                                  PatternRewriter *rewriter) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);

  DenseElementsAttr attr;
  if (auto float_ty = ty.dyn_cast_or_null<FloatType>()) {
    APFloat neg_inf =
        APFloat::getInf(float_ty.getFloatSemantics(), /*negative=*/true);
    attr = DenseElementsAttr::get(scalar_ty, neg_inf);
  } else {
    auto int_ty = ty.cast<IntegerType>();
    APInt min_val = APInt::getSignedMinValue(int_ty.getWidth());
    attr = DenseElementsAttr::get(scalar_ty, min_val);
  }
  return rewriter->create<ConstOp>(loc, attr);
}

// Returns maximal value for the given int or float element type.
static ConstOp GetMaxValueForType(Type ty, Location loc,
                                  PatternRewriter *rewriter) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);

  DenseElementsAttr attr;
  if (auto float_ty = ty.dyn_cast_or_null<FloatType>()) {
    APFloat pos_inf =
        APFloat::getInf(float_ty.getFloatSemantics(), /*negative=*/false);
    attr = DenseElementsAttr::get(scalar_ty, pos_inf);
  } else {
    auto int_ty = ty.cast<IntegerType>();
    APInt max_val = APInt::getSignedMaxValue(int_ty.getWidth());
    attr = DenseElementsAttr::get(scalar_ty, max_val);
  }
  return rewriter->create<ConstOp>(loc, attr);
}

// Returns int or float scalar DenseElementsAttr attribute with the given
// element type and the value.
static ConstOp GetScalarConstOfType(Type ty, Location loc, int64_t raw_value,
                                    PatternRewriter *rewriter) {
  return rewriter->create<ConstOp>(loc, xla::GetScalarOfType(ty, raw_value));
}

// Builds body for reduce op by using the using the template binary op as the
// reducer op.
template <typename Op>
static void BuildReduceBody(Type element_type, Region *body,
                            OpBuilder *builder) {
  OpBuilder::InsertionGuard guard(*builder);
  Block *block = builder->createBlock(body);

  // Block arguments are scalars of the given element type.
  Type type = RankedTensorType::get(/*shape=*/{}, element_type);
  block->addArguments({type, type});

  Location loc = body->getLoc();
  auto reducer =
      builder->create<Op>(loc, block->getArgument(0), block->getArgument(1),
                          /*broadcast_dimensions=*/nullptr);
  builder->create<ReturnOp>(loc, reducer.getResult());
}

// Builds region taking two arguments and returning second argument as the
// result. Corresponds to the function f(x, y) = y.
// Used in Scatter op's computation to update specific elements.
static void BuildBinaryAssignmentRegion(Type element_type, Region *region,
                                        OpBuilder *builder) {}

// Builds a set of operations for applying reduction on the input value. A
// tf.sum op is created and will be legalized to tfl ops automatically.
static Value ApplyReduction(Location loc, Value input,
                            DenseIntElementsAttr reduce_dims,
                            OpBuilder *builder) {
  auto reduce_dims_op = builder->create<ConstOp>(loc, reduce_dims);
  return builder->create<TF::SumOp>(loc, input, reduce_dims_op,
                                    builder->getBoolAttr(false));
}

// Creates a xla_hlo.rng_uniform op with `builder` to generate `num_elements`
// 32-bit integer numbers in the range of [`lower_limit`, `upper_limit`).
static xla_hlo::RngUniformOp CreateRngUniform32(Location loc, int num_elements,
                                                int lower_limit,
                                                int upper_limit,
                                                OpBuilder *builder) {
  auto i32_type = builder->getIntegerType(32);
  auto key_type = RankedTensorType::get({num_elements}, i32_type);
  auto shape_tensor = builder->create<xla_hlo::ConstOp>(
      loc, GetI64ElementsAttr({num_elements}, builder));

  auto lower = builder->create<xla_hlo::ConstOp>(
      loc, builder->getI32IntegerAttr(lower_limit));
  auto upper = builder->create<xla_hlo::ConstOp>(
      loc, builder->getI32IntegerAttr(upper_limit));

  return builder->create<xla_hlo::RngUniformOp>(loc, key_type, lower, upper,
                                                shape_tensor);
}

using WhileBodyFnType = llvm::function_ref<void(
    Location loc, Value iteration, ArrayRef<Value> old_values,
    SmallVectorImpl<Value> *new_values, OpBuilder *builder)>;

// Creates a xla_hlo.while op with `builder` to loop `num_interations` times,
// each time calling the given `body_fn` on a set of values to generate a new
// set of values. Returns the final set of values via `final_values`. The
// initial set of values is passed in via `init_values`.
//
// This effectively does:
//
// ```c++
// SmallVector<Values, 4> old_values = init_values;
// SmallVector<Values, 4> new_values;
// for (int i = 0; i < num_iterations; ++i) {
//   body_fn(old_values, &new_values, ...);
//   old_values = new_values;
// }
// ```
//
// Under the hood an induction variable is prepended to values to control the
// number of iterations, but that is transparent to `body_fn`, which does not
// need to care about that.
static void CreateWhile32(Location loc, int num_iterations,
                          WhileBodyFnType body_fn, ArrayRef<Value> init_values,
                          SmallVectorImpl<Value> *final_values,
                          OpBuilder *builder) {
  int value_count = init_values.size() + 1;

  // Prepend a loop induction variable to the initial values.
  SmallVector<Value, 2> init_values_with_loop_iv;
  init_values_with_loop_iv.reserve(value_count);
  // The initial value for the loop induction variable is 0.
  init_values_with_loop_iv.push_back(
      builder->create<xla_hlo::ConstOp>(loc, builder->getI32IntegerAttr(0)));
  init_values_with_loop_iv.append(init_values.begin(), init_values.end());

  // Prepare the initial tuple for the while op.
  auto init_tuple =
      builder->create<xla_hlo::TupleOp>(loc, init_values_with_loop_iv);
  auto tuple_type = init_tuple.getType();

  // Create the while op.
  auto while_op = builder->create<xla_hlo::WhileOp>(loc, init_tuple);

  {
    OpBuilder::InsertionGuard guard(*builder);

    // Build up the only block in the condition region. It should take one
    // argument of the loop's tuple type.
    Region &condition = while_op.cond();
    Block *block = builder->createBlock(&condition);
    BlockArgument arg = block->addArgument(tuple_type);

    // Get the loop induction variable and compare it against the upper limit.
    auto loop_iv = builder->create<GetTupleElementOp>(loc, arg, 0);
    auto upper_limit = builder->create<xla_hlo::ConstOp>(
        loc, builder->getI32IntegerAttr(num_iterations));
    StringAttr compare_direction = StringAttr::get("LT", builder->getContext());
    Value compare = builder->create<xla_hlo::CompareOp>(
        loc, loop_iv, upper_limit,
        /*broadcast_dimensions=*/nullptr, compare_direction);

    builder->create<xla_hlo::ReturnOp>(loc, compare);
  }

  {
    OpBuilder::InsertionGuard guard(*builder);

    // Build up the only block in the body region. It should take one
    // argument of the loop's tuple type.
    Region &body = while_op.body();
    Block *block = builder->createBlock(&body);
    BlockArgument arg = block->addArgument(tuple_type);

    SmallVector<Value, 4> old_values;  // From the previous iteration
    SmallVector<Value, 4> new_values;  // Generated by this iteration
    old_values.reserve(value_count);
    new_values.reserve(value_count);

    // Unpack the tuple value from the last iteration.
    for (int i = 0; i < value_count; ++i)
      old_values.push_back(builder->create<GetTupleElementOp>(loc, arg, i));

    // Feed all values excluding the loop induction variable to body_fn.
    body_fn(loc, old_values[0], llvm::makeArrayRef(old_values).drop_front(),
            &new_values, builder);

    // Increment the loop induction variable by one.
    auto one =
        builder->create<xla_hlo::ConstOp>(loc, builder->getI32IntegerAttr(1));
    auto no_broadcast_dims = GetI64ElementsAttr({}, builder);
    auto plus_one = builder->create<xla_hlo::AddOp>(loc, old_values[0], one,
                                                    no_broadcast_dims);
    // Prepend with the updated loop induction variable.
    new_values.insert(new_values.begin(), plus_one);

    Value updated_tuple = builder->create<xla_hlo::TupleOp>(loc, new_values);

    builder->create<xla_hlo::ReturnOp>(loc, updated_tuple);
  }

  final_values->reserve(init_values.size());
  for (int i = 0, e = init_values.size(); i < e; ++i)
    final_values->push_back(
        builder->create<GetTupleElementOp>(loc, while_op, i + 1));
}

//===----------------------------------------------------------------------===//
// BatchNorm op utilities.
//===----------------------------------------------------------------------===//

static IntegerAttr getFeatureDimensionAttr(Builder &b, StringAttr format,
                                           Value input) {
  return b.getI64IntegerAttr(
      GetFeatureDimension(format, input.getType().cast<RankedTensorType>()));
}

//===----------------------------------------------------------------------===//
// Bias op utilities.
//===----------------------------------------------------------------------===//

// Return a 1D DenseIntElementsAttr for the feature dimension of a BiasAdd.
// Requires input to have ranked tensor.
static DenseIntElementsAttr getBiasFeatureDimension(Builder &b,
                                                    StringAttr format,
                                                    Value input) {
  auto inputType = input.getType().cast<RankedTensorType>();
  size_t featureDim = GetFeatureDimension(format, inputType);
  RankedTensorType type = RankedTensorType::get(1, b.getIntegerType(64));
  return DenseIntElementsAttr::get(type, featureDim);
}

//===----------------------------------------------------------------------===//
// MatMul op utilities.
//===----------------------------------------------------------------------===//

// If the 'transpose' attribute is true returns ElementsAttr to transpose 2D
// matrix. Otherwise, returns ElementsAttr for identity transpose.
static DenseIntElementsAttr Get2DTransposePerm(BoolAttr transpose, Builder *b) {
  if (transpose.getValue()) return GetI64ElementsAttr({1, 0}, b);
  return GetI64ElementsAttr({0, 1}, b);
}

//===----------------------------------------------------------------------===//
// Pad op utilities.
//===----------------------------------------------------------------------===//

// Slices input attribute of rank two and returns the specified column.
//
// Always returns 64 bit integer attribute regardless of bitwidth of the input
// attribute.
static DenseIntElementsAttr SliceDenseIntElementsAttrColumn2D(
    ElementsAttr input, int column) {
  auto int_attr = input.cast<DenseIntElementsAttr>();
  auto shaped_type = int_attr.getType();
  auto shape = shaped_type.getShape();

  if (shape.size() != 2) return DenseIntElementsAttr();

  llvm::SmallVector<int64_t, 4> values;
  values.reserve(shaped_type.getNumElements() / shape[1]);

  for (auto it : llvm::enumerate(int_attr.getIntValues())) {
    if (it.index() % shape[1] == column) {
      values.push_back(it.value().getSExtValue());
    }
  }

  auto element_type = IntegerType::get(64, input.getContext());
  return DenseIntElementsAttr::get(
      RankedTensorType::get({shape[0]}, element_type), values);
}

// Returns interior padding to use in HLO Pad op based on the TensorFlow padding
// in TensorFlow PadV2 op.
static DenseIntElementsAttr GetInteriorPadding(ElementsAttr tf_padding) {
  auto length = tf_padding.getType().getShape()[0];
  auto element_type = IntegerType::get(64, tf_padding.getContext());
  return DenseIntElementsAttr::get<int64_t>(
      RankedTensorType::get({length}, element_type), 0);
}

//===----------------------------------------------------------------------===//
// Binary op utilities.
//===----------------------------------------------------------------------===//

// Returns whether the two values are guaranteed to be broadcastable to the
// same shape, this broadcasts size 1 tensors up to any rank. Dynamic dimensions
// must be broadcasted with a size 1 tensor or another dynamic dimension.
// Returns false on rankless.
static bool AreBroadcastCompatible(Value x, Value y) {
  auto x_rankless = x.getType().dyn_cast<RankedTensorType>();
  auto y_rankless = y.getType().dyn_cast<RankedTensorType>();
  if (!x_rankless || !y_rankless) {
    return false;
  }

  // Check that the shapes can be broadcasted.
  auto shape_x = x_rankless.getShape();
  auto shape_y = y_rankless.getShape();

  int rank_diff = shape_x.size() - shape_y.size();
  int offset_x = rank_diff > 0 ? rank_diff : 0;
  int offset_y = rank_diff < 0 ? -rank_diff : 0;
  for (int i = 0, s = std::min(shape_x.size(), shape_y.size()); i < s; i++) {
    int index_x = i + offset_x;
    int index_y = i + offset_y;
    if ((shape_x[index_x] == -1 && shape_y[index_y] != 1) ||
        (shape_y[index_y] == -1 && shape_x[index_x] != 1)) {
      return false;
    }
  }

  return true;
}

// Return a new TensorType the same rank and dimensions as the input with an
// updated element type.
static Type ChangeTensorElementType(Builder *b, Type tensor_type,
                                    Type element_type) {
  RankedTensorType ranked_type = tensor_type.dyn_cast<RankedTensorType>();
  if (ranked_type) {
    return RankedTensorType::get(ranked_type.getShape(), element_type);
  }

  return UnrankedTensorType::get(element_type);
}

//===----------------------------------------------------------------------===//
// Softmax op utilities.
//===----------------------------------------------------------------------===//

// Returns a 1-d i64 elements attribute populated with numbers from start to
// end, excluding.
static DenseIntElementsAttr GetI64ElementsAttrForSeq(int start, int end,
                                                     Builder *builder) {
  int size = end - start;

  SmallVector<int64_t, 4> vals;
  vals.resize(size);
  std::iota(vals.begin(), vals.end(), start);

  TensorType ty = RankedTensorType::get({size}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, vals);
}

// Returns the type to use for accumulating the given type.
static Type GetAccumulationType(Type ty) {
  // Upcast 16 bit sum reductions to 32 bit to reduce the precision loss from
  // repeated floating point additions.
  return (ty.isF16() || ty.isBF16()) ? FloatType::getF32(ty.getContext()) : ty;
}

//===----------------------------------------------------------------------===//
// ArgMax/ArgMin op utilities.
//===----------------------------------------------------------------------===//

static void BuildArgMinMaxReductionBody(Type input_element_type,
                                        Type index_element_type,
                                        StringRef direction, Region *body,
                                        OpBuilder *builder) {
  OpBuilder::InsertionGuard insertion_point_gurad(*builder);

  Type input_type = RankedTensorType::get(/*shape=*/{}, input_element_type);
  Type index_type = RankedTensorType::get(/*shape=*/{}, index_element_type);
  Block *block = builder->createBlock(body);
  block->addArguments({input_type, index_type, input_type, index_type});

  Location loc = body->getLoc();
  StringAttr compare_direction =
      StringAttr::get(direction, builder->getContext());
  Value compare = builder->create<CompareOp>(
      loc, block->getArgument(0), block->getArgument(2),
      /*broadcast_dimensions=*/nullptr, compare_direction);

  Value selected_input = builder->create<SelectOp>(
      loc, input_type, compare, block->getArgument(0), block->getArgument(2));
  Value selected_index = builder->create<SelectOp>(
      loc, index_type, compare, block->getArgument(1), block->getArgument(3));

  Value return_values[] = {selected_input, selected_index};
  builder->create<ReturnOp>(loc, return_values);
}

//===----------------------------------------------------------------------===//
// PartitionedCall op utilities.
//===----------------------------------------------------------------------===//

// Verify that the arguments to be passed into the function are the same types
// as the function paramter types.
static bool ArgTypesMatchCallee(mlir::Operation *op, OperandRange args,
                                SymbolRefAttr func) {
  auto module = op->getParentOfType<ModuleOp>();
  auto function =
      dyn_cast_or_null<FuncOp>(SymbolTable::lookupSymbolIn(module, func));
  FunctionType function_ty = function.getType();

  for (auto arg_in : llvm::zip(args, function_ty.getInputs())) {
    if (std::get<0>(arg_in).getType() != std::get<1>(arg_in)) {
      // Argument type and input type mismatch.
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Slice op utilities.
//===----------------------------------------------------------------------===//

static bool CanBeTranslatedToDynamicSlice(Value input, Value start_indices,
                                          DenseIntElementsAttr slice_sizes) {
  auto input_ty = input.getType().dyn_cast<RankedTensorType>();
  int64_t input_rank = input_ty.getRank();
  ArrayRef<int64_t> input_shape = input_ty.getShape();
  DenseIntElementsAttr constant_start_indices;
  if (!matchPattern(start_indices, m_Constant(&constant_start_indices))) {
    for (int64_t i = 0; i < input_rank; ++i) {
      int64_t slice_size = slice_sizes.getValue<IntegerAttr>(i).getInt();
      int64_t input_size = input_shape[i];
      if (slice_size < 0 || (input_size != -1 && slice_size > input_size)) {
        return false;
      }
    }
    return true;
  }

  for (int64_t i = 0; i < input_rank; ++i) {
    int64_t input_size = input_shape[i];
    int64_t start_index =
        constant_start_indices.getValue<IntegerAttr>(i).getInt();
    int64_t slice_size = slice_sizes.getValue<IntegerAttr>(i).getInt();
    if (start_index < 0) return false;
    // A slice_size of -1 means "all elements from start_index to the end".
    // We can't support this semantics for dynamic shapes.
    if (slice_size == -1) {
      if (input_size == -1) return false;
      slice_size = input_size - start_index;
    }
    if (input_size != -1 && start_index + slice_size > input_size) {
      return false;
    }
  }

  return true;
}

// TF slice size can be -1, which represents all elements from start_index to
// the end. HLO slice size can't be -1. As such, we need to translate TF slice
// size -1 to HLO slice size.
static DenseIntElementsAttr TFSliceSizes2HLOSliceSizes(
    Value input, Value start_indices, DenseIntElementsAttr slice_sizes,
    Builder *builder) {
  DenseIntElementsAttr constant_start_indices;
  if (!matchPattern(start_indices, m_Constant(&constant_start_indices))) {
    return xla::ConvertElementsAttr(slice_sizes, builder->getIntegerType(64))
        .cast<DenseIntElementsAttr>();
  }

  auto input_ty = input.getType().dyn_cast<RankedTensorType>();
  int64_t input_rank = input_ty.getRank();
  ArrayRef<int64_t> input_shape = input_ty.getShape();
  SmallVector<int64_t, 4> normalized_sizes;

  for (int64_t i = 0; i < input_rank; ++i) {
    int64_t input_size = input_shape[i];
    int64_t start_index =
        constant_start_indices.getValue<IntegerAttr>(i).getInt();
    int64_t slice_size = slice_sizes.getValue<IntegerAttr>(i).getInt();
    normalized_sizes.push_back(slice_size == -1 ? input_size - start_index
                                                : slice_size);
  }

  return GetI64ElementsAttr(normalized_sizes, builder);
}

//===----------------------------------------------------------------------===//
// Sort op utilities.
//===----------------------------------------------------------------------===//

// Builds the region `body` for xla_hlo.sort's comparator: for each type in
// `element_types`, create two block arguments, one for lhs and one for rhs, and
// generates xla_hlo.compare op to compare them with the given `direction`.
//
// Note that this right now only does comparision on the first pair of block
// arguments.
static void BuildSortComparisonBody(llvm::ArrayRef<Type> element_types,
                                    StringRef direction, Region *body,
                                    OpBuilder *builder) {
  OpBuilder::InsertionGuard insertion_point_gurad(*builder);

  Block *block = builder->createBlock(body);
  // Add two arguments for each element type.
  for (Type element_type : element_types) {
    TensorType tensor_type = RankedTensorType::get({}, element_type);
    block->addArguments({tensor_type, tensor_type});
  }

  Location loc = body->getLoc();
  StringAttr compare_direction =
      StringAttr::get(direction, builder->getContext());
  Value compare = builder->create<xla_hlo::CompareOp>(
      loc, block->getArgument(0), block->getArgument(1),
      /*broadcast_dimensions=*/nullptr, compare_direction);

  builder->create<xla_hlo::ReturnOp>(loc, compare);
}

//===----------------------------------------------------------------------===//
// Op converters.
//===----------------------------------------------------------------------===//

NamedAttribute GetConvDimensionNumbersAttr(
    ArrayRef<int64_t> spatial_dim_indices, tensorflow::TensorFormat format,
    Builder *builder) {
  int64_t num_spatial_dims = spatial_dim_indices.size();
  int64_t num_dims = num_spatial_dims + 2;

  IntegerAttr batch_dim =
      builder->getI64IntegerAttr(GetTensorBatchDimIndex(num_dims, format));
  IntegerAttr feature_dim =
      builder->getI64IntegerAttr(GetTensorFeatureDimIndex(num_dims, format));
  DenseIntElementsAttr spatial_dims =
      GetI64ElementsAttr(spatial_dim_indices, builder);

  // Filters data_format is always HWIO so input channels dimension is after
  // all spatial dimensions.
  IntegerAttr kernel_input_feature_dim =
      builder->getI64IntegerAttr(num_spatial_dims);
  IntegerAttr kernel_output_feature_dim =
      builder->getI64IntegerAttr(num_spatial_dims + 1);
  DenseIntElementsAttr kernel_spatial_dimensions =
      GetI64ElementsAttrForSeq(0, num_spatial_dims, builder);

  return builder->getNamedAttr(
      "dimension_numbers",
      ConvDimensionNumbers::get(
          batch_dim, feature_dim, spatial_dims, kernel_input_feature_dim,
          kernel_output_feature_dim, kernel_spatial_dimensions, batch_dim,
          feature_dim, spatial_dims, builder->getContext()));
}

// Converts the TensorFlow conv op in template to the generic HLO conv op by
// converting TensorFlow op attributes to HLO op attributes.
//
// Sample result for Conv2D:
//
//   %conv = "xla_hlo.conv"(%input, %filter) {
//     strides = [1, 2],
//     paddings = [[1, 0], [1, 1]],
//     ...
//   }
//
// This pattern is not defined using declarative rewrite rules as computation of
// the paddings attribute anyway requires multiple source op attributes and
// result op attributes. Defining it as declarative rewrite rule will introduce
// some duplication in the C++ helper methods.
template <typename OpT, int num_spatial_dims>
class ConvertConv : public OpRewritePattern<OpT> {
 public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(OpT op,
                                     PatternRewriter &rewriter) const override {
    tensorflow::TensorFormat format;
    std::string data_format = op.data_format().str();
    if (!FormatFromString(data_format, &format)) return Pattern::matchFailure();

    auto input_ty = op.input().getType().template dyn_cast<RankedTensorType>();
    auto filter_ty =
        op.filter().getType().template dyn_cast<RankedTensorType>();
    auto result_ty = op.getType().template dyn_cast<RankedTensorType>();

    // Input, filter and the result needs to have static shape for calculation
    // of HLO paddings and feature group count attributes.
    for (RankedTensorType ty : {input_ty, filter_ty, result_ty}) {
      if (!ty || !ty.hasStaticShape()) return Pattern::matchFailure();
    }

    int num_dims = num_spatial_dims + 2;
    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.padding().str(), &padding).ok())
      return Pattern::matchFailure();

    auto get_int = [](Attribute attr) {
      return attr.template cast<IntegerAttr>().getInt();
    };

    SmallVector<int64_t, 4> spatial_dim_indices;
    SmallVector<int64_t, 4> rhs_dilations;
    SmallVector<int64_t, 4> window_strides;
    SmallVector<int64_t, 8> paddings;

    ArrayRef<Attribute> dilations = op.dilations().getValue();
    ArrayRef<Attribute> strides = op.strides().getValue();
    ArrayRef<Attribute> explicit_paddings;
    if (padding == tensorflow::Padding::EXPLICIT) {
      // EXPLICIT padding mode and the associated attribute is limited to
      // Conv2D. So, fetch attribute by identifier instead of the
      // op.explicit_paddings() attribute getter.
      explicit_paddings =
          op.template getAttrOfType<ArrayAttr>("explicit_paddings").getValue();
    }

    for (int i = 0; i < num_spatial_dims; ++i) {
      int64_t dim = GetTensorSpatialDimIndex(num_dims, format, i);
      spatial_dim_indices.push_back(dim);

      int64_t stride = get_int(strides[dim]);
      int64_t dilation = get_int(dilations[dim]);
      window_strides.push_back(stride);
      rhs_dilations.push_back(dilation);

      int64_t pad_low, pad_high;
      if (padding == tensorflow::Padding::EXPLICIT) {
        pad_low = get_int(explicit_paddings[2 * dim]);
        pad_high = get_int(explicit_paddings[2 * dim + 1]);
      } else {
        tensorflow::int64 output_size;
        tensorflow::int64 pad_low_int64;
        tensorflow::int64 pad_high_int64;
        tensorflow::Status status = tensorflow::GetWindowedOutputSizeVerboseV2(
            input_ty.getDimSize(dim), filter_ty.getDimSize(i), dilation, stride,
            padding, &output_size, &pad_low_int64, &pad_high_int64);
        if (!status.ok()) return Pattern::matchFailure();
        pad_low = pad_low_int64;
        pad_high = pad_high_int64;
      }
      paddings.push_back(pad_low);
      paddings.push_back(pad_high);
    }

    auto rhs_dilations_attr = rewriter.getNamedAttr(
        "rhs_dilation", GetI64ElementsAttr(rhs_dilations, &rewriter));

    auto window_strides_attr = rewriter.getNamedAttr(
        "window_strides", GetI64ElementsAttr(window_strides, &rewriter));

    auto dimension_numbers_attr =
        GetConvDimensionNumbersAttr(spatial_dim_indices, format, &rewriter);

    int64_t input_channels =
        GetDimSize(input_ty, GetTensorFeatureDimIndex(num_dims, format));
    // Filters data_format is always HWIO so input channels dimension is after
    // all spatial dimensions.
    int64_t filter_channels = GetDimSize(filter_ty, num_spatial_dims);
    // TensorFlow convolution op verifies that the number of input channels is
    // divisible by the number of filter channels.
    int64_t feature_group_count = input_channels / filter_channels;
    auto feature_group_count_attr = rewriter.getNamedAttr(
        "feature_group_count", rewriter.getI64IntegerAttr(feature_group_count));

    auto batch_group_count_attr = rewriter.getNamedAttr(
        "batch_group_count", rewriter.getI64IntegerAttr(1));

    RankedTensorType paddings_ty = RankedTensorType::get(
        {num_spatial_dims, 2}, rewriter.getIntegerType(64));
    auto paddings_attr = rewriter.getNamedAttr(
        "padding", DenseElementsAttr::get<int64_t>(paddings_ty, paddings));

    SmallVector<Value, 2> operands(op.getOperands());
    NamedAttribute attrs[] = {rhs_dilations_attr,     window_strides_attr,
                              dimension_numbers_attr, feature_group_count_attr,
                              batch_group_count_attr, paddings_attr};
    rewriter.replaceOpWithNewOp<ConvOp>(op, op.getType(), operands,
                                        llvm::makeArrayRef(attrs));
    return Pattern::matchSuccess();
  }
};

using ConvertConv2D = ConvertConv<TF::Conv2DOp, /*num_spatial_dims=*/2>;

// Converts BF16 FloorDiv op to have casting operators on either end as BF16
// division can result in strange behavior.
//
//      floordiv = cast(floordiv(cast(left), cast(right))))
//
//   %left_cast = cast(%left)
//   %right_cast = cast(%right)
//   %div = div(%left, %left)
//   %floored = floor(%div)
//   %floored_cast = cast(%floored)
//
// Required to manually specify the intermediate types.
class ConvertBF16FloorDivOp : public OpRewritePattern<TF::FloorDivOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::FloorDivOp op,
                                     PatternRewriter &rewriter) const override {
    auto l = op.x();
    auto r = op.y();
    auto element_type = getElementTypeOrSelf(l.getType());
    if (!element_type.isBF16()) return matchFailure();

    auto out_type = op.z().getType().cast<TensorType>();

    l = rewriter.create<ConvertOp>(op.getLoc(), l, rewriter.getF32Type());
    r = rewriter.create<ConvertOp>(op.getLoc(), r, rewriter.getF32Type());

    auto intermediate = rewriter.create<TF::FloorDivOp>(
        op.getLoc(),
        ChangeTensorElementType(&rewriter, out_type, rewriter.getF32Type()), l,
        r);

    auto floor_op =
        rewriter.create<ConvertOp>(op.getLoc(), out_type, intermediate);
    rewriter.replaceOp(op, floor_op.getResult());
    return Pattern::matchSuccess();
  }
};

// Converts TensorFlow EinsumOp to either HLO EinsumOp or UnaryEinsumOp
// depending on arity of the op.
class ConvertEinsumOp : public OpRewritePattern<TF::EinsumOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::EinsumOp op,
                                     PatternRewriter &rewriter) const override {
    StringAttr equation = op.getAttrOfType<StringAttr>("equation");
    if (op.N() == 1) {
      rewriter.replaceOpWithNewOp<UnaryEinsumOp>(
          op, op.getType(), *op.inputs().begin(), equation);
    } else if (op.N() == 2) {
      ValueRange inputs = op.inputs();
      rewriter.replaceOpWithNewOp<EinsumOp>(op, op.getType(), inputs[0],
                                            inputs[1], equation);
    } else {
      // TensorFlow EinsumOp verifies that the number of operands are at most
      // two.
      return Pattern::matchFailure();
    }
    return Pattern::matchSuccess();
  }
};

// The base class to convert TensorFlow FusedBatchNormGrad*Op to HLO
// BatchNormGradOp for training and a sequence of binary ops for inference.
// TODO(b/145536565): move to legalize_tf_patterns.td if it applies.
template <typename FusedBatchNormGradOpT>
class ConvertFusedBatchNormGradBase
    : public OpRewritePattern<FusedBatchNormGradOpT> {
 public:
  using OpRewritePattern<FusedBatchNormGradOpT>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(FusedBatchNormGradOpT op,
                                     PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value grad = op.y_backprop();
    Value act = op.x();
    Value scale = op.scale();
    Value mean = op.reserve_space_1();
    Value var = op.reserve_space_2();

    // TODO(b/141785544): Update this to not require static shapes.
    // activation shape needs to be static to convert negative indices in
    // TensorFlow to absolute indices required by HLO.
    RankedTensorType act_type =
        act.getType().template dyn_cast<RankedTensorType>();
    if (!act_type) return Pattern::matchFailure();
    Type act_ele_type = act_type.getElementType();
    // To support mixed precision, the statistics type, which maybe more
    // precise than the input types, are used for this op.
    Type kernel_type =
        scale.getType().template cast<TensorType>().getElementType();
    grad = rewriter.create<ConvertOp>(loc, grad, kernel_type);
    act = rewriter.create<ConvertOp>(loc, act, kernel_type);

    auto feature_dim_attr =
        getFeatureDimensionAttr(rewriter, op.data_formatAttr(), act);
    auto feature_dim = feature_dim_attr.getValue().getSExtValue();

    // Gets the result values.
    Value x_backprop, scale_backprop, offset_backprop;
    if (op.is_training()) {  // training
      // TODO(b/145536565): handle GPU logic separately.
      // Infers the output type with the converted `act`.
      Type feature_type = RankedTensorType::get(
          {GetDimSize(act_type, feature_dim)}, kernel_type);
      Type result_type = TupleType::get(
          {act.getType(), feature_type, feature_type}, rewriter.getContext());

      auto training_op = rewriter.create<BatchNormGradOp>(
          loc, result_type, act, scale, mean, var, grad, op.epsilon(),
          feature_dim_attr.getValue());

      x_backprop =
          rewriter.create<GetTupleElementOp>(loc, training_op.getResult(), 0);

      scale_backprop =
          rewriter.create<GetTupleElementOp>(loc, training_op.getResult(), 1);

      offset_backprop =
          rewriter.create<GetTupleElementOp>(loc, training_op.getResult(), 2);
    } else {  // inference
      SmallVector<int64_t, 4> non_feature_dims;
      for (int64_t i = 0; i < act_type.getRank(); ++i) {
        if (i == feature_dim) continue;
        non_feature_dims.push_back(i);
      }
      auto reduce_dims = GetI64ElementsAttr(non_feature_dims, &rewriter);
      auto broadcast_dims = GetI64ElementsAttr({feature_dim}, &rewriter);
      auto no_broadcast_dims = GetI64ElementsAttr({}, &rewriter);

      // scratch1 = rsqrt(var + epsilon)
      RankedTensorType scalar_float = RankedTensorType::get({}, kernel_type);
      auto epsilon = rewriter.create<ConstOp>(
          loc, DenseFPElementsAttr::get(scalar_float, {op.epsilon()}));
      auto add_op = rewriter.create<AddOp>(loc, var, epsilon.getResult(),
                                           no_broadcast_dims);
      Value scratch1 = rewriter.create<RsqrtOp>(loc, add_op);

      // scratch2 = sum(y_backprop * (x - mean))
      auto sub_op = rewriter.create<SubOp>(loc, act, mean, broadcast_dims);
      auto weighted_grad =
          rewriter.create<MulOp>(loc, grad, sub_op, no_broadcast_dims);
      Value scratch2 =
          ApplyReduction(loc, weighted_grad, reduce_dims, &rewriter);

      // x_backprop = y_backprop * (scale * scratch1)
      auto scaled_grad =
          rewriter.create<MulOp>(loc, op.scale(), scratch1, no_broadcast_dims);
      x_backprop =
          rewriter.create<MulOp>(loc, grad, scaled_grad, broadcast_dims);

      // scale_backprop = scratch2 * scratch1
      scale_backprop =
          rewriter.create<MulOp>(loc, scratch1, scratch2, no_broadcast_dims);

      // offset_backprop = sum(y_backprop)
      offset_backprop = ApplyReduction(loc, grad, reduce_dims, &rewriter);
    }

    x_backprop = rewriter.create<ConvertOp>(loc, x_backprop, act_ele_type);
    // It doesn't matter what values we provide for the last 2 results.
    rewriter.replaceOp(op,
                       {/*x_backprop=*/x_backprop,
                        /*scale_backprop=*/scale_backprop,
                        /*offset_backprop=*/offset_backprop, op.x(), op.x()});
    return Pattern::matchSuccess();
  }
};

using ConvertFusedBatchNormGradOp =
    ConvertFusedBatchNormGradBase<TF::FusedBatchNormGradOp>;
using ConvertFusedBatchNormGradV2Op =
    ConvertFusedBatchNormGradBase<TF::FusedBatchNormGradV2Op>;
using ConvertFusedBatchNormGradV3Op =
    ConvertFusedBatchNormGradBase<TF::FusedBatchNormGradV3Op>;

// Converts TensorFlow FusedBatchNormV3Op to either HLO BatchNormTrainingOp or
// HLO BatchNormInferenceOp, depending on the value of the 'is_training'
// parameter.
class ConvertFusedBatchNormV3Op
    : public OpRewritePattern<TF::FusedBatchNormV3Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::FusedBatchNormV3Op op,
                                     PatternRewriter &rewriter) const override {
    auto feature_dim =
        getFeatureDimensionAttr(rewriter, op.data_formatAttr(), op.x());

    auto input_type_tensor = op.x().getType().dyn_cast<TensorType>();
    auto input_element_type = input_type_tensor.getElementType();

    auto scale_type_tensor = op.scale().getType().dyn_cast<TensorType>();
    auto scale_element_type = scale_type_tensor.getElementType();
    // In the training case, dimensions of input tensors must be static.
    if (op.is_training() && ((!input_type_tensor.hasStaticShape()) ||
                             (!scale_type_tensor.hasStaticShape()))) {
      return matchFailure();
    }

    // TODO(b/69928690): Support mixed precision in the XLA batch
    // normalization operators. As a workaround, create a new x with the same
    // element type as scale (which may be more precise than the input type).
    Value bn_train_input = rewriter.create<xla_hlo::ConvertOp>(
        op.getLoc(), op.x(), scale_element_type);
    TensorType bn_train_input_type_tensor =
        bn_train_input.getType().cast<TensorType>();

    if (op.is_training()) {
      // Training case.
      auto operand_shape = bn_train_input_type_tensor.getShape();
      // The mean and variance are each 1 dimensional arrays the size of the
      // feature dimension, with the same element type as the operand (x).
      // This shape must be constructed manually because the mean and variance
      // inputs are empty in the training case.
      Type mean_var_type = RankedTensorType::get(
          {operand_shape[feature_dim.getInt()]}, scale_element_type);
      // Op result type is a tuple of 3 values: output with same shape as input;
      // batch_mean, and batch_var.
      SmallVector<Type, 3> operand_types = {bn_train_input_type_tensor,
                                            mean_var_type, mean_var_type};
      Type result_type = TupleType::get(operand_types, rewriter.getContext());

      auto bn_train_op = rewriter.create<xla_hlo::BatchNormTrainingOp>(
          op.getLoc(), result_type, bn_train_input, op.scale(), op.offset(),
          op.epsilon(), feature_dim.getValue());
      // HLO op outputs a tuple of tensors. Extract those results.
      auto bn_train_op_result = bn_train_op.getResult();
      Value y_out = rewriter.create<xla_hlo::GetTupleElementOp>(
          op.getLoc(), bn_train_op_result, 0);
      Value batch_mean = rewriter.create<xla_hlo::GetTupleElementOp>(
          op.getLoc(), bn_train_op_result, 1);
      Value batch_variance = rewriter.create<xla_hlo::GetTupleElementOp>(
          op.getLoc(), bn_train_op_result, 2);

      // Apply Bessel's correction on the variance.
      int total_input_size = bn_train_input_type_tensor.getNumElements();
      int total_scale_size = scale_type_tensor.getNumElements();
      int sample_size = total_input_size / total_scale_size;
      int sample_size_minus_one = std::max(1, sample_size - 1);
      double factor = static_cast<double>(sample_size) /
                      static_cast<double>(sample_size_minus_one);
      auto factor_const_op = rewriter.create<xla_hlo::ConstOp>(
          op.getLoc(), rewriter.getFloatAttr(scale_element_type, factor));

      auto corrected_variance = rewriter.create<xla_hlo::MulOp>(
          op.getLoc(), batch_variance.getType(), batch_variance,
          factor_const_op, /*DenseIntElementsAttr=*/DenseIntElementsAttr());

      // Convert back to input type to stay aligned with expected output type
      // for TF op.
      y_out = rewriter.create<xla_hlo::ConvertOp>(op.getLoc(), y_out,
                                                  input_element_type);

      // TF FusedBatchNormV3 op expects 5 outputs. Outputs 3 and 4 are
      // currently marked as "reserved spaces 1 and 2". They are used to
      // pass the per-batch mean and variance to the gradiant. Here we
      // maintain the same behavior by setting them to the mean and variance
      // calculated by BatchNormTraining. Output 5 is unused; it doesn't
      // matter what we pass there.
      rewriter.replaceOp(op, {y_out, /*batch_mean=*/batch_mean,
                              /*batch_variance=*/corrected_variance,
                              /*reserve_space_1=*/batch_mean,
                              /*reserve_space_2=*/corrected_variance,
                              /*reserve_space_3=*/op.x()});
    } else {  // Inference case.
      auto bn_train_op = rewriter.create<BatchNormInferenceOp>(
          op.getLoc(),
          /*result_type=*/bn_train_input_type_tensor, bn_train_input,
          op.scale(), op.offset(), op.mean(), op.variance(), op.epsilon(),
          feature_dim.getValue());

      // Convert back to input type to stay aligned with expected output type
      // for TF op.
      auto y_out = rewriter.create<xla_hlo::ConvertOp>(op.getLoc(), bn_train_op,
                                                       input_element_type);

      // The mean, variance, and reserved space outputs of the batch norm op are
      // not used for inference. It doesn't matter what values we provide for
      // the last 5 results.
      rewriter.replaceOp(
          op, {/*y=*/y_out, /*batch_mean=*/op.x(),
               /*batch_variance=*/op.x(), /*reserve_space_1=*/op.x(),
               /*reserve_space_2=*/op.x(), /*reserve_space_3=*/op.x()});
    }
    return Pattern::matchSuccess();
  }
};

// Returns padding attribute for ReduceWindow op with given params.
//
// Requires padding to be either 'SAME' or 'VALID' and the number of input
// dimensions to be equal to the size of window dimensions and window strides.
static DenseIntElementsAttr GetReduceWindowPadding(
    llvm::ArrayRef<int64_t> input_dims, ArrayAttr window_dims,
    ArrayAttr window_strides, StringRef padding, Builder *builder) {
  if (padding == "VALID") return {};
  DCHECK_EQ(padding.str(), "SAME");

  llvm::SmallVector<tensorflow::int64, 4> input_shape, window_shape, strides;
  input_shape.reserve(input_dims.size());
  window_shape.reserve(window_shape.size());
  strides.reserve(window_strides.size());

  for (const auto &dim : input_dims) input_shape.push_back(dim);
  for (Attribute attr : window_dims)
    window_shape.push_back(attr.cast<IntegerAttr>().getInt());
  for (Attribute attr : window_strides)
    strides.push_back(attr.cast<IntegerAttr>().getInt());

  std::vector<std::pair<tensorflow::int64, tensorflow::int64>> paddings =
      ::xla::MakePadding(input_shape, window_shape, strides,
                         ::xla::Padding::kSame);
  int64_t rank = paddings.size();
  llvm::SmallVector<int64_t, 8> flatten_paddings(rank * 2);
  for (int i = 0; i < rank; i++) {
    flatten_paddings[2 * i] = paddings[i].first;
    flatten_paddings[2 * i + 1] = paddings[i].second;
  }
  return DenseIntElementsAttr::get(
      RankedTensorType::get({rank, 2}, builder->getIntegerType(64)),
      flatten_paddings);
}

// Converts MaxPool op to HLO ReduceWindow op by setting appropriate window
// dimensions with add as the reduction function. The reduction result is
// then divided by the number of elements in the window.
class ConvertAvgPoolOp : public OpRewritePattern<TF::AvgPoolOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::AvgPoolOp op,
                                     PatternRewriter &rewriter) const override {
    auto input_type = op.value().getType().dyn_cast<RankedTensorType>();
    if (!input_type) return matchFailure();

    // TODO(b/147217034): support other data formats.
    if (!IsDefaultDataFormat(op.data_format())) return matchFailure();
    // TODO(b/147217034): support "SAME" padding.
    if (op.padding() != "VALID") return matchFailure();

    // We will do accumulation first; use a larger bitwidth if suitable.
    Type input_element_type = input_type.getElementType();
    Type sum_element_type = GetSumAccumulationType(input_element_type);
    Type result_type;

    // The result type for reduction and division with the proper element type.
    if (auto ranked_type = op.getType().dyn_cast<RankedTensorType>())
      result_type =
          RankedTensorType::get(ranked_type.getShape(), sum_element_type);
    else
      result_type = UnrankedTensorType::get(sum_element_type);

    Value input_value = op.value();

    // Convert if we need enlarge the element type's bitwidth.
    if (input_element_type != sum_element_type)
      input_value = rewriter.create<ConvertOp>(op.getLoc(), input_value,
                                               sum_element_type);

    // Create the tf.ReduceWindow op.
    Value init =
        GetScalarConstOfType(sum_element_type, op.getLoc(), 0, &rewriter);
    DenseIntElementsAttr paddings_attr =
        GetReduceWindowPadding(input_type.getShape(), op.ksize(), op.strides(),
                               op.padding(), &rewriter);
    auto reduce = rewriter.create<ReduceWindowOp>(
        op.getLoc(), result_type, input_value, init,
        GetI64ElementsAttr(op.ksize()), GetI64ElementsAttr(op.strides()),
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(), paddings_attr);
    BuildReduceBody<AddOp>(sum_element_type, &reduce.body(), &rewriter);

    // Count the number of elements in the window. The following calculation
    // is only valid for no paddings.
    SmallVector<int64_t, 4> ksize;
    GetI64ArrayAttrValues(op.ksize(), &ksize);
    int64_t count = std::accumulate(ksize.begin(), ksize.end(), 1,
                                    std::multiplies<int64_t>());

    // Divide by the number of elements in the window.
    Value divisor =
        GetScalarConstOfType(sum_element_type, op.getLoc(), count, &rewriter);
    auto batch_dims =
        GetI64ElementsAttrForSeq(0, input_type.getRank(), &rewriter);
    Value result = rewriter.create<DivOp>(op.getLoc(), result_type, reduce,
                                          divisor, batch_dims);

    // Convert back if we enlarged the element type's bitwidth.
    if (input_element_type != sum_element_type)
      result =
          rewriter.create<ConvertOp>(op.getLoc(), result, input_element_type);

    rewriter.replaceOp(op, result);
    return matchSuccess();
  }
};

// Converts MaxPool op to HLO ReduceWindow op by setting appropriate window
// dimensions with max as the reduction function.
//
// Sample result for VALID padding mode:
//
//   %init = constant dense<...> : tensor<i32>
//   %max_pool = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.max"]
//               {window_dimensions = ..., window_strides = ... }
//
class ConvertMaxPoolOp : public OpRewritePattern<TF::MaxPoolOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::MaxPoolOp op,
                                     PatternRewriter &rewriter) const override {
    Type element_type =
        op.input().getType().cast<TensorType>().getElementType();
    if (!element_type.isIntOrFloat()) return matchFailure();
    Location loc = op.getLoc();
    ConstOp init = GetMinValueForType(element_type, loc, &rewriter);

    auto input_ty = op.input().getType().dyn_cast<RankedTensorType>();
    if (!input_ty) return matchFailure();
    DenseIntElementsAttr paddings_attr = GetReduceWindowPadding(
        input_ty.getShape(), op.ksize(), op.strides(), op.padding(), &rewriter);
    auto reduce = rewriter.create<ReduceWindowOp>(
        loc, op.getType(), op.input(), init.getResult(),
        GetI64ElementsAttr(op.ksize()), GetI64ElementsAttr(op.strides()),
        /*base_dilations=*/DenseIntElementsAttr(),
        /*window_dilations=*/DenseIntElementsAttr(), paddings_attr);
    BuildReduceBody<MaxOp>(element_type, &reduce.body(), &rewriter);

    rewriter.replaceOp(op, reduce.getResult());
    return matchSuccess();
  }
};

// Converts SelectV2 to HLO Select op and necessary BroadcastInDim ops on
// operands.
//
// For example, the following source IR:
//
//   %select = "tf.SelectV2"(%condition, %t, %e) :
//               (tensor<1xi1>, tensor<2xi32>, tensor<1xi32>) -> tensor<2xi32>
//
// will be converted into:
//
//   %pred = "xla_hlo.broadcast_in_dim"(%cond)
//             {broadcast_dimensions = dense<[0]> : tensor<1xi64>} :
//               (tensor<1xi1>) -> tensor<2xi1>
//   %on_false = "xla_hlo.broadcast_in_dim"(%e)
//                 {broadcast_dimensions = dense<[0]> : tensor<1xi64>} :
//                   (tensor<1xi32>) -> tensor<2xi32>
//   %select = "xla_hlo.select"(%pred, %t, %on_false) :
//               (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
class ConvertSelectV2Op : public OpRewritePattern<TF::SelectV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::SelectV2Op op,
                                     PatternRewriter &rewriter) const override {
    llvm::SmallVector<int64_t, 4> broadcast_then_else_shape;
    auto ranked_then_type = op.t().getType().dyn_cast<RankedTensorType>();
    auto ranked_else_type = op.e().getType().dyn_cast<RankedTensorType>();
    auto ranked_cond_type =
        op.condition().getType().dyn_cast<RankedTensorType>();
    if (!ranked_then_type || !ranked_then_type.hasStaticShape() ||
        !ranked_else_type || !ranked_else_type.hasStaticShape() ||
        !ranked_cond_type || !ranked_cond_type.hasStaticShape())
      return matchFailure();

    if (!OpTrait::util::getBroadcastedShape(ranked_then_type.getShape(),
                                            ranked_else_type.getShape(),
                                            broadcast_then_else_shape))
      return matchFailure();

    llvm::SmallVector<int64_t, 4> broadcast_shape;
    if (!OpTrait::util::getBroadcastedShape(broadcast_then_else_shape,
                                            ranked_cond_type.getShape(),
                                            broadcast_shape))
      return matchFailure();

    auto broadcast_or_self = [&](Value value) {
      RankedTensorType type = value.getType().cast<RankedTensorType>();
      auto output_type =
          RankedTensorType::get(broadcast_shape, type.getElementType());
      if (output_type == type) return value;

      int64_t rank = type.getRank();
      SmallVector<int64_t, 4> broadcast_dimensions(rank);
      std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(),
                broadcast_shape.size() - rank);

      return rewriter
          .create<BroadcastInDimOp>(
              op.getLoc(), output_type, value,
              GetI64ElementsAttr(broadcast_dimensions, &rewriter))
          .getResult();
    };

    // HLO SelectOp supports broadcasting for predicate/condition if
    // predicate/condition is a scalar.
    Value pred = ranked_cond_type.getRank() == 0
                     ? op.condition()
                     : broadcast_or_self(op.condition());
    Value on_true = broadcast_or_self(op.t());
    Value on_false = broadcast_or_self(op.e());

    rewriter.replaceOpWithNewOp<SelectOp>(op, on_true.getType(), pred, on_true,
                                          on_false);

    return matchSuccess();
  };
};

// Converts Sigmoid op to HLO ops computing sigmoid with the following formula:
//
//     sigmoid = add(mul(tanh(mul(logits, 0.5)), 0.5), 0.5)
//
// Sample result with 2-d f16 inputs with B batches of with N elements each.
//
//    // Create an array of 0.5 the shape of the input array.
//    %half = xla_hlo.constant dense<5.000000e-01> : tensor<f32>
//    %half_array = "xla_hlo.broadcast"(half)
//                           {broadcast_sizes = dense<2> : tensor<1xi64>}
//                           : (tensor<f32>) -> tensor<2xf32>
//
//    // Compute Tanh of half the logits of the values.
//    %halved_logits = xla_hlo.mul %logits, %half_array : tensor<2xf32>
//    %tanh = "xla_hlo.tanh"(%halved_logits) : (tensor<2xf32>) -> tensor<2xf32>
//
//    // Have the result of Tanh and add 0.5.
//    %halved_tanh = xla_hlo.mul %tanh, %half : tensor<2xf32>
//    %sigmoid = xla_hlo.add %halved_tanh, %half : tensor<2xf32>
//
class ConvertSigmoidOp : public OpRewritePattern<TF::SigmoidOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::SigmoidOp op,
                                     PatternRewriter &rewriter) const override {
    auto operand = op.getOperand();

    auto scalar_one = rewriter.create<ConstOp>(
        op.getLoc(),
        rewriter.getFloatAttr(getElementTypeOrSelf(operand.getType()), 0.5));

    auto shaped_type = operand.getType().cast<ShapedType>();
    auto constant_ones = rewriter.create<BroadcastOp>(
        op.getLoc(), shaped_type, scalar_one,
        DenseIntElementsAttr::get(
            RankedTensorType::get({shaped_type.getRank()},
                                  rewriter.getIntegerType(64)),
            shaped_type.getShape()));

    auto scaled_input = rewriter.create<MulOp>(
        op.getLoc(), operand, constant_ones, DenseIntElementsAttr());
    auto tanh_op =
        rewriter.create<TanhOp>(op.getLoc(), operand.getType(), scaled_input);
    auto mul_op =
        rewriter.create<MulOp>(op.getLoc(), tanh_op, constant_ones,
                               /*DenseIntElementsAttr=*/DenseIntElementsAttr());
    auto add_op =
        rewriter.create<AddOp>(op.getLoc(), mul_op, constant_ones,
                               /*DenseIntElementsAttr=*/DenseIntElementsAttr());

    rewriter.replaceOp(op, add_op.getResult());
    return matchSuccess();
  }
};

// Converts Softmax and LogSoftmax to HLO ops, computing softmax with the
// following formulas:
//
//     softmax = div(exp(logits), sum(exp(logits)))

//     log_softmax = sub(logits, log(sum(exp(logits))))
//
// Sample result with 2-d f16 inputs with B batches of with N elements each.
//
//    %reduce_dim = tf.Const dense<[1]> : tensor<1xi64>
//
//    // Subtract each element by their batches' max to improve numerical
//    // stability.
//    %max = "tf.Max"(%input, %reduce_dim)
//           : (tensor<BxNxf16>, tensor<1xi64>) -> tensor<Bxf16>
//    %sub = "xla_hlo.sub"(%inp, %max) {broadcast_dimensions = 0}
//            : (tensor<BxNxf16>, tensor<Bxf16>) -> tensor<BxNxf16>
//
//    %exp = "xla_hlo.exp"(%sub) : (tensor<BxNxf16>) -> tensor<BxNxf16>
//    %sum = "tf.Sum"(%exp, %reduce_dim)
//            : (tensor<BxNxf32>, tensor<1xi64>) -> tensor<Bxf32>
//
//    // Softmax computation:
//    %softmax = "xla_hlo.div"(%exp, %sum_f16) {broadcast_dimensions = 0}
//            : (tensor<BxNxf16>, tensor<Bxf16>) -> tensor<BxNxf16>
template <typename OpTy, bool use_log = true>
class ConvertSoftmaxOp : public OpRewritePattern<OpTy> {
 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(OpTy op,
                                     PatternRewriter &rewriter) const override {
    Value logits = op.logits();

    // Softmax converter requires ranked type because the XLA reduce ops used
    // while lowering requires dimensions attribute to reduce along.
    RankedTensorType type = logits.getType().dyn_cast<RankedTensorType>();
    if (!type) return Pattern::matchFailure();

    auto loc = op.getLoc();
    int rank = type.getRank();

    // Note that the TensorFlow Softmax op verifies that the input rank is
    // greater than or equal to one so both of the following sequences are
    // valid.
    auto batch_dims = GetI64ElementsAttrForSeq(0, rank - 1, &rewriter);
    auto reduce_dim = rewriter.create<TF::ConstOp>(
        loc, GetI64ElementsAttr({rank - 1}, &rewriter));

    // Exponential of input values and then their sum can be very large here.
    // Division with large denominator is numerically unstable. To improve
    // numerical stability, subtract each batch with their max element so that
    // the maximum input value is zero. It can be shown that softmax computed
    // after adding or subtracting all inputs in a batch using a common value
    // gives mathematically equivalent result.
    auto max_logits =
        rewriter.create<TF::MaxOp>(loc, logits, reduce_dim,
                                   /*keep_dims=*/rewriter.getBoolAttr(false));
    auto shifted_logits =
        rewriter.create<SubOp>(loc, type, logits, max_logits, batch_dims);

    // Exponentiate the inputs.
    Value exp = rewriter.create<ExpOp>(loc, type, shifted_logits);

    // Compute summation of the exponentials.
    auto exp_sum =
        rewriter.create<TF::SumOp>(loc, exp, reduce_dim,
                                   /*keep_dims=*/rewriter.getBoolAttr(false));
    Value sum = exp_sum.getResult();

    if (use_log) {
      Value log = rewriter.create<LogOp>(loc, sum);
      rewriter.replaceOpWithNewOp<SubOp>(op, shifted_logits, log, batch_dims);
    } else {
      rewriter.replaceOpWithNewOp<DivOp>(op, exp, sum, batch_dims);
    }
    return Pattern::matchSuccess();
  }
};

// Converts Size to HLO ops, computing the size of a ranked input tensor.
// TODO(b/145253252): Update this to not require ranked input tensor shapes.
//
// The main logic of this pattern is to calculate the size by multiplying every
// dimension of the input tensor's shape together.
//
// For example, the following source IR:
//
//   %size = "tf.Size"(%input) : (tensor<2x?x8xf32>) -> tensor<i32>
//
// will be converted into:
//
//   %const = xla_hlo.constant dense<1> : tensor<i32>
//   %dim_0 = "xla_hlo.get_dimension_size"(%input) {dimension = 0 : i32} :
//                                         (tensor<2x?x8xf32>) -> tensor<i32>
//   %prod_0 = xla_hlo.mul %const, %dim_0 : tensor<i32>
//   %dim_1 = "xla_hlo.get_dimension_size"(%input) {dimension = 1 : i32} :
//                                         (tensor<2x?x8xf32>) -> tensor<i32>
//   %prod_1 = xla_hlo.mul %prod_0, %dim_1 : tensor<i32>
//   %dim_2 = "xla_hlo.get_dimension_size"(%input) {dimension = 2 : i32} :
//                                         (tensor<2x?x8xf32>) -> tensor<i32>
//   %size = xla_hlo.mul %prod_1, %dim_2 : tensor<i32>
class ConvertSizeOp : public OpRewritePattern<TF::SizeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::SizeOp op,
                                     PatternRewriter &rewriter) const override {
    Value input = op.input();
    auto input_ty = input.getType().dyn_cast<RankedTensorType>();
    if (!input_ty) return Pattern::matchFailure();

    const int64_t rank = input_ty.getRank();
    auto result_type = op.getResult().getType();
    Operation *size =
        GetScalarConstOfType(result_type.cast<TensorType>().getElementType(),
                             op.getLoc(), 1, &rewriter);
    for (int64_t i = 0; i < rank; ++i) {
      auto dim = rewriter.create<GetDimensionSizeOp>(
          op.getLoc(), result_type, input,
          rewriter.getIntegerAttr(rewriter.getIntegerType(32), i));
      size = rewriter.create<MulOp>(
          op.getLoc(), size->getResult(0), dim.getResult(),
          /*DenseIntElementsAttr=*/DenseIntElementsAttr());
    }
    rewriter.replaceOp(op, size->getResult(0));

    return Pattern::matchSuccess();
  }
};

// Converts the tf.Split op into a series of HLO slice ops when the tensor to be
// split has fully static shape and the dimension to split is a constant.
//
// The main logic of this pattern is to calculate the index start and end range
// for each slice. And this happens only on the dimension to be split; for all
// other dimensions, all resultant slices' index start and end range covers the
// input tensor's full range. Strides for all resultant slices are all one.
//
// For example, the following source IR:
//
//   %dim = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
//   %0:3 = "tf.Split"(%dim, %input) : (tensor<i32>, tensor<4x6xf32>) ->
//                (tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>)
//
// will be converted into:
//
//   %0 = "xla_hlo.slice"(%input) {
//             limit_indices = dense<[4, 2]> : tensor<2xi64>,
//             start_indices = dense<0> : tensor<2xi64>,
//             strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
//   %1 = "xla_hlo.slice"(%input) {
//             limit_indices = dense<4> : tensor<2xi64>,
//              start_indices = dense<[0, 2]> : tensor<2xi64>,
//            strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
//    %2 = "xla_hlo.slice"(%input) {
//            limit_indices = dense<[4, 6]> : tensor<2xi64>,
//            start_indices = dense<[0, 4]> : tensor<2xi64>,
//             strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
// TODO(antiagainst): consider lowering into TF ops so the pattern can be more
// applicable.
class ConvertSplitOp : public OpRewritePattern<TF::SplitOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::SplitOp op,
                                     PatternRewriter &rewriter) const override {
    // We can only split along static dimensions.
    auto input_type = op.value().getType().dyn_cast<RankedTensorType>();
    if (!input_type) return matchFailure();

    // We can only match when the split dimension is a constant scalar.
    DenseIntElementsAttr split_dim_attr;
    if (!matchPattern(op.split_dim(), m_Constant(&split_dim_attr)))
      return matchFailure();

    // Get the dimension we are splitting at. Offset properly if it's negative.
    int64_t input_rank = input_type.getRank();
    int64_t dim_index = (*split_dim_attr.begin()).getSExtValue();
    if (dim_index < 0) dim_index += input_rank;

    // Calculate the dimension size for each slice along the split dimension.
    int64_t input_dim_size = input_type.getDimSize(dim_index);
    // If we are splitting along the dynamic dimension then we cannot compute
    // the static dimension length.
    if (TensorType::isDynamic(input_dim_size)) return matchFailure();

    int64_t num_splits = op.getNumResults();
    int64_t slice_size = input_dim_size / num_splits;

    // Get each slice's type.
    auto slice_shape = llvm::to_vector<4>(input_type.getShape());
    slice_shape[dim_index] = slice_size;
    Type slice_type =
        RankedTensorType::get(slice_shape, input_type.getElementType());

    // Parameters for constructing each slice.
    SmallVector<int64_t, 4> begin_indices(input_rank, 0);
    auto end_indices = llvm::to_vector<4>(input_type.getShape());
    SmallVector<int64_t, 4> strides(input_rank, 1);

    // All HLO slice results used to replace the original tf.Split op.
    SmallVector<Value, 4> slices;
    slices.reserve(num_splits);

    for (int i = 0; i < num_splits; ++i) {
      begin_indices[dim_index] = i * slice_size;
      end_indices[dim_index] = (i + 1) * slice_size;
      slices.push_back(
          rewriter.create<SliceOp>(op.getLoc(), slice_type, op.value(),
                                   GetI64ElementsAttr(begin_indices, &rewriter),
                                   GetI64ElementsAttr(end_indices, &rewriter),
                                   GetI64ElementsAttr(strides, &rewriter)));
    }

    rewriter.replaceOp(op, slices);
    return matchSuccess();
  }
};

// Converts the tf.SplitV op into a series of HLO slice ops when the tensor to
// be split has fully static shape and the dimension to split and split sizes
// are constants.
//
// This is similar to the conversion for tf.Split op other than that the size of
// each chunk on the dimension to split is explicitly given as an op operand
// and they are not necessarily the same.
//
// For example, given the following IR:
//
// %split_sizes = "tf.Const"() {value = dense<[1, -1, 3]> : tensor<3xi32>}
// %split_dim = "tf.Const"() {value = dense<1> : tensor<i32>}
// %0:3 = "tf.SplitV"(%input, %split_sizes, %split_dim) :
//                   (tensor<4x6xf32>, tensor<3xi32>, tensor<i32>) ->
//                   (tensor<4x1xf32>, tensor<4x2xf32>, tensor<4x3xf32>)
//
// We will generate slices following slices:
// %0 = "xla_hlo.slice"(%input) {
//        limit_indices = dense<[4, 1]> : tensor<2xi64>,
//        start_indices = dense<0> : tensor<2xi64>,
//        strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x1xf32>
// %1 = "xla_hlo.slice"(%input) {
//        limit_indices = dense<[4, 3]> : tensor<2xi64>,
//        start_indices = dense<[0, 1]> : tensor<2xi64>,
//        strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x2xf32>
// %2 = "xla_hlo.slice"(%input) {
//        limit_indices = dense<[4, 6]> : tensor<2xi64>,
//        start_indices = dense<[0, 3]> : tensor<2xi64>,
//        strides = dense<1> : tensor<2xi64>} :
//        (tensor<4x6xf32>) -> tensor<4x3xf32>
class ConvertSplitVOp : public OpRewritePattern<TF::SplitVOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::SplitVOp op,
                                     PatternRewriter &rewriter) const override {
    // We can only split along static dimensions.
    // TODO(b/145731001): enhance to support dynamic-shaped inputs.
    auto input_type = op.value().getType().dyn_cast<RankedTensorType>();
    if (!input_type) return matchFailure();

    // We can only match when the split dimension is a constant scalar.
    DenseIntElementsAttr split_dim_attr;
    if (!matchPattern(op.split_dim(), m_Constant(&split_dim_attr)))
      return matchFailure();

    // We can only match when the split sizes is a constant int vector.
    DenseIntElementsAttr split_sizes_attr;
    if (!matchPattern(op.size_splits(), m_Constant(&split_sizes_attr)))
      return matchFailure();

    // Get each chunck's size along the dimension to split. It may contain
    // dynamic sizes and we need to update it if so.
    SmallVector<int64_t, 4> split_sizes;
    int64_t total_dim_size = 0;  // Total dimension size assigned to splits
    llvm::Optional<int> dynamic_dim_index;
    split_sizes.reserve(
        split_sizes_attr.getType().cast<ShapedType>().getNumElements());
    for (auto dim : llvm::enumerate(split_sizes_attr)) {
      int64_t dim_val = dim.value().getSExtValue();
      split_sizes.push_back(dim_val);
      if (dim_val == ShapedType::kDynamicSize) {
        // We cannot have more than one dynamic dimension.
        assert(!dynamic_dim_index && "invalid split sizes");
        dynamic_dim_index = dim.index();
      } else {
        total_dim_size += dim_val;
      }
    }

    // Get the dimension we are splitting at. Offset properly if it's negative.
    int64_t input_rank = input_type.getRank();
    int64_t dim_index = (*split_dim_attr.begin()).getSExtValue();
    if (dim_index < 0) dim_index += input_rank;

    int64_t input_dim_size = input_type.getDimSize(dim_index);
    if (TensorType::isDynamic(input_dim_size)) return matchFailure();

    assert(((dynamic_dim_index && total_dim_size <= input_dim_size) ||
            (!dynamic_dim_index && total_dim_size == input_dim_size)) &&
           "invalid split sizes");

    // Update the dynamic dimension with calculated concrete size.
    if (dynamic_dim_index)
      split_sizes[*dynamic_dim_index] = input_dim_size - total_dim_size;

    // Parameters for constructing each slice.
    SmallVector<int64_t, 4> begin_indices(input_rank, 0);
    auto end_indices = llvm::to_vector<4>(input_type.getShape());
    SmallVector<int64_t, 4> strides(input_rank, 1);

    // All HLO slice results used to replace the original tf.Split op.
    SmallVector<Value, 4> slices;
    slices.reserve(op.getNumResults());

    for (int i = 0; i < op.getNumResults(); ++i) {
      end_indices[dim_index] = begin_indices[dim_index] + split_sizes[i];
      slices.push_back(rewriter.create<xla_hlo::SliceOp>(
          op.getLoc(), op.value(), GetI64ElementsAttr(begin_indices, &rewriter),
          GetI64ElementsAttr(end_indices, &rewriter),
          GetI64ElementsAttr(strides, &rewriter)));
      // Prepare the begin indice for the next slice.
      begin_indices[dim_index] = end_indices[dim_index];
    }

    rewriter.replaceOp(op, slices);
    return matchSuccess();
  }
};

// Converts StridedSlice op to HLO Slice op along with Reverse op to handle
// negative strides and Reshape op to update the output shape. Indices and
// strides operands are converted to attributes with non-negative indexing.
//
// For example with an op like following,
//   tf.StridedSlice(%input, %begin, %end, %strides) {shrink_axis_mask = 1}
//     : tensor<AxBxf32> -> tensor<Pxf32>
//
// Output would be:
//   %reversed = "xla_hlo.Reverse" (%input) {dimensions = ...}
//   %sliced = "xla_hlo.Slice" (%input)
//             {start_indices = ..., limit_indices = ..., strides = ...}
//   %output = "xla_hlo.Reshape" (%sliced) : tensor<1xPxf32> -> tensor<Pxf32>
//
class ConvertStridedSliceOp : public OpRewritePattern<TF::StridedSliceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::StridedSliceOp op,
                                     PatternRewriter &rewriter) const override {
    // Input shape needs to be static to convert negative indices in TensorFlow
    // to absolute indices required by HLO.
    //
    // TODO(hinsu): Relax this constraint for ops without negative indices and
    // strides.
    auto input_ty = op.input().getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) return matchFailure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();

    // Output shape needs to be static to apply 'new_axis_mask' or
    // 'shrink_axis_mask' by reshaping tensor after slice.
    //
    // TODO(hinsu): Relax this constraint for ops without the above masks.
    auto result_ty = op.getType().dyn_cast<RankedTensorType>();
    if (!result_ty || !result_ty.hasStaticShape()) return matchFailure();

    SmallVector<int64_t, 4> begin_indices, end_indices, strides;
    if (!op.GetSlicedBoundRanges(&begin_indices, &end_indices, &strides))
      return matchFailure();

    SmallVector<int64_t, 4> hlo_begin_indices, hlo_end_indices, hlo_strides,
        dims_to_reverse;
    int64_t input_rank = input_ty.getRank();
    hlo_begin_indices.reserve(input_rank);
    hlo_end_indices.reserve(input_rank);
    hlo_strides.reserve(input_rank);

    int64_t indices_elements = begin_indices.size();
    if (input_rank < indices_elements) return matchFailure();

    // Convert from TensorFlow negative or out of range indices and strides
    // values to legal HLO Slice attributes.
    for (int i = 0, e = indices_elements; i != e; i++) {
      int64_t begin = begin_indices[i];
      int64_t end = end_indices[i];
      int64_t stride = strides[i];

      if (stride < 0) {
        // Negative stride means that the output values are computed starting
        // from end until begin. Mark the dimension for reversal before slice
        // and compute indices for the reversed input.
        dims_to_reverse.push_back(i);
        begin = (input_shape[i] - 1) - begin;
        end = (input_shape[i] - 1) - end;
        stride = -stride;
      }

      // Unlike TensorFlow, HLO requires begin and end values to be within
      // range.
      begin = std::max(int64_t(0), begin);
      end = std::max(begin, end);
      end = std::min(end, input_shape[i]);

      hlo_begin_indices.push_back(begin);
      hlo_end_indices.push_back(end);
      hlo_strides.push_back(stride);
    }

    Location loc = op.getLoc();
    Value input = op.input();
    if (!dims_to_reverse.empty())
      input = rewriter.create<ReverseOp>(
          loc, input_ty, op.input(),
          GetI64ElementsAttr(dims_to_reverse, &rewriter));
    auto sliced = rewriter.create<SliceOp>(
        loc, input, GetI64ElementsAttr(hlo_begin_indices, &rewriter),
        GetI64ElementsAttr(hlo_end_indices, &rewriter),
        GetI64ElementsAttr(hlo_strides, &rewriter));

    // Reshape slice result so that the shape is updated depending on
    // 'new_axis_mask' or 'shrink_axis_mask' attributes.
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), sliced);
    return matchSuccess();
  }
};

// Converts tf.StridedSliceGrad to HLO reshape, reverse and padding ops.
//
// tf.StridedSlice is taking slice of the input tensor. tf.StridedSliceGrad does
// the reverse: it propagates the graident for the sliced tensor to the original
// input tensor by doing padding with zeros. The main logic is calculating the
// indices and strides for padding.
class ConvertStridedSliceGradOp
    : public OpRewritePattern<TF::StridedSliceGradOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::StridedSliceGradOp op,
                                     PatternRewriter &rewriter) const override {
    // We need constant input shape to perform padding calculations later.
    DenseIntElementsAttr input_shape_attr;
    if (!matchPattern(op.shape(), m_Constant(&input_shape_attr)))
      return matchFailure();

    // We also need constant begin/end indices and strides to perform padding
    // calculations.
    // Bounded shape after performing strided slice
    SmallVector<int64_t, 4> shape;
    // Bounded begin, end, and strides for strided slice
    SmallVector<int64_t, 4> begin_indices, end_indices, strides;
    if (!op.GetSlicedShapeAndBoundRanges(&shape, &begin_indices, &end_indices,
                                         &strides))
      return matchFailure();

    Value grad = op.dy();
    Type element_type = grad.getType().cast<ShapedType>().getElementType();

    // Perform reshape to undo any new/shrink axies done by strided slice.
    grad = rewriter.create<xla_hlo::ReshapeOp>(
        op.getLoc(), RankedTensorType::get(shape, element_type), grad);

    SmallVector<int64_t, 4> padding_low, padding_high, padding_interm;
    SmallVector<int64_t, 4> dims_to_reverse;
    padding_low.reserve(shape.size());
    padding_high.reserve(shape.size());
    padding_interm.reserve(shape.size());

    // Prepare padding parameters for each dimension.
    for (int i = 0, e = shape.size(); i < e; ++i) {
      int64_t input_dim = (*(input_shape_attr.begin() + i)).getSExtValue();
      if (strides[i] > 0) {
        padding_low.push_back(begin_indices[i]);
        padding_interm.push_back(strides[i] - 1);

        // Pad the upper dimension up to the expected input shape. It's not
        // sufficient simply to use end_indices[i] to compute the padding in
        // cases where the stride does not divide evenly into the interval
        // between begin_indices[i] and end_indices[i].
        int64_t size =
            padding_low[i] + shape[i] + (shape[i] - 1) * padding_interm[i];
        padding_high.push_back(input_dim - size);
      } else {
        dims_to_reverse.push_back(i);
        padding_high.push_back(input_dim - begin_indices[i] - 1);
        padding_interm.push_back(-strides[i] - 1);

        // Pad the lower dimension up to the expected input shape.
        int64_t size =
            padding_high[i] + shape[i] + (shape[i] - 1) * padding_interm[i];
        padding_low.push_back(input_dim - size);
      }
    }

    if (!dims_to_reverse.empty()) {
      grad = rewriter.create<xla_hlo::ReverseOp>(
          op.getLoc(), grad.getType(), grad,
          GetI64ElementsAttr(dims_to_reverse, &rewriter));
    }

    auto zero = GetScalarConstOfType(element_type, op.getLoc(), 0, &rewriter);
    rewriter.replaceOpWithNewOp<xla_hlo::PadOp>(
        op, op.getType(), grad, zero,
        GetI64ElementsAttr(padding_low, &rewriter),
        GetI64ElementsAttr(padding_high, &rewriter),
        GetI64ElementsAttr(padding_interm, &rewriter));
    return matchSuccess();
  }
};

/// Converts the RangeOp tensorflow op to a xla_hlo.iota op with a scaling and
/// offset applied to generate the range values. The output tensor needs to
/// have a static shape.
///
/// For example an op like the following:
///   %result = "tf.Range"(%start, %limit, %delta) {Tidx = "tfdtype$DT_FLOAT"}
///      : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<5xf32>
///
/// Output would be:
///   %iota = "xla_hlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5xf32>
///   %scaled = "xla_hlo.mul"(%iota, %delta)
///       {broadcast_dimensions = dense<[]> : tensor<0xi64>} :
///       (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
///   %result = "xla_hlo.add"(%scaled, %offset)
///       {broadcast_dimensions = dense<[]> : tensor<0xi64>} :
///       (tensor<5xf32>, tensor<f32>) -> tensor<5xf32>
///
/// Implementation is defined in C++ due to no type interface for the iota op.
class ConvertRangeOp : public OpRewritePattern<TF::RangeOp> {
  using OpRewritePattern<TF::RangeOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::RangeOp op,
                                     PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    auto result_type = result.getType();
    if (!result_type.cast<ShapedType>().hasStaticShape()) {
      return matchFailure();
    }

    auto iota = rewriter.create<IotaOp>(op.getLoc(), result_type,
                                        rewriter.getI64IntegerAttr(0));
    auto scaled = rewriter.create<MulOp>(
        op.getLoc(), result_type, iota, op.delta(),
        xla::getBroadcastDimensionsAttr(&rewriter, iota, op.delta()));
    rewriter.replaceOpWithNewOp<AddOp>(
        op, result_type, scaled, op.start(),
        xla::getBroadcastDimensionsAttr(&rewriter, scaled, op.start()));
    return matchSuccess();
  }
};

/// Converts the LinSpace tensorflow op to a xla_hlo.iota op with a scaling
/// and offset applied to generate the linspace values. The output tensor needs
/// to have a static shape.  The implementation is defined in C++ because there
/// is no type inference for the iota op.
class ConvertLinSpaceOp : public OpRewritePattern<TF::LinSpaceOp> {
  using OpRewritePattern<TF::LinSpaceOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::LinSpaceOp op,
                                     PatternRewriter &rewriter) const override {
    auto result = op.getResult();
    auto result_type = result.getType().dyn_cast<ShapedType>();
    if (!result_type || !result_type.hasStaticShape()) {
      return matchFailure();
    }

    // Calculate the scaling that needs to be applied to the iota.
    auto step_numerator = rewriter.create<SubOp>(
        op.getLoc(), op.start().getType(), op.stop(), op.start(),
        xla::getBroadcastDimensionsAttr(&rewriter, op.stop(), op.start()));
    Value step_denominator = rewriter.create<ConvertOp>(
        op.getLoc(), op.num(), result_type.getElementType());
    if (op.num() > 1) {
      Value one = GetScalarConstOfType(result_type.getElementType(),
                                       op.getLoc(), 1, &rewriter);
      step_denominator = rewriter.create<SubOp>(
          op.getLoc(), step_denominator.getType(), step_denominator, one,
          xla::getBroadcastDimensionsAttr(&rewriter, step_denominator, one));
    }
    auto step = rewriter.create<DivOp>(
        op.getLoc(), step_numerator.getType(), step_numerator, step_denominator,
        xla::getBroadcastDimensionsAttr(&rewriter, step_numerator,
                                        step_denominator));

    // Scale the iota and add the offset.
    auto iota = rewriter.create<IotaOp>(op.getLoc(), result_type,
                                        rewriter.getI64IntegerAttr(0));
    auto scaled = rewriter.create<MulOp>(
        op.getLoc(), result_type, iota, step,
        xla::getBroadcastDimensionsAttr(&rewriter, iota, step));
    rewriter.replaceOpWithNewOp<AddOp>(
        op, result_type, scaled, op.start(),
        xla::getBroadcastDimensionsAttr(&rewriter, scaled, op.start()));
    return matchSuccess();
  }
};

/// Converts a generic OpTy tensorflow op to a xla_hlo.reduce op over
/// ReductionOp.
/// `is_accumulation` controls whether it uses higher precision for the actual
/// reduction. This is set to false for ops like max where there is no precision
/// concerns.
template <typename Derived, typename OpTy, typename ReductionOp,
          bool is_accumulation = true>
class GenericConvertReductionOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(OpTy op,
                                     PatternRewriter &rewriter) const override {
    // TODO(b/141785544): Update this to not require static shapes.
    // Input shape needs to be static to convert negative indices in TensorFlow
    // to absolute indices required by HLO.
    auto input_ty = op.input().getType().template dyn_cast<RankedTensorType>();
    if (!input_ty) return this->matchFailure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();

    DenseIntElementsAttr dimensions;
    if (!matchPattern(op.reduction_indices(), m_Constant(&dimensions)))
      return this->matchFailure();

    // Build the final shape from input_shape and dimensions using a bitmap
    // to mark the reduced dimensions.
    SmallVector<bool, 4> reduced_dimensions_bitmap(input_shape.size(), false);
    SmallVector<int64_t, 4> xla_dimensions;
    for (APInt index_raw : dimensions.getValues<APInt>()) {
      int64_t index = index_raw.getSExtValue();
      int64_t rank = input_shape.size();
      if ((index < -rank || index >= rank)) return this->matchFailure();
      index = (index + rank) % rank;
      reduced_dimensions_bitmap[index] = true;
      xla_dimensions.push_back(index);
    }

    Location loc = op.getLoc();
    Type element_type = input_ty.getElementType();
    // Convert to an accumulation type to not lose precision when doing
    // repeated arithmetic operations.
    Type reduce_element_type =
        is_accumulation ? GetAccumulationType(element_type) : element_type;
    auto casted_input =
        rewriter.create<ConvertOp>(loc, op.input(), reduce_element_type);

    // Each reduction op can have a different initial value.
    Value init = Derived::GetInitialValue(reduce_element_type, loc, &rewriter);

    auto reduction = rewriter.create<ReduceOp>(
        loc, casted_input.getResult(), init,
        GetI64ElementsAttr(xla_dimensions, &rewriter));
    BuildReduceBody<ReductionOp>(reduce_element_type, &reduction.body(),
                                 &rewriter);
    Value result = reduction.getResult(0);

    // The mean op needs to divide by the product of the reduced dimensions.
    if (std::is_same<OpTy, TF::MeanOp>::value) {
      int64_t divisor_count = 1;
      for (size_t i = 0; i < input_shape.size(); ++i) {
        if (reduced_dimensions_bitmap[i]) {
          if (TensorType::isDynamic(input_shape[i])) {
            return this->matchFailure();
          }
          divisor_count *= input_shape[i];
        }
      }
      auto divisor = GetScalarConstOfType(reduce_element_type, loc,
                                          divisor_count, &rewriter);
      auto broadcast_dims = GetI64ElementsAttr({}, &rewriter);
      result = rewriter.create<DivOp>(loc, result, divisor.getResult(),
                                      broadcast_dims);
    }

    result = rewriter.create<ConvertOp>(loc, result, element_type);

    // Need to reshape back after the reduction if we're keeping the reduced
    // dimensions.
    if (op.keep_dims()) {
      result = rewriter.create<ReshapeOp>(loc, op.getType(), result);
    }
    rewriter.replaceOp(op, {result});

    return this->matchSuccess();
  }
};

// Converts Mean op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %sum = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.add"]
//               {dimensions = ...}
//   %divisor = constant dense<...> : tensor<T>
//   %mean = "xla_hlo.div"(%sum, %divisor)
class ConvertMeanOp
    : public GenericConvertReductionOp<ConvertMeanOp, TF::MeanOp, AddOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;
  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 0, rewriter);
  }
};

// Converts Sum op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %sum = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.add"]
//               {dimensions = ...}
class ConvertSumOp
    : public GenericConvertReductionOp<ConvertSumOp, TF::SumOp, AddOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 0, rewriter);
  }
};

// Converts Max op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %max = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.max"]
//               {dimensions = ...}
class ConvertMaxOp
    : public GenericConvertReductionOp<ConvertMaxOp, TF::MaxOp, MaxOp,
                                       /* is_accumulation= */ false> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetMinValueForType(reduce_element_type, loc, rewriter);
  }
};

// Converts Min op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %min = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.min"]
//               {dimensions = ...}
class ConvertMinOp
    : public GenericConvertReductionOp<ConvertMinOp, TF::MinOp, MinOp,
                                       /* is_accumulation= */ false> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetMaxValueForType(reduce_element_type, loc, rewriter);
  }
};

// Converts Prod op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %prod = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.mul"]
//               {dimensions = ...}
class ConvertProdOp
    : public GenericConvertReductionOp<ConvertProdOp, TF::ProdOp, MulOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 1, rewriter);
  }
};

// Converts All op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %max = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.and"]
//               {dimensions = ...}
class ConvertAllOp
    : public GenericConvertReductionOp<ConvertAllOp, TF::AllOp, AndOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;
  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 1, rewriter);
  }
};

// Converts Any op to HLO Reduce op.
//
//   %init = constant dense<...> : tensor<T>
//   %max = "xla_hlo.reduce"(%inp, %init) ["xla_hlo.or"]
//               {dimensions = ...}
class ConvertAnyOp
    : public GenericConvertReductionOp<ConvertAnyOp, TF::AnyOp, OrOp> {
 public:
  using GenericConvertReductionOp::GenericConvertReductionOp;
  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 0, rewriter);
  }
};

// Converts tensorflow ArgMin or ArgMax op to xla_hlo operations that perform
// a reduction on the original input and the corresponding index. The reduction
// sub-computation selects the max (or min) value and the index for the value.
//   Derived: is the resulting derived class of this class.
//   OpTy: is TF::ArgMaxOp or TF::ArgMinOp.
template <typename Derived, typename OpTy>
class ConvertArgMinMaxOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(OpTy op,
                                     PatternRewriter &rewriter) const override {
    RankedTensorType input_type =
        op.input().getType().template dyn_cast<RankedTensorType>();
    if (!input_type) {
      return this->matchFailure();
    }

    Type input_element_type = input_type.getElementType();
    // TODO(bixia): Clarify whether tf.ArgMax supports complex data types. If
    // tf.ArgMax doesn't support complex data types, this check can be removed.
    if (!input_element_type.isIntOrFloat()) return this->matchFailure();

    Location loc = op.getLoc();
    Value init_value =
        Derived::GetInitialValue(input_element_type, loc, rewriter);

    RankedTensorType output_type =
        op.output().getType().template dyn_cast<RankedTensorType>();
    if (!output_type) {
      return this->matchFailure();
    }

    Type index_element_type = output_type.getElementType();
    Value index_init_value =
        GetScalarConstOfType(index_element_type, loc, 0, &rewriter);

    RankedTensorType index_type =
        RankedTensorType::get(input_type.getShape(), index_element_type);

    llvm::Optional<int64_t> optional_axis =
        GetIntegerHLOAxisFromTFAxis(op.dimension(), input_type.getRank());
    if (!optional_axis.hasValue()) {
      return this->matchFailure();
    }
    int64_t axis = optional_axis.getValue();

    IntegerAttr iota_dimension =
        IntegerAttr::get(rewriter.getIntegerType(64), axis);
    Value index_values =
        rewriter.create<IotaOp>(loc, index_type, iota_dimension);

    std::vector<int64_t> dimensions = input_type.getShape();
    dimensions.erase(dimensions.begin() + axis);
    ArrayRef<int64_t> reduction_result_shape(dimensions);

    Value operands[] = {op.input(), index_values};
    Value init_values[] = {init_value, index_init_value};
    DenseIntElementsAttr reduction_dimensions =
        GetI64ElementsAttr({axis}, &rewriter);

    auto reduction = rewriter.create<ReduceOp>(
        loc, llvm::ArrayRef<Value>(operands),
        llvm::ArrayRef<Value>(init_values), reduction_dimensions);
    StringRef direction = Derived::GetDirection();
    BuildArgMinMaxReductionBody(input_element_type, index_element_type,
                                direction, &reduction.body(), &rewriter);

    rewriter.replaceOp(op, {reduction.getResult(1)});
    return this->matchSuccess();
  }
};

// Converts tensorflow ArgMax op to xla_hlo operations. The actual
// implementation is in class ConvertArgMinMaxOp:
//
//   %init_index = constant dense<...> : tensor<T>
//   %init = constant dense<...> : tensor<T>
//   %reduce = "xla_hlo.reduce"(%selected_input, %select_index, %init,
//                              %init_index) ["xla_hlo.arg_max"]
class ConvertArgMaxOp
    : public ConvertArgMinMaxOp<ConvertArgMaxOp, TF::ArgMaxOp> {
 public:
  using ConvertArgMinMaxOp::ConvertArgMinMaxOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter &rewriter) {
    return GetMinValueForType(reduce_element_type, loc, &rewriter);
  }

  static StringRef GetDirection() { return "GT"; }
};

// Converts TF TensorScatterUpdate op into Scatter Op with assignment:
//
//   %result = "xla_hlo.scatter"(%tensor, %indices, %updates)
//     { dimensions = ... }
//
class ConvertTensorScatterUpdateOp
    : public OpRewritePattern<TF::TensorScatterUpdateOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::TensorScatterUpdateOp op,
                                     PatternRewriter &rewriter) const override {
    auto tensor_ty = op.tensor().getType().dyn_cast<RankedTensorType>();
    auto indices_ty = op.indices().getType().dyn_cast<RankedTensorType>();
    auto updates_ty = op.updates().getType().dyn_cast<RankedTensorType>();

    if (!tensor_ty || !indices_ty || !updates_ty) return matchFailure();
    // Last dimension of the indices needs to known at compile time for
    // computation of the 'update_window_dims' attribute in the dimensions
    // struct.
    int64_t num_index_dims = indices_ty.getShape().back();
    if (ShapedType::isDynamic(num_index_dims)) return matchFailure();

    int64_t tensor_rank = tensor_ty.getRank();
    int64_t indices_rank = indices_ty.getRank();
    int64_t updates_rank = updates_ty.getRank();

    int64_t window_dims = tensor_rank - num_index_dims;
    auto dims_attr = ScatterDimensionNumbers::get(
        GetI64ElementsAttrForSeq(updates_rank - window_dims, updates_rank,
                                 &rewriter),
        GetI64ElementsAttrForSeq(0, num_index_dims, &rewriter),
        GetI64ElementsAttrForSeq(0, num_index_dims, &rewriter),
        rewriter.getI64IntegerAttr(indices_rank - 1), rewriter.getContext());

    Location loc = op.getLoc();
    auto scatter = rewriter.create<ScatterOp>(
        loc, op.getType(), op.tensor(), op.indices(), op.updates(), dims_attr);

    // Build region to assign the new value.
    [&](Region *region) {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block = rewriter.createBlock(region);

      // Block arguments are scalars of the given element type.
      Type type =
          RankedTensorType::get(/*shape=*/{}, tensor_ty.getElementType());
      block->addArguments({type, type});
      rewriter.create<ReturnOp>(loc, block->getArgument(1));
    }(&scatter.update_computation());

    rewriter.replaceOp(op, scatter.getResult());
    return matchSuccess();
  }
};

// Converts Tile op to HLO BroadcastInDim and Reshape ops.
//   For shape [S1, S2] and multiples [M1, M2],
//     MS1 = M1 * S1; MS2 = M2 * S2
//
//   %broadcast = xla_hlo.broadcast_in_dim(%input) {
//     broadcast_dimensions = [0, 2]
//   }
//   %result = "xla_hlo.reshape"(%broadcast) : (tensor<S1xM1xS2xM2xf32>)
//      -> tensor<MS1xMS2xf32>
class ConvertTileOp : public OpRewritePattern<TF::TileOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::TileOp op,
                                     PatternRewriter &rewriter) const override {
    auto input_ty = op.input().getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) return matchFailure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();
    Type element_type = input_ty.getElementType();

    DenseIntElementsAttr multiples;
    if (!matchPattern(op.multiples(), m_Constant(&multiples)) ||
        multiples.getType().getRank() != 1)
      return matchFailure();

    if (multiples.getNumElements() != input_shape.size()) return matchFailure();

    SmallVector<int64_t, 8> broadcasted_shape;
    SmallVector<int64_t, 4> broadcast_dimensions;
    broadcasted_shape.reserve(input_shape.size() * 2);
    broadcast_dimensions.reserve(input_shape.size());
    for (auto multiple_and_input :
         llvm::zip(multiples.getValues<APInt>(), input_shape)) {
      int64_t multiple = std::get<0>(multiple_and_input).getSExtValue();
      int64_t input_size = std::get<1>(multiple_and_input);

      if (multiple < 0) return matchFailure();

      // Line input up with the next dimension in broadcasted_shape
      // when broadcasting.
      broadcast_dimensions.push_back(broadcasted_shape.size());
      int64_t output_size = input_size * multiple;
      if (input_size == 1 || multiple == 1) {
        // Special case for when normal broadcasting will just work.
        broadcasted_shape.push_back(output_size);
      } else {
        // Tiling will happen for this dimension during the ReshapeOp below.
        broadcasted_shape.push_back(input_size);
        broadcasted_shape.push_back(multiple);
      }
    }
    Location loc = op.getLoc();
    Type broadcasted_type =
        RankedTensorType::get(broadcasted_shape, element_type);
    Type output_type = op.getType();

    Value result = rewriter.create<BroadcastInDimOp>(
        loc, broadcasted_type, op.input(),
        GetI64ElementsAttr(broadcast_dimensions, &rewriter));

    if (output_type != broadcasted_type) {
      result = rewriter.create<ReshapeOp>(loc, output_type, result);
    }

    rewriter.replaceOp(op, {result});

    return matchSuccess();
  }
};

class ConvertMaxPoolGradOp : public OpRewritePattern<TF::MaxPoolGradOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::MaxPoolGradOp op,
                                     PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type element_type =
        op.orig_input().getType().cast<TensorType>().getElementType();

    // Compute paddings using the original input and kernel shape and strides.
    // Here, ReduceWindow op as used as the MaxPool op is lowered to the
    // ReduceWindow op.
    auto input_ty = op.orig_input().getType().dyn_cast<RankedTensorType>();
    if (!input_ty) return matchFailure();
    DenseIntElementsAttr paddings_attr = GetReduceWindowPadding(
        input_ty.getShape(), op.ksize(), op.strides(), op.padding(), &rewriter);

    auto result = rewriter.create<SelectAndScatterOp>(
        loc, op.getType(), op.orig_input(), op.grad(),
        GetScalarConstOfType(element_type, loc, 0, &rewriter),
        GetI64ElementsAttr(op.ksize()), GetI64ElementsAttr(op.strides()),
        paddings_attr);

    BuildReduceBody<AddOp>(element_type, &result.scatter(), &rewriter);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      Block *block = rewriter.createBlock(&result.select());

      // Block arguments are scalars of the given element type.
      Type type = RankedTensorType::get(/*shape=*/{}, element_type);
      block->addArguments({type, type});

      auto reducer = rewriter.create<CompareOp>(
          loc, block->getArgument(0), block->getArgument(1),
          /*broadcast_dimensions=*/nullptr,
          StringAttr::get("GE", rewriter.getContext()));
      rewriter.create<ReturnOp>(loc, reducer.getResult());
    }

    rewriter.replaceOp(op, {result});

    return matchSuccess();
  }
};

// Converts hlo.Conv2DBackpropInputOp into:
//   %rev_filter = "xla_hlo.reverse"(%filter)
//   %result = "xla_hlo.conv"(%out_backprop, %rev_filter)
class ConvertConv2DBackpropInputOp
    : public OpRewritePattern<TF::Conv2DBackpropInputOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::Conv2DBackpropInputOp op,
                                     PatternRewriter &rewriter) const override {
    // Unpack all of the attributes.
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.data_format().str(), &data_format)) {
      return matchFailure();
    }
    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.padding().str(), &padding).ok())
      return Pattern::matchFailure();

    auto out_backprop_ty =
        op.out_backprop().getType().dyn_cast<RankedTensorType>();
    if (!out_backprop_ty || !out_backprop_ty.hasStaticShape())
      return matchFailure();
    ArrayRef<int64_t> out_backprop_shape = out_backprop_ty.getShape();
    auto filter_ty = op.filter().getType().dyn_cast<RankedTensorType>();
    if (!filter_ty || !filter_ty.hasStaticShape()) return matchFailure();
    ArrayRef<int64_t> filter_shape = filter_ty.getShape();
    int num_spatial_dims = 2;
    Location loc = op.getLoc();

    int num_dims = num_spatial_dims + 2;
    int batch_dim = tensorflow::GetTensorBatchDimIndex(num_dims, data_format);
    int feature_dim =
        tensorflow::GetTensorFeatureDimIndex(num_dims, data_format);

    DenseIntElementsAttr input_shape_attr;
    if (!matchPattern(op.input_sizes(), m_Constant(&input_shape_attr)) ||
        input_shape_attr.getType().getRank() != 1) {
      return matchFailure();
    }
    auto input_shape =
        llvm::to_vector<4>(input_shape_attr.getValues<int32_t>());
    if (input_shape.size() != num_dims) return matchFailure();

    auto batch_dim_attr = rewriter.getI64IntegerAttr(batch_dim);
    auto feature_dim_attr = rewriter.getI64IntegerAttr(feature_dim);

    auto strides_attr = GetI64ElementsAttr(op.strides());
    std::vector<tensorflow::int32> strides{
        strides_attr.getValues<int64_t>().begin(),
        strides_attr.getValues<int64_t>().end()};
    auto dilations_attr = GetI64ElementsAttr(op.dilations());
    std::vector<int> dilations{dilations_attr.getValues<int64_t>().begin(),
                               dilations_attr.getValues<int64_t>().end()};
    auto explicit_paddings_attr = GetI64ElementsAttr(op.explicit_paddings());
    std::vector<tensorflow::int64> explicit_paddings{
        explicit_paddings_attr.getValues<int64_t>().begin(),
        explicit_paddings_attr.getValues<int64_t>().end()};

    int64_t in_depth = input_shape[feature_dim];
    int64_t filter_in_depth = filter_shape[num_spatial_dims];
    int64_t feature_group_count = in_depth / filter_in_depth;

    // Reuse dimension computation logic from conv_grad_shape_utils.cc.
    tensorflow::ConvBackpropDimensions dims;
    if (!tensorflow::ConvBackpropComputeDimensionsV2(
             "", num_spatial_dims, ToTensorShape<int>(input_shape),
             ToTensorShape<int64_t>(filter_shape),
             ToTensorShape<int64_t>(out_backprop_shape), dilations, strides,
             padding, explicit_paddings, data_format, &dims)
             .ok()) {
      return matchFailure();
    }

    // Compute ConvDimensionNumbers, dilation, and padding.
    SmallVector<int64_t, 4> kernel_spatial_dims(num_spatial_dims);
    SmallVector<int64_t, 4> conv_paddings(num_spatial_dims * 2);
    SmallVector<int64_t, 4> lhs_dilation(num_spatial_dims);
    SmallVector<int64_t, 4> rhs_dilation(num_spatial_dims);
    SmallVector<int64_t, 4> ones(num_spatial_dims, 1);
    SmallVector<int64_t, 4> spatial_dims(num_spatial_dims);
    for (int i = 0; i < num_spatial_dims; ++i) {
      int64_t dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
      spatial_dims[i] = dim;
      kernel_spatial_dims[i] = i;

      conv_paddings[i * 2] = dims.spatial_dims[i].pad_before;
      conv_paddings[i * 2 + 1] = dims.spatial_dims[i].pad_after;
      lhs_dilation[i] = dims.spatial_dims[i].stride;
      rhs_dilation[i] = dilations[dim];
    }
    RankedTensorType paddings_ty = RankedTensorType::get(
        {num_spatial_dims, 2}, rewriter.getIntegerType(64));
    auto paddings_attr = DenseIntElementsAttr::get(paddings_ty, conv_paddings);
    auto spatial_dims_attr = GetI64ElementsAttr(spatial_dims, &rewriter);

    Value filter = op.filter();

    if (feature_group_count != 1) {
      /*
      // TODO(parkers): Convert this code to mlir.
    filter = TransposeFilterForGroupConvolutionBackpropInput(
        filter, filter_shape, feature_group_count, attrs.num_spatial_dims);
        */
      return matchFailure();
    }

    // Mirror the filter in the spatial dimensions.
    filter = rewriter.create<ReverseOp>(
        loc, filter, GetI64ElementsAttr(kernel_spatial_dims, &rewriter));

    // activation gradients
    //   = gradients (with padding and dilation) <conv> mirrored_weights
    Value result = rewriter.create<ConvOp>(
        loc, op.getType(), op.out_backprop(), filter,
        /*window_strides=*/GetI64ElementsAttr(ones, &rewriter),
        /*padding=*/paddings_attr, GetI64ElementsAttr(lhs_dilation, &rewriter),
        GetI64ElementsAttr(rhs_dilation, &rewriter),
        ConvDimensionNumbers::get(
            /*input_batch_dimension=*/batch_dim_attr,
            /*input_feature_dimension=*/feature_dim_attr,
            /*input_spatial_dimensions=*/spatial_dims_attr,
            // TF filter shape is [ H, W, ..., inC, outC ]
            // Transpose the input and output features for computing the
            // gradient.
            /*kernel_input_feature_dimension=*/
            rewriter.getI64IntegerAttr(num_spatial_dims + 1),
            /*kernel_output_feature_dimension=*/
            rewriter.getI64IntegerAttr(num_spatial_dims),
            /*kernel_spatial_dimensions=*/
            GetI64ElementsAttr(kernel_spatial_dims, &rewriter),
            /*output_batch_dimension=*/batch_dim_attr,
            /*output_feature_dimension=*/feature_dim_attr,
            /*output_spatial_dimensions=*/spatial_dims_attr,
            rewriter.getContext()),
        rewriter.getI64IntegerAttr(feature_group_count),
        /*batch_group_count=*/rewriter.getI64IntegerAttr(1),
        /*precision_config=*/ArrayAttr());

    rewriter.replaceOp(op, {result});

    return matchSuccess();
  }
};

// Converts tf.Conv2DBackpropFilterOp into:
//   %result = "xla_hlo.conv"(%input, %out_backprop)
class ConvertConv2DBackpropFilterOp
    : public OpRewritePattern<TF::Conv2DBackpropFilterOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::Conv2DBackpropFilterOp op,
                                     PatternRewriter &rewriter) const override {
    // Unpack all of the attributes.
    tensorflow::TensorFormat data_format;
    if (!FormatFromString(op.data_format().str(), &data_format)) {
      return matchFailure();
    }
    tensorflow::Padding padding;
    if (!GetPaddingFromString(op.padding().str(), &padding).ok())
      return Pattern::matchFailure();

    auto out_backprop_ty =
        op.out_backprop().getType().dyn_cast<RankedTensorType>();
    if (!out_backprop_ty || !out_backprop_ty.hasStaticShape())
      return matchFailure();
    ArrayRef<int64_t> out_backprop_shape = out_backprop_ty.getShape();
    auto input_ty = op.input().getType().dyn_cast<RankedTensorType>();
    if (!input_ty || !input_ty.hasStaticShape()) return matchFailure();
    ArrayRef<int64_t> input_shape = input_ty.getShape();

    DenseIntElementsAttr filter_shape_attr;
    if (!matchPattern(op.filter_sizes(), m_Constant(&filter_shape_attr)) ||
        filter_shape_attr.getType().getRank() != 1) {
      return matchFailure();
    }

    auto strides_attr = GetI64ElementsAttr(op.strides());
    std::vector<tensorflow::int32> strides{
        strides_attr.getValues<int64_t>().begin(),
        strides_attr.getValues<int64_t>().end()};
    auto dilations_attr = GetI64ElementsAttr(op.dilations());
    SmallVector<int, 4> dilations{dilations_attr.getValues<int64_t>().begin(),
                                  dilations_attr.getValues<int64_t>().end()};
    auto explicit_paddings_attr = GetI64ElementsAttr(op.explicit_paddings());
    SmallVector<tensorflow::int64, 4> explicit_paddings{
        explicit_paddings_attr.getValues<int64_t>().begin(),
        explicit_paddings_attr.getValues<int64_t>().end()};

    int num_spatial_dims = 2;
    int num_dims = num_spatial_dims + 2;
    int batch_dim = tensorflow::GetTensorBatchDimIndex(num_dims, data_format);
    int feature_dim =
        tensorflow::GetTensorFeatureDimIndex(num_dims, data_format);

    auto filter_shape =
        llvm::to_vector<4>(filter_shape_attr.getValues<int32_t>());
    if (filter_shape.size() != num_dims) return matchFailure();

    // Reuse dimension computation logic from conv_grad_shape_utils.cc.
    tensorflow::ConvBackpropDimensions dims;
    if (!tensorflow::ConvBackpropComputeDimensionsV2(
             "", num_spatial_dims, ToTensorShape<int64_t>(input_shape),
             ToTensorShape<int>(filter_shape),
             ToTensorShape<int64_t>(out_backprop_shape), dilations, strides,
             padding, explicit_paddings, data_format, &dims)
             .ok()) {
      return matchFailure();
    }

    // The activations (inputs) form the LHS of the convolution.
    // Activations have shape: [batch, in_rows, in_cols, ..., in_depth]
    // For the gradient computation, we need to:
    // 1. In the case of group convolution, move the num_groups dimension before
    // the batch dimension
    // 2. Swap the roles of the batch and feature dimensions.
    int64_t in_depth = input_shape[feature_dim];
    int64_t filter_in_depth = filter_shape[num_spatial_dims];
    int64_t feature_group_count = in_depth / filter_in_depth;
    if (feature_group_count != 1) {
      /*
          // TODO(parkers): translate this code to mlir.
          activations = TransposeInputForGroupConvolutionBackpropFilter(
              activations, input_shape, feature_group_count, batch_dim,
         feature_dim);
      */
      return matchFailure();
    }

    // Compute ConvDimensionNumbers, dilation, and padding.
    SmallVector<int64_t, 8> conv_padding(num_spatial_dims * 2);
    SmallVector<int64_t, 4> rhs_dilation(num_spatial_dims);
    SmallVector<int64_t, 4> window_strides(num_spatial_dims);
    SmallVector<int64_t, 4> lhs_dilation(num_spatial_dims, 1);
    SmallVector<int64_t, 4> spatial_dims(num_spatial_dims);
    SmallVector<int64_t, 4> kernel_spatial_dims(num_spatial_dims);

    // The filter gradients are computed by a convolution of the input
    // activations and the output gradients, with some appropriate padding.
    // See the comment at the top of conv_grad_ops.h for details.

    for (int64_t i = 0; i < num_spatial_dims; ++i) {
      int64_t dim =
          tensorflow::GetTensorSpatialDimIndex(num_dims, data_format, i);
      kernel_spatial_dims[i] = dim;
      // Besides padding the input, we will also expand output_rows to
      //    expanded_out_rows = (output_rows - 1) * stride + 1
      // with zeros in between:
      //
      //      a . . . b . . . c . . . d . . . e
      //
      // This is done by specifying the window dilation factors in the
      // convolution HLO below.
      rhs_dilation[i] = dims.spatial_dims[i].stride;
      window_strides[i] = dilations[dim];

      // We will also need to pad the input with zeros such that after the
      // convolution, we get the right size for the filter.
      // The padded_in_rows should be such that when we convolve this with the
      // expanded_out_rows as a filter, we should get filter_rows back.

      const int64_t padded_in_size =
          dims.spatial_dims[i].expanded_output_size +
          (dims.spatial_dims[i].filter_size - 1) * dilations[dim];

      // However it can be smaller than input_rows: in this
      // case it means some of the inputs are not used.
      //
      // An example is to have input_cols = 3, filter_cols = 2 and stride = 2:
      //
      // INPUT =  [ A  B  C ]
      //
      // FILTER = [ x y ]
      //
      // and the output will only have one column: a = A * x + B * y
      //
      // and input "C" is not used at all.
      //
      // We apply negative padding in this case.
      const int64_t pad_total =
          padded_in_size - dims.spatial_dims[i].input_size;

      // + For the EXPLICIT padding, we pad the top/left side with the explicit
      //   padding and pad the bottom/right side with the remaining space.
      // + For the VALID padding, we don't pad anything on the top/left side
      //   and pad the bottom/right side with the remaining space.
      // + For the SAME padding, we pad top/left side the same as bottom/right
      //   side.
      //
      // In addition, if the padded input size is smaller than the input size,
      // we need to ignore some training elements of the input. We do this by
      // applying negative padding on the right/bottom.
      const int64_t pad_before = padding == tensorflow::Padding::EXPLICIT
                                     ? explicit_paddings[2 * dim]
                                     : padding == tensorflow::Padding::SAME
                                           ? std::max<int64_t>(pad_total / 2, 0)
                                           : 0;
      conv_padding[i * 2] = pad_before;
      conv_padding[i * 2 + 1] = pad_total - pad_before;
    }

    RankedTensorType paddings_ty = RankedTensorType::get(
        {num_spatial_dims, 2}, rewriter.getIntegerType(64));
    auto paddings_attr = DenseIntElementsAttr::get(paddings_ty, conv_padding);
    auto out_spatial_dims_attr =
        GetI64ElementsAttrForSeq(0, num_spatial_dims, &rewriter);
    auto kernel_spatial_dims_attr =
        GetI64ElementsAttr(kernel_spatial_dims, &rewriter);

    auto batch_dim_attr = rewriter.getI64IntegerAttr(batch_dim);
    auto feature_dim_attr = rewriter.getI64IntegerAttr(feature_dim);

    Location loc = op.getLoc();
    Value result = rewriter.create<ConvOp>(
        loc, op.getType(), op.input(), op.out_backprop(),
        /*window_strides=*/GetI64ElementsAttr(window_strides, &rewriter),
        /*padding=*/paddings_attr, GetI64ElementsAttr(lhs_dilation, &rewriter),
        GetI64ElementsAttr(rhs_dilation, &rewriter),
        ConvDimensionNumbers::get(
            // Swap batch_dim and feature_dim in the activations.
            /*input_batch_dimension=*/feature_dim_attr,
            /*input_feature_dimension=*/batch_dim_attr,
            /*input_spatial_dimensions=*/kernel_spatial_dims_attr,
            // The gradients become the RHS of the convolution.
            // The gradients have shape [batch, out_rows, out_cols, ...,
            // out_depth] where the batch becomes the input feature for the
            // convolution.
            /*kernel_input_feature_dimension=*/batch_dim_attr,
            /*kernel_output_feature_dimension=*/feature_dim_attr,
            /*kernel_spatial_dimensions=*/kernel_spatial_dims_attr,
            /*output_batch_dimension=*/
            rewriter.getI64IntegerAttr(num_spatial_dims),
            /*output_feature_dimension=*/
            rewriter.getI64IntegerAttr(num_spatial_dims + 1),
            /*output_spatial_dimensions=*/out_spatial_dims_attr,
            rewriter.getContext()),
        rewriter.getI64IntegerAttr(feature_group_count),
        /*batch_group_count=*/rewriter.getI64IntegerAttr(1),
        /*precision_config=*/ArrayAttr());

    rewriter.replaceOp(op, {result});

    return matchSuccess();
  }
};

class ConvertOneHotOp : public OpRewritePattern<TF::OneHotOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::OneHotOp op,
                                     PatternRewriter &rewriter) const override {
    auto indices_ty = op.indices().getType().dyn_cast<RankedTensorType>();
    if (!indices_ty || !indices_ty.hasStaticShape()) return matchFailure();
    ArrayRef<int64_t> indices_shape = indices_ty.getShape();
    Type element_type = indices_ty.getElementType();

    DenseIntElementsAttr depth_attr;
    if (!matchPattern(op.depth(), m_Constant(&depth_attr))) {
      return matchFailure();
    }

    int64_t depth = depth_attr.getValue<APInt>({}).getSExtValue();
    int64_t axis = op.axis().getSExtValue();
    if (axis == -1) axis = indices_shape.size();

    llvm::SmallVector<int64_t, 4> broadcast_dims(indices_shape.size());
    std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
    std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);

    llvm::SmallVector<int64_t, 4> output_dims =
        llvm::to_vector<4>(indices_shape);
    output_dims.insert(output_dims.begin() + axis, depth);

    Location loc = op.getLoc();
    auto index_type = RankedTensorType::get(output_dims, element_type);
    Value compare = rewriter.create<CompareOp>(
        loc, op.indices(),
        rewriter.create<IotaOp>(
            loc, index_type,
            IntegerAttr::get(rewriter.getIntegerType(64), axis)),
        GetI64ElementsAttr(broadcast_dims, &rewriter),
        StringAttr::get("EQ", rewriter.getContext()));
    Value on_value = rewriter.create<BroadcastOp>(
        loc, op.getType(), op.on_value(),
        GetI64ElementsAttr(output_dims, &rewriter));
    Value off_value = rewriter.create<BroadcastOp>(
        loc, op.getType(), op.off_value(),
        GetI64ElementsAttr(output_dims, &rewriter));
    Value result = rewriter.create<SelectOp>(loc, op.getType(), compare,
                                             on_value, off_value);

    rewriter.replaceOp(op, {result});

    return matchSuccess();
  }
};

// Converts InfeedEnqueueTuple to XLA HLO after_all, infeed and
// get_tuple_element ops.
//
// All HLO infeed ops expect a HLO token type operand and produce a tuple
// containing a token. This HLO token type is used to order multiple infeed
// operations within a computation. The token type can come from other
// infeed/outfeed/send/recv ops or can be generated using an after_all op with
// no operands. Here we emit an after_all op to generate the token type operand
// of infeed.
//
// For example the following IR:
// %0:2 = "tf.InfeedDequeueTuple"() : () -> (tensor<3xi32>, tensor<4xf32>)
//
// would be lowered to
//
// %token = "xla_hlo.after_all"() : () -> !xla_hlo.token
// %data_and_token = "xla_hlo.infeed"(%token) {infeed_config = ""} :
//      (!xla_hlo.token) -> tuple<tuple<tensor<3xi32>, tensor<4xf32>>,
//      !xla_hlo.token>
// %data = "xla_hlo.get_tuple_element"(%data_and_token) {index = 0}
// %0#0 = "xla_hlo.get_tuple_element"(%data) {index = 0}
// %0#1 = "xla_hlo.get_tuple_element"(%data) {index = 1}
//
class ConvertInfeedDequeueTupleOp
    : public OpRewritePattern<TF::InfeedDequeueTupleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::InfeedDequeueTupleOp op,
                                     PatternRewriter &rewriter) const override {
    std::vector<Type> result_types(op.outputs().size());
    for (auto idx_and_output : llvm::enumerate(op.outputs())) {
      result_types[idx_and_output.index()] = (idx_and_output.value().getType());
    }
    // Infeed takes a single token operand. Generate the token using after_all
    // op to pass to the infeed op.
    auto afterall = rewriter.create<AfterAllOp>(
        op.getLoc(), xla_hlo::TokenType::get(rewriter.getContext()),
        ValueRange());

    // Emit infeed op.
    // The result type of infeed is a tuple(tuple(result types), token type).
    auto data_tuple_type =
        mlir::TupleType::get(result_types, rewriter.getContext());
    auto data_and_token_type = mlir::TupleType::get(
        {data_tuple_type, afterall.getType()}, rewriter.getContext());

    auto data_and_token =
        rewriter.create<InfeedOp>(op.getLoc(), data_and_token_type, afterall,
                                  /*infeed_config=*/rewriter.getStringAttr(""));

    // The infeed instruction produces a tuple of the infeed data and a token
    // type. Emit get_tuple_element to get infeed data tuple.
    auto data_tuple = rewriter.create<GetTupleElementOp>(
        op.getLoc(), data_tuple_type, data_and_token,
        rewriter.getI32IntegerAttr(0));

    // Emit get_tuple_element for each result.
    std::vector<Value> results;
    for (auto idx_and_type : llvm::enumerate(result_types)) {
      auto tuple_element = rewriter.create<GetTupleElementOp>(
          op.getLoc(), idx_and_type.value(), data_tuple,
          rewriter.getI32IntegerAttr(idx_and_type.index()));
      results.push_back(tuple_element);
    }
    rewriter.replaceOp(op, ValueRange(results));
    return matchSuccess();
  }
};

// Converts tf.OutfeedEnqueueTuple to XLA HLO tuple, after_all and outfeed ops.
//
// XLA HLO outfeed op expects a token, which we generate by emitting an
// after_all op.
//
// For example the following IR:
// "tf.OutfeedEnqueueTuple"(%val_1, %val_2) : (tensor<3xi32>, tensor<4xf32>) ->
//      ()
//
// would be lowered to
//
// %tuple = "xla_hlo.tuple"(%val_1, %val_2) : (tensor<3xi32>, tensor<4xf32>) ->
//      tuple<tensor<3xi32>, tensor<4xf32>>
// %token = "xla_hlo.after_all"() : () -> !xla_hlo.token
// %outfeed_token = "xla_hlo.outfeed"(%tuple, %token) {outfeed_config = ""} :
//      (tuple<tensor<3xi32>, tensor<4xf32>>, !xla_hlo.token) -> !xla_hlo.token
//
class ConvertOutfeedEnqueueTupleOp
    : public OpRewritePattern<TF::OutfeedEnqueueTupleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::OutfeedEnqueueTupleOp op,
                                     PatternRewriter &rewriter) const override {
    auto token_type = xla_hlo::TokenType::get(rewriter.getContext());
    auto tuple = rewriter.create<TupleOp>(op.getLoc(), op.inputs());
    auto afterall =
        rewriter.create<AfterAllOp>(op.getLoc(), token_type, ValueRange());
    rewriter.create<OutfeedOp>(op.getLoc(), token_type, tuple, afterall,
                               /*outfeed_config=*/rewriter.getStringAttr(""));
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

// Converts tf.TopKV2 to XLA HLO iota, sort, and slice ops when k is a constant.
//
// tf.TopKV2 sorts along last dimension of the input tensor and then returns
// the top K components' values and indices. This is translated into a few
// ops in XLA HLO: first generating an integer sequence for the indices,
// then sort both the original input tensor and the indices togheter, and
// at last slice out the top K components.
//
// For example, for the following IR:
//
// %k = "tf.Const"() {value = dense<8> : tensor<i32>} : () -> tensor<i32>
// %0:2 = "tf.TopKV2"(%input, %k): (tensor<16x16xf32>, tensor<i32>) ->
//                                 (tensor<16x8xf32>, tensor<16x8xi32>)
//
// We will get:
//
// %1 = "xla_hlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<16x16xi32>
// %2 = "xla_hlo.sort"(%input, %1) ( {
// ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>,
//      %arg3: tensor<i32>, %arg4: tensor<i32>):
//   %7 = "xla_hlo.compare"(%arg1, %arg2) {comparison_direction = "GT"}: ...
//   "xla_hlo.return"(%7) : (tensor<i1>) -> ()
// }) {dimension = 1 : i64, is_stable = true} : ...
// %3 = "xla_hlo.get_tuple_element"(%2) {index = 0 : i32} : ...
// %4 = "xla_hlo.get_tuple_element"(%2) {index = 1 : i32} : ...
// %5 = "xla_hlo.slice"(%3) {limit_indices = dense<[16, 8]> : tensor<2xi64>,
//                           start_indices dense<0> : tensor<2xi64>,
//                           strides = dense<1> : tensor<2xi64>} :
//                              (tensor<16x16xf32>) -> tensor<16x8xf32>
// %6 = "xla_hlo.slice"(%4) ...
class ConvertTopKV2Op : public OpRewritePattern<TF::TopKV2Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::TopKV2Op op,
                                     PatternRewriter &rewriter) const override {
    // We can only match when the `k` operand is a constant scalar.
    DenseIntElementsAttr k_attr;
    if (!matchPattern(op.k(), m_Constant(&k_attr))) return matchFailure();

    // The last dimension of the input tensor's shape should be known so we can
    // have clamped end_indices for slices.
    TensorType input_type = op.input().getType().cast<TensorType>();
    if (!input_type.hasRank()) return matchFailure();
    int64_t input_rank = input_type.getRank();
    int64_t last_dim_index = input_rank - 1;
    int64_t last_dim_size = input_type.getDimSize(last_dim_index);
    if (last_dim_size == ShapedType::kDynamicSize) return matchFailure();

    // Create an Itoa op for indices.
    auto i32_type = rewriter.getIntegerType(32);
    Type iota_type = RankedTensorType::get(input_type.getShape(), i32_type);
    Value iota_op = rewriter.create<xla_hlo::IotaOp>(
        op.getLoc(), iota_type, rewriter.getI64IntegerAttr(last_dim_index));

    // Create the sort op. It takes two inputs, one for the original input, the
    // other for the indices.
    auto sort_op = rewriter.create<xla_hlo::SortOp>(
        op.getLoc(), llvm::ArrayRef<Value>{op.input(), iota_op}, last_dim_index,
        /*is_stable=*/true);
    BuildSortComparisonBody({input_type.getElementType(), i32_type},
                            /*direction=*/"GT", &sort_op.comparator(),
                            &rewriter);

    // Get the sorted input and index tuple element.
    auto tuple_first_element =
        rewriter.create<xla_hlo::GetTupleElementOp>(op.getLoc(), sort_op, 0);
    auto tuple_second_element =
        rewriter.create<xla_hlo::GetTupleElementOp>(op.getLoc(), sort_op, 1);

    SmallVector<int64_t, 4> begin_indices(input_rank, 0);
    auto end_indices = llvm::to_vector<4>(input_type.getShape());
    end_indices.back() =
        std::min((*k_attr.begin()).getSExtValue(), last_dim_size);
    SmallVector<int64_t, 4> strides(input_rank, 1);

    // Get the slice for the top K elements.

    Value values = rewriter.create<xla_hlo::SliceOp>(
        op.getLoc(), tuple_first_element,
        GetI64ElementsAttr(begin_indices, &rewriter),
        GetI64ElementsAttr(end_indices, &rewriter),
        GetI64ElementsAttr(strides, &rewriter));

    Value indices = rewriter.create<xla_hlo::SliceOp>(
        op.getLoc(), tuple_second_element,
        GetI64ElementsAttr(begin_indices, &rewriter),
        GetI64ElementsAttr(end_indices, &rewriter),
        GetI64ElementsAttr(strides, &rewriter));

    rewriter.replaceOp(op, {values, indices});
    return matchSuccess();
  }
};

// Converts tf.Unpack to a series of XLA HLO slice ops.
//
// Each slice takes one element along the dimension to unpack and takes the full
// range for all other dimensions. Each slice is then reshaped to drop the
// dimension to unpack (which is always of size 1).
// TODO(antiagainst): consider changing this into a TF internal lowering pass.
class ConvertUnpackOp : public OpRewritePattern<TF::UnpackOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::UnpackOp op,
                                     PatternRewriter &rewriter) const override {
    auto value_type = op.value().getType().cast<RankedTensorType>();
    if (!value_type) return matchFailure();

    int64_t value_rank = value_type.getRank();
    int64_t axis = op.axis().getSExtValue();
    if (axis < 0) axis += value_rank;

    // Parameters for constructing each slice.
    SmallVector<int64_t, 4> begin_indices(value_rank, 0);
    auto end_indices = llvm::to_vector<4>(value_type.getShape());
    SmallVector<int64_t, 4> strides(value_rank, 1);

    // All HLO slice+reshape results used to replace the original tf.Unpack op.
    SmallVector<Value, 4> results;
    results.reserve(op.getNumResults());

    for (int i = 0; i < op.getNumResults(); ++i) {
      begin_indices[axis] = i;
      end_indices[axis] = i + 1;

      auto slice_op = rewriter.create<xla_hlo::SliceOp>(
          op.getLoc(), op.value(), GetI64ElementsAttr(begin_indices, &rewriter),
          GetI64ElementsAttr(end_indices, &rewriter),
          GetI64ElementsAttr(strides, &rewriter));
      // Reshape to drop the axis dimension.
      auto reshape_op = rewriter.create<xla_hlo::ReshapeOp>(
          op.getLoc(), op.getType(i), slice_op);
      results.push_back(reshape_op);
    }

    rewriter.replaceOp(op, results);
    return matchSuccess();
  }
};

// Converts TF unsorted segment reduction ops to XLA HLO scatter op.
//
// TF unsorted segment reduction op peforms the following calculation:
//
// Assume segment ids' shape is [SI0, SI1, ..., SIm] and data's  shape is
// [D0, D1, ..., Dn]. Note that segment ids' shape must be a prefix of data's
// shape, so we can have data's shape represented as [SI0, SI1, ..., SIm,
// Dm+1, ..., Dn]. Then
//   output[segment_ids[SI_i0, SI_i1, ..., SI_im], D_im+1, ..., D_in] =
//      <ReductionOp> over data[SI_i0, SI_i1, ..., SI_im, D_im+1, ..., D_in]
// where SI_iN is in the range of [0, SIN) and D_iN is in the range of [0, DN).
//
// The op will be translated to XLA HLO scatter with the following parameters:
// * Update window dims is [segment_id_rank, data_rank).
// * Inserted window dims is {0}.
// * Scatter dims to operand dims mapping is {0}.
// * Index vector dim is segment_id_rank.
template <typename ConcreteClass, typename OpTy, typename ReductionOp>
class GenericConvertUnsortedSegmentReductionOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(OpTy op,
                                     PatternRewriter &rewriter) const override {
    auto data_type = op.data().getType().template dyn_cast<RankedTensorType>();
    if (!data_type) return this->matchFailure();
    int64_t data_rank = data_type.getRank();

    auto segment_ids_type =
        op.segment_ids().getType().template dyn_cast<RankedTensorType>();
    if (!segment_ids_type) return this->matchFailure();
    int64_t segment_ids_rank = segment_ids_type.getRank();

    DenseIntElementsAttr num_segments_attr;
    if (!matchPattern(op.num_segments(), m_Constant(&num_segments_attr)))
      return this->matchFailure();

    // The final shape for TF unsorted segment reduction op is [num_segments] +
    // data_shape[segment_ids_rank:].
    SmallVector<int64_t, 4> output_shape;
    output_shape.push_back((*num_segments_attr.begin()).getSExtValue());
    auto suffix = data_type.getShape().drop_front(segment_ids_rank);
    output_shape.append(suffix.begin(), suffix.end());
    auto output_type =
        RankedTensorType::get(output_shape, data_type.getElementType());

    // Broadccast the initial value for reduction. This will become the
    // 'operand' parameter to scatter to for the final scatter op.
    Value init = ConcreteClass::GetInitialValue(data_type.getElementType(),
                                                op.getLoc(), &rewriter);
    auto broadcasted_init = rewriter.create<xla_hlo::BroadcastOp>(
        op.getLoc(), output_type, init,
        GetI64ElementsAttr(output_shape, &rewriter));

    // Parameters for the generated scatter op.
    SmallVector<int64_t, 1> inserted_window_dims(1, 0);
    SmallVector<int64_t, 1> scatter_dims_to_operand_dims(1, 0);
    int64_t index_vector_dim = segment_ids_rank;

    // Put all parameters in a StructAttr.
    auto dims_attr = ScatterDimensionNumbers::get(
        GetI64ElementsAttrForSeq(segment_ids_rank, data_rank, &rewriter),
        GetI64ElementsAttr(inserted_window_dims, &rewriter),
        GetI64ElementsAttr(scatter_dims_to_operand_dims, &rewriter),
        rewriter.getI64IntegerAttr(index_vector_dim), rewriter.getContext());

    auto scatter =
        rewriter.create<ScatterOp>(op.getLoc(), op.getType(), broadcasted_init,
                                   op.segment_ids(), op.data(), dims_attr);
    BuildReduceBody<ReductionOp>(data_type.getElementType(),
                                 &scatter.update_computation(), &rewriter);

    rewriter.replaceOp(op, scatter.getResult());
    return this->matchSuccess();
  }
};

class ConvertUnsortedSegmentMaxOp
    : public GenericConvertUnsortedSegmentReductionOp<
          ConvertUnsortedSegmentMaxOp, TF::UnsortedSegmentMaxOp, MaxOp> {
 public:
  using GenericConvertUnsortedSegmentReductionOp::
      GenericConvertUnsortedSegmentReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetMinValueForType(reduce_element_type, loc, rewriter);
  }
};

class ConvertUnsortedSegmentMinOp
    : public GenericConvertUnsortedSegmentReductionOp<
          ConvertUnsortedSegmentMinOp, TF::UnsortedSegmentMinOp, MinOp> {
 public:
  using GenericConvertUnsortedSegmentReductionOp::
      GenericConvertUnsortedSegmentReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetMaxValueForType(reduce_element_type, loc, rewriter);
  }
};

class ConvertUnsortedSegmentProdOp
    : public GenericConvertUnsortedSegmentReductionOp<
          ConvertUnsortedSegmentProdOp, TF::UnsortedSegmentProdOp, MulOp> {
 public:
  using GenericConvertUnsortedSegmentReductionOp::
      GenericConvertUnsortedSegmentReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 1, rewriter);
  }
};

class ConvertUnsortedSegmentSumOp
    : public GenericConvertUnsortedSegmentReductionOp<
          ConvertUnsortedSegmentSumOp, TF::UnsortedSegmentSumOp, AddOp> {
 public:
  using GenericConvertUnsortedSegmentReductionOp::
      GenericConvertUnsortedSegmentReductionOp;

  static Value GetInitialValue(Type reduce_element_type, Location loc,
                               PatternRewriter *rewriter) {
    return GetScalarConstOfType(reduce_element_type, loc, 0, rewriter);
  }
};

// Converts tf.RandomShuffle op into a series of XLA HLO ops.
//
// tf.RandomShuffle shuffles tensors along the first dimension. If the input
// tensor's rank is 1, then it is translated into HLO sort op(s) according to
// indices randomly generated via HLO rng_uniform ops. Otherwise, it is
// translated into an HLO while op to first emulate shuffling indices using
// HLO dynamic_slice and dynamic_update_slice ops, then finally HLO gather
// with the shuffled indices.
class ConvertRandomShuffleOp : public OpRewritePattern<TF::RandomShuffleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::RandomShuffleOp op,
                                     PatternRewriter &rewriter) const override {
    auto input_type = op.value().getType().dyn_cast<RankedTensorType>();
    if (!input_type) return matchFailure();

    int64_t input_rank = input_type.getRank();
    int64_t first_dim_size = input_type.getDimSize(0);
    if (ShapedType::isDynamic(first_dim_size)) return matchFailure();

    // We are shuffling along the first dimension. If its size is <= 1, then
    // shuffling is a no-op.
    if (first_dim_size <= 1) {
      rewriter.replaceOp(op, op.value());
      return matchSuccess();
    }

    // For vectors, shuffle values by sorting instead of the obvious
    // Fisher-Yates algorithm. Fisher-Yates is simple to implement and correct,
    // but not easily parallelizable. For a sufficiently parallel architecture,
    // it is faster to sort many times, than Fisher-Yates shuffle once.
    if (input_rank == 1) {
      // Shuffle values by assigning each value a random key and sorting the
      // keys. Keys can collide causing detectable patterns in the shuffled
      // output. Collisions translates into more ascending sub-sequences in the
      // shuffled output than would be expected by chance. To avoid collisions,
      // the number of possible key values must be sufficiently large.

      // How are more than 2^32 keys created? In each loop iteration, the
      // algorithm sorts by random keys. Conceptually, the earlier iterations
      // are sorting on the lower-order bits of larger keys that are never
      // actually assembled.

      // The expected number of collisions is n - d + d(1 - 1/d)^n, where d is
      // the number of possible keys and n is the number of values. If d = n^2,
      // then the limit as n goes to infinity is 1/2. If d = n^3, then the limit
      // as n goes to infinity is zero.

      // This implementation ensures that the key-space is greater than or equal
      // to the cube of the number of values. The risk of collisions can be
      // further reduced by increasing Exponent at the expense of
      // performance.

      // For Exponent = 2, the expected number of collisions per shuffle is
      // maximized at n = floor((2^32-1)^(1/2)) = 65535 where the expectation is
      // about 1/2.

      // For Exponent = 3, the expected number of collisions per shuffle is
      // maximized at n = floor((2^32-1)^(1/3)) = 1625 where the expectation is
      // about 1/3255.

      // For Exponent = 4, the expected number of collisions per shuffle is
      // maximized at n = floor((2^32-1)^(1/4)) = 255 where the expectation is
      // about 1/132622.
      constexpr int exponent = 3;
      int64_t num_elements = input_type.getNumElements();
      uint32_t u32_max = std::numeric_limits<uint32_t>::max();
      int rounds =
          std::ceil(exponent * std::log(num_elements) / std::log(u32_max));

      Value current = op.value();
      for (int i = 0; i < rounds; ++i) {
        auto keys =
            CreateRngUniform32(op.getLoc(), num_elements, /*lower_limit=*/0,
                               /*upper_limit=*/u32_max, &rewriter);
        auto sorted = rewriter.create<xla_hlo::SortOp>(
            op.getLoc(), llvm::ArrayRef<Value>{keys, current});
        auto i32_type = rewriter.getIntegerType(32);
        BuildSortComparisonBody({i32_type, input_type.getElementType()},
                                /*direction=*/"LT", &sorted.comparator(),
                                &rewriter);
        current = rewriter.create<GetTupleElementOp>(op.getLoc(),
                                                     sorted.getResult(), 1);
      }
      rewriter.replaceOp(op, current);
      return matchSuccess();
    }

    // The Fisher-Yates algorithm.

    // Generate range(n) as the initial value for the indices to be swapped.
    auto indices_type =
        RankedTensorType::get({first_dim_size}, rewriter.getIntegerType(32));
    Value indices = rewriter.create<xla_hlo::IotaOp>(
        op.getLoc(), indices_type, rewriter.getI64IntegerAttr(first_dim_size));

    // Generate random numbers to be used as swaps for the indices.
    Value swaps = CreateRngUniform32(op.getLoc(), first_dim_size, 0,
                                     first_dim_size, &rewriter);

    // While loop body to perform index swaps.
    auto swap_body_fn = [&](Location loc, Value i, ArrayRef<Value> old_values,
                            SmallVectorImpl<Value> *new_values,
                            OpBuilder *builder) {
      Value swaps = old_values[0];
      Value indices = old_values[1];

      auto vec1_i32_type =
          RankedTensorType::get({1}, builder->getIntegerType(32));
      auto scalar_i32_type =
          RankedTensorType::get({}, builder->getIntegerType(32));
      auto scalar_i64_type =
          RankedTensorType::get({}, builder->getIntegerType(64));

      auto scalar_one =
          DenseIntElementsAttr::get(scalar_i64_type, ArrayRef<int64_t>(1));

      // We need to swap the indices[i] with indices[swaps[i]]. First get
      // these index values.
      Value source_index = builder->create<xla_hlo::DynamicSliceOp>(
          loc, vec1_i32_type, indices, i, scalar_one);
      Value swap_index = builder->create<xla_hlo::ReshapeOp>(
          loc, scalar_i32_type,
          builder->create<xla_hlo::DynamicSliceOp>(loc, vec1_i32_type, swaps, i,
                                                   scalar_one));
      Value target_index = builder->create<xla_hlo::DynamicSliceOp>(
          loc, vec1_i32_type, indices, swap_index, scalar_one);

      // Then perform the swap.
      // indices[i] <- indices[swaps[i]]
      indices = builder->create<xla_hlo::DynamicUpdateSliceOp>(
          loc, indices.getType(), indices, target_index, llvm::makeArrayRef(i));
      // indices[swaps[i]] <- indices[i]
      indices = builder->create<xla_hlo::DynamicUpdateSliceOp>(
          loc, indices.getType(), indices, source_index,
          llvm::makeArrayRef(swap_index));

      // Update new values.
      new_values->assign({swaps, indices});
    };

    // Create a while op to swap indices.
    SmallVector<Value, 2> while_output;
    CreateWhile32(op.getLoc(), first_dim_size, swap_body_fn, {swaps, indices},
                  &while_output, &rewriter);
    Value swaped_indices = while_output[1];

    // Gather the data using the swapped indices as the shuffled order.
    ArrayRef<int64_t> input_shape = input_type.getShape();
    SmallVector<int64_t, 4> slice_sizes(input_shape.begin(), input_shape.end());
    slice_sizes[0] = 1;
    auto dims_attr = GatherDimensionNumbers::get(
        /*offset_dims=*/GetI64ElementsAttrForSeq(1, first_dim_size, &rewriter),
        /*collapsed_slice_dims=*/GetI64ElementsAttr({0}, &rewriter),
        /*start_index_map=*/GetI64ElementsAttr({0}, &rewriter),
        /*index_vector_dim=*/rewriter.getI64IntegerAttr(1),
        rewriter.getContext());
    rewriter.replaceOpWithNewOp<xla_hlo::GatherOp>(
        op, op.getType(), op.value(), swaped_indices, dims_attr,
        GetI64ElementsAttr(slice_sizes, &rewriter));

    return matchSuccess();
  }
};

// Converts tf.VariableShape op to a XLA HLO constant representing the variable
// shape.
class ConvertVariableShapeOp : public OpRewritePattern<TF::VariableShapeOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::VariableShapeOp op,
                                     PatternRewriter &rewriter) const override {
    // The input type should be a tensor<!tf.resource<resource-type>>. We need
    // to get the inner resource type.
    auto input_type = op.input().getType().cast<TensorType>();
    auto subtypes =
        input_type.getElementType().cast<TF::ResourceType>().getSubtypes();
    // It can be missing; then we cannot convert.
    if (subtypes.empty()) return matchFailure();

    auto resource_type = subtypes[0].cast<TensorType>();
    if (!resource_type.hasStaticShape()) return matchFailure();

    auto resource_shape = resource_type.getShape();
    Attribute const_attr;

    // We need to match the original op result's element type.
    auto element_type = op.getType().cast<TensorType>().getElementType();
    unsigned bitwidth = element_type.cast<IntegerType>().getWidth();
    if (bitwidth == 32) {
      SmallVector<int32_t, 4> shape(resource_shape.begin(),
                                    resource_shape.end());
      const_attr = GetI32ElementsAttr(shape, &rewriter);
    } else {
      assert(bitwidth == 64);
      const_attr = GetI64ElementsAttr(resource_shape, &rewriter);
    }

    rewriter.replaceOpWithNewOp<xla_hlo::ConstOp>(op, const_attr);
    return matchSuccess();
  }
};

// Converts an XlaSharding op to a XLA HLO shard op with sharding attributes.
class ConvertXlaShardingOp : public OpRewritePattern<TF::XlaShardingOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::XlaShardingOp op,
                                     PatternRewriter &rewriter) const override {
    // TODO(b/148313088): define sharding attribute struct in MLIR intead of
    // using a string.
    auto sharding = op.getAttrOfType<StringAttr>("_XlaSharding");
    if (!sharding) {
      return matchFailure();
    }

    // _XlaSharding attribute in TF is a serialized string of the OpSharding
    // proto, so convert to a text form here.
    ::xla::OpSharding sharding_proto;
    std::string sharding_str;
    if (!sharding_proto.ParseFromString(sharding.getValue().str())) {
      return matchFailure();
    }
    if (!::tensorflow::protobuf::TextFormat::PrintToString(sharding_proto,
                                                           &sharding_str)) {
      return matchFailure();
    }

    auto custom_call = rewriter.create<xla_hlo::CustomCallOp>(
        op.getLoc(), op.getType(), op.input(),
        /*call_target_name=*/rewriter.getStringAttr("Sharding"),
        /*has_side_effect=*/rewriter.getBoolAttr(false),
        /*backend_config=*/rewriter.getStringAttr(""));
    custom_call.setAttr("xla_hlo.sharding",
                        rewriter.getStringAttr(sharding_str));
    rewriter.replaceOp(op, custom_call.getResult());

    return matchSuccess();
  }
};

// Converts a TF XlaDynamicUpdateSlice op to DynamicUpdateSlice HLO.
class ConvertXlaDynamicUpdateSliceOp
    : public OpRewritePattern<TF::XlaDynamicUpdateSliceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  PatternMatchResult matchAndRewrite(TF::XlaDynamicUpdateSliceOp op,
                                     PatternRewriter &rewriter) const override {
    auto indices_type = op.indices().getType().dyn_cast<RankedTensorType>();
    if (!indices_type) return matchFailure();

    SmallVector<Type, 2> unpacked_indices_type(
        2, RankedTensorType::get({}, indices_type.getElementType()));
    auto unpacked_indices = rewriter.create<TF::UnpackOp>(
        op.getLoc(), unpacked_indices_type, op.indices(),
        IntegerAttr::get(rewriter.getIntegerType(64), 0));
    rewriter.replaceOpWithNewOp<xla_hlo::DynamicUpdateSliceOp>(
        op, op.getType(), op.input(), op.update(), unpacked_indices.output());
    return matchSuccess();
  }
};

#include "tensorflow/compiler/mlir/xla/transforms/generated_legalize_tf.inc"

LogicalResult legalizeTF(Operation *op, bool allow_partial_conversion) {
  MLIRContext *context = op->getContext();

  // Add lowering patterns to the list.
  OwningRewritePatternList patterns;
  populateWithGenerated(context, &patterns);

  // Add patterns that lower some of the high level TensorFlow ops to lower
  // level TensorFlow ops. So, we don't have to target all the TensorFlow ops
  // here for lowering to HLO.
  TF::PopulateLoweringTFPatterns(context, &patterns);
  patterns.insert<
      ConvertAllOp, ConvertAnyOp, ConvertArgMaxOp, ConvertBF16FloorDivOp,
      ConvertConv2D, ConvertConv2DBackpropFilterOp,
      ConvertConv2DBackpropInputOp, ConvertEinsumOp,
      ConvertFusedBatchNormGradOp, ConvertFusedBatchNormGradV2Op,
      ConvertFusedBatchNormGradV3Op, ConvertFusedBatchNormV3Op,
      ConvertInfeedDequeueTupleOp, ConvertLinSpaceOp, ConvertMaxOp,
      ConvertMinOp, ConvertAvgPoolOp, ConvertMaxPoolOp, ConvertMaxPoolGradOp,
      ConvertMeanOp, ConvertOneHotOp, ConvertOutfeedEnqueueTupleOp,
      ConvertProdOp, ConvertRangeOp, ConvertSelectV2Op, ConvertSigmoidOp,
      ConvertSizeOp, ConvertSoftmaxOp<TF::LogSoftmaxOp, true>,
      ConvertSoftmaxOp<TF::SoftmaxOp, false>, ConvertSplitOp, ConvertSplitVOp,
      ConvertStridedSliceOp, ConvertStridedSliceGradOp, ConvertSumOp,
      ConvertTensorScatterUpdateOp, ConvertTileOp, ConvertTopKV2Op,
      ConvertUnpackOp, ConvertUnsortedSegmentMaxOp, ConvertUnsortedSegmentMinOp,
      ConvertUnsortedSegmentProdOp, ConvertUnsortedSegmentSumOp,
      ConvertRandomShuffleOp, ConvertVariableShapeOp, ConvertXlaShardingOp,
      ConvertXlaDynamicUpdateSliceOp>(op->getContext());

  ConversionTarget target(*context);
  target.addLegalDialect<XlaHloDialect>();
  target.addLegalOp<CallOp>();

  if (!allow_partial_conversion) {
    // Fully qualify ReturnOp here as xla_hlo dialect also defines a ReturnOp.
    target.addLegalOp<ModuleOp, FuncOp, ModuleTerminatorOp, ::mlir::ReturnOp>();
    return applyFullConversion(op, target, patterns);
  }

  return applyPartialConversion(op, target, patterns);
}

/// Performs the lowering to XLA dialect.
void LegalizeTF::runOnFunction() {
  if (failed(legalizeTF(getFunction(), allow_partial_conversion_)))
    signalPassFailure();
}

static PassRegistration<LegalizeTF> pass(
    "xla-legalize-tf", "Legalize from TensorFlow to the XLA dialect");

}  // end namespace

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeTFPass(
    bool allow_partial_conversion) {
  return std::make_unique<LegalizeTF>(allow_partial_conversion);
}

}  // end namespace xla_hlo
}  // end namespace mlir
