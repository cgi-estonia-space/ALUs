/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* MLIR XLA Builders                                                          *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

mlir::LogicalResult ExportXlaOp(mlir::mhlo::AbsOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Abs(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::AddOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Add(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::AfterAllOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  std::vector<xla::XlaOp> xla_arg_0;
  for (auto operand : op.getODSOperands(0)) {
    xla::XlaOp result;
    if (failed(GetXlaOp(operand, value_map, &result, op)))
      return mlir::failure();
    xla_arg_0.push_back(result);
  }
  auto xla_result = xla::AfterAll(ctx.builder, Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::AllToAllOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convertuint64_t(op.split_dimension());
  auto xla_arg_2 = Convertuint64_t(op.concat_dimension());
  auto xla_arg_3 = Convertuint64_t(op.split_count());
  auto xla_arg_4 = Convert_replica_groups(op.replica_groups());
  auto xla_result = xla::AllToAll(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3), Unwrap(xla_arg_4));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::AndOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::And(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::Atan2Op op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Atan2(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::BatchNormGradOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_2;
  if (failed(GetXlaOp(*op.getODSOperands(2).begin(), value_map, &xla_arg_2, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_3;
  if (failed(GetXlaOp(*op.getODSOperands(3).begin(), value_map, &xla_arg_3, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_4;
  if (failed(GetXlaOp(*op.getODSOperands(4).begin(), value_map, &xla_arg_4, op)))
    return mlir::failure();
  auto xla_arg_5 = ConvertAPFloat(op.epsilon());
  auto xla_arg_6 = Convertuint64_t(op.feature_index());
  auto xla_result = xla::BatchNormGrad(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3), Unwrap(xla_arg_4), Unwrap(xla_arg_5), Unwrap(xla_arg_6));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::BatchNormInferenceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_2;
  if (failed(GetXlaOp(*op.getODSOperands(2).begin(), value_map, &xla_arg_2, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_3;
  if (failed(GetXlaOp(*op.getODSOperands(3).begin(), value_map, &xla_arg_3, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_4;
  if (failed(GetXlaOp(*op.getODSOperands(4).begin(), value_map, &xla_arg_4, op)))
    return mlir::failure();
  auto xla_arg_5 = ConvertAPFloat(op.epsilon());
  auto xla_arg_6 = Convertuint64_t(op.feature_index());
  auto xla_result = xla::BatchNormInference(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3), Unwrap(xla_arg_4), Unwrap(xla_arg_5), Unwrap(xla_arg_6));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::BatchNormTrainingOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_2;
  if (failed(GetXlaOp(*op.getODSOperands(2).begin(), value_map, &xla_arg_2, op)))
    return mlir::failure();
  auto xla_arg_3 = ConvertAPFloat(op.epsilon());
  auto xla_arg_4 = Convertuint64_t(op.feature_index());
  auto xla_result = xla::BatchNormTraining(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3), Unwrap(xla_arg_4));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::BroadcastOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_broadcast_sizes(op.broadcast_sizes());
  auto xla_result = xla::Broadcast(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CbrtOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Cbrt(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CeilOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Ceil(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CholeskyOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convertbool(op.lower());
  auto xla_result = xla::Cholesky(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ClampOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_2;
  if (failed(GetXlaOp(*op.getODSOperands(2).begin(), value_map, &xla_arg_2, op)))
    return mlir::failure();
  auto xla_result = xla::Clamp(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ClzOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Clz(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CollectivePermuteOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_source_target_pairs(op.source_target_pairs());
  auto xla_result = xla::CollectivePermute(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CompareOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convert_comparison_direction(op.comparison_direction());
  auto xla_result = xla::Compare(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ComplexOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Complex(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ConcatenateOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  std::vector<xla::XlaOp> xla_arg_0;
  for (auto operand : op.getODSOperands(0)) {
    xla::XlaOp result;
    if (failed(GetXlaOp(operand, value_map, &result, op)))
      return mlir::failure();
    xla_arg_0.push_back(result);
  }
  auto xla_arg_1 = Convertuint64_t(op.dimension());
  auto xla_result = xla::ConcatInDim(ctx.builder, Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ConvOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convert_window_strides(op.window_strides());
  auto xla_arg_3 = Convert_padding(op.padding());
  auto xla_arg_4 = Convert_lhs_dilation(op.lhs_dilation());
  auto xla_arg_5 = Convert_rhs_dilation(op.rhs_dilation());
  auto xla_arg_6 = Convert_dimension_numbers(op.dimension_numbers());
  auto xla_arg_7 = Convertuint64_t(op.feature_group_count());
  auto xla_arg_8 = Convertuint64_t(op.batch_group_count());
  auto xla_arg_9 = Convert_precision_config(op.precision_config());
  auto xla_result = xla::ConvGeneralDilated(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3), Unwrap(xla_arg_4), Unwrap(xla_arg_5), Unwrap(xla_arg_6), Unwrap(xla_arg_7), Unwrap(xla_arg_8), Unwrap(xla_arg_9));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CopyOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Copy(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CosOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Cos(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CreateTokenOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  auto xla_result = xla::CreateToken(ctx.builder);
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CrossReplicaSumOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_replica_groups(op.replica_groups());
  auto xla_result = xla::CrossReplicaSum(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::DivOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Div(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::DotGeneralOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convert_dot_dimension_numbers(op.dot_dimension_numbers());
  auto xla_arg_3 = Convert_precision_config(op.precision_config());
  auto xla_result = xla::DotGeneral(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::DotOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convert_precision_config(op.precision_config());
  auto xla_result = xla::Dot(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::DynamicSliceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  std::vector<xla::XlaOp> xla_arg_1;
  for (auto operand : op.getODSOperands(1)) {
    xla::XlaOp result;
    if (failed(GetXlaOp(operand, value_map, &result, op)))
      return mlir::failure();
    xla_arg_1.push_back(result);
  }
  auto xla_arg_2 = Convert_slice_sizes(op.slice_sizes());
  auto xla_result = xla::DynamicSlice(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::DynamicUpdateSliceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  std::vector<xla::XlaOp> xla_arg_2;
  for (auto operand : op.getODSOperands(2)) {
    xla::XlaOp result;
    if (failed(GetXlaOp(operand, value_map, &result, op)))
      return mlir::failure();
    xla_arg_2.push_back(result);
  }
  auto xla_result = xla::DynamicUpdateSlice(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::EinsumOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = ConvertStringRef(op.einsum_config());
  auto xla_result = xla::Einsum(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ExpOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Exp(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::Expm1Op op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Expm1(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::FftOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_fft_type(op.fft_type());
  auto xla_arg_2 = Convert_fft_length(op.fft_length());
  auto xla_result = xla::Fft(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::FloorOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Floor(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::GatherOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convert_dimension_numbers(op.dimension_numbers());
  auto xla_arg_3 = Convert_slice_sizes(op.slice_sizes());
  auto xla_arg_4 = Convertbool(op.indices_are_sorted());
  auto xla_result = xla::Gather(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3), Unwrap(xla_arg_4));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::GetDimensionSizeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convertuint32_t(op.dimension());
  auto xla_result = xla::GetDimensionSize(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::GetTupleElementOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convertuint32_t(op.index());
  auto xla_result = xla::GetTupleElement(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ImagOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Imag(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::IsFiniteOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::IsFinite(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::Log1pOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Log1p(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::LogOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Log(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::LogisticOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Logistic(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::MaxOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Max(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::MinOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Min(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::MulOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Mul(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::NegOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Neg(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::NotOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Not(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::OrOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Or(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::PopulationCountOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::PopulationCount(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::PowOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Pow(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::RealOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Real(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ReducePrecisionOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convertuint32_t(op.exponent_bits());
  auto xla_arg_2 = Convertuint32_t(op.mantissa_bits());
  auto xla_result = xla::ReducePrecision(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::RemOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Rem(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ReplicaIdOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  auto xla_result = xla::ReplicaId(ctx.builder);
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ReverseOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_dimensions(op.dimensions());
  auto xla_result = xla::Rev(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::RoundOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Round(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::RsqrtOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Rsqrt(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SelectOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_2;
  if (failed(GetXlaOp(*op.getODSOperands(2).begin(), value_map, &xla_arg_2, op)))
    return mlir::failure();
  auto xla_result = xla::Select(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SetDimensionSizeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convertuint32_t(op.dimension());
  auto xla_result = xla::SetDimensionSize(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ShiftLeftOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::ShiftLeft(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ShiftRightArithmeticOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::ShiftRightArithmetic(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ShiftRightLogicalOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::ShiftRightLogical(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SignOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Sign(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SinOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Sin(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SliceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_start_indices(op.start_indices());
  auto xla_arg_2 = Convert_limit_indices(op.limit_indices());
  auto xla_arg_3 = Convert_strides(op.strides());
  auto xla_result = xla::Slice(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SqrtOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Sqrt(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SubOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Sub(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::TanhOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Tanh(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::TorchIndexSelectOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convertuint64_t(op.dim());
  auto xla_arg_3 = Convertuint64_t(op.batch_dims());
  auto xla_result = xla::TorchIndexSelect(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::TransposeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_permutation(op.permutation());
  auto xla_result = xla::Transpose(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::TriangularSolveOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convertbool(op.left_side());
  auto xla_arg_3 = Convertbool(op.lower());
  auto xla_arg_4 = Convertbool(op.unit_diagonal());
  auto xla_arg_5 = Convert_transpose_a(op.transpose_a());
  auto xla_result = xla::TriangularSolve(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3), Unwrap(xla_arg_4), Unwrap(xla_arg_5));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::TupleOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  std::vector<xla::XlaOp> xla_arg_0;
  for (auto operand : op.getODSOperands(0)) {
    xla::XlaOp result;
    if (failed(GetXlaOp(operand, value_map, &result, op)))
      return mlir::failure();
    xla_arg_0.push_back(result);
  }
  auto xla_result = xla::Tuple(ctx.builder, Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::XorOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Xor(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOperator(
mlir::Operation* op, OpLoweringContext lowering_context) {

  xla::XlaScopedShardingAssignment sharding(lowering_context.builder, CreateOpShardingFromAttribute(op));

  xla::XlaScopedFrontendAttributesAssignment frontend_attributes(lowering_context.builder, CreateOpFrontendAttributesFromAttribute(op));

  xla::XlaScopedOpMetadataAssignment op_metadata(lowering_context.builder, CreateOpMetadataFromLocation(op));

  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AbsOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AddOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AfterAllOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AllReduceOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AllToAllOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AndOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::Atan2Op>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BatchNormGradOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BatchNormInferenceOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BatchNormTrainingOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BitcastConvertOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BitcastOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BroadcastInDimOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BroadcastOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CaseOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CbrtOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CeilOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CholeskyOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ClampOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ClzOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CollectivePermuteOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CompareOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ComplexOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ConcatenateOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ConstOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ConvOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ConvertOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CopyOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CosOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CreateTokenOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CrossReplicaSumOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CustomCallOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DequantizeOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DivOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DotGeneralOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DotOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicBroadcastInDimOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicIotaOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicReshapeOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicSliceOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicUpdateSliceOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::EinsumOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ExpOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::Expm1Op>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::FftOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::FloorOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::FusionOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::GatherOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::GetDimensionSizeOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::GetTupleElementOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::IfOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ImagOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::InfeedOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::IotaOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::IsFiniteOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::Log1pOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::LogOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::LogisticOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::MapOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::MaxOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::MinOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::MulOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::NegOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::NotOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::OrOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::OutfeedOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::PadOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::PopulationCountOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::PowOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RealOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RecvOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReduceOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReducePrecisionOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReduceWindowOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RemOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReplicaIdOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReshapeOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReturnOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReverseOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RngBitGeneratorOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RngNormalOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RngUniformOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RoundOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RsqrtOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ScatterOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SelectAndScatterOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SelectOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SendOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SetDimensionSizeOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ShiftLeftOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ShiftRightArithmeticOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ShiftRightLogicalOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SignOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SinOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SliceOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SortOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SqrtOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SubOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TanhOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TorchIndexSelectOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TraceOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TransposeOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TriangularSolveOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TupleOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::UnaryEinsumOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::WhileOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::XorOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  return mlir::failure();
}
