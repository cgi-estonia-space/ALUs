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
#include "tensorflow/lite/experimental/delegates/hexagon/utils.h"

#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace {

bool IsActivationReluOrNone(TfLiteFusedActivation activation) {
  return (activation == kTfLiteActRelu || activation == kTfLiteActRelu6 ||
          activation == kTfLiteActRelu1 || activation == kTfLiteActNone);
}

bool TensorTypeMatch(int tensor_id, TfLiteContext* context,
                     TfLiteType tensor_type) {
  const auto& tensor = context->tensors[tensor_id];
  return tensor.type == tensor_type;
}

bool InputsWithCorrectTypes(const TfLiteNode* node, TfLiteContext* context,
                            const std::vector<TfLiteType>& input_types) {
  if (node->inputs->size != input_types.size()) return false;
  for (int i = 0; i < input_types.size(); ++i) {
    if (!TensorTypeMatch(node->inputs->data[i], context, input_types[i]))
      return false;
  }
  return true;
}

}  // namespace

TfLiteStatus Get4DShape(unsigned int* batch_size, unsigned int* height_size,
                        unsigned int* width_size, unsigned int* depth_size,
                        TfLiteIntArray* dims) {
  if (dims->size > 4) return kTfLiteError;
  unsigned int* dim[] = {batch_size, height_size, width_size, depth_size};
  for (int i = 0; i < 4; ++i) *(dim[i]) = 1;
  for (int i = 4 - dims->size; i < 4; ++i) {
    *dim[i] = dims->data[i - (4 - dims->size)];
  }
  return kTfLiteOk;
}

bool IsNodeSupportedByHexagon(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context) {
  // Ensure all inputs & outputs have dim <= 4.
  int tensor_id;
  for (int i = 0; i < node->inputs->size; ++i) {
    tensor_id = node->inputs->data[i];
    const auto& tensor = context->tensors[tensor_id];
    if (tensor.dims->size > 4) return false;
  }
  for (int i = 0; i < node->outputs->size; ++i) {
    tensor_id = node->outputs->data[i];
    const auto& tensor = context->tensors[tensor_id];
    if (tensor.dims->size > 4) return false;
  }

  // Most hexagon kernels are not compatible with op versions > 1.
  // We maintain a 'whitelist' here to ensure we don't accept unintended nodes.
  if (registration->version > 1) {
    if (registration->builtin_code == kTfLiteBuiltinDepthwiseConv2d &&
        registration->version == 2) {
      return true;
    }
    return false;
  }

  switch (registration->builtin_code) {
    case kTfLiteBuiltinAdd: {
      if (!InputsWithCorrectTypes(node, context, {kTfLiteUInt8, kTfLiteUInt8}))
        return false;
      const TfLiteAddParams* add_params =
          reinterpret_cast<const TfLiteAddParams*>(node->builtin_data);
      return IsActivationReluOrNone(add_params->activation);
    }
    case kTfLiteBuiltinMul: {
      if (!InputsWithCorrectTypes(node, context, {kTfLiteUInt8, kTfLiteUInt8}))
        return false;
      const TfLiteMulParams* mul_params =
          reinterpret_cast<const TfLiteMulParams*>(node->builtin_data);
      // TODO(b/129276536): Add support for activation on Mul node.
      return mul_params->activation == kTfLiteActNone;
    }
    case kTfLiteBuiltinSub: {
      if (!InputsWithCorrectTypes(node, context, {kTfLiteUInt8, kTfLiteUInt8}))
        return false;
      const TfLiteSubParams* sub_params =
          reinterpret_cast<const TfLiteSubParams*>(node->builtin_data);
      return IsActivationReluOrNone(sub_params->activation);
    }
    case kTfLiteBuiltinSum:
    case kTfLiteBuiltinMean: {
      // TODO(b/139277813): Enable these when they pass unit tests. These seem
      // to recompute the output min/max instead of taking them as inputs, which
      // causes an unexpected shift in dequantized values.
      return false;
    }
    case kTfLiteBuiltinPad: {
      // TODO(b/139277813): Currently we only support padding with the default
      // of 0. Add support for user-defined constant if required.
      return (
          node->inputs->size == 2 &&
          InputsWithCorrectTypes(node, context, {kTfLiteUInt8, kTfLiteInt32}) &&
          IsConstantTensor(&context->tensors[node->inputs->data[1]]));
    }
    case kTfLiteBuiltinFullyConnected: {
      if (!InputsWithCorrectTypes(node, context,
                                  {kTfLiteUInt8, kTfLiteUInt8, kTfLiteInt32}))
        return false;
      const TfLiteFullyConnectedParams* matmul_params =
          reinterpret_cast<const TfLiteFullyConnectedParams*>(
              node->builtin_data);
      return (IsActivationReluOrNone(matmul_params->activation) &&
              matmul_params->keep_num_dims == false &&
              matmul_params->weights_format ==
                  kTfLiteFullyConnectedWeightsFormatDefault);
    }
    case kTfLiteBuiltinConcatenation: {
      // All concatenated tensors must be Uint8 type.
      for (int i = 0; i < node->inputs->size; ++i) {
        if (!TensorTypeMatch(node->inputs->data[i], context, kTfLiteUInt8))
          return false;
      }
      // Hexagon only supports concatenation at axis 3.
      const TfLiteConcatenationParams* concat_params =
          reinterpret_cast<const TfLiteConcatenationParams*>(
              node->builtin_data);
      return (concat_params->axis == 3);
    }
    case kTfLiteBuiltinMaxPool2d: {
      if (!InputsWithCorrectTypes(node, context, {kTfLiteUInt8})) return false;
      // TODO(b/129276536): Add support for activation here.
      const TfLitePoolParams* pool_params =
          reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
      return pool_params->activation == kTfLiteActNone;
    }
    case kTfLiteBuiltinAveragePool2d: {
      if (!InputsWithCorrectTypes(node, context, {kTfLiteUInt8})) return false;
      // AvgPool works fine for filter dim <=7.
      const TfLitePoolParams* pool_params =
          reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
      return (node->inputs->size == 1 &&
              pool_params->activation == kTfLiteActNone);
    }
    case kTfLiteBuiltinTransposeConv: {
      if (!InputsWithCorrectTypes(node, context,
                                  {kTfLiteInt32, kTfLiteUInt8, kTfLiteUInt8}))
        return false;
      const TfLiteTransposeConvParams* params =
          reinterpret_cast<const TfLiteTransposeConvParams*>(
              node->builtin_data);
      return (params->stride_height <= 3 && params->stride_width <= 3 &&
              (params->padding == kTfLitePaddingSame ||
               params->padding == kTfLitePaddingValid));
    }
    case kTfLiteBuiltinConv2d: {
      if (!InputsWithCorrectTypes(node, context,
                                  {kTfLiteUInt8, kTfLiteUInt8, kTfLiteInt32}))
        return false;
      const TfLiteConvParams* conv_params =
          reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);
      return (IsActivationReluOrNone(conv_params->activation) &&
              conv_params->stride_height <= 3 &&
              conv_params->stride_width <= 3 &&
              conv_params->dilation_height_factor == 1 &&
              conv_params->dilation_width_factor == 1);
    }
    case kTfLiteBuiltinDepthwiseConv2d: {
      if (!InputsWithCorrectTypes(node, context,
                                  {kTfLiteUInt8, kTfLiteUInt8, kTfLiteInt32}))
        return false;
      // Hexagon only supports width of 3 for Depthwise Conv.
      const auto& tensor = context->tensors[node->inputs->data[1]];
      if (tensor.dims->data[2] != 3) return false;
      const TfLiteDepthwiseConvParams* conv_params =
          reinterpret_cast<const TfLiteDepthwiseConvParams*>(
              node->builtin_data);
      const bool dilation = conv_params->dilation_height_factor != 1 ||
                            conv_params->dilation_width_factor != 1;
      if (dilation) {
        // We only support dilations when stride == 1.
        if (conv_params->stride_height != 1 || conv_params->stride_width != 1)
          return false;
      }
      return (IsActivationReluOrNone(conv_params->activation) &&
              conv_params->stride_height <= 3 &&
              conv_params->stride_width <= 3 &&
              conv_params->depth_multiplier == 1);
    }
    case kTfLiteBuiltinReshape: {
      if (node->inputs->size > 2 ||
          !TensorTypeMatch(node->inputs->data[0], context, kTfLiteUInt8))
        return false;
      return true;
    }
    case kTfLiteBuiltinSoftmax:
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinLogistic: {
      return InputsWithCorrectTypes(node, context, {kTfLiteUInt8});
    }
    case kTfLiteBuiltinResizeNearestNeighbor: {
      return InputsWithCorrectTypes(node, context,
                                    {kTfLiteUInt8, kTfLiteInt32});
    }
    case kTfLiteBuiltinL2Normalization: {
      if (!InputsWithCorrectTypes(node, context, {kTfLiteUInt8})) return false;
      const TfLiteL2NormParams* norm_params =
          reinterpret_cast<const TfLiteL2NormParams*>(node->builtin_data);
      return (norm_params->activation == kTfLiteActNone);
    }
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin:
      return InputsWithCorrectTypes(node, context,
                                    {kTfLiteUInt8, kTfLiteInt32});
    case kTfLiteBuiltinSplit: {
      if (!InputsWithCorrectTypes(node, context, {kTfLiteInt32, kTfLiteUInt8}))
        return false;
      const auto& input_tensor = context->tensors[node->inputs->data[1]];
      const bool is_four_dim_or_less = input_tensor.dims->size < 5;
      // We need splitting axis to be constant, so Hexagon knows output shapes.
      return is_four_dim_or_less &&
             IsConstantTensor(&context->tensors[node->inputs->data[0]]);
    }
    case kTfLiteBuiltinResizeBilinear: {
      if (!InputsWithCorrectTypes(node, context,
                                  {kTfLiteUInt8, kTfLiteInt32}) ||
          !IsConstantTensor(&context->tensors[node->inputs->data[1]])) {
        return false;
      }
      const auto& size_tensor = context->tensors[node->inputs->data[1]];
      // TODO(b/143105433): Latency increase significantly with large size
      // value. Limiting to 65 for now.
      return NumElements(&size_tensor) == 2 && size_tensor.data.i32[0] < 66 &&
             size_tensor.data.i32[1] < 66;
    }
    case kTfLiteBuiltinNeg: {
      return InputsWithCorrectTypes(node, context, {kTfLiteUInt8});
    }
    case kTfLiteBuiltinTranspose: {
      return InputsWithCorrectTypes(node, context,
                                    {kTfLiteUInt8, kTfLiteInt32});
    }
    case kTfLiteBuiltinSpaceToDepth: {
      return InputsWithCorrectTypes(node, context, {kTfLiteUInt8});
    }
    case kTfLiteBuiltinDepthToSpace: {
      return InputsWithCorrectTypes(node, context, {kTfLiteUInt8});
    }
    default:
      return false;
  }
  return false;
}

}  // namespace tflite
