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

#include "tensorflow/lite/kernels/internal/reference/quantize.h"

#include <xtensa/tie/xt_hifi2.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa-hifimini/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa-hifimini/utils.h"

namespace tflite {
namespace ops {
namespace micro {

namespace xtensa {
namespace hifimini {

void AffineQuantize(int scale_multiplier,
                    const tflite::QuantizationParams& op_params,
                    const RuntimeShape& input_shape, const int16_t* input_data,
                    const RuntimeShape& output_shape, int8_t* output_data) {
  const int32 zero_point = op_params.zero_point;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  ae_q56s min_val_56 = AE_CVTQ48A32S(INT16_MIN);
  ae_q56s max_val_56 = AE_CVTQ48A32S(INT16_MAX);
  ae_q56s zero_point_56 = AE_CVTQ48A32S(zero_point);

  const ae_p16x2s* input_data_ptr = (const ae_p16x2s*)(input_data - 2);

  ae_p24x2s scale_multiplier_24x2 = AE_CONVERT_INT32_24x2(scale_multiplier);

  int iters = flat_size / 2;
  for (int i = 0; i < iters; i++) {
    // Load two 16bit pairs into the 2x24bit register PR:
    // Values need to be right shifted 8 bits to align from upper 16bits to a
    // 24bit value:
    ae_p24x2s inputs_24x2;
    AE_LP16X2F_IU(inputs_24x2, input_data_ptr, 4);
    inputs_24x2 = AE_P24X2S_SRAI(inputs_24x2, 8);

    // Q0.23 * Q16.0 == Q16.23
    ae_q56s sum_56 = AE_ZEROQ56();

    {
      AE_MULAS56P24S_HH(sum_56, scale_multiplier_24x2, inputs_24x2);

      // Q16.23 -> Q16.0
      // Shift right only 7 bits (23 - 16). This truncated shift aligns the
      // 16bit value at the truncation line for 32bit in the QR register. The
      // lower 16 bits will be used for rounding in AE_ROUNDSQ32SYM.
      sum_56 = AE_Q56S_SRAI(sum_56, 7);

      // Round and truncate 32 bits
      sum_56 = AE_ROUNDSQ32SYM(sum_56);

      // Add offset (zero_point_56 is already aligned at 32bits.
      sum_56 = AE_ADDQ56(sum_56, zero_point_56);

      // Saturate:
      sum_56 = AE_MINQ56S(sum_56, max_val_56);
      sum_56 = AE_MAXQ56S(sum_56, min_val_56);

      output_data[i * 2] = static_cast<int16_t>(AE_TRUNCA32Q48(sum_56));
    }

    sum_56 = AE_ZEROQ56();
    {
      AE_MULAS56P24S_LL(sum_56, scale_multiplier_24x2, inputs_24x2);

      // Q16.23 -> Q16.0
      // Shift right only 7 bits (23 - 16). This truncated shift aligns the
      // 16bit value at the truncation line for 32bit in the QR register. The
      // lower 16 bits will be used for rounding in AE_ROUNDSQ32SYM.
      sum_56 = AE_Q56S_SRAI(sum_56, 23 - 16);

      // Round and truncate 32 bits
      sum_56 = AE_ROUNDSQ32SYM(sum_56);

      // Add offset (zero_point_56 is already aligned at 32bits.
      sum_56 = AE_ADDQ56(sum_56, zero_point_56);

      // Saturate:
      sum_56 = AE_MINQ56S(sum_56, max_val_56);
      sum_56 = AE_MAXQ56S(sum_56, min_val_56);

      output_data[i * 2 + 1] = static_cast<int16_t>(AE_TRUNCA32Q48(sum_56));
    }
  }
}

}  // namespace hifimini
}  // namespace xtensa

namespace quantize {

struct OpData {
  int scale_multiplier = 0;
};

static OpData kStaticOpData;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

  // TODO(b/132070898): Use statically slotted OpData structures until a
  // scratch memory API is ready.
  OpData* op_data = &kStaticOpData;
  node->user_data = op_data;

  op_data->scale_multiplier =
      xtensa::hifimini::CreateQConstantForInt24(0, 1.f / output->params.scale);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* input = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

  tflite::QuantizationParams op_params;
  op_params.zero_point = output->params.zero_point;
  op_params.scale = static_cast<double>(output->params.scale);

  if (input->type != kTfLiteInt16 && output->type != kTfLiteInt8) {
    TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                       TfLiteTypeGetName(input->type),
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  xtensa::hifimini::AffineQuantize(
      op_data->scale_multiplier, op_params, GetTensorShape(input),
      GetTensorData<int16_t>(input), GetTensorShape(output),
      GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

}  // namespace quantize

// This Op (QUANTIZE) quantizes the input and produces quantized output.
// AffineQuantize takes scale and zero point and quantizes the float value to
// quantized output, in int8 or uint8 format.
TfLiteRegistration* Register_QUANTIZE() {
  static TfLiteRegistration r = {};
  r.init = quantize::Init;
  r.free = quantize::Free;
  r.prepare = quantize::Prepare;
  r.invoke = quantize::Eval;
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
