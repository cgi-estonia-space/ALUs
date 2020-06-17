/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/kernels/winograd.h"

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

std::string GetWinograd4x4To36Code(
    const OperationDef& op_def, const LinearStorage& bt_arr,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor(
      "src_data",
      WHSBPoint{"src_size.x", "src_size.y", "src_size.z", "src_size.w"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data",
      WHSBPoint{"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w"},
      op_def.dst_tensors[0]);

  const std::string batch_id = op_def.IsBatchSupported() ? "batch_id" : "";
  std::string c = GetCommonDefines(op_def.precision);

  const auto src_tensor_type = op_def.src_tensors[0].storage_type;
  const bool is_image_buffer =
      src_tensor_type == TensorStorageType::IMAGE_BUFFER;
  const bool is_buffer = src_tensor_type == TensorStorageType::BUFFER;

  c += R"(
  constant FLT Bt[36] = {
   1.0000000000f,  0.0000000887f, -2.3075673580f,  0.0000000089f,  0.8519787788f, -0.0000000000f,
  -0.0000000000f,  0.9057970643f,  1.2307025194f, -0.4180375934f, -0.5679858327f,  0.0000000000f,
   0.0000000000f, -0.9057970643f,  1.2307025194f,  0.4180375934f, -0.5679858327f, -0.0000000000f,
  -0.0000000000f, -0.1132246330f, -0.0769189075f,  0.2090187818f,  0.1419964582f,  0.0000000000f,
   0.0000000000f,  0.1132246330f, -0.0769189224f, -0.2090187967f,  0.1419964582f, -0.0000000000f,
  -0.0000000000f,  1.1737382412f, -0.0000000532f, -2.7084801197f, -0.0000000355f,  1.0000000000f,
};
)";
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += bt_arr.GetDeclaration();
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,                              \n";
  c += "    int4 dst_size,                              \n";
  c += "    int2 padding,                               \n";
  c += "    int tiles_total,                            \n";
  c += "    int tiles_x                                 \n";
  c += ") {\n";
  c += "  int DST_X = get_global_id(0);\n";
  c += "  int DST_Y = get_global_id(1);\n";
  c += "  int DST_Z = get_global_id(2);\n";
  c += "  if (DST_X >= tiles_total || DST_Y >= 6 || DST_Z >= dst_size.z) {\n";
  c += "    return; \n";
  c += "  }\n";
  c += "  int tile_x = (DST_X % tiles_x) * 4;\n";
  c += "  int tile_y = (DST_X / tiles_x) * 4;\n";
  c += "  FLT4 I0, I1, I2, I3, I4, I5;\n";
  c += "  FLT bt_ar[6];\n";
  c += "  FLT4 t0 = " + bt_arr.ReadLinearFLT4("DST_Y * 2 + 0") + ";\n";
  c += "  FLT4 t1 = " + bt_arr.ReadLinearFLT4("DST_Y * 2 + 1") + ";\n";
  c += "  DST_Y *= 6;\n";
  c += "  bt_ar[0] = t0.x;\n";
  c += "  bt_ar[1] = t0.y;\n";
  c += "  bt_ar[2] = t0.z;\n";
  c += "  bt_ar[3] = t0.w;\n";
  c += "  bt_ar[4] = t1.x;\n";
  c += "  bt_ar[5] = t1.y;\n";
  auto read_src = [&](const std::string& src, const std::string& xs) {
    if (is_image_buffer) {
      c += "    FLT4 " + src + " = " +
           src_tensor.Read("src_a_" + xs + " + offset") + ";\n";
    } else if (is_buffer) {
      c += "    FLT4 " + src + " = " +
           src_tensor.Read("src_a_" + xs + " + offset") + " * m" + xs + "_x;\n";
    } else {
      c += "    FLT4 " + src + " = " +
           src_tensor.ReadWHSB("tile_x + padding.x + " + xs, "yc", "DST_Z",
                               batch_id) +
           ";\n";
    }
  };
  if (is_buffer || is_image_buffer) {
    for (int x = 0; x < 6; ++x) {
      const std::string xs = std::to_string(x);
      c += "  int xc" + xs + " = tile_x + padding.x + " + xs + ";\n";
      c += "  FLT m" + xs + "_x = (FLT)(xc" + xs + " >= 0 && xc" + xs +
           " < src_size.x);\n";
      c += "  bool inx" + xs + " = (xc" + xs + " >= 0 && xc" + xs +
           " < src_size.x);\n";
      c += "  xc" + xs + " = clamp(xc" + xs + ", 0, src_size.x - 1);\n";
      c += "  " + src_tensor.GetAddressWHSB("src_a_" + xs, "xc" + xs, "0",
                                            "DST_Z", batch_id);
      if (is_image_buffer) {
        c += "  src_a_" + xs + " = select(-src_size.x * src_size.y, src_a_" +
             xs + ", inx" + xs + ");\n";
      }
    }
  }
  c += "  {\n";
  c += "    int yc = tile_y + padding.y;\n";
  if (is_buffer || is_image_buffer) {
    c += "    bool iny = (yc >= 0 && yc < src_size.y);\n";
    c += "    int offset = select(0, yc * src_size.x, iny);\n";
    c += "    FLT bt = bt_ar[0] * (FLT)(iny);\n";
  } else {
    c += "    FLT bt = bt_ar[0];\n";
  }
  for (int x = 0; x < 6; ++x) {
    const std::string xs = std::to_string(x);
    const std::string src = "src" + xs;
    read_src(src, xs);
    c += "    I" + xs + " = bt * " + src + ";\n";
  }
  c += "  }\n";
  for (int y = 1; y < 6; ++y) {
    const std::string ys = std::to_string(y);
    c += "  {\n";
    c += "    int yc = tile_y + padding.y + (" + ys + ");\n";
    if (is_buffer || is_image_buffer) {
      c += "    bool iny = (yc >= 0 && yc < src_size.y);\n";
      c += "    int offset = select(0, yc * src_size.x, iny);\n";
      c += "    FLT bt = bt_ar[" + ys + "] * (FLT)(iny);\n";
    } else {
      c += "    FLT bt = bt_ar[" + ys + "];\n";
    }
    for (int x = 0; x < 6; ++x) {
      const std::string xs = std::to_string(x);
      const std::string src = "src" + xs;
      read_src(src, xs);
      c += "    I" + xs + " += bt * " + src + ";\n";
    }
    c += "  }\n";
  }
  const LinkingContext context{"r0", "DST_X", "DST_Y", "DST_Z"};
  c += "  {\n";
  c += "    FLT4 r0 = I0 + Bt[2] * I2 + Bt[4] * I4;\n";
  c += PostProcess(linked_operations, context);
  c += "    " + dst_tensor.WriteWHSB("r0", "DST_X", "DST_Y", "DST_Z", batch_id);
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "  {\n";
  c += "    FLT4 r0 = Bt[7] * I1 + Bt[8] * I2 + Bt[9] * I3 + Bt[10] * I4;\n";
  c += PostProcess(linked_operations, context);
  c += "    " + dst_tensor.WriteWHSB("r0", "DST_X", "DST_Y", "DST_Z", batch_id);
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "  {\n";
  c += "    FLT4 r0 = Bt[13] * I1 + Bt[14] * I2 + Bt[15] * I3 + Bt[16] * I4;\n";
  c += PostProcess(linked_operations, context);
  c += "    " + dst_tensor.WriteWHSB("r0", "DST_X", "DST_Y", "DST_Z", batch_id);
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "  {\n";
  c += "    FLT4 r0 = Bt[19] * I1 + Bt[20] * I2 + Bt[21] * I3 + Bt[22] * I4;\n";
  c += PostProcess(linked_operations, context);
  c += "    " + dst_tensor.WriteWHSB("r0", "DST_X", "DST_Y", "DST_Z", batch_id);
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "  {\n";
  c += "    FLT4 r0 = Bt[25] * I1 + Bt[26] * I2 + Bt[27] * I3 + Bt[28] * I4;\n";
  c += PostProcess(linked_operations, context);
  c += "    " + dst_tensor.WriteWHSB("r0", "DST_X", "DST_Y", "DST_Z", batch_id);
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "  {\n";
  c += "    FLT4 r0 = Bt[31] * I1 + Bt[33] * I3 + I5;\n";
  c += PostProcess(linked_operations, context);
  c += "    " + dst_tensor.WriteWHSB("r0", "DST_X", "DST_Y", "DST_Z", batch_id);
  c += "    DST_Y++;\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

std::string GetWinograd36To4x4Code(
    const OperationDef& op_def, const LinearStorage& at_arr,
    const LinearStorage& biases,
    const std::vector<ElementwiseOperation*>& linked_operations) {
  TensorCodeGenerator src_tensor(
      "src_data",
      WHSBPoint{"src_size.x", "src_size.y", "src_size.z", "src_size.w"},
      op_def.src_tensors[0]);
  TensorCodeGenerator dst_tensor(
      "dst_data",
      WHSBPoint{"dst_size.x", "dst_size.y", "dst_size.z", "dst_size.w"},
      op_def.dst_tensors[0]);

  const std::string batch_id = op_def.IsBatchSupported() ? "batch_id" : "";
  std::string c = GetCommonDefines(op_def.precision);

  c += R"(
constant FLT At[24] = {
  1.0000000000f, 1.0000000000f,  1.0000000000f, 1.0000000000f,  1.0000000000f, 0.0000000000f,
  0.0000000000f, 0.7360000014f, -0.7360000014f, 1.4720000029f, -1.4720000029f, 0.0000000000f,
  0.0000000000f, 0.5416960120f,  0.5416960120f, 2.1667840481f,  2.1667840481f, 0.0000000000f,
  0.0000000000f, 0.3986882567f, -0.3986882567f, 3.1895060539f, -3.1895060539f, 1.0000000000f,
};
)";
  c += "__kernel void main_function(\n";
  c += src_tensor.GetDeclaration(AccessType::READ) + ",\n";
  c += at_arr.GetDeclaration() + ",\n";
  c += biases.GetDeclaration();
  c += GetArgsDeclaration(linked_operations);
  c += dst_tensor.GetDeclaration(AccessType::WRITE) + ",\n";
  c += "    int4 src_size,                              \n";
  c += "    int4 dst_size,                              \n";
  c += "    int tiles_x                                 \n";
  c += ") {\n";
  c += "  int tile_id = get_global_id(0);\n";
  c += "  int DST_Y = get_global_id(1);\n";
  c += "  int DST_Z = get_global_id(2);\n";
  c += "  int tile_x = (tile_id % tiles_x) * 4;\n";
  c += "  int tile_y = (tile_id / tiles_x) * 4 + DST_Y;\n";
  c += "  if (tile_x >= dst_size.x || tile_y >= dst_size.y || DST_Z >= "
       "dst_size.z) {\n";
  c += "    return; \n";
  c += "  }\n";
  c += "  FLT4 I0, I1, I2, I3, I4, I5;\n";
  c += "  FLT at_ar[6];\n";
  c += "  FLT4 t00 = " + at_arr.ReadLinearFLT4("DST_Y * 2 + 0") + ";\n";
  c += "  FLT4 t01 = " + at_arr.ReadLinearFLT4("DST_Y * 2 + 1") + ";\n";
  c += "  at_ar[0] = t00.x;\n";
  c += "  at_ar[1] = t00.y;\n";
  c += "  at_ar[2] = t00.z;\n";
  c += "  at_ar[3] = t00.w;\n";
  c += "  at_ar[4] = t01.x;\n";
  c += "  at_ar[5] = t01.y;\n";
  c += "  {\n";
  c += "    FLT at = at_ar[0];\n";
  for (int x = 0; x < 6; ++x) {
    const std::string yc = std::to_string(x);
    const std::string src = "src" + std::to_string(x);
    c += "    FLT4 " + src + " = " +
         src_tensor.ReadWHSB("tile_id", yc, "DST_Z", batch_id) + ";\n";
    c += "    I" + std::to_string(x) + " = at * " + src + ";\n";
  }
  c += "  }\n";
  for (int y = 1; y < 6; ++y) {
    c += "  {\n";
    c += "    FLT at = at_ar[" + std::to_string(y) + "];\n";
    for (int x = 0; x < 6; ++x) {
      const std::string yc = std::to_string(y * 6 + x);
      const std::string src = "src" + std::to_string(x);
      c += "    FLT4 " + src + " = " +
           src_tensor.ReadWHSB("tile_id", yc, "DST_Z", batch_id) + ";\n";
      c += "    I" + std::to_string(x) + " += at * " + src + ";\n";
    }
    c += "  }\n";
  }
  c += "  FLT4 t0 = I1 + I2;\n";
  c += "  FLT4 t1 = I3 + I4;\n";
  c += "  FLT4 bias_val = " + biases.ReadLinearFLT4("DST_Z") + ";\n";
  c += "  {\n";
  const LinkingContext context{"r0", "tile_x", "tile_y", "DST_Z"};
  c += "    FLT4 r0 = I0 + t0 + t1 + bias_val;\n";
  c += PostProcess(linked_operations, context);
  c += "    " +
       dst_tensor.WriteWHSB("r0", "tile_x", "tile_y", "DST_Z", batch_id);
  c += "    tile_x++;\n";
  c += "  }\n";
  c += "  FLT4 t2 = I1 - I2;\n";
  c += "  FLT4 t3 = I3 - I4;\n";
  c += "  if (tile_x < dst_size.x) {\n";
  c += "    FLT4 r0 = t2 * At[7] + t3 * At[9] + bias_val;\n";
  c += PostProcess(linked_operations, context);
  c += "    " +
       dst_tensor.WriteWHSB("r0", "tile_x", "tile_y", "DST_Z", batch_id);
  c += "    tile_x++;\n";
  c += "  }\n";
  c += "  if (tile_x < dst_size.x) {\n";
  c += "    FLT4 r0 = t0 * At[13] + t1 * At[15] + bias_val;\n";
  c += PostProcess(linked_operations, context);
  c += "    " +
       dst_tensor.WriteWHSB("r0", "tile_x", "tile_y", "DST_Z", batch_id);
  c += "    tile_x++;\n";
  c += "  }\n";
  c += "  if (tile_x < dst_size.x) {\n";
  c += "    FLT4 r0 = t2 * At[19] + t3 * At[21] + I5 + bias_val;\n";
  c += PostProcess(linked_operations, context);
  c += "    " +
       dst_tensor.WriteWHSB("r0", "tile_x", "tile_y", "DST_Z", batch_id);
  c += "    tile_x++;\n";
  c += "  }\n";
  c += "}\n";
  return c;
}
}  // namespace

Winograd4x4To36::Winograd4x4To36(Winograd4x4To36&& operation)
    : GPUOperation(std::move(operation)),
      bt_(std::move(operation.bt_)),
      padding_(operation.padding_),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

Winograd4x4To36& Winograd4x4To36::operator=(Winograd4x4To36&& operation) {
  if (this != &operation) {
    bt_ = std::move(operation.bt_);
    std::swap(padding_, operation.padding_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status Winograd4x4To36::Compile(const CreationContext& creation_context) {
  std::vector<CompilerOptions> options;
  if (creation_context.device->IsAdreno()) {
    options.push_back(CompilerOptions::ADRENO_MORE_WAVES);
  }
  RETURN_IF_ERROR(UploadBt(creation_context.context));
  const auto code =
      GetWinograd4x4To36Code(definition_, bt_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", options, *creation_context.context,
      *creation_context.device, &kernel_);
}

Status Winograd4x4To36::UploadBt(CLContext* context) {
  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> Bt;
  Bt.shape = Linear(48);
  Bt.data = {1.0000000000f,  0.0000000887f,  -2.3075673580f, 0.0000000089f,
             0.8519787788f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
             0.0000000000f,  0.9057970643f,  1.2307025194f,  -0.4180375934f,
             -0.5679858327f, 0.0000000000f,  0.0000000000f,  0.0000000000f,
             0.0000000000f,  -0.9057970643f, 1.2307025194f,  0.4180375934f,
             -0.5679858327f, -0.0000000000f, 0.0000000000f,  0.0000000000f,
             0.0000000000f,  -0.1132246330f, -0.0769189075f, 0.2090187818f,
             0.1419964582f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
             0.0000000000f,  0.1132246330f,  -0.0769189224f, -0.2090187967f,
             0.1419964582f,  0.0000000000f,  0.0000000000f,  0.0000000000f,
             0.0000000000f,  1.1737382412f,  -0.0000000532f, -2.7084801197f,
             -0.0000000355f, 1.0000000000f,  0.0000000000f,  0.0000000000f};

  LinearStorageCreateInfo create_info;
  create_info.storage_type = LinearStorageType::TEXTURE_2D;
  create_info.data_type = definition_.GetDataType();
  create_info.name = "bt_arr";
  return CreateLinearStorage(create_info, Bt, context, &bt_);
}

Status Winograd4x4To36::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(bt_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWHSB()));
  const int tiles_x = IntegralDivideRoundUp(
      src_[0]->Width() + padding_.prepended.w + padding_.appended.w - 2, 4);
  const int tiles_y = IntegralDivideRoundUp(
      src_[0]->Height() + padding_.prepended.h + padding_.appended.h - 2, 4);
  const int tiles_total = tiles_x * tiles_y;
  RETURN_IF_ERROR(
      kernel_.SetBytesAuto(int2(-padding_.prepended.w, -padding_.prepended.h)));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(tiles_total));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(tiles_x));

  return OkStatus();
}

int3 Winograd4x4To36::GetGridSize() const {
  const int grid_x = dst_[0]->Width() * dst_[0]->Batch();
  const int grid_y = 6;
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Status Winograd4x4To36::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status Winograd4x4To36::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Status CreateWinograd4x4To36(const CreationContext& creation_context,
                             const OperationDef& definition,
                             const Padding2D& padding,
                             Winograd4x4To36* result) {
  *result = Winograd4x4To36(definition, padding);
  return result->UploadBt(creation_context.context);
}

Winograd36To4x4::Winograd36To4x4(Winograd36To4x4&& operation)
    : GPUOperation(std::move(operation)),
      at_(std::move(operation.at_)),
      biases_(std::move(operation.biases_)),
      kernel_(std::move(operation.kernel_)),
      work_group_size_(operation.work_group_size_) {}

Winograd36To4x4& Winograd36To4x4::operator=(Winograd36To4x4&& operation) {
  if (this != &operation) {
    at_ = std::move(operation.at_);
    biases_ = std::move(operation.biases_);
    kernel_ = std::move(operation.kernel_);
    std::swap(work_group_size_, operation.work_group_size_);
    GPUOperation::operator=(std::move(operation));
  }
  return *this;
}

Status Winograd36To4x4::Compile(const CreationContext& creation_context) {
  const auto code =
      GetWinograd36To4x4Code(definition_, at_, biases_, linked_operations_);
  return creation_context.cache->GetOrCreateCLKernel(
      code, "main_function", *creation_context.context,
      *creation_context.device, &kernel_);
}

Status Winograd36To4x4::UploadAt(CLContext* context) {
  ::tflite::gpu::Tensor<Linear, DataType::FLOAT32> At;
  At.shape = Linear(32);
  At.data = {1.0000000000f,  1.0000000000f, 1.0000000000f,  1.0000000000f,
             1.0000000000f,  0.0000000000f, 0.0000000000f,  0.0000000000f,
             0.0000000000f,  0.7360000014f, -0.7360000014f, 1.4720000029f,
             -1.4720000029f, 0.0000000000f, 0.0000000000f,  0.0000000000f,
             0.0000000000f,  0.5416960120f, 0.5416960120f,  2.1667840481f,
             2.1667840481f,  0.0000000000f, 0.0000000000f,  0.0000000000f,
             0.0000000000f,  0.3986882567f, -0.3986882567f, 3.1895060539f,
             -3.1895060539f, 1.0000000000f, 0.0000000000f,  0.0000000000f};

  LinearStorageCreateInfo create_info;
  create_info.storage_type = LinearStorageType::TEXTURE_2D;
  create_info.data_type = definition_.GetDataType();
  create_info.name = "at_arr";
  return CreateLinearStorage(create_info, At, context, &at_);
}

Status Winograd36To4x4::BindArguments() {
  kernel_.ResetBindingCounter();
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(src_[0]->GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(at_.GetMemoryPtr()));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(biases_.GetMemoryPtr()));
  RETURN_IF_ERROR(BindArgs(&kernel_, linked_operations_));
  RETURN_IF_ERROR(kernel_.SetMemoryAuto(dst_[0]->GetMemoryPtrForWriting()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(src_[0]->GetWHSB()));
  RETURN_IF_ERROR(kernel_.SetBytesAuto(dst_[0]->GetWHSB()));
  const int tiles_x = IntegralDivideRoundUp(dst_[0]->Width(), 4);
  RETURN_IF_ERROR(kernel_.SetBytesAuto(tiles_x));

  return OkStatus();
}

int3 Winograd36To4x4::GetGridSize() const {
  const int tiles_x = IntegralDivideRoundUp(dst_[0]->Width(), 4);
  const int tiles_y = IntegralDivideRoundUp(dst_[0]->Height(), 4);
  const int grid_x = tiles_x * tiles_y * dst_[0]->Batch();
  const int grid_y = 4;
  const int grid_z = dst_[0]->Slices();
  return int3(grid_x, grid_y, grid_z);
}

Status Winograd36To4x4::Tune(const TuningParameters& params) {
  RETURN_IF_ERROR(BindArguments());
  return GetBestWorkGroup(params, kernel_, GetGridSize(), &work_group_size_);
}

Status Winograd36To4x4::AddToQueue(CLCommandQueue* queue) {
  RETURN_IF_ERROR(BindArguments());
  return queue->DispatchImplicit(kernel_, GetGridSize(), work_group_size_);
}

Status CreateWinograd36To4x4(
    const CreationContext& creation_context, const OperationDef& definition,
    const ::tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases,
    Winograd36To4x4* result) {
  *result = Winograd36To4x4(definition);
  LinearStorageCreateInfo create_info;
  create_info.storage_type = LinearStorageType::TEXTURE_2D;
  create_info.data_type = definition.GetDataType();
  create_info.name = "biases";
  RETURN_IF_ERROR(CreateLinearStorage(
      create_info, biases, creation_context.context, &result->biases_));

  return result->UploadAt(creation_context.context);
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
