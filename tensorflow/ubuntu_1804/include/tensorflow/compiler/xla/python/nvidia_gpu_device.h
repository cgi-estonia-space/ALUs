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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_NVIDIA_GPU_DEVICE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_NVIDIA_GPU_DEVICE_H_

#include <memory>

#include "tensorflow/compiler/xla/python/local_client.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class GpuDevice : public Device {
 public:
  GpuDevice(int id, std::unique_ptr<LocalDeviceState> local_device_state);
};

struct GpuAllocatorConfig {
  enum class Kind {
    kDefault,   // Client picks the best option for the platform.
    kPlatform,  // The platform's default.
    kBFC,  // Allocator using a "Best-Fit with Coalescing" algorithm. Currently
           // only available for GPU.
  };
  Kind kind = Kind::kDefault;

  // Only used if kind == kBFC. The maximum fraction of available memory to
  // allocate.
  double memory_fraction = 0.9;

  // Only used if kind == kBFC. If true, the allocator will immediately allocate
  // the maximum amount allowed by `memory_fraction`. This reduces
  // fragmentation, allowing more of the total memory to be used. If false, the
  // allocator will allocate more memory as allocations are requested.
  bool preallocate = true;
};

StatusOr<std::shared_ptr<PyLocalClient>> GetNvidiaGpuClient(
    bool asynchronous, const GpuAllocatorConfig& allocator_config);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_NVIDIA_GPU_DEVICE_H_
