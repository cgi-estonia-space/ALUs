/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_DETECT_X86_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_DETECT_X86_H_

#include "tensorflow/lite/experimental/ruy/platform.h"

namespace ruy {

#if RUY_PLATFORM(X86)
#if RUY_PLATFORM(X86_ENHANCEMENTS)

// This also checks ABM support, which implies LZCNT and POPCNT.
bool DetectCpuSse42();
bool DetectCpuAvx2();
bool DetectCpuAvx512();
// TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
// Optimization is not finished. In particular the dimensions of the kernel
// blocks can be changed as desired.
//
// TODO(b/146646451): Introduce and activate.
inline bool DetectCpuAvxVnni() { return false; }

#else  // RUY_PLATFORM(X86_ENHANCEMENTS)

inline bool DetectCpuSse42() { return false; }
inline bool DetectCpuAvx2() { return false; }
inline bool DetectCpuAvx512() { return false; }
inline bool DetectCpuAvxVnni() { return false; }

#endif  // !RUY_PLATFORM(X86_ENHANCEMENTS)
#endif  // RUY_PLATFORM(X86)

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_DETECT_X86_H_
