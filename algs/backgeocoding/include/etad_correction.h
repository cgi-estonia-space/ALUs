/** Full-resolution pair-specific ETAD correction generation. SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_ptr.h"
#include "etad_preparation.h"
#include "shapes.h"

namespace alus::backgeocoding {

/** Device-addressable correction layers and timing geometry for one SLC burst. */
struct EtadBurstDescriptor {
    s1tbx::etad::SlcBurstGeometry slc_geometry;
    s1tbx::etad::GridGeometry etad_geometry;
    s1tbx::etad::GridView phase;
    s1tbx::etad::GridView height;
    s1tbx::etad::GridView gradient;
    size_t first_output_line;
};

static_assert(std::is_standard_layout_v<EtadBurstDescriptor> && std::is_trivially_copyable_v<EtadBurstDescriptor>);

class EtadCorrection {
public:
    explicit EtadCorrection(const s1tbx::etad::PreparedPair& pair);

    EtadCorrection(const EtadCorrection&) = delete;
    EtadCorrection& operator=(const EtadCorrection&) = delete;

    /** Generate positive etad_ifg in radians. Invalid correction pixels are NaN. */
    void Compute(int reference_burst_index, int secondary_burst_index, Rectangle target_area,
                 const double* secondary_x, const double* secondary_y, float* output) const;

private:
    struct DeviceBurst {
        cuda::CudaPtr<double> phase;
        cuda::CudaPtr<double> height;
        cuda::CudaPtr<double> gradient;
        EtadBurstDescriptor descriptor;
    };

    static DeviceBurst Upload(const s1tbx::etad::PreparedBurst& burst);

    std::vector<DeviceBurst> reference_;
    std::vector<DeviceBurst> secondary_;
};

cudaError_t LaunchEtadCorrection(EtadBurstDescriptor reference, EtadBurstDescriptor secondary,
                                 Rectangle target_area, const double* secondary_x, const double* secondary_y,
                                 float* output);

}  // namespace alus::backgeocoding
