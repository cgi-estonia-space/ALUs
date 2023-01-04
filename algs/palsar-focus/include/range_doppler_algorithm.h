#pragma once

#include "cuda_workplace.h"
#include "device_padded_image.h"
#include "sar_metadata.h"

namespace alus::palsar {

/**
 * Basic RDA implementation as described in "Digital Processing of Synthethic Aperture Radar Data"
 * RCMC use no interpolator/nearest neighbor
 */
void RangeDopplerAlgorithm(const SARMetadata& metadata, palsar::DevicePaddedImage& src_img,
                           palsar::DevicePaddedImage& out_img, CudaWorkspace d_workspace);
}  // namespace alus::palsar