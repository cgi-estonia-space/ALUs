/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#pragma once

#include <driver_types.h>
#include <map>
#include <memory>

#include "kernel_array.h"
#include "s1tbx-commons/noise_azimuth_vector.h"
#include "s1tbx-commons/noise_vector.h"
#include "shapes.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "thermal_noise_data_structures.h"
#include "thermal_noise_info.h"
#include "time_maps.h"

namespace alus::tnr {

/**
 * Gathers metadata required for execution of Thermal Noise Removal operator. For SLC data.
 *
 * @param polarisation Selected polarisation.
 * @param sub_swath Selected subswath.
 * @param origin_metadata_root Metadata of the product.
 * @return Parsed metadata.
 */
ThermalNoiseInfo GetThermalNoiseInfoForBursts(std::string_view polarisation, std::string_view sub_swath,
                                              const std::shared_ptr<snapengine::MetadataElement>& origin_metadata_root);

/**
 * Gathers metadata required for execution of Thermal Noise Removal operator. For GRD data.
 *
 * @param polarisation Selected polarisation.
 * @param sub_swath Selected subswath.
 * @param origin_metadata_root Metadata of the product.
 * @return Parsed metadata.
 */
ThermalNoiseInfo GetThermalNoiseInfoForGrd(std::string_view polarisation,
                                           const std::shared_ptr<snapengine::MetadataElement>& origin_metadata_root);

/**
 * Fills time maps with T0 and DeltaTS values acquired from the product metadata.
 *
 * This is a port of SNAP's Sentinel1RemoveThermalNoiseOP.getT0andDeltaTS() method.
 *
 * @param image_name Name of the image (band). Usually looks like s1a-iw-slc-hh.
 * @param origin_metadata_root The MetadataElement of the Product's root metadata.
 * @param time_maps Reference to TimeMaps object. Its t0 and delta_t members will be updated.
 */
void FillTimeMapsWithT0AndDeltaTS(const std::string_view& image_name,
                                  const std::shared_ptr<snapengine::MetadataElement>& origin_metadata_root,
                                  TimeMaps& time_maps);

/**
 * Retrieves NoiseAzimuthVectors from the given noiseAzimuthVectorList metadata element.
 *
 * This method is a ported version of SNAP's
 * Sentinel1Itils.getAzimuthNoiseVector(final MetadataElement azimNoiseVectorListElem). It was moved here as it is used
 * only by thermal noise removal operator and there is no reason for it to be under Sentinel1Utils.
 *
 * @param azimuth_noise_vector_list_element noiseAzimuthVectorList metadata element.
 * @return Vector with NoiseAzimuthVector objects.
 */
std::vector<s1tbx::NoiseAzimuthVector> GetAzimuthNoiseVectorList(
    const std::shared_ptr<snapengine::MetadataElement>& azimuth_noise_vector_list_element);

/**
 * Retrieves NoiseAzimuthVectors from the given noiseRangeVectorList metadata element.
 *
 * This method is a ported version of SNAP's
 * Sentinel1Itils.getNoiseVector(final MetadataElement noiseVectorListElem). It was moved here as it is used
 * only by thermal noise removal operator and there is no reason for it to be under Sentinel1Utils.
 *
 * @param azimuth_noise_vector_list_element noiseRangeVectorList metadata element.
 * @return Vector with NoiseVector objects.
 */
std::vector<s1tbx::NoiseVector> GetNoiseVectorList(
    const std::shared_ptr<snapengine::MetadataElement>& noise_vector_list_element);

/**
 * Finds the NoiseVector line value of which is closest to the burst_center_line value.
 *
 * @param burst_center_line Index of the line in the middle of the burst.
 * @param noise_range_vectors List of computed NoiseVectors.
 * @return NoiseVector, which is closest to the center of the burst.
 */
s1tbx::NoiseVector GetBurstRangeVector(int burst_center_line,
                                       const std::vector<s1tbx::NoiseVector>& noise_range_vectors);

/**
 * Find index of the given line.
 *
 * @param line Line value.
 * @param lines Vector of lines.
 * @return Index of the provided line.
 */
size_t GetLineIndex(int line, const std::vector<int>& lines);

device::Matrix<double> BuildNoiseLutForTOPSSLC(Rectangle tile, const ThermalNoiseInfo& thermal_noise_info,
                                               ThreadData* thread_data);

device::Matrix<double> BuildNoiseLutForTOPSGRD(Rectangle tile, const ThermalNoiseInfo& thermal_noise_info,
                                               ThreadData* thread_data);

cuda::KernelArray<int> CalculateBurstIndices(Rectangle tile, int lines_per_burst, ThreadData* thread_data);

std::vector<size_t> DetermineNoiseVectorIndices(double start_az_time, double end_az_time,
                                                const std::vector<s1tbx::NoiseVector>& noise_range);

void FillRangeNoiseWithInterpolatedValues(const s1tbx::NoiseVector& nv, int first_range_sample, int last_range_sample,
                                          std::vector<double>& to_compute);

inline int GetSampleIndex(int sample, const std::vector<int>& pixels) {
    for (size_t i = 0; i < pixels.size(); i++) {
        if (sample < pixels.at(i)) {
            return (i > 0) ? i - 1 : 0;
        }
    }

    return pixels.size() - 2;
}

void FillAzimuthNoiseVectorWithInterpolatedValues(const s1tbx::NoiseAzimuthVector& v, int first_azimuth_line,
                                                  int last_azimuth_line, std::vector<double>& to_compute);

void ComputeNoiseMatrix(int tile_offset_x, int tile_offset_y, int nx0, int nx_max, int ny0, int ny_max,
                        const std::vector<int>& noise_range_vector_line,
                        const std::vector<std::vector<double>>& interpolated_range_vectors,
                        const std::vector<double>& interpolated_azimuth_vector,
                        std::vector<std::vector<double>>& values);

}  // namespace alus::tnr