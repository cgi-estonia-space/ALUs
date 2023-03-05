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

#include "get_position.h"
#include "position_data.h"
#include "s1tbx-commons/sar_geocoding.cuh"
#include "snap-engine-utilities/engine-utilities/eo/geo_utils.cuh"

namespace alus {
namespace terraincorrection {
inline __device__ __host__ snapengine::PosVector GetPositionWithLut(
    double time, cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors, const double* osv_lut) {
    const int nv{8};
    const int vectorsSize = vectors.size;
    // TODO: This should be done once.
    const double dt =
        (vectors.array[vectorsSize - 1].timeMjd_ - vectors.array[0].timeMjd_) / static_cast<double>(vectorsSize - 1);

    int i0;
    int iN;
    if (vectorsSize <= nv) {
        i0 = 0;
        iN = static_cast<int>(vectorsSize - 1);
    } else {
        i0 = std::max((int)((time - vectors.array[0].timeMjd_) / dt) - nv / 2 + 1, 0);
        iN = std::min(i0 + nv - 1, vectorsSize - 1);
        i0 = (iN < vectorsSize - 1 ? i0 : iN - nv + 1);
    }

    snapengine::PosVector result{0, 0, 0};
    for (int i = i0; i <= iN; ++i) {
        auto const orbI = vectors.array[i];

        double weight = 1;
        for (int j = i0; j <= iN; ++j) {
            if (j != i) {
                double const time2 = vectors.array[j].timeMjd_;
                // the following code line replaces the equivalent line of
                // weight *= (time - time2) / (orbI.timeMjd_ - time2);
                // on lowend gpus with double flops bottleneck this decrease is about 33% of the total GetSourceRect
                // kernel time and this means can be seconds of the total GPU time
                weight *= (time - time2) * osv_lut[i * vectorsSize + j];
            }
        }
        result.x += weight * orbI.xPos_;
        result.y += weight * orbI.yPos_;
        result.z += weight * orbI.zPos_;
    }
    return result;
}

inline __device__ __host__ double ComputeSlantRangeWithLut(
    double time, cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors, double* osv_lut,
    snapengine::PosVector earth_point, snapengine::PosVector& sensor_pos) {
    sensor_pos = GetPositionWithLut(time, vectors, osv_lut);
    double const xDiff = sensor_pos.x - earth_point.x;
    double const yDiff = sensor_pos.y - earth_point.y;
    double const zDiff = sensor_pos.z - earth_point.z;

    return std::sqrt(xDiff * xDiff + yDiff * yDiff + zDiff * zDiff);
}

inline __device__ __host__ bool GetPositionImpl(double lat, double lon, double alt, s1tbx::PositionData& satellite_pos,
                                                const GetPositionMetadata& metadata) {
    snapengine::geoutils::Geo2xyzWgs84Impl(lat, lon, alt, satellite_pos.earth_point);
    const auto zero_doppler_time = s1tbx::sargeocoding::GetEarthPointZeroDopplerTimeImpl(
        metadata.first_line_utc, metadata.line_time_interval, metadata.wavelength, satellite_pos.earth_point,
        metadata.sensor_position, metadata.sensor_velocity);
    if (zero_doppler_time == s1tbx::sargeocoding::NON_VALID_ZERO_DOPPLER_TIME) {
        return false;
    }
    satellite_pos.slant_range =
        ComputeSlantRangeWithLut(zero_doppler_time, metadata.orbit_state_vectors, metadata.orbit_state_vector_lut.array,
                                 satellite_pos.earth_point, satellite_pos.sensor_pos);

    satellite_pos.range_index = s1tbx::sargeocoding::ComputeRangeIndexSlcImpl(
        metadata.range_spacing, satellite_pos.slant_range, metadata.near_edge_slant_range);

    if (satellite_pos.range_index == -1.0) {
        return false;
    }

    satellite_pos.azimuth_index = (zero_doppler_time - metadata.first_line_utc) / metadata.line_time_interval;

    return true;
}
}  // namespace terraincorrection
}  // namespace alus