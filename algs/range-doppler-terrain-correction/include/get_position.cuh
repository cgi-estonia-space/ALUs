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
#include "interpolations.h"
#include "kernel_array.h"
#include "polynomials.h"
#include "position_data.h"
#include "s1tbx-commons/sar_geocoding.cuh"
#include "snap-engine-utilities/engine-utilities/eo/geo_utils.cuh"
#include "srgr_coefficients.h"

namespace alus::terraincorrection {
inline __device__ __host__ snapengine::PosVector GetPositionWithLut(
    double time, cuda::KernelArray<snapengine::OrbitStateVectorComputation> vectors, double* osv_lut) {
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

/**
 * Compute ground range for given slant range. Ported from SNAP's s1tbx SARGeocoding.java.
 *
 * @param sourceImageWidth    The source image width.
 * @param groundRangeSpacing  The ground range spacing.
 * @param slantRange          The slant range in meters.
 * @param srgrCoeff           The SRGR coefficients for converting ground range to slant range.
 *                            Here it is assumed that the polynomial is given by
 *                            c0 + c1*x + c2*x^2 + ... + cn*x^n, where {c0, c1, ..., cn} are the SRGR coefficients.
 * @param ground_range_origin The ground range origin.
 * @return The ground range in meters.
 */
inline __device__ __host__ double ComputeGroundRange(int source_image_width, double ground_range_spacing,
                                                     double slant_range, cuda::KernelArray<double> srgr_coefficients,
                                                     double ground_range_origin) {
    // binary search is used in finding the ground range for given slant range
    double lower_bound = ground_range_origin;
    const double lower_bound_slant_range =
        math::polynomials::CalculateValue(lower_bound, srgr_coefficients.array, srgr_coefficients.size);
    if (slant_range < lower_bound_slant_range) {
        return -1.0;
    }

    double upper_bound = ground_range_origin + source_image_width * ground_range_spacing;
    const double upper_bound_slant_range =
        math::polynomials::CalculateValue(upper_bound, srgr_coefficients.array, srgr_coefficients.size);
    if (slant_range > upper_bound_slant_range) {
        return -1.0;
    }

    // start binary search
    double mid_slant_range;
    while (upper_bound - lower_bound > 0.0) {
        const double mid = (lower_bound + upper_bound) / 2.0;
        mid_slant_range = math::polynomials::CalculateValue(mid, srgr_coefficients.array, srgr_coefficients.size);
        if (mid_slant_range < slant_range) {
            lower_bound = mid;
        } else if (mid_slant_range > slant_range) {
            upper_bound = mid;
        } else {
            const double a = mid_slant_range - slant_range;
            if ((a > 0 && a < 0.1) || (a <= 0.0 && 0.0 - a < 0.1)) {
                return mid;
            }
        }
    }

    return -1.0;
}

/**
 * Compute range index in source image for earth point with given zero Doppler time and slant range. Ported from SNAP's
 * s1tbx SARGeocoding.java
 *
 * @param zeroDopplerTime The zero Doppler time in MJD.
 * @param slantRange      The slant range in meters.
 * @return The range index.
 */
inline __device__ __host__ double ComputeRangeIndexGrdImpl(cuda::KernelArray<SrgrCoefficientsDevice>& srgr_coefficients,
                                                           cuda::KernelArray<double> srgr_polynomial_calc_buf,
                                                           double zero_doppler_time, int source_image_width,
                                                           double range_spacing, double slant_range) {
    constexpr auto NO_RESULT{-1.0};

    if (srgr_coefficients.size == 0 || srgr_coefficients.array == nullptr) {
        return NO_RESULT;
    }

    if (srgr_coefficients.size == 1) {
        const auto ground_range =
            ComputeGroundRange(source_image_width, range_spacing, slant_range, srgr_coefficients.array[0].coefficients,
                               srgr_coefficients.array[0].ground_range_origin);
        if (ground_range < 0.0) {
            return -1.0;
        } else {
            return (ground_range - srgr_coefficients.array[0].ground_range_origin) / range_spacing;
        }
    }

    size_t target_index = 0;
    for (size_t i = 0; i < srgr_coefficients.size && zero_doppler_time >= srgr_coefficients.array[i].time_mjd; i++) {
        target_index = i;
    }

    const auto interpolation_length = srgr_coefficients.array[target_index].coefficients.size;
    if (interpolation_length > srgr_polynomial_calc_buf.size) {
        return NO_RESULT;
    }
    //    final double[] srgrCoefficients = new double[srgr_coefficients[target_index].coefficients.size];
    if (target_index == srgr_coefficients.size - 1) {
        target_index--;
    }

    const auto mu =
        (zero_doppler_time - srgr_coefficients.array[target_index].time_mjd) /
        (srgr_coefficients.array[target_index + 1].time_mjd - srgr_coefficients.array[target_index].time_mjd);
    if (mu > 1.0) {
        return NO_RESULT;
    }
    for (size_t i = 0; i < interpolation_length; i++) {
        srgr_polynomial_calc_buf.array[i] =
            math::interpolations::Linear(srgr_coefficients.array[target_index].coefficients.array[i],
                                         srgr_coefficients.array[target_index + 1].coefficients.array[i], mu);
    }
    const auto ground_range =
        ComputeGroundRange(source_image_width, range_spacing, slant_range, srgr_polynomial_calc_buf,
                           srgr_coefficients.array[target_index].ground_range_origin);
    if (ground_range < 0.0) {
        return NO_RESULT;
    } else {
        return (ground_range - srgr_coefficients.array[target_index].ground_range_origin) / range_spacing;
    }
}

inline __device__ __host__ bool GetPositionImpl(double lat, double lon, double alt, s1tbx::PositionData& satellite_pos,
                                                const GetPositionMetadata& metadata,
                                                cuda::KernelArray<SrgrCoefficientsDevice>& srgr_coefficients,
                                                cuda::KernelArray<double> srgr_polynomial_calc_buf) {
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

    if (srgr_coefficients.size > 0) {
        satellite_pos.range_index =
            ComputeRangeIndexGrdImpl(srgr_coefficients, srgr_polynomial_calc_buf, zero_doppler_time,
                                     metadata.source_image_width, metadata.range_spacing, satellite_pos.slant_range);
    } else {
        satellite_pos.range_index = s1tbx::sargeocoding::ComputeRangeIndexSlcImpl(
            metadata.range_spacing, satellite_pos.slant_range, metadata.near_edge_slant_range);
    }

    if (satellite_pos.range_index == -1.0) {
        return false;
    }

    satellite_pos.azimuth_index = (zero_doppler_time - metadata.first_line_utc) / metadata.line_time_interval;

    return true;
}
}  // namespace alus::terraincorrection
