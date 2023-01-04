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

#include <cmath>
#include <complex>
#include <vector>

#include <cuComplex.h>

#include "sar_metadata.h"

namespace alus::palsar {

constexpr double SOL = 299792458;

inline int NextPower2(int value) {
    int r = 1;
    while (r < value) {
        r *= 2;
    }
    return r;
}

// https://en.wikipedia.org/wiki/Hann_function
inline void ApplyHanningWindow(std::vector<std::complex<float>>& data) {
    const size_t N = data.size();
    for (size_t i = 0; i < N; i++) {
        size_t n = i;
        double term = (2 * M_PI * n) / N;
        double m = 0.5 * (1 - cos(term));
        data[i] *= m;
    }
}

inline void ApplyHammingWindow(std::vector<std::complex<float>>& data) {
    const size_t N = data.size();
    for (size_t i = 0; i < N; i++) {
        size_t n = i;
        double term = (2 * M_PI * n) / N;
        double m = 0.54 - 0.46 * cos(term);
        data[i] *= m;
    }
}

inline double CalcVelocity(const OrbitInfo& info) {
    return sqrt(info.x_vel * info.x_vel + info.y_vel * info.y_vel + info.z_vel * info.z_vel);
}

inline void InplaceComplexToIntensity(cuComplex* data, size_t n) {
    for (size_t idx = 0; idx < n; idx++) {
        float i = data[idx].x;
        float q = data[idx].y;

        float intensity = i * i + q * q;

        auto& dest = data[idx / 2];
        if ((idx % 2) == 0) {
            dest.x = intensity;
        } else {
            dest.y = intensity;
        }
    }
}

inline double CalcR0(const SARMetadata& metadata, int range_pixel) {
    return metadata.slant_range_first_sample + range_pixel * metadata.range_spacing;
}

inline double CalcKa(const SARMetadata& metadata, int range_pixel) {
    const double Vr = metadata.results.Vr;
    const double R0 = CalcR0(metadata, range_pixel);
    return (2 * metadata.carrier_frequency * Vr * Vr) / (SOL * R0);
}

inline int CalcAperturePixels(const SARMetadata& metadata, int range_pixel) {
    double Ka = CalcKa(metadata, range_pixel);
    double fmax = (metadata.results.doppler_centroid +
                   (metadata.pulse_repetition_frequency * metadata.azimuth_bandwidth_fraction) / 2);
    double pixels = (fmax / Ka) * metadata.pulse_repetition_frequency;

    return std::round(pixels);
}

}  // namespace alus::palsar
