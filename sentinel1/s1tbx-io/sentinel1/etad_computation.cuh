/**
 * Inline ETAD calculations usable by ordinary C++ and CUDA translation units, without runtime or I/O dependencies.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cmath>
#include <limits>

#include "etad_grid.h"

#ifdef __CUDACC__
#define ALUS_ETAD_HOST_DEVICE __host__ __device__
#else
#define ALUS_ETAD_HOST_DEVICE
#endif

namespace alus::s1tbx::etad {

/** Convert seconds of range delay (or seconds/metre of gradient) to radians (or radians/metre). */
ALUS_ETAD_HOST_DEVICE inline double DelayToPhase(double delay, double frequency_hz) {
    constexpr double TWO_PI = 6.283185307179586476925286766559;
    return -TWO_PI * frequency_hz * delay;
}

/** InSAR range phase; unlike a geometric range correction, the ionospheric contribution is subtracted. */
ALUS_ETAD_HOST_DEVICE inline double RangePhase(double troposphere, double geodetic, double ionosphere,
                                               double calibration_seconds, double frequency_hz) {
    const double phase = DelayToPhase(troposphere + geodetic - ionosphere + calibration_seconds, frequency_hz);
    return std::isfinite(phase) ? phase : std::numeric_limits<double>::quiet_NaN();
}

/** Pair correction in radians; secondary fields must already be sampled in reference geometry. */
ALUS_ETAD_HOST_DEVICE inline double DifferentialPhase(double reference_phase, double secondary_phase,
                                                      double reference_height, double secondary_height,
                                                      double secondary_gradient) {
    return reference_phase - secondary_phase - secondary_gradient * (reference_height - secondary_height);
}

/**
 * Bilinear interpolation with no edge clamping. Returns NaN without four valid neighbours.
 * Caller guarantees values points to rows*columns doubles; dimensions and finite positive steps are host-validated.
 */
ALUS_ETAD_HOST_DEVICE inline double SampleGrid(GridView grid, GridGeometry geometry, double azimuth_time,
                                               double range_time) {
    constexpr double INVALID = std::numeric_limits<double>::quiet_NaN();
    const double row = (azimuth_time - geometry.azimuth_origin) / geometry.azimuth_step;
    const double column = (range_time - geometry.range_origin) / geometry.range_step;
    if (grid.rows < 2 || grid.columns < 2 || !std::isfinite(row) || !std::isfinite(column) || row < 0 || column < 0 ||
        row >= static_cast<double>(grid.rows - 1) || column >= static_cast<double>(grid.columns - 1)) {
        return INVALID;
    }
    const auto r = static_cast<size_t>(row);
    const auto c = static_cast<size_t>(column);
    const double dr = row - static_cast<double>(r);
    const double dc = column - static_cast<double>(c);
    const double c00 = grid.values[r * grid.columns + c];
    const double c01 = grid.values[r * grid.columns + c + 1];
    const double c10 = grid.values[(r + 1) * grid.columns + c];
    const double c11 = grid.values[(r + 1) * grid.columns + c + 1];
    if (!std::isfinite(c00) || !std::isfinite(c01) || !std::isfinite(c10) || !std::isfinite(c11)) {
        return INVALID;
    }
    return (1 - dr) * ((1 - dc) * c00 + dc * c01) + dr * ((1 - dc) * c10 + dc * c11);
}

/**
 * One pixel of SNAP 14.x's 25x25 troposphere-to-height regression (seconds/metre).
 * Caller guarantees equally sized valid buffers. Singular or invalid windows yield NaN; final column is zero.
 * Each output is independent, permitting a CUDA thread per grid point without intermediate allocations.
 */
ALUS_ETAD_HOST_DEVICE inline double TroposphericGradientAt(GridView troposphere, GridView height, size_t y, size_t x) {
    constexpr size_t HALF_WINDOW = 12;
    constexpr double INVALID = std::numeric_limits<double>::quiet_NaN();
    if (y >= height.rows || x >= height.columns || height.rows < 2 || height.columns < 2) {
        return INVALID;
    }
    if (x == height.columns - 1) {
        return 0.0;
    }
    const size_t x_start = x > HALF_WINDOW ? x - HALF_WINDOW : 0;
    const size_t x_end = height.columns - 1 - x > HALF_WINDOW ? x + HALF_WINDOW : height.columns - 1;
    const size_t y_start = y > HALF_WINDOW ? y - HALF_WINDOW : 0;
    const size_t y_end = height.rows - 1 - y > HALF_WINDOW ? y + HALF_WINDOW : height.rows - 1;
    double sum_x{};
    double sum_x2{};
    double sum_y{};
    double sum_xy{};
    for (size_t r = y_start; r <= y_end; ++r) {
        for (size_t c = x_start; c < x_end; ++c) {
            const size_t i = r * height.columns + c;
            const double dh = height.values[i + 1] - height.values[i];
            const double dt = troposphere.values[i + 1] - troposphere.values[i];
            sum_x += dh;
            sum_x2 += dh * dh;
            sum_y += dt;
            sum_xy += dh * dt;
        }
    }
    // Same 2x2 normal equations as SNAP/Jama, with partial pivoting.
    const double count = static_cast<double>((y_end - y_start + 1) * (x_end - x_start));
    const bool swap_rows = std::abs(sum_x) > std::abs(sum_x2);
    const double a = swap_rows ? sum_x : sum_x2;
    const double b = swap_rows ? count : sum_x;
    const double c = swap_rows ? sum_x2 : sum_x;
    const double d = swap_rows ? sum_x : count;
    const double u = swap_rows ? sum_y : sum_xy;
    const double v = swap_rows ? sum_xy : sum_y;
    if (a == 0 || !std::isfinite(a)) {
        return INVALID;
    }
    const double factor = c / a;
    const double pivot = d - factor * b;
    if (pivot == 0 || !std::isfinite(pivot)) {
        return INVALID;
    }
    const double intercept = (v - factor * u) / pivot;
    const double slope = (u - b * intercept) / a;
    return std::isfinite(slope) ? slope : INVALID;
}

}  // namespace alus::s1tbx::etad

#undef ALUS_ETAD_HOST_DEVICE
