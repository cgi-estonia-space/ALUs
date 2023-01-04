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

#include <array>
#include <map>

#include <cufft.h>

namespace alus::palsar {

// Ease of use wrappers around cuFFT plan APIs

// 2D image C2C float 1D FFT on each row(aka range direction)
[[nodiscard]] cufftHandle PlanRangeFFT(int range_size, int azimuth_size, bool auto_alloc);

// 2D image C2C float 1D FFT on each column(aka azimuth direction)
[[nodiscard]] cufftHandle PlanAzimuthFFT(int range_size, int azimuth_size, int range_stride, bool auto_alloc);

// 2D image C2C float 2D FFT
[[nodiscard]] cufftHandle Plan2DFFT(int range_size, int azimuth_size);

size_t EstimateRangeFFT(int range_size, int azimuth_size);

size_t EstimateAzimuthFFT(int range_size, int azimuth_size, int range_stride);

size_t Estimate2DFFT(int range_size, int azimuth_size);

std::map<int, std::array<int, 4>> GetOptimalFFTSizes();

}  // namespace alus::palsar
