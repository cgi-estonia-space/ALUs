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

#include <cstddef>
#include <vector>

namespace alus::featurextractiongabor {

std::vector<float> GenerateFrequencies(size_t frequency_count);
std::vector<float> GenerateOrientations(size_t orientation_count);
size_t ComputeFilterDimensionSeed(float sigma_value);
inline size_t ComputeFilterDimension(size_t filter_dimension_seed) { return 2 * filter_dimension_seed + 1; }
std::vector<float> ComputeWavelengthsFrom(const std::vector<float>& frequencies);
std::vector<float> ComputeSigmasFrom(const std::vector<float>& wavelengths);
void ComputeThetaX(float theta, size_t filter_dimension_seed, float* buf, size_t buf_size);
void ComputeThetaY(float theta, size_t filter_dimension_seed, float* buf, size_t buf_size);

struct FilterBankItemParameters {
    const std::vector<float>& theta_x;
    const std::vector<float>& theta_y;
    const float sigma;
    const float lambda;
    const float phy;
    const float gamma;
    const size_t filter_dimension_seed;
};
void ComputeFilterBankItem(float* buf, size_t buf_size, const FilterBankItemParameters& params);

struct FilterBankItem {
    size_t orientation_index;
    size_t frequency_index;
    size_t edge_size;
    std::vector<float> filter_buffer;
};
std::vector<FilterBankItem> CreateGaborFilterBank(size_t orientations, size_t frequencies);
std::vector<size_t> GetFilterDimensions(const std::vector<std::vector<float>>& filters);

}  // namespace alus::featurextractiongabor
