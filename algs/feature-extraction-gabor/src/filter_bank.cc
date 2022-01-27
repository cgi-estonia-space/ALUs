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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "../include/filter_bank.h"

namespace {
void ThrowIfDimensionsDoNotAlign(size_t filter_dimension_seed, size_t buf_size) {
    if (alus::featurextractiongabor::ComputeFilterDimension(filter_dimension_seed) !=
        static_cast<size_t>(std::sqrt(static_cast<double>(buf_size)))) {
        throw std::invalid_argument("Filter dimension seed does not match the buffer size.");
    }
}
}  // namespace

namespace alus::featurextractiongabor {

std::vector<float> GenerateFrequencies(size_t frequency_count) {
    std::vector<float> values(frequency_count);
    for (size_t i = 0; i < frequency_count; i++) {
        values.at(i) = (static_cast<float>(i) + 0.5f) / static_cast<float>(frequency_count);
    }

    return values;
}

std::vector<float> GenerateOrientations(size_t orientation_count) {
    std::vector<float> values(orientation_count);
    const auto oc_divide_f = static_cast<float>(orientation_count);
    for (size_t o = 0; o < orientation_count; o++) {
        values.at(o) = (static_cast<float>(o) / oc_divide_f) * M_PIf32;
    }

    return values;
}

void ComputeThetaX(float theta, size_t filter_dimension_seed, float* buf, size_t buf_size) {
    ThrowIfDimensionsDoNotAlign(filter_dimension_seed, buf_size);
    const auto theta_cos = std::cos(theta);
    const auto theta_sin = std::sin(theta);
    const auto dimension = static_cast<int>(ComputeFilterDimension(filter_dimension_seed));
    const auto filter_dimension_seed_offset = -1 * static_cast<int>(filter_dimension_seed);
    for (auto x{0}; x < dimension; x++) {
        for (auto y{0}; y < dimension; y++) {
            buf[x * dimension + y] = static_cast<float>(filter_dimension_seed_offset + y) * theta_cos +
                                     static_cast<float>(filter_dimension_seed_offset + x) * theta_sin;
        }
    }
}

void ComputeThetaY(float theta, size_t filter_dimension_seed, float* buf, size_t buf_size) {
    ThrowIfDimensionsDoNotAlign(filter_dimension_seed, buf_size);
    const auto theta_cos = std::cos(theta);
    const auto theta_sin = std::sin(theta);
    const auto dimension = static_cast<int>(ComputeFilterDimension(filter_dimension_seed));
    const auto filter_dimension_seed_negative = -1 * static_cast<int>(filter_dimension_seed);
    const auto filter_dimension_seed_int = static_cast<int>(filter_dimension_seed);
    for (auto x{0}; x < dimension; x++) {
        for (auto y{0}; y < dimension; y++) {
            buf[x * dimension + y] = static_cast<float>(filter_dimension_seed_int - y) * theta_sin +
                                     static_cast<float>(filter_dimension_seed_negative + x) * theta_cos;
        }
    }
}

size_t ComputeFilterDimensionSeed(float sigma_value) { return static_cast<size_t>(std::round(3 * sigma_value)); }

std::vector<float> ComputeWavelengthsFrom(const std::vector<float>& frequencies) {
    std::vector<float> wl;
    std::transform(frequencies.cbegin(), frequencies.cend(), std::back_inserter(wl), [](float f) { return 1 / f; });
    return wl;
}
std::vector<float> ComputeSigmasFrom(const std::vector<float>& wavelengths) {
    std::vector<float> sigmas;
    std::transform(wavelengths.cbegin(), wavelengths.cend(), std::back_inserter(sigmas),
                   [](float lambda) { return 0.56f * lambda; });
    return sigmas;
}
void ComputeFilterBankItem(float* buf, size_t buf_size, const FilterBankItemParameters& params) {
    ThrowIfDimensionsDoNotAlign(params.filter_dimension_seed, buf_size);
    const auto dimension = static_cast<int>(ComputeFilterDimension(params.filter_dimension_seed));
    const auto sigma_squared = params.sigma * params.sigma;
    constexpr auto two_pi_f = 2 * M_PIf32;
    constexpr auto two_pi_d = 2 * M_PIf64;
    for (auto x{0}; x < dimension; x++) {
        for (auto y{0}; y < dimension; y++) {
            const auto index = x * dimension + y;
            // Not using squared variable, since multiplication of floats not associative.
            const float a = 1 / (two_pi_f * params.sigma * params.sigma);
            const float prod1 = (params.theta_x[index] * params.theta_x[index]) / sigma_squared;
            const float prod2 =
                (params.gamma * params.gamma) * (params.theta_y[index] * params.theta_y[index]) / sigma_squared;
            const float expression = -1.0f * (prod1 + prod2) / 2;
            const float exp = std::exp(expression);

            // Using M_PIf64 here to trigger std::cos(double) overload which does not generate different results
            // after every pi/2 compared to original implementation
            buf[index] =
                a * exp * static_cast<float>(std::cos(two_pi_d * params.theta_x[index] / params.lambda + params.phy));
        }
    }
}

std::vector<FilterBankItem> CreateGaborFilterBank(size_t orientations, size_t frequencies) {
    const auto& theta_raw = GenerateOrientations(orientations);
    const auto& f_raw = GenerateFrequencies(frequencies);
    const auto& lambda = ComputeWavelengthsFrom(f_raw);
    const auto& sigmas = ComputeSigmasFrom(lambda);
    const float phy{0.0f};
    const float gamma{0.5f};

    assert((f_raw.size() == lambda.size()) && (lambda.size() == sigmas.size()) && (f_raw.size() == frequencies));
    assert(theta_raw.size() == orientations);

    std::vector<FilterBankItem> filter_bank(orientations * frequencies);
    for (size_t freq_i{0}; freq_i < frequencies; freq_i++) {
        const auto filter_dimension_seed = ComputeFilterDimensionSeed(sigmas.at(freq_i));
        const auto matrix_dim = ComputeFilterDimension(filter_dimension_seed);
        for (size_t theta_i{0}; theta_i < orientations; theta_i++) {
            std::vector<float> theta_x(matrix_dim * matrix_dim);
            ComputeThetaX(theta_raw.at(theta_i), filter_dimension_seed, theta_x.data(), theta_x.size());
            std::vector<float> theta_y(matrix_dim * matrix_dim);
            ComputeThetaY(theta_raw.at(theta_i), filter_dimension_seed, theta_y.data(), theta_y.size());
            const auto filter_item_index = freq_i * orientations + theta_i;
            auto& bank_item = filter_bank.at(filter_item_index);
            bank_item.frequency_index = freq_i;
            bank_item.orientation_index = theta_i;
            bank_item.edge_size = matrix_dim;
            bank_item.filter_buffer.resize(matrix_dim * matrix_dim);
            ComputeFilterBankItem(
                bank_item.filter_buffer.data(), bank_item.filter_buffer.size(),
                {theta_x, theta_y, sigmas.at(freq_i), lambda.at(freq_i), phy, gamma, filter_dimension_seed});
        }
    }

    return filter_bank;
}

}  // namespace alus::featurextractiongabor
