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

#include "thermal_noise_utils.h"

#include <boost/algorithm/string.hpp>

#include <driver_types.h>
#include <cmath>
#include <cstddef>
#include <set>
#include <string_view>

#include "s1tbx-commons/sentinel1_utils.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"
#include "thermal_noise_data_structures.h"
#include "thermal_noise_kernel.h"
#include "time_maps.h"

namespace alus::tnr {

ThermalNoiseInfo GetThermalNoiseInfoForBursts(
    std::string_view polarisation, std::string_view sub_swath,
    const std::shared_ptr<snapengine::MetadataElement>& origin_metadata_root) {
    ThermalNoiseInfo thermal_noise_info;

    const auto root_noise_element = origin_metadata_root->GetElement(snapengine::AbstractMetadata::NOISE);

    for (const auto& list_element : root_noise_element->GetElements()) {
        const auto image_name = list_element->GetName();
        if (boost::algorithm::icontains(image_name, polarisation) &&
            boost::algorithm::icontains(image_name, sub_swath)) {
            const auto noise_element = list_element->GetElement(snapengine::AbstractMetadata::NOISE);

            thermal_noise_info.noise_azimuth_vectors = GetAzimuthNoiseVectorList(
                noise_element->GetElement(snapengine::AbstractMetadata::NOISE_AZIMUTH_VECTOR_LIST));
            thermal_noise_info.noise_range_vectors =
                GetNoiseVectorList(noise_element->GetElement(snapengine::AbstractMetadata::NOISE_RANGE_VECTOR_LIST));
        }
    }

    for (const auto& image_element :
         origin_metadata_root->GetElement(snapengine::AbstractMetadata::ANNOTATION)->GetElements()) {
        const auto image_name = image_element->GetName();
        if (boost::algorithm::icontains(image_name, polarisation) &&
            boost::algorithm::icontains(image_name, sub_swath)) {
            const auto swath_timing_element = image_element->GetElement(snapengine::AbstractMetadata::PRODUCT)
                                                  ->GetElement(snapengine::AbstractMetadata::SWATH_TIMING);

            thermal_noise_info.lines_per_burst =
                swath_timing_element->GetAttributeInt(snapengine::AbstractMetadata::LINES_PER_BURST);
            const auto burst_list_array =
                swath_timing_element->GetElement(snapengine::AbstractMetadata::BURST_LIST)->GetElements();
            thermal_noise_info.burst_to_range_vector_map.resize(burst_list_array.size());
            for (size_t i = 0; i < burst_list_array.size(); i++) {
                const auto burst_center_line =
                    i * thermal_noise_info.lines_per_burst + thermal_noise_info.lines_per_burst / 2;
                thermal_noise_info.burst_to_range_vector_map.at(i) =
                    GetBurstRangeVector(static_cast<int>(burst_center_line), thermal_noise_info.noise_range_vectors);
            }
        }
    }

    return thermal_noise_info;
}

ThermalNoiseInfo GetThermalNoiseInfoForGrd(std::string_view polarisation,
                                           const std::shared_ptr<snapengine::MetadataElement>& origin_metadata_root) {
    ThermalNoiseInfo thermal_noise_info;

    const auto root_noise_element = origin_metadata_root->GetElement(snapengine::AbstractMetadata::NOISE);

    for (const auto& list_element : root_noise_element->GetElements()) {
        const auto image_name = list_element->GetName();
        if (boost::algorithm::icontains(image_name, polarisation)) {
            const auto noise_element = list_element->GetElement(snapengine::AbstractMetadata::NOISE);

            thermal_noise_info.noise_azimuth_vectors = GetAzimuthNoiseVectorList(
                noise_element->GetElement(snapengine::AbstractMetadata::NOISE_AZIMUTH_VECTOR_LIST));
            thermal_noise_info.noise_range_vectors =
                GetNoiseVectorList(noise_element->GetElement(snapengine::AbstractMetadata::NOISE_RANGE_VECTOR_LIST));
        }
    }

    return thermal_noise_info;
}

void FillTimeMapsWithT0AndDeltaTS(const std::string_view& image_name,
                                  const std::shared_ptr<snapengine::MetadataElement>& origin_metadata_root,
                                  TimeMaps& time_maps) {
    const auto annotation_elements =
        origin_metadata_root->GetElement(snapengine::AbstractMetadata::ANNOTATION)->GetElements();

    for (const auto& annotation_element : annotation_elements) {
        if (annotation_element->GetName().find(image_name) != std::string::npos) {
            const auto image_information_element = annotation_element->GetElement(snapengine::AbstractMetadata::PRODUCT)
                                                       ->GetElement(snapengine::AbstractMetadata::IMAGE_ANNOTATION)
                                                       ->GetElement(snapengine::AbstractMetadata::IMAGE_INFORMATION);

            const auto t_0 = s1tbx::Sentinel1Utils::GetTime(image_information_element,
                                                            snapengine::AbstractMetadata::PRODUCT_FIRST_LINE_UTC_TIME)
                                 ->GetMjd();
            time_maps.t_0_map.insert_or_assign(image_name.data(), t_0);

            const auto delta_ts =
                image_information_element->GetAttributeDouble(snapengine::AbstractMetadata::AZIMUTH_TIME_INTERVAL) /
                snapengine::eo::constants::SECONDS_IN_DAY;
            time_maps.delta_t_map.insert_or_assign(image_name.data(), delta_ts);

            return;
        }
    }
}

std::vector<s1tbx::NoiseAzimuthVector> GetAzimuthNoiseVectorList(
    const std::shared_ptr<snapengine::MetadataElement>& azimuth_noise_vector_list_element) {
    const auto elements_list = azimuth_noise_vector_list_element->GetElements();
    std::vector<s1tbx::NoiseAzimuthVector> noise_vector_list;
    noise_vector_list.reserve(elements_list.size());

    for (const auto& noise_vector_element : elements_list) {
        const auto line_element = noise_vector_element->GetElement(snapengine::AbstractMetadata::LINE);
        const auto line_attribute = line_element->GetAttributeString(snapengine::AbstractMetadata::LINE);
        const auto line_count = line_element->GetAttributeString(snapengine::AbstractMetadata::COUNT);

        const auto count = std::stoi(line_count);
        std::vector<int> line_vector(count);

        const std::string delimiter = boost::icontains(line_attribute, "\t") ? "\t" : " ";
        s1tbx::Sentinel1Utils::AddToArray(line_vector, 0, line_attribute, delimiter);

        const auto noise_lut_element =
            noise_vector_element->GetElement(snapengine::AbstractMetadata::NOISE_AZIMUTH_LUT);
        const auto noise_lut_attribute =
            noise_lut_element->GetAttributeString(snapengine::AbstractMetadata::NOISE_AZIMUTH_LUT);
        std::vector<float> noise_lut_vector(count);
        s1tbx::Sentinel1Utils::AddToArray(noise_lut_vector, 0, noise_lut_attribute, delimiter);

        const auto swath = noise_vector_element->ContainsAttribute(snapengine::AbstractMetadata::SWATH)
                               ? noise_vector_element->GetAttributeString(snapengine::AbstractMetadata::SWATH)
                               : "";
        const auto first_azimuth_line =
            noise_vector_element->ContainsAttribute(snapengine::AbstractMetadata::FIRST_AZIMUTH_LINE)
                ? std::stoi(noise_vector_element->GetAttributeString(snapengine::AbstractMetadata::FIRST_AZIMUTH_LINE))
                : -1;
        const auto last_azimuth_line =
            noise_vector_element->ContainsAttribute(snapengine::AbstractMetadata::LAST_AZIMUTH_LINE)
                ? std::stoi(noise_vector_element->GetAttributeString(snapengine::AbstractMetadata::LAST_AZIMUTH_LINE))
                : -1;
        const auto first_range_sample =
            noise_vector_element->ContainsAttribute(snapengine::AbstractMetadata::FIRST_RANGE_SAMPLE)
                ? std::stoi(noise_vector_element->GetAttributeString(snapengine::AbstractMetadata::FIRST_RANGE_SAMPLE))
                : -1;
        const auto last_range_sample =
            noise_vector_element->ContainsAttribute(snapengine::AbstractMetadata::LAST_RANGE_SAMPLE)
                ? std::stoi(noise_vector_element->GetAttributeString(snapengine::AbstractMetadata::LAST_RANGE_SAMPLE))
                : -1;

        noise_vector_list.push_back({swath, first_azimuth_line, first_range_sample, last_azimuth_line,
                                     last_range_sample, line_vector, noise_lut_vector});
    }

    return noise_vector_list;
}

std::vector<s1tbx::NoiseVector> GetNoiseVectorList(
    const std::shared_ptr<snapengine::MetadataElement>& noise_vector_list_element) {
    const auto list = noise_vector_list_element->GetElements();
    std::vector<s1tbx::NoiseVector> noise_vector_list;
    noise_vector_list.reserve(list.size());

    for (const auto& noise_vector_element : list) {
        const auto time =
            s1tbx::Sentinel1Utils::GetTime(noise_vector_element, snapengine::AbstractMetadata::AZIMUTH_TIME)->GetMjd();
        const auto line = std::stoi(noise_vector_element->GetAttributeString(snapengine::AbstractMetadata::LINE));

        const auto pixel_element = noise_vector_element->GetElement(snapengine::AbstractMetadata::PIXEL);
        const auto pixel_attribute = pixel_element->GetAttributeString(snapengine::AbstractMetadata::PIXEL);
        const auto count = std::stoi(pixel_element->GetAttributeString(snapengine::AbstractMetadata::COUNT));
        std::vector<int> pixel_vector(count);
        s1tbx::Sentinel1Utils::AddToArray(pixel_vector, 0, pixel_attribute, " ");

        const auto noise_lut_element = noise_vector_element->GetElement(snapengine::AbstractMetadata::NOISE_RANGE_LUT);
        const auto noise_lut_attribute =
            noise_lut_element->GetAttributeString(snapengine::AbstractMetadata::NOISE_RANGE_LUT);
        std::vector<float> noise_lut_vector(count);
        s1tbx::Sentinel1Utils::AddToArray(noise_lut_vector, 0, noise_lut_attribute, " ");

        noise_vector_list.push_back({time, line, pixel_vector, noise_lut_vector});
    }

    return noise_vector_list;
}
s1tbx::NoiseVector GetBurstRangeVector(int burst_center_line,
                                       const std::vector<s1tbx::NoiseVector>& noise_range_vectors) {
    size_t closest{0};
    for (size_t i = 1; i < noise_range_vectors.size(); i++) {
        if (std::abs(burst_center_line - noise_range_vectors.at(i).line) <
            std::abs(burst_center_line - noise_range_vectors.at(closest).line)) {
            closest = i;
        }
    }
    return noise_range_vectors.at(closest);
}
size_t GetLineIndex(int line, const std::vector<int>& lines) {
    // NB! Lines length is assumed to be larger than 2.
    for (size_t i = 0; i < lines.size(); ++i) {
        if (line < lines.at(i)) {
            return i > 0 ? i - 1 : 0;
        }
    }

    return lines.size() - 2;
}
device::Matrix<double> BuildNoiseLutForTOPSSLC(Rectangle tile, const ThermalNoiseInfo& thermal_noise_info,
                                               ThreadData* thread_data) {
    // INTERPOLATE NOISE AZIMUTH KERNEL
    const auto d_azimuth_vector = thermal_noise_info.noise_azimuth_vectors.at(0).ToDeviceVector();
    const auto starting_line_index = GetLineIndex(tile.y, thermal_noise_info.noise_azimuth_vectors.at(0).lines);
    const auto interpolated_azimuth_vector = LaunchInterpolateNoiseAzimuthVectorKernel(
        d_azimuth_vector, tile.y, tile.y + tile.height + 1, starting_line_index, thread_data->stream);

    // GET SAMPLE INDICES
    std::vector<s1tbx::DeviceNoiseVector> h_burst_to_range_map(thermal_noise_info.burst_to_range_vector_map.size());
    std::transform(std::begin(thermal_noise_info.burst_to_range_vector_map),
                   std::end(thermal_noise_info.burst_to_range_vector_map), std::begin(h_burst_to_range_map),
                   [](auto& vector) { return vector.ToDeviceVector(); });
    cuda::KernelArray<s1tbx::DeviceNoiseVector> d_burst_to_range_map{nullptr, h_burst_to_range_map.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_burst_to_range_map.array, d_burst_to_range_map.ByteSize()));
    CHECK_CUDA_ERR(cudaMemcpy(d_burst_to_range_map.array, h_burst_to_range_map.data(), d_burst_to_range_map.ByteSize(),
                              cudaMemcpyHostToDevice));

    const auto d_burst_indices = CalculateBurstIndices(tile, thermal_noise_info.lines_per_burst, thread_data);
    const auto d_sample_indices =
        LaunchGetSampleIndexKernel(tile, d_burst_to_range_map, d_burst_indices, thread_data->stream);

    // INTERPOLATE NOISE RANGE VECTOR
    const auto index_to_interpolated_range_vector_map = LaunchInterpolateNoiseRangeVectorsKernel(
        tile, d_burst_indices, d_sample_indices, d_burst_to_range_map, thread_data->stream);
    // Calculate noise matrix
    const auto noise_matrix =
        CalculateNoiseMatrix(tile, thermal_noise_info.lines_per_burst, interpolated_azimuth_vector,
                             index_to_interpolated_range_vector_map, thread_data->stream);

    // DEALLOCATE MEMORY
    CHECK_CUDA_ERR(cudaFree(d_azimuth_vector.lines.array));
    CHECK_CUDA_ERR(cudaFree(d_azimuth_vector.noise_azimuth_lut.array));
    CHECK_CUDA_ERR(cudaFree(interpolated_azimuth_vector.array));
    CHECK_CUDA_ERR(cudaFree(d_sample_indices.array));
    for (auto& vector : h_burst_to_range_map) {
        CHECK_CUDA_ERR(cudaFree(vector.pixels.array));
        CHECK_CUDA_ERR(cudaFree(vector.noise_lut.array));
    }
    CHECK_CUDA_ERR(cudaFree(d_burst_to_range_map.array));
    CHECK_CUDA_ERR(cudaFree(d_burst_indices.array));
    device::DestroyBurstIndexToInterpolatedRangeVectorMap(index_to_interpolated_range_vector_map);

    return noise_matrix;
}

device::Matrix<double> BuildNoiseLutForTOPSGRD(Rectangle tile, const ThermalNoiseInfo& thermal_noise_info,
                                               ThreadData*) {
    const auto x_max = tile.x + tile.width - 1;
    const auto y_max = tile.y + tile.height - 1;
    bool has_data{false};
    // Although it is GRD, the noise azimuth vectors are segregated by the originating swath in the metadata.
    // Hence 3 vectors, each for a subswath.
    std::vector<std::vector<double>> noise_matrix(tile.height);
    for (auto& v : noise_matrix) {
        v.resize(tile.width);
    }
    for (auto& nav : thermal_noise_info.noise_azimuth_vectors) {
        const auto nx0 = std::max(tile.x, nav.first_range_sample);
        const auto nx_max = std::min(x_max, nav.last_range_sample);
        const auto ny0 = std::max(tile.y, nav.first_azimuth_line);
        const auto ny_max = std::min(y_max, nav.last_azimuth_line);

        if (nx0 >= nx_max || ny0 >= ny_max) {
            continue;
        }

        has_data = true;
        // Contains a single polarisation data only for GRD. If needed more can be added later.
        const auto first_line_time = thermal_noise_info.time_maps.t_0_map.begin()->second;
        const auto line_time_interval = thermal_noise_info.time_maps.delta_t_map.begin()->second;
        const auto start_azim_time = first_line_time + nav.first_azimuth_line * line_time_interval;
        const auto end_azim_time = first_line_time + nav.last_azimuth_line * line_time_interval;
        const auto noise_vector_indices =
            DetermineNoiseVectorIndices(start_azim_time, end_azim_time, thermal_noise_info.noise_range_vectors);
        if (noise_vector_indices.size() == 0) {
            throw std::runtime_error("There were no matching noise values from noise vectors for azimuth range [" +
                                     std::to_string(start_azim_time) + ", " + std::to_string(end_azim_time) +
                                     "]. Tile properties x:" + std::to_string(tile.x) + " y:" + std::to_string(tile.y) +
                                     " w:" + std::to_string(tile.width) + " h:" + std::to_string(tile.height));
        }

        const auto interpolated_range_value_count = nx_max - nx0 + 1;
        std::vector<std::vector<double>> interpolated_range_vectors(noise_vector_indices.size());
        std::vector<int> noise_range_vector_line(noise_vector_indices.size());
        for (int i{0}; i < static_cast<int>(noise_vector_indices.size()); i++) {
            const auto& noise_range_vector = thermal_noise_info.noise_range_vectors.at(noise_vector_indices.at(i));
            noise_range_vector_line[i] = noise_range_vector.line;

            interpolated_range_vectors.at(i) = std::vector<double>(interpolated_range_value_count);
            FillRangeNoiseWithInterpolatedValues(noise_range_vector, nx0, nx_max, interpolated_range_vectors.at(i));
        }

        std::vector<double> interpolated_azimuth_vector(ny_max - ny0 + 1);
        FillAzimuthNoiseVectorWithInterpolatedValues(nav, ny0, ny_max, interpolated_azimuth_vector);
        ComputeNoiseMatrix(tile.x, tile.y, nx0, nx_max, ny0, ny_max, noise_range_vector_line,
                           interpolated_range_vectors, interpolated_azimuth_vector, noise_matrix);
    }

    if (!has_data) {
        throw std::runtime_error(
            "Could not locate a tile for a given range and azimuth set of a product for thermal noise removal "
            "calculation. Tile properties x:" +
            std::to_string(tile.x) + " y:" + std::to_string(tile.y) + " w:" + std::to_string(tile.width) +
            " h:" + std::to_string(tile.height));
    }

    const auto noise_matrix_dev = device::CreateKernelMatrix<double>(tile.width, tile.height, noise_matrix);
    return noise_matrix_dev;
}

cuda::KernelArray<int> CalculateBurstIndices(Rectangle tile, int lines_per_burst, ThreadData* thread_data) {
    const int average_burst_count{9};
    std::vector<int> burst_indices;
    burst_indices.reserve(average_burst_count);
    for (int i = 0; i < tile.height; ++i) {
        auto y = i + tile.y;
        const auto burst_index = y / lines_per_burst;
        burst_indices.emplace_back(burst_index);
        i += lines_per_burst;
        if (i >= tile.height) {  // Case for when lines_per_burst is larger than tile.
            const auto last_index = (tile.height + tile.y - 1) / lines_per_burst;
            if (last_index != burst_index) {
                burst_indices.emplace_back(last_index);
            }
        }
    }
    (void)thread_data;
    cuda::KernelArray<int> d_burst_indices{nullptr, burst_indices.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_burst_indices.array, d_burst_indices.ByteSize()));
    CHECK_CUDA_ERR(
        cudaMemcpy(d_burst_indices.array, burst_indices.data(), d_burst_indices.ByteSize(), cudaMemcpyHostToDevice));

    return d_burst_indices;
}

std::vector<size_t> DetermineNoiseVectorIndices(double start_az_time, double end_az_time,
                                                const std::vector<s1tbx::NoiseVector>& noise_range) {
    std::vector<size_t> results;
    for (size_t i{0}; i < noise_range.size(); i++) {
        if (noise_range.at(i).time_mjd >= start_az_time && noise_range.at(i).time_mjd <= end_az_time) {
            results.push_back(i);
        }
    }

    return results;
}

inline double InterpolateNoise(int p1, int p2, double noise1, double noise2, int sample_index) {
    return noise1 + (static_cast<double>(sample_index - p1) / static_cast<double>(p2 - p1)) * (noise2 - noise1);
}

void FillRangeNoiseWithInterpolatedValues(const s1tbx::NoiseVector& nv, int first_range_sample, int last_range_sample,
                                          std::vector<double>& to_compute) {
    const int nv_pix_len{static_cast<int>(nv.pixels.size())};
    if (nv_pix_len < 2) {
        throw std::runtime_error(std::string(__FUNCTION__) + " expects more elements in noise vector than " +
                                 std::to_string(nv_pix_len) + ". Please check the input metadata.");
    }

    size_t computed_index{0};
    auto sample_index = GetSampleIndex(first_range_sample, nv.pixels);
    for (int s{first_range_sample}; s <= last_range_sample; s++) {
        if (s > nv.pixels.at(sample_index + 1) && sample_index < (nv_pix_len - 2)) {
            sample_index++;
        }

        to_compute.at(computed_index++) =
            InterpolateNoise(nv.pixels.at(sample_index), nv.pixels.at(sample_index + 1), nv.noise_lut.at(sample_index),
                             nv.noise_lut.at(sample_index + 1), s);
    }
}

void FillAzimuthNoiseVectorWithInterpolatedValues(const s1tbx::NoiseAzimuthVector& v, int first_azimuth_line,
                                                  int last_azimuth_line, std::vector<double>& to_compute) {
    const int nv_pix_len{static_cast<int>(v.lines.size())};
    if (nv_pix_len < 2) {
        for (int line = first_azimuth_line; line <= last_azimuth_line; line++) {
            to_compute.at(line - first_azimuth_line) = v.noise_azimuth_lut.front();
        }
    } else {
        int line_index = GetSampleIndex(first_azimuth_line, v.lines);
        for (int line = first_azimuth_line; line <= last_azimuth_line; line++) {
            if (line > v.lines.at(line_index + 1) && line_index < nv_pix_len - 2) {
                line_index++;
            }

            to_compute.at(line - first_azimuth_line) =
                InterpolateNoise(v.lines.at(line_index), v.lines.at(line_index + 1), v.noise_azimuth_lut.at(line_index),
                                 v.noise_azimuth_lut.at(line_index + 1), line);
        }
    }
}

void ComputeNoiseMatrix(int tile_offset_x, int tile_offset_y, int nx0, int nx_max, int ny0, int ny_max,
                        const std::vector<int>& noise_range_vector_line,
                        const std::vector<std::vector<double>>& interpolated_range_vectors,
                        const std::vector<double>& interpolated_azimuth_vector,
                        std::vector<std::vector<double>>& values) {
    if (noise_range_vector_line.size() == 1) {
        for (int x = nx0; x <= nx_max; x++) {
            const int xx = x - nx0;
            for (int y = ny0; y <= ny_max; y++) {
                values.at(y - tile_offset_y).at(x - tile_offset_x) =
                    interpolated_azimuth_vector.at(y - ny0) * interpolated_range_vectors.front().at(xx);
            }
        }

    } else {
        const int line0_index = GetSampleIndex(ny0, noise_range_vector_line);

        for (int x = nx0; x <= nx_max; x++) {
            const int xx = x - nx0;
            int line_index = line0_index;

            for (int y = ny0; y <= ny_max; y++) {
                if (y > noise_range_vector_line.at(line_index + 1) &&
                    line_index < static_cast<int>(noise_range_vector_line.size()) - 2) {
                    line_index++;
                }

                values.at(y - tile_offset_y).at(x - tile_offset_x) =
                    interpolated_azimuth_vector.at(y - ny0) *
                    InterpolateNoise(noise_range_vector_line.at(line_index), noise_range_vector_line.at(line_index + 1),
                                     interpolated_range_vectors.at(line_index).at(xx),
                                     interpolated_range_vectors.at(line_index + 1).at(xx), y);
            }
        }
    }
}

}  // namespace alus::tnr