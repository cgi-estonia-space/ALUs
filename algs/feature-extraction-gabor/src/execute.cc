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

#include "execute.h"

#include <gdal_priv.h>

#include "alus_log.h"
#include "conv_kernel.h"
#include "cuda_copies.h"
#include "cuda_ptr.h"
#include "cuda_util.h"
#include "filter_bank.h"
#include "gdal_management.h"
#include "gdal_util.h"
#include "patch_assembly.h"
#include "patch_reduction.h"

namespace alus::featurextractiongabor {

Execute::Execute(size_t orientation_count, size_t frequency_count, size_t patch_edge_dimension, std::string_view input)
    : orientation_count_{orientation_count},
      frequency_count_{frequency_count},
      patch_edge_dimension_{patch_edge_dimension},
      patched_image_(input) {
    alus::gdalmanagement::Initialize();
}

void Execute::GenerateInputs() {
    filter_banks_ = alus::featurextractiongabor::CreateGaborFilterBank(orientation_count_, frequency_count_);
    const auto& filter_edges = ExtractfilterEdgeSizes();
    patched_image_.CreatePatchedImagesFor(filter_edges, patch_edge_dimension_);
}

void Execute::CalculateGabor() {
    const auto band_count = patched_image_.GetBandCount();
    for (size_t band_i{}; band_i < band_count; band_i++) {
        for (const auto& filter : filter_banks_) {
            const auto& patch_item = patched_image_.GetPatchedImageFor(band_i, filter.edge_size);
            const auto patch_edge = static_cast<int>(patch_item.padding.padded_patch_edge_size);
            const auto patch_size = static_cast<int>(patch_item.padding.origin_patch_edge_size);
            const auto patches_x = static_cast<int>(patch_item.width / patch_edge);
            const auto patches_y = static_cast<int>(patch_item.height / patch_edge);
            const auto n_patches = patches_x * patches_y;

            // setup filter on device
            std::vector<float> h_filt = filter.filter_buffer;
            std::reverse(h_filt.begin(), h_filt.end());
            cuda::DeviceBuffer<float> d_filt(h_filt.size());
            cuda::CopyArrayH2D(d_filt.Get(), h_filt.data(), h_filt.size());

            // move input to GPU
            cuda::DeviceBuffer<float> d_src(patch_item.buffer.size());
            cuda::CopyArrayH2D(d_src.Get(), patch_item.buffer.data(), d_src.size()); // optimization: do this once per frequency

            // convolution
            cuda::DeviceBuffer<float> d_conv(d_src.size());
            LaunchConvKernel(d_src.Get(), static_cast<int>(patch_item.width), static_cast<int>(patch_item.height), patch_size, d_conv.Get(), d_filt.Get(),
                             static_cast<int>(filter.edge_size));

            // mean & std deviation calculation
            cuda::DeviceBuffer<float> d_means(n_patches);
            cuda::DeviceBuffer<float> d_std_devs(n_patches);
            LaunchPatchMeanReduction(d_conv.Get(), d_means.Get(), patch_size, static_cast<int>(filter.edge_size), patches_x, patches_y);
            LaunchPatchStdDevReduction(d_conv.Get(), d_means.Get(), d_std_devs.Get(), patch_size, static_cast<int>(filter.edge_size),
                                       patches_x, patches_y);

            CHECK_CUDA_ERR(cudaDeviceSynchronize());
            CHECK_CUDA_ERR(cudaGetLastError());

            std::vector<float> h_means(n_patches);
            std::vector<float> h_std_devs(n_patches);
            cuda::CopyArrayD2H(h_means.data(), d_means.Get(), h_means.size());
            cuda::CopyArrayD2H(h_std_devs.data(), d_std_devs.Get(), h_std_devs.size());

            // keep the patch results
            std::vector<PatchResult>& results_vec =
                result_.GetResultRef(band_i, filter.orientation_index, filter.frequency_index);
            for (int y = 0; y < patches_y; y++) {
                for (int x = 0; x < patches_x; x++) {
                    PatchResult r = {};
                    r.x = x;
                    r.y = y;
                    size_t idx = x + y * patches_x;
                    r.mean = h_means.at(idx);
                    r.std_dev = h_std_devs.at(idx);
                    results_vec.push_back(r);
                }
            }
        }
    }
}

void Execute::SaveGaborInputsTo(std::string_view path) const {
    SaveFilterBanks(path);
    SavePatchedImages(path);
}

void Execute::SaveResultsTo(std::string_view) const {
    result_.LogConsoleResult();
    // result as file/image in future task
}

void Execute::SaveFilterBanks(std::string_view path) const {
    GDALDriver* output_driver = GetGdalGeoTiffDriver();

    size_t dimension_last_bank{filter_banks_.at(0).edge_size};
    std::vector<float> buffer{};
    for (size_t i{0}; i < filter_banks_.size(); i++) {
        const auto& b = filter_banks_.at(i);

        if (const size_t filter_dimension = b.edge_size;
            filter_dimension != dimension_last_bank || i == filter_banks_.size() - 1) {
            if (i == filter_banks_.size() - 1) {
                std::copy(b.filter_buffer.cbegin(), b.filter_buffer.cend(), std::back_inserter(buffer));
            }
            const size_t filters = buffer.size() / (dimension_last_bank * dimension_last_bank);
            std::string filepath = std::string(path)
                                       .append("/gabor_filter_bank_")
                                       .append(std::to_string(dimension_last_bank))
                                       .append("_")
                                       .append(std::to_string(dimension_last_bank))
                                       .append(".tif");
            LOGI << "Saving " << filters << " filters to " << filepath;

            auto* output_dataset = output_driver->Create(filepath.c_str(), static_cast<int>(dimension_last_bank),
                                                         static_cast<int>(filters * dimension_last_bank), 1,
                                                         GDALDataType::GDT_Float32, nullptr);
            CHECK_GDAL_PTR(output_dataset);

            auto err = output_dataset->GetRasterBand(1)->RasterIO(
                GF_Write, 0, 0, static_cast<int>(dimension_last_bank), static_cast<int>(filters * dimension_last_bank),
                buffer.data(), static_cast<int>(dimension_last_bank), static_cast<int>(filters * dimension_last_bank),
                GDALDataType::GDT_Float32, 0, 0, nullptr);

            GDALClose(output_dataset);
            CHECK_GDAL_ERROR(err);

            if (i == filter_banks_.size() - 1) {
                break;
            }

            dimension_last_bank = filter_dimension;
            buffer = {};
        }

        std::copy(b.filter_buffer.cbegin(), b.filter_buffer.cend(), std::back_inserter(buffer));
    }
}

void Execute::SavePatchedImages(std::string_view path) const {
    GDALDriver* output_driver = GetGdalGeoTiffDriver();
    CPLStringList option_list;
    option_list.AddNameValue("PHOTOMETRIC", "RGB");

    const auto band_count = patched_image_.GetBandCount();
    const auto& filter_edge_sizes = patched_image_.GetFilterEdgeSizes();
    for (const auto& filter_edge_size : filter_edge_sizes) {
        const auto& patch = patched_image_.GetPatchedImageFor(0, filter_edge_size);
        std::string filepath = std::string(path)
                                   .append("/gabor_patch_")
                                   .append(std::to_string(patch.width))
                                   .append("_")
                                   .append(std::to_string(patch.height))
                                   .append(".tif");
        LOGI << "Saving patch " << filepath;
        auto* output_dataset =
            output_driver->Create(filepath.c_str(), static_cast<int>(patch.width), static_cast<int>(patch.height),
                                  static_cast<int>(band_count), GDALDataType::GDT_Byte, option_list.List());
        CHECK_GDAL_PTR(output_dataset);

        for (size_t band_i{}; band_i < band_count; band_i++) {
            const auto* patch_buffer = patched_image_.GetPatchedImageFor(band_i, filter_edge_size).buffer.data();
            const auto err_out =
                output_dataset->GetRasterBand(static_cast<int>(band_i) + 1)
                    ->RasterIO(GF_Write, 0, 0, static_cast<int>(patch.width), static_cast<int>(patch.height),
                               const_cast<float*>(patch_buffer), static_cast<int>(patch.width),
                               static_cast<int>(patch.height), GDALDataType::GDT_Float32, 0, 0, nullptr);

            if (err_out != CE_None) {
                GDALClose(output_dataset);
                CHECK_GDAL_ERROR(err_out);
            }
        }

        GDALClose(output_dataset);
    }
}

std::vector<size_t> Execute::ExtractfilterEdgeSizes() const {
    std::vector<size_t> edge_sizes;
    for (const auto& filter : filter_banks_) {
        if (edge_sizes.empty() || edge_sizes.back() != filter.edge_size) {
            edge_sizes.push_back(filter.edge_size);
        }
    }

    return edge_sizes;
}

Execute::~Execute() { alus::gdalmanagement::Deinitialize(); }

}  // namespace alus::featurextractiongabor
