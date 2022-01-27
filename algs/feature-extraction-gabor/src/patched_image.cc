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

#include "../include/patched_image.h"

#include <cmath>

#include "algorithm_exception.h"
#include "dataset.h"
#include "../include/patch_assembly.h"

namespace alus::featurextractiongabor {

PatchedImage::PatchedImage(std::string_view input_path) : input_path_{input_path} {}

void PatchedImage::CreatePatchedImagesFor(const std::vector<size_t>& filter_edge_sizes, size_t patch_edge_size) {
    Dataset<float> in_ds(input_path_);
    band_count_ = in_ds.GetGdalDataset()->GetBands().size();
    const auto in_x_size = in_ds.GetRasterSizeX();
    const auto in_y_size = in_ds.GetRasterSizeY();

    filter_edge_sizes_ = filter_edge_sizes;
    for (size_t band_i{0}; band_i < band_count_; band_i++) {
        in_ds.LoadRasterBand(static_cast<int>(band_i) + 1);

        patched_images_.try_emplace(band_i, 0);
        for (const auto& filter_edge_size : filter_edge_sizes) {
            const auto& padded_patches_image_dim = alus::featurextractiongabor::GetPaddedPatchImageDimension(
                patch_edge_size, {in_x_size, in_y_size}, filter_edge_size);
            const auto padding_params =
                alus::featurextractiongabor::CreatePaddedPatchParameters(patch_edge_size, filter_edge_size);
            const auto patch_count_x = alus::featurextractiongabor::GetPatchCountFor(
                padded_patches_image_dim.columnsX, padding_params.padded_patch_edge_size);
            const auto patch_count_y = alus::featurextractiongabor::GetPatchCountFor(
                padded_patches_image_dim.rowsY, padding_params.padded_patch_edge_size);
            alus::featurextractiongabor::StrideFillParameters stride_parameters{
                static_cast<size_t>(in_x_size), static_cast<size_t>(padded_patches_image_dim.columnsX)};

            patched_images_.at(band_i).push_back(
                {padding_params, static_cast<size_t>(padded_patches_image_dim.columnsX),
                 static_cast<size_t>(padded_patches_image_dim.rowsY), filter_edge_size,
                 std::vector<float>(padded_patches_image_dim.columnsX * padded_patches_image_dim.rowsY)});
            float* output_buffer = patched_images_.at(band_i).back().buffer.data();
            for (size_t patch_x = 0; patch_x < patch_count_x; patch_x++) {
                for (size_t patch_y = 0; patch_y < patch_count_y; patch_y++) {
                    const size_t in_offset = patch_y * (patch_count_x * padding_params.origin_patch_edge_size *
                                                        padding_params.origin_patch_edge_size) +
                                             (patch_x * padding_params.origin_patch_edge_size);
                    const size_t out_offset = (patch_y * (patch_count_x * padding_params.padded_patch_edge_size *
                                                          padding_params.padded_patch_edge_size)) +
                                              (patch_x * padding_params.padded_patch_edge_size);
                    alus::featurextractiongabor::FillPaddingPatch(in_ds.GetHostDataBuffer().data() + in_offset,
                                                                  output_buffer + out_offset, padding_params,
                                                                  stride_parameters);
                }
            }
        }
    }
}

const PatchedImage::Item& PatchedImage::GetPatchedImageFor(size_t band, size_t filter_edge_size) const {
    const auto& band_patches = patched_images_.at(band);
    for (const auto& patch : band_patches) {
        if (patch.filter_edge_size == filter_edge_size) {
            return patch;
        }
    }

    THROW_ALGORITHM_EXCEPTION("Gabor feature extraction", "No patch found for band " + std::to_string(band) +
                                                              " with filter edge " + std::to_string(filter_edge_size));
}
}  // namespace alus::featurextractiongabor
