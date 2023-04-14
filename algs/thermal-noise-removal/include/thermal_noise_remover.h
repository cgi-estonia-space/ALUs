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

#include <driver_types.h>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gdal_priv.h>

#include "dataset.h"
#include "gdal_util.h"
#include "shapes.h"
#include "snap-core/core/datamodel/product.h"
#include "thermal_noise_data_structures.h"
#include "thermal_noise_info.h"
#include "thermal_noise_utils.h"
#include "tile_queue.h"

namespace alus::tnr {

struct SharedData {
    // synchronised internally
    ThreadSafeTileQueue<Rectangle> tile_queue;
    Dataset<Iq16>* src_dataset;

    // explicit mutex
    std::mutex dst_mutex;
    std::mutex exception_mutex;
    std::exception_ptr exception;
    GDALRasterBand* dst_band;
    std::mutex dataset_mutex;

    bool use_pinned_memory;
    int max_height{};
    int max_width{};
};

class ThermalNoiseRemover {
public:
    ThermalNoiseRemover(std::shared_ptr<snapengine::Product> source_product, Dataset<Iq16>* pixel_reader,
                        std::string_view subswath, std::string_view polarisation, std::string_view output_path,
                        int tile_width = 2000, int tile_height = 2000);

    ThermalNoiseRemover(const ThermalNoiseRemover&) = delete;
    ThermalNoiseRemover(ThermalNoiseRemover&&) = delete;
    ThermalNoiseRemover& operator=(const ThermalNoiseRemover&) = delete;
    ThermalNoiseRemover& operator=(ThermalNoiseRemover&&) = delete;

    ~ThermalNoiseRemover() = default;

    void Execute();

    [[nodiscard]] const std::shared_ptr<snapengine::Product>& GetTargetProduct() const;
    [[nodiscard]] std::pair<std::shared_ptr<GDALDataset>, std::string> GetOutputDataset() const;

private:
    std::shared_ptr<snapengine::Product> source_product_;
    Dataset<Iq16>* pixel_reader_;
    std::string subswath_;
    std::string polarisation_;
    std::string output_path_;
    std::shared_ptr<snapengine::Product> target_product_;
    std::map<std::string, std::vector<std::string>, std::less<>> target_band_name_to_source_band_names_;
    std::shared_ptr<snapengine::MetadataElement> abstract_metadata_root_;
    std::shared_ptr<GDALDataset> target_dataset_;
    std::string target_path_;
    ThermalNoiseInfo thermal_noise_info_;
    int subset_offset_x_{0};
    int subset_offset_y_{0};
    bool is_tnr_done_{false};
    bool was_absolute_calibration_performed_{false};
    int tile_width_{};
    int tile_height_{};

    bool is_complex_data_{
        true};  // TNR is intended to be the first operator in chain, but that might not always be the case.

    double target_no_data_value_{0.0};
    const double target_floor_value_{1e-5};

    static constexpr std::array<std::string_view, 1> SUPPORTED_ACQUISITION_MODES{"IW" /*, "EW", "SM"*/};
    static constexpr std::array<std::string_view, 2> SUPPORTED_PRODUCT_TYPES{"SLC", "GRD"};
    static constexpr std::string_view ALG_NAME{"Thermal Noise Removal"};
    static constexpr std::string_view PRODUCT_SUFFIX{"_tnr"};

    void ComputeTileImage(ThreadData* context, SharedData* tnr_data);
    void ComputeComplexTile(Rectangle target_tile, ThreadData* context, SharedData* tnr_data);

    /**
     * Initialises the TNR operator.
     *
     * Reads the input product metadata and creates the output product.
     */
    void Initialise();

    /**
     * Gets the subset offsets from the source product.
     */
    void GetSubsetOffset();

    /**
     * Checks whether the Thermal Noise Correction was already performed.
     *
     * @throws AlgorithmException if the TNR was previously performed.
     * @note Currently this operator does not support thermal noise reintroduction.
     */
    void GetThermalNoiseCorrectionFlag();

    /**
     * Checks whether the absolute calibration is being performed.
     *
     * @throws AlgorithmException if calibration was previously performed.
     * @note Currently this operator does not support processing already calibrated products.
     */
    void GetCalibrationFlag();

    /**
     * Creates the target product.
     */
    void CreateTargetProduct();

    /**
     * Adds bands to the target product.
     */
    void AddSelectedBands();

    /**
     * Calculates the tiles in the target product.
     */
    void SetTargetImages();

    /**
     * Calculates dimensions of the tiles.
     *
     * @param target_band Band for which the tiles will be calculated.
     * @return Vector of tiles.
     */
    [[nodiscard]] std::vector<Rectangle> CalculateTiles(snapengine::Band& target_band) const;

    /**
     * Creates a GDALDataset from the product.
     */
    void CreateTargetDatasetFromProduct();
};
}  // namespace alus::tnr