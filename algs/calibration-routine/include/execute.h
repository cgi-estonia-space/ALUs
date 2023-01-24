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

#include <memory>
#include <string>
#include <vector>

#include "cuda_device_init.h"
#include "dem_assistant.h"
#include "gdal_image_writer.h"
#include "sentinel1_calibrate.h"
#include "topsar_split.h"

namespace alus::calibrationroutine {

class Execute final {
public:
    struct Parameters {
        std::string input;
        std::string output;
        bool wif;
        std::string subswath;
        std::string polarisation;
        std::string aoi;
        size_t burst_first_index;
        size_t burst_last_index;
        std::string calibration_type;
    };

    Execute() = delete;
    Execute(Parameters params, const std::vector<std::string>& dem_files);

    void Run(alus::cuda::CudaInit& cuda_init, size_t gpu_mem_percentage);

    Execute(const Execute& other) = delete;
    Execute& operator=(const Execute& other) = delete;

    ~Execute();

private:
    void PrintProcessingParameters() const;
    void ParseCalibrationType(std::string_view type_string);
    void ValidateSubSwath() const;
    void ValidatePolarisation() const;
    void ValidateParameters() const;
    void Split(const std::string& path, size_t burst_index_start, size_t burst_index_end,
               std::vector<std::shared_ptr<topsarsplit::TopsarSplit>>& splits,
               std::vector<std::string>& swath_selection) const;
    void ThermalNoiseRemoval(const std::vector<std::shared_ptr<topsarsplit::TopsarSplit>>& splits,
                             const std::vector<std::string>& subswaths, std::string_view output_dir,
                             std::vector<std::shared_ptr<snapengine::Product>>& tnr_products,
                             std::vector<std::shared_ptr<GDALDataset>>& tnr_datasets) const;

    void Calibration(const std::vector<std::shared_ptr<snapengine::Product>>& tnr_products,
                     const std::vector<std::shared_ptr<GDALDataset>>& tnr_datasets,
                     const std::vector<std::string>& subswaths, std::string_view output_dir,
                     std::vector<std::string>& output_names,
                     std::vector<std::shared_ptr<snapengine::Product>>& calib_products,
                     std::vector<std::shared_ptr<GDALDataset>>& calib_datasets) const;

    void Deburst(const std::vector<std::shared_ptr<snapengine::Product>>& calib_products,
                 std::vector<std::shared_ptr<GDALDataset>>& calib_datasets,
                 std::vector<std::string>& output_names,
                 std::vector<std::shared_ptr<snapengine::Product>>& deburst_products) const;

    void Merge(const std::vector<std::shared_ptr<snapengine::Product>>& deburst_products,
               std::vector<std::string>& output_names, std::shared_ptr<snapengine::Product>& merge_output) const;

    std::string TerrainCorrection(const std::shared_ptr<snapengine::Product>& merge_product, size_t deb_product_count,
                                  std::string_view output_name, std::shared_ptr<app::DemAssistant> dem_assistant,
                                  std::string_view predefined_output_name) const;
    std::string ConditionAoi(const std::string& aoi) const;

    Parameters params_;
    sentinel1calibrate::SelectedCalibrationBands calibration_types_selected_{};
    const std::vector<std::string>& dem_files_;
};
}  // namespace alus::calibrationroutine