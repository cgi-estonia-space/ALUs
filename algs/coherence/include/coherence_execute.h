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

#include <memory>
#include <string>
#include <vector>

#include <gdal_priv.h>

#include "alg_bond.h"
#include "algorithm_parameters.h"
#include "product.h"

namespace alus {
class CoherenceExecuter final : public AlgBond {
public:
    CoherenceExecuter() = default;

    [[nodiscard]] int Execute() override;

    void SetInputFilenames(const std::vector<std::string>& input_datasets,
                           const std::vector<std::string>& metadata_paths) override {
        input_name_ = input_datasets;
        aux_location_ = metadata_paths;
    }

    void SetInputDataset(const std::vector<GDALDataset*>& inputs,
                         const std::vector<std::string>& metadata_paths) override {
        input_dataset_ = inputs;
        aux_location_ = metadata_paths;
    }

    void SetParameters(const app::AlgorithmParameters::Table& param_values) override;

    void SetTileSize(size_t width, size_t height) override {
        tile_width_ = width;
        tile_height_ = height;
    }

    void SetInputProducts(std::shared_ptr<snapengine::Product> main,
                     std::shared_ptr<snapengine::Product> secondary) {
        main_product_ = main;
        secondary_product_ = secondary;
    }

    void SetOutputFilename(const std::string& output_name) override { output_name_ = output_name; }

    [[nodiscard]] std::string GetArgumentsHelp() const override;

    void SetOutputDriver(GDALDriver* output_driver) override { output_driver_ = output_driver; }

    [[nodiscard]] GDALDataset* GetProcessedDataset() const override { return output_dataset_; }

    ~CoherenceExecuter() override = default;

private:
    void PrintProcessingParameters() const override;
    void ExecuteBeamDimap();
    void ExecuteSafe();

    std::vector<std::string> input_name_{};
    std::vector<GDALDataset*> input_dataset_{};
    std::string output_name_{};
    GDALDriver* output_driver_{nullptr};
    GDALDataset* output_dataset_{nullptr};
    std::vector<std::string> aux_location_{};
    size_t tile_width_{};
    size_t tile_height_{};
    int srp_number_points_{501};
    int srp_polynomial_degree_{5};
    bool subtract_flat_earth_phase_{true};
    int coherence_window_range_{15};
    int coherence_window_azimuth_{3};
    // orbit interpolation degree
    int orbit_degree_{3};
    bool fetch_transform_{true};
    double per_process_fpu_memory_fraction_{1.0};
    std::vector<int> band_map_in_{1, 2, 3, 4};
    std::shared_ptr<snapengine::Product> main_product_{};
    std::shared_ptr<snapengine::Product> secondary_product_{};
};
}  // namespace alus