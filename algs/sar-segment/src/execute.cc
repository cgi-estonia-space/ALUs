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

#include <chrono>
#include <cstddef>

#include <gdal_priv.h>
#include <boost/algorithm/string/predicate.hpp>

#include "algorithm_exception.h"
#include "alus_log.h"
#include "constants.h"
#include "dataset_util.h"
#include "dem_assistant.h"
#include "gdal_management.h"
#include "metadata_record.h"
#include "product.h"
#include "product_name.h"
#include "sentinel1_calibrate.h"
#include "sentinel1_product_reader_plug_in.h"
#include "shapes.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"
#include "thermal_noise_remover.h"

namespace {

// Single GRD is ~8-900M times the polarizations (2) plus the third band.
constexpr size_t GDAL_CACHE_SIZE{static_cast<size_t>(900e6 * 3)};
constexpr size_t TILE_SIZE_DIMENSION{2048};

void ThermalNoiseRemoval(const std::vector<std::shared_ptr<alus::snapengine::Product>>& prods,
                         const std::vector<GDALDataset*>& datasets, const std::vector<alus::Rectangle>& ds_areas,
                         const std::vector<std::string>& subswaths, std::string output_dir,
                         std::vector<std::shared_ptr<alus::snapengine::Product>>& tnr_products,
                         std::vector<std::shared_ptr<GDALDataset>>& tnr_datasets) {
    const auto tnr_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < prods.size(); i++) {
        alus::tnr::ThermalNoiseRemover thermal_noise_remover{prods.at(i),         datasets.at(i),     ds_areas.at(i),
                                                             subswaths.at(i),     subswaths.at(i),    output_dir,
                                                             TILE_SIZE_DIMENSION, TILE_SIZE_DIMENSION};
        thermal_noise_remover.Execute();
        tnr_products.at(i) = thermal_noise_remover.GetTargetProduct();
        tnr_datasets.at(i) = thermal_noise_remover.GetOutputDataset().first;
    }

    LOGI << "Thermal noise removal done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tnr_start).count()
         << "ms";
}

void Calibration(const std::vector<std::shared_ptr<alus::snapengine::Product>>& tnr_products,
                 const std::vector<std::shared_ptr<GDALDataset>>& tnr_datasets,
                 const std::vector<std::string>& subswaths,
                 alus::sentinel1calibrate::SelectedCalibrationBands calib_type, std::string_view output_dir,
                 std::vector<std::string>& output_names,
                 std::vector<std::shared_ptr<alus::snapengine::Product>>& calib_products,
                 std::vector<std::shared_ptr<GDALDataset>>& calib_datasets) {
    const auto cal_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < tnr_products.size(); i++) {
        alus::sentinel1calibrate::Sentinel1Calibrator calibrator{tnr_products.at(i),
                                                                 tnr_datasets.at(i).get(),
                                                                 {subswaths.at(i)},
                                                                 {subswaths.at(i)},
                                                                 calib_type,
                                                                 output_dir,
                                                                 false,
                                                                 TILE_SIZE_DIMENSION,
                                                                 TILE_SIZE_DIMENSION};
        calibrator.Execute();
        calib_products.at(i) = calibrator.GetTargetProduct();
        calib_datasets.at(i) = calibrator.GetOutputDatasets().begin()->second;
        output_names.at(i) = calibrator.GetTargetPath(subswaths.at(i));
    }

    LOGI << "Sentinel1 calibration done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cal_start).count()
         << "ms";
}

std::string TerrainCorrection(const std::shared_ptr<alus::snapengine::Product>& merge_product, GDALDataset* in_ds,
                              std::string_view output_name, std::shared_ptr<alus::dem::Assistant> dem_assistant,
                              std::string_view predefined_output_name) {
    const auto tc_start = std::chrono::steady_clock::now();

    alus::terraincorrection::Metadata metadata(merge_product);

    const auto* d_dem_tiles = dem_assistant->GetElevationManager()->GetBuffers();
    const size_t dem_tiles_length = dem_assistant->GetElevationManager()->GetTileCount();
    const int selected_band{1};

    const auto total_dimension_edge = 4096;
    const auto x_tile_size = static_cast<int>(
        (in_ds->GetRasterXSize() / static_cast<double>(in_ds->GetRasterXSize() + in_ds->GetRasterYSize())) *
        total_dimension_edge);
    const auto y_tile_size = total_dimension_edge - x_tile_size;

    alus::terraincorrection::TerrainCorrection tc(
        in_ds, metadata.GetMetadata(), metadata.GetLatTiePointGrid(), metadata.GetLonTiePointGrid(), d_dem_tiles,
        dem_tiles_length, dem_assistant->GetElevationManager()->GetProperties(), dem_assistant->GetType(),
        dem_assistant->GetElevationManager()->GetPropertiesValue(), selected_band);
    std::string tc_output_file = predefined_output_name.empty()
                                     ? boost::filesystem::change_extension(output_name.data(), "").string() + "_tc.tif"
                                     : std::string(predefined_output_name);
    // tc.RegisterMetadata(metadata_);
    tc.ExecuteTerrainCorrection(tc_output_file, x_tile_size, y_tile_size, true);
    LOGI << "Terrain correction done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tc_start).count()
         << "ms";

    return tc_output_file;
}

}  // namespace

namespace alus::sarsegment {

Execute::Execute(Parameters params, const std::vector<std::string>& dem_files)
    : params_{std::move(params)}, dem_files_{dem_files} {
    alus::gdalmanagement::Initialize();
}

void Execute::Run(alus::cuda::CudaInit& cuda_init, size_t gpu_memory_percentage) {
    ParseCalibrationType(params_.calibration_type);
    PrintProcessingParameters();
    (void)gpu_memory_percentage;

    auto final_product_name = common::ProductName(params_.output);
    alus::gdalmanagement::SetCacheMax(GDAL_CACHE_SIZE);
    auto dem_assistant = dem::Assistant::CreateFormattedDemTilesOnGpuFrom(dem_files_);

    auto reader_plug_in = std::make_shared<s1tbx::Sentinel1ProductReaderPlugIn>();
    auto reader = reader_plug_in->CreateReaderInstance();
    auto product = reader->ReadProductNodes(boost::filesystem::canonical(params_.input), nullptr);

    std::vector<std::shared_ptr<snapengine::Product>> tnr_in_products;
    std::shared_ptr<Dataset<uint16_t>> grd_ds_vv;
    std::shared_ptr<Dataset<uint16_t>> grd_ds_vh;
    std::vector<GDALDataset*> tnr_in_ds;
    std::vector<Rectangle> tnr_in_ds_areas;
    // For GRD the current product implementation uses polarization for the subswath.
    std::vector<std::string> swath_selection{"VV", "VH"};

    tnr_in_products.push_back(product);
    // For GRD the current product implementation uses polarization for the subswath.
    grd_ds_vv = dataset::OpenSentinel1SafeRaster<Dataset<uint16_t>>(params_.input, swath_selection.front(),
                                                                    swath_selection.front());
    tnr_in_ds.push_back(grd_ds_vv->GetGdalDataset());
    tnr_in_ds_areas.push_back(grd_ds_vv->GetReadingArea());

    tnr_in_products.push_back(product);
    // For GRD the current product implementation uses polarization for the subswath.
    grd_ds_vh = dataset::OpenSentinel1SafeRaster<Dataset<uint16_t>>(params_.input, swath_selection.back(),
                                                                    swath_selection.back());
    tnr_in_ds.push_back(grd_ds_vh->GetGdalDataset());
    tnr_in_ds_areas.push_back(grd_ds_vh->GetReadingArea());

    while (!cuda_init.IsFinished())
        ;
    cuda_init.CheckErrors();
    const auto cuda_device = cuda_init.GetDevices().front();
    cuda_device.Set();

    std::vector<std::shared_ptr<snapengine::Product>> tnr_products(2);
    std::vector<std::shared_ptr<GDALDataset>> tnr_datasets(2);
    ThermalNoiseRemoval(tnr_in_products, tnr_in_ds, tnr_in_ds_areas, swath_selection, final_product_name.GetDirectory(),
                        tnr_products, tnr_datasets);

    std::vector<std::shared_ptr<snapengine::Product>> calib_products(tnr_products.size());
    std::vector<std::shared_ptr<GDALDataset>> calib_datasets(tnr_datasets.size());
    std::vector<std::string> output_names(tnr_products.size());
    Calibration(tnr_products, tnr_datasets, swath_selection, calibration_types_selected_,
                final_product_name.GetDirectory(), output_names, calib_products, calib_datasets);
    tnr_products.clear();
    tnr_datasets.clear();

    dem_assistant->GetElevationManager()->TransferToDevice();
    // std::string TerrainCorrection(const std::shared_ptr<alus::snapengine::Product>& merge_product, GDALDataset*
    // in_ds,
    //                               std::string_view output_name, std::shared_ptr<alus::dem::Assistant> dem_assistant,
    //                               std::string_view predefined_output_name) {
    TerrainCorrection(product, calib_datasets.front().get(), "VV", dem_assistant,
                      final_product_name.GetDirectory() + "S1A_tnr_cal_tc_seg_vv.tif");
    TerrainCorrection(product, calib_datasets.back().get(), "VH", dem_assistant,
                      final_product_name.GetDirectory() + "S1A_tnr_cal_tc_seg_vh.tif");
}

void Execute::ParseCalibrationType(std::string_view type) {
    if (boost::iequals(CALIBRATION_TYPE_BETA, type)) {
        calibration_types_selected_.get_beta_lut = true;
    } else if (boost::iequals(CALIBRATION_TYPE_GAMMA, type)) {
        calibration_types_selected_.get_gamma_lut = true;
    } else if (boost::iequals(CALIBRATION_TYPE_SIGMA, type)) {
        calibration_types_selected_.get_sigma_lut = true;
    } else if (boost::iequals(CALIBRATION_TYPE_DN, type)) {
        calibration_types_selected_.get_dn_lut = true;
    } else {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Unknown calibration type specified - " + std::string(type) + ".");
    }
}

void Execute::PrintProcessingParameters() const {
    LOGI << "Processing parameters:" << std::endl
         << "Input product - " << params_.input << std::endl
         << "Calibration type - " << params_.calibration_type << std::endl
         << "Write intermediate files - " << (params_.wif ? "YES" : "NO") << std::endl;
}

Execute::~Execute() { alus::gdalmanagement::Deinitialize(); }

}  // namespace alus::sarsegment