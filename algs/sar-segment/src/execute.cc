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

#include <algorithm>
#include <chrono>
#include <cstddef>

#include <gdal_priv.h>
#include <boost/algorithm/string/predicate.hpp>

#include "abstract_metadata.h"
#include "algorithm_exception.h"
#include "alus_log.h"
#include "constants.h"
#include "cuda_copies.h"
#include "cuda_mem_arena.h"
#include "dataset_util.h"
#include "dem_assistant.h"
#include "gdal_management.h"
#include "gdal_util.h"
#include "kernel_array.h"
#include "memory_policy.h"
#include "product.h"
#include "product_name.h"
#include "sar_segment_kernels.h"
#include "sentinel1_calibrate.h"
#include "sentinel1_product_reader_plug_in.h"
#include "shapes.h"
#include "target_dataset.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"
#include "thermal_noise_remover.h"

namespace {

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

alus::SimpleDataset<float> TerrainCorrection(const std::shared_ptr<alus::snapengine::Product>& merge_product,
                                             GDALDataset* in_ds, std::shared_ptr<alus::dem::Assistant> dem_assistant) {
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
    // tc.RegisterMetadata(metadata_);
    const auto simple_dataset = tc.ExecuteTerrainCorrection(x_tile_size, y_tile_size, false);
    LOGI << "Terrain correction done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tc_start).count()
         << "ms";

    return simple_dataset;
}

void ProcessDespeckle(alus::SimpleDataset<float>& vv, alus::SimpleDataset<float>& vh, size_t window,
                      const alus::cuda::CudaDevice& dev, size_t gpu_mem_percentage) {
    LOGI << "Despeckle using Refined Lee filter";
    const auto time_start = std::chrono::steady_clock::now();
    // Pass also gpu memory percentage and how much available? Single TC GRD takes ~3.5Gb
    // Try to do in one pass the following - tile of VH and VV + tile of div_VH_VV - now all to dB, then move back,
    // delete from GPU memory, start new calculation and then write to GeoTIFF async.
    const auto allocation_step = static_cast<size_t>(vh.width);
    const auto memory_police = alus::cuda::MemoryFitPolice(dev, gpu_mem_percentage);
    constexpr size_t CUSHION{200 << 20};  // 200 MiB leverage for not pushing it too tight.

    const auto tile_width{allocation_step};
    const auto scene_height{static_cast<size_t>(vh.height)};
    size_t tile_height{1};
    for (size_t i = tile_height; i <= static_cast<size_t>(vh.height); i++) {
        auto memory_budget_registry = alus::cuda::MemoryAllocationForecast(dev.GetMemoryAlignment());
        // We need to allocate for VH, VV plus an extra buffer for despeckle.
        const auto proposed_tile_size_bytes = tile_width * i * sizeof(float);
        memory_budget_registry.Add(proposed_tile_size_bytes);  // This counts the alignment as well.
        memory_budget_registry.Add(proposed_tile_size_bytes);
        memory_budget_registry.Add(proposed_tile_size_bytes);
        if (memory_police.CanFit(memory_budget_registry.Get() + CUSHION)) {
            tile_height = i;
        } else {
            if (i == 1) {
                throw std::runtime_error("Given GPU memory is not enough for VH and VV despeckle processing.");
            }
            break;
        }
    }

    LOGD << "Suitable tile size for GPU device " << tile_width << "x" << tile_height;

    alus::cuda::MemArena vh_dev_arena(tile_width * tile_height * sizeof(float));
    alus::cuda::KernelArray<float> vh_dev;
    vh_dev.array = vh_dev_arena.AllocArray<float>(tile_width * tile_height);
    alus::cuda::MemArena vv_dev_arena(tile_width * tile_height * sizeof(float));
    alus::cuda::KernelArray<float> vv_dev;
    vv_dev.array = vv_dev_arena.AllocArray<float>(tile_width * tile_height);
    alus::cuda::MemArena despeckle_buffer_arena(tile_width * tile_height * sizeof(float));
    alus::cuda::KernelArray<float> despeckle_buffer;
    despeckle_buffer.array = despeckle_buffer_arena.AllocArray<float>(tile_width * tile_height);
    for (auto i{0u}; i < scene_height; i += tile_height) {
        const auto computed_tile_height = std::min(tile_height, scene_height - i);
        const auto host_buffer_offset = i * tile_width;
        const auto computed_tile_pixel_count = tile_width * computed_tile_height;
        vh_dev.size = computed_tile_pixel_count;
        alus::cuda::CopyArrayH2D(vh_dev.array, vh.buffer.get() + host_buffer_offset, vh_dev.size);
        vv_dev.size = computed_tile_pixel_count;
        alus::cuda::CopyArrayH2D(vv_dev.array, vv.buffer.get() + host_buffer_offset, vv_dev.size);
        alus::sarsegment::Despeckle(vv_dev, despeckle_buffer, tile_width, tile_height, window);
        alus::cuda::CopyArrayD2H(vv.buffer.get() + host_buffer_offset, despeckle_buffer.array, vv_dev.size);
        alus::sarsegment::Despeckle(vh_dev, despeckle_buffer, tile_width, tile_height, window);
        alus::cuda::CopyArrayD2H(vh.buffer.get() + host_buffer_offset, despeckle_buffer.array, vh_dev.size);
    }

    LOGI << "Despeckle processing done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_start).count()
         << "ms";
}

alus::SimpleDataset<float> CalculateDivideAndDb(const alus::SimpleDataset<float>& vh,
                                                const alus::SimpleDataset<float>& vv, const alus::cuda::CudaDevice& dev,
                                                size_t gpu_mem_percentage) {
    LOGI << "Dividing VH/VV and calculating dB values";
    const auto time_start = std::chrono::steady_clock::now();
    // Pass also gpu memory percentage and how much available? Single TC GRD takes ~3.5Gb
    // Try to do in one pass the following - tile of VH and VV + tile of div_VH_VV - now all to dB, then move back,
    // delete from GPU memory, start new calculation and then write to GeoTIFF async.
    const auto allocation_step = static_cast<size_t>(vh.width);
    const auto memory_police = alus::cuda::MemoryFitPolice(dev, gpu_mem_percentage);
    constexpr size_t CUSHION{200 << 20};  // 200 MiB leverage for not pushing it too tight.

    const auto tile_width{allocation_step};
    const auto scene_height{static_cast<size_t>(vh.height)};
    size_t tile_height{1};
    for (size_t i = tile_height; i <= static_cast<size_t>(vh.height); i++) {
        auto memory_budget_registry = alus::cuda::MemoryAllocationForecast(dev.GetMemoryAlignment());
        // We need to allocate for VH, VV and VH/VV all are preserved in the memory for dB calculation as well.
        const auto proposed_tile_size_bytes = tile_width * i * sizeof(float);
        memory_budget_registry.Add(proposed_tile_size_bytes);  // This counts the alignment as well.
        memory_budget_registry.Add(proposed_tile_size_bytes);
        memory_budget_registry.Add(proposed_tile_size_bytes);
        if (memory_police.CanFit(memory_budget_registry.Get() + CUSHION)) {
            tile_height = i;
        } else {
            if (i == 1) {
                throw std::runtime_error("Given GPU memory is not enough for VH, VV and VH/VV (dB) processing.");
            }
            break;
        }
    }

    LOGD << "Suitable tile size for GPU device " << tile_width << "x" << tile_height;
    alus::SimpleDataset<float> vh_div_vv = vh;
    vh_div_vv.buffer = std::shared_ptr<float[]>(new float[vh.width * vh.height]);
    alus::cuda::MemArena vh_dev_arena(tile_width * tile_height * sizeof(float));
    alus::cuda::KernelArray<float> vh_dev;
    vh_dev.array = vh_dev_arena.AllocArray<float>(tile_width * tile_height);
    alus::cuda::MemArena vv_dev_arena(tile_width * tile_height * sizeof(float));
    alus::cuda::KernelArray<float> vv_dev;
    vv_dev.array = vv_dev_arena.AllocArray<float>(tile_width * tile_height);
    alus::cuda::MemArena vh_div_vv_dev_arena(tile_width * tile_height * sizeof(float));
    alus::cuda::KernelArray<float> vh_div_vv_dev;
    vh_div_vv_dev.array = vh_div_vv_dev_arena.AllocArray<float>(tile_width * tile_height);
    for (auto i{0u}; i < scene_height; i += tile_height) {
        const auto computed_tile_height = std::min(tile_height, scene_height - i);
        const auto host_buffer_offset = i * tile_width;
        const auto computed_tile_pixel_count = tile_width * computed_tile_height;
        // Create VH/VV
        alus::cuda::CopyArrayH2D(vh_dev.array, vh.buffer.get() + host_buffer_offset, computed_tile_pixel_count);
        vh_dev.size = computed_tile_pixel_count;
        alus::cuda::CopyArrayH2D(vv_dev.array, vv.buffer.get() + host_buffer_offset, computed_tile_pixel_count);
        vv_dev.size = computed_tile_pixel_count;
        alus::sarsegment::ComputeDivision(vh_div_vv_dev, vh_dev, vv_dev, tile_width, computed_tile_height, vh.no_data);
        vh_div_vv_dev.size = computed_tile_pixel_count;
        // dB
        alus::sarsegment::ComputeDecibel(vh_dev, tile_width, computed_tile_height, vh.no_data);
        alus::sarsegment::ComputeDecibel(vv_dev, tile_width, computed_tile_height, vv.no_data);
        alus::sarsegment::ComputeDecibel(vh_div_vv_dev, tile_width, computed_tile_height, vh_div_vv.no_data);
        alus::cuda::CopyArrayD2H(vh.buffer.get() + host_buffer_offset, vh_dev.array, vh_dev.size);
        alus::cuda::CopyArrayD2H(vv.buffer.get() + host_buffer_offset, vv_dev.array, vv_dev.size);
        alus::cuda::CopyArrayD2H(vh_div_vv.buffer.get() + host_buffer_offset, vh_div_vv_dev.array, vh_div_vv_dev.size);
    }

    LOGI << "Segmentation representation done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time_start).count()
         << "ms";

    return vh_div_vv;
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

    auto final_product_name = common::ProductName(params_.output);
    auto dem_assistant = dem::Assistant::CreateFormattedDemTilesOnGpuFrom(dem_files_);

    auto reader_plug_in = std::make_shared<s1tbx::Sentinel1ProductReaderPlugIn>();
    auto reader = reader_plug_in->CreateReaderInstance();
    auto product = reader->ReadProductNodes(boost::filesystem::canonical(params_.input), nullptr);

    common::ProductName prod_name(params_.output);
    bool create_prod_name = !prod_name.IsFinal();

    if (create_prod_name) {
        prod_name.Add(product->GetName());
    }

    metadata_.Add("BANDS", "VV VH VH/VV");
    metadata_.Add("UNIT", "dB");
    metadata_.Add(common::metadata::sentinel1::SENSING_START,
                  snapengine::AbstractMetadata::GetAbstractedMetadata(product)
                      ->GetAttributeUtc(alus::snapengine::AbstractMetadata::FIRST_LINE_TIME)
                      ->ToString());
    metadata_.Add(common::metadata::sentinel1::SENSING_END,
                  snapengine::AbstractMetadata::GetAbstractedMetadata(product)
                      ->GetAttributeUtc(alus::snapengine::AbstractMetadata::LAST_LINE_TIME)
                      ->ToString());

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
    if (create_prod_name) {
        prod_name.Add("tnr");
    }

    std::vector<std::shared_ptr<snapengine::Product>> calib_products(tnr_products.size());
    std::vector<std::shared_ptr<GDALDataset>> calib_datasets(tnr_datasets.size());
    std::vector<std::string> output_names(tnr_products.size());
    Calibration(tnr_products, tnr_datasets, swath_selection, calibration_types_selected_,
                final_product_name.GetDirectory(), output_names, calib_products, calib_datasets);
    tnr_products.clear();
    tnr_datasets.clear();
    if (create_prod_name) {
        prod_name.Add("cal");
    }
    metadata_.Add("CALIBRATION", params_.calibration_type);

    if (params_.remove_speckle) {
        SimpleDataset<float> simple_ds_vv;
        FetchSimpleDatasetFromGdalDataset(simple_ds_vv, calib_datasets.front().get());
        SimpleDataset<float> simple_ds_vh;
        FetchSimpleDatasetFromGdalDataset(simple_ds_vh, calib_datasets.back().get());
        ProcessDespeckle(simple_ds_vv, simple_ds_vh, params_.refined_lee_window_size, cuda_device,
                         gpu_memory_percentage);
        StoreSimpleDatasetToGdalRasterBand(
            simple_ds_vv, calib_datasets.front()->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND));
        StoreSimpleDatasetToGdalRasterBand(
            simple_ds_vh, calib_datasets.back()->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND));
        if (create_prod_name) {
            prod_name.Add("speckle");
            metadata_.Add("DESPECKLE_WINDOW", std::to_string(params_.refined_lee_window_size));
        }
    }

    dem_assistant->GetElevationManager()->TransferToDevice();

    auto simple_ds_vv = TerrainCorrection(product, calib_datasets.front().get(), dem_assistant);
    auto simple_ds_vh = TerrainCorrection(product, calib_datasets.back().get(), dem_assistant);
    if (create_prod_name) {
        prod_name.Add("tc");
    }
    dem_assistant->GetElevationManager()->ReleaseFromDevice();

    auto vh_div_vv_buffer = CalculateDivideAndDb(simple_ds_vh, simple_ds_vv, cuda_device, gpu_memory_percentage);
    if (create_prod_name) {
        prod_name.Add("segment");
    }

    const auto result_filepath = prod_name.Construct(".tif");
    LOGI << "Start write " << result_filepath;
    std::string x_tile_sz = FindOptimalTileSize(simple_ds_vv.width);
    std::string y_tile_sz = FindOptimalTileSize(simple_ds_vv.height);
    std::vector<std::pair<std::string, std::string>> driver_options;
    driver_options.emplace_back("TILED", "YES");
    driver_options.emplace_back("BLOCKXSIZE", x_tile_sz.c_str());
    driver_options.emplace_back("BLOCKYSIZE", y_tile_sz.c_str());
    driver_options.emplace_back("COMPRESS", "LZW");
    driver_options.emplace_back("BIGTIFF", "YES");
    std::vector<SimpleDataset<float>> bands{simple_ds_vv, simple_ds_vh, vh_div_vv_buffer};
    // Reset counters for buffer, need to erase memory after write.
    simple_ds_vv.buffer.reset();
    simple_ds_vh.buffer.reset();
    vh_div_vv_buffer.buffer.reset();
    WriteSimpleDatasetToGeoTiff(std::move(bands), result_filepath, driver_options, metadata_, true);
    LOGI << "Done";
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
    std::string despeckle_params =
        "Despeckle - " + (params_.remove_speckle
                              ? std::string("YES") + " window size " + std::to_string(params_.refined_lee_window_size)
                              : std::string("NO"));
    LOGI << "Processing parameters:" << std::endl
         << "Input product - " << params_.input << std::endl
         << "Calibration type - " << params_.calibration_type << std::endl
         << despeckle_params << std::endl;
}

Execute::~Execute() { alus::gdalmanagement::Deinitialize(); }

}  // namespace alus::sarsegment