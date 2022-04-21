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
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "algorithm_exception.h"
#include "alus_log.h"
#include "constants.h"
#include "dem_assistant.h"
#include "gdal_image_reader.h"
#include "gdal_image_writer.h"
#include "gdal_management.h"
#include "sentinel1_calibrate.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"
#include "topsar_deburst_op.h"
#include "topsar_split.h"

namespace {
template <typename T>
bool EqualsAnyOf(std::string_view comparand, T begin, T end) {
    return std::any_of(begin, end, [&comparand](std::string_view value) { return boost::iequals(comparand, value); });
}

constexpr size_t TILE_SIZE_DIMENSION{2048};
constexpr size_t GDAL_CACHE_SIZE{static_cast<size_t>(4e9)};
}  // namespace

namespace alus::calibrationroutine {

Execute::Execute(Parameters params, const std::vector<std::string>& dem_files)
    : params_{std::move(params)}, dem_files_{dem_files} {
    alus::gdalmanagement::Initialize();
}

void Execute::Run(alus::cuda::CudaInit& cuda_init, size_t) {
    ParseCalibrationType(params_.calibration_type);
    PrintProcessingParameters();
    ValidateParameters();

    std::string output_dir;
    std::string final_path;
    auto output_path = boost::filesystem::path(params_.output);

    if (boost::filesystem::is_directory(output_path)) {
        // final name comes from input
        output_dir = params_.output + "/";
    } else {
        output_dir = output_path.parent_path().string() + "/";
        final_path = params_.output;
    }

    // SLC input x 1 = ~1.25GB
    // TC ouput = ~1GB
    alus::gdalmanagement::SetCacheMax(GDAL_CACHE_SIZE);

    // split
    const auto subswath_case_up = boost::to_upper_copy(params_.subswath);
    const auto split_start = std::chrono::steady_clock::now();
    std::unique_ptr<topsarsplit::TopsarSplit> split_op{};
    if (!params_.aoi.empty()) {
        split_op = std::make_unique<topsarsplit::TopsarSplit>(params_.input, subswath_case_up, params_.polarisation,
                                                              params_.aoi);
    } else if (params_.burst_first_index != INVALID_BURST_INDEX && params_.burst_last_index != INVALID_BURST_INDEX) {
        split_op = std::make_unique<topsarsplit::TopsarSplit>(params_.input, subswath_case_up, params_.polarisation,
                                                              params_.burst_first_index, params_.burst_last_index);
    } else {
        split_op = std::make_unique<topsarsplit::TopsarSplit>(params_.input, subswath_case_up, params_.polarisation);
    }

    split_op->initialize();
    auto split_product = split_op->GetTargetProduct();
    auto* pixel_reader = split_op->GetPixelReader()->GetDataset();
    LOGI
        << "Sentinel 1 split done - "
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - split_start).count()
        << "ms";

    while (!cuda_init.IsFinished());
    cuda_init.CheckErrors();
    const auto cuda_device = cuda_init.GetDevices().front();
    cuda_device.Set();

    LOGI << "Using '" << cuda_device.GetName() << "' device nr " << cuda_device.GetDeviceNr()
         << " for calculations";

    // calibration
    const auto cal_start = std::chrono::steady_clock::now();
    std::shared_ptr<snapengine::Product> calibrated_product;
    std::shared_ptr<GDALDataset> calibrated_ds;
    sentinel1calibrate::Sentinel1Calibrator calibrator{split_product,
                                                       pixel_reader,
                                                       {subswath_case_up},
                                                       {params_.polarisation},
                                                       calibration_types_selected_,
                                                       output_dir,
                                                       false,
                                                       TILE_SIZE_DIMENSION,
                                                       TILE_SIZE_DIMENSION};
    calibrator.Execute();
    calibrated_product = calibrator.GetTargetProduct();
    const auto calibration_tmp_file = calibrator.GetTargetPath(subswath_case_up);
    calibrated_ds = calibrator.GetOutputDatasets().begin()->second;

    LOGI << "Sentinel1 calibration done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cal_start).count()
         << "ms";

    auto dem_assistant = app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(dem_files_);
    // start thread for srtm calculations parallel with CPU deburst
    // rethink this part if we add support for larger GPUs with full chain on GPU processing
    dem_assistant->GetSrtm3Manager()->HostToDevice();

    if (params_.wif) {
        LOGI << "Calibration output @ " << calibration_tmp_file << ".tif";
        GeoTiffWriteFile(calibrated_ds.get(), calibration_tmp_file);
    }

    // deburst
    const auto deb_start = std::chrono::steady_clock::now();
    auto data_reader = std::make_shared<alus::snapengine::custom::GdalImageReader>();
    data_reader->TakeExternalDataset(calibrated_ds.get());
    calibrated_ds.reset();

    calibrated_product->SetImageReader(data_reader);
    auto data_writer = std::make_shared<alus::snapengine::custom::GdalImageWriter>();
    auto deburst_op = alus::s1tbx::TOPSARDeburstOp::CreateTOPSARDeburstOp(calibrated_product);
    auto debursted_product = deburst_op->GetTargetProduct();

    const auto deburst_tmp_path = boost::filesystem::change_extension(calibration_tmp_file, "").string() + "_deb.tif";
    data_writer->Open(deburst_tmp_path, deburst_op->GetTargetProduct()->GetSceneRasterWidth(),
                      deburst_op->GetTargetProduct()->GetSceneRasterHeight(), data_reader->GetGeoTransform(),
                      data_reader->GetDataProjection(), true);
    debursted_product->SetImageWriter(data_writer);
    deburst_op->Compute();

    LOGI << "TOPSAR Deburst done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - deb_start).count()
         << "ms";
    data_reader->ReleaseDataset();

    if (params_.wif) {
        LOGI << "Deburst output @ " << deburst_tmp_path;
        GeoTiffWriteFile(data_writer->GetDataset(), deburst_tmp_path);
    }

    // TC
    const auto tc_start = std::chrono::steady_clock::now();
    terraincorrection::Metadata metadata(debursted_product);

    const auto* d_srtm_3_tiles = dem_assistant->GetSrtm3Manager()->GetSrtmBuffersInfo();
    const size_t srtm_3_tiles_length = dem_assistant->GetSrtm3Manager()->GetDeviceSrtm3TilesCount();
    const int selected_band{1};
    auto* tc_in_dataset = data_writer->GetDataset();
    terraincorrection::TerrainCorrection tc(tc_in_dataset, metadata.GetMetadata(), metadata.GetLatTiePointGrid(),
                                            metadata.GetLonTiePointGrid(), d_srtm_3_tiles, srtm_3_tiles_length,
                                            selected_band);

    std::string tc_output_file = final_path.empty()
                                     ? boost::filesystem::change_extension(deburst_tmp_path, "").string() + "_tc.tif"
                                     : final_path;
    const auto total_dimension_edge = 4096;
    const auto x_tile_size =
        static_cast<int>((tc_in_dataset->GetRasterXSize() /
                          static_cast<double>(tc_in_dataset->GetRasterXSize() + tc_in_dataset->GetRasterYSize())) *
                         total_dimension_edge);
    const auto y_tile_size = total_dimension_edge - x_tile_size;

    tc.ExecuteTerrainCorrection(tc_output_file, x_tile_size, y_tile_size);

    LOGI << "Terrain correction done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tc_start).count()
         << "ms";

    LOGI << "Algorithm completed, output file @ " << tc_output_file;
}

void Execute::PrintProcessingParameters() const {
    LOGI << "Processing parameters:" << std::endl
         << "Input product - " << params_.input << std::endl
         << "Subswath - " << params_.subswath << std::endl
         << "Polarisation - " << params_.polarisation << std::endl
         << "Calibration type - " << params_.calibration_type << std::endl
         << "First burst index - " << params_.burst_first_index << std::endl
         << "Last burst index - " << params_.burst_last_index << std::endl
         << "AOI - " << params_.aoi << std::endl
         << "Write intermediate files - " << (params_.wif ? "YES" : "NO") << std::endl;
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

void Execute::ValidateParameters() const {
    ValidateSubSwath();
    ValidatePolarisation();
}

void Execute::ValidateSubSwath() const {
    if (!EqualsAnyOf(params_.subswath, SUBSWATHS.cbegin(), SUBSWATHS.cend())) {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Subswath value not supported - " + params_.subswath);
    }
}

void Execute::ValidatePolarisation() const {
    if (!EqualsAnyOf(params_.polarisation, POLARISATIONS.cbegin(), POLARISATIONS.cend())) {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Polarisation value not supported - " + params_.polarisation);
    }
}

Execute::~Execute() { alus::gdalmanagement::Deinitialize(); }

}  // namespace alus::calibrationroutine