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
#include "calibration_routine.h"

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/program_options.hpp>

#include "alg_bond.h"
#include "algorithm_parameters.h"
#include "alus_log.h"
#include "custom/gdal_image_reader.h"
#include "custom/gdal_image_writer.h"
#include "dataset.h"
#include "sentinel1_calibrate.h"
#include "snap-core/datamodel/product.h"
#include "srtm3_elevation_model.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"
#include "topsar_deburst_op.h"
#include "topsar_split.h"

namespace {
// TODO share theese and some of the implemenation with sentinel_calibrate_executor.cc
const std::vector<std::string> ALLOWED_SUB_SWATHES{"IW1", "IW2", "IW3"};
const std::vector<std::string> ALLOWED_POLARISATIONS{"VV", "VH"};
// const std::vector<std::string> ALLOWED_CALIBRATION_TYPES{"sigma", "beta", "gamma", "dn"};

constexpr std::string_view PARAMETER_SUB_SWATH{"subswath"};
constexpr std::string_view PARAMETER_POLARISATION{"polarisation"};
constexpr std::string_view PARAMETER_CALIBRATION_TYPE{"calibration_type"};
constexpr std::string_view PARAMETER_COMPLEX_OUTPUT{"complex"};  // TODO(anton): SNAPGPU-254
constexpr std::string_view PARAMETER_VALUE_BETA{"beta"};
constexpr std::string_view PARAMETER_VALUE_SIGMA{"sigma"};
constexpr std::string_view PARAMETER_VALUE_GAMMA{"gamma"};
constexpr std::string_view PARAMETER_VALUE_DN{"dn"};
constexpr std::string_view PARAMETER_WRITE_INTERMEDIATE_FILES("wif");
}  // namespace

namespace alus::sentinel1calibrate {

void CalibrationRoutine::SetParameters(const app::AlgorithmParameters::Table& param_values) {
    if (const auto sub_swath = param_values.find(PARAMETER_SUB_SWATH.data()); sub_swath != param_values.end()) {
        sub_swaths_.emplace_back(boost::to_upper_copy<std::string>(sub_swath->second));
    }

    if (const auto polarisation = param_values.find(PARAMETER_POLARISATION.data());
        polarisation != param_values.end()) {
        polarisations_.emplace(boost::to_upper_copy<std::string>(polarisation->second));
    }

    if (const auto calibration_type = param_values.find(PARAMETER_CALIBRATION_TYPE.data());
        calibration_type != param_values.end()) {
        ParseCalibrationType(calibration_type->second);
    }
    if (const auto write_intermediate_files = param_values.find(PARAMETER_WRITE_INTERMEDIATE_FILES.data());
        write_intermediate_files != param_values.end()) {
        write_intermediate_files_ = boost::iequals(write_intermediate_files->second, "true");
    }
}

void CalibrationRoutine::SetTileSize(size_t width, size_t height) {
    tile_width_ = width;
    tile_height_ = height;
}

int CalibrationRoutine::Execute() {
    try {
        ValidateParameters();

        const auto cal_start = std::chrono::steady_clock::now();
        const auto input_path = input_dataset_filenames_.at(0);
        const auto subswath = sub_swaths_.at(0);
        const auto polarization = *polarisations_.begin();
        std::string output_dir;
        std::string final_path;
        auto output_path = boost::filesystem::path(output_path_);

        if (boost::filesystem::is_directory(boost::filesystem::path(output_path_))) {
            // final name comes from input
            output_dir = output_path_ + "/";
        } else {
            output_dir = boost::filesystem::path(output_path_).parent_path().string() + "/";
            final_path = output_path_;
        }

        // SLC input x 1 = ~1.25GB
        // TC ouput = ~1GB
        GDALSetCacheMax64(4e9);
        
        // split
        alus::topsarsplit::TopsarSplit split_op(input_path, subswath, polarization);
        split_op.initialize();
        auto split_product = split_op.GetTargetProduct();

        // calibration

        std::shared_ptr<snapengine::Product> calibrated_product;
        std::shared_ptr<GDALDataset> calibrated_ds;
        std::string calibration_tmp_file;
        Sentinel1Calibrator calibrator{split_product,
                                       input_path,
                                       sub_swaths_,
                                       polarisations_,
                                       calibration_bands_,
                                       output_dir,
                                       false,
                                       static_cast<int>(tile_width_),
                                       static_cast<int>(tile_height_)};
        calibrator.Execute();
        calibrated_product = calibrator.GetTargetProduct();
        calibration_tmp_file = calibrator.GetTargetPath(subswath);
        calibrated_ds = calibrator.GetOutputDatasets().begin()->second;

        LOGI << "Sentinel1 calibration done - "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cal_start)
                 .count() << "ms";

        if(write_intermediate_files_) {
            LOGI << "Calibration output @ " << calibration_tmp_file;
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

        const auto deburst_tmp_path =
            boost::filesystem::change_extension(calibration_tmp_file, "").string() + "_deb.tif";
        data_writer->Open(deburst_tmp_path, deburst_op->GetTargetProduct()->GetSceneRasterWidth(),
                          deburst_op->GetTargetProduct()->GetSceneRasterHeight(), data_reader->GetGeoTransform(),
                          data_reader->GetDataProjection(), true);
        debursted_product->SetImageWriter(data_writer);
        deburst_op->Compute();

        LOGI << "TOPSAR Deburst done - "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - deb_start)
                 .count() << "ms";
        data_reader->ReleaseDataset();

        if(write_intermediate_files_) {
            LOGI << "Deburst output @ " << deburst_tmp_path;
            GeoTiffWriteFile(data_writer->GetDataset(), deburst_tmp_path);
        }

        // TC
        const auto tc_start = std::chrono::steady_clock::now();
        terraincorrection::Metadata metadata(debursted_product);
        srtm3_manager_->HostToDevice();
        const auto* d_srtm_3_tiles = srtm3_manager_->GetSrtmBuffersInfo();
        const size_t srtm_3_tiles_length = srtm3_manager_->GetDeviceSrtm3TilesCount();
        const int selected_band{1};
        terraincorrection::TerrainCorrection tc(data_writer->GetDataset(), metadata.GetMetadata(),
                                                metadata.GetLatTiePointGrid(), metadata.GetLonTiePointGrid(),
                                                d_srtm_3_tiles, srtm_3_tiles_length, selected_band);

        std::string tc_output_file =
            final_path.empty() ? boost::filesystem::change_extension(deburst_tmp_path, "").string() + "_tc.tif"
                               : final_path;
        tc.ExecuteTerrainCorrection(tc_output_file, tile_width_, tile_height_);

        LOGI << "Terrain correction done - "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tc_start)
                 .count() << "ms";



        LOGI << "Algorithm completed, output file @ " << tc_output_file;
    } catch (const std::exception& e) {
        LOGE << "Exception caught while running Sentinel-1 Calibration - " << e.what();
        return 1;
    } catch (...) {
        LOGE << "Unknown exception caught while running Sentinel-1 Calibration.";
        return 2;
    }

    return 0;
}
std::string CalibrationRoutine::GetArgumentsHelp() const {
    boost::program_options::options_description options("Sentinel1 Calibration routine argument list");
    options.add_options()(PARAMETER_SUB_SWATH.data(),
                          "Subswath for which the calibration will be performed (IW1, IW2, IW3)")(
        PARAMETER_POLARISATION.data(), "Polarisation for which the calibration will be performed (VV or VH)")(
        PARAMETER_CALIBRATION_TYPE.data(),
        "Type of calibration (sigma, beta, "
        "gamma, dn)")(PARAMETER_WRITE_INTERMEDIATE_FILES.data(),
                      "write intermediate files - true/false (default:false)");
    std::stringstream result;
    result << options;
    return boost::algorithm::replace_all_copy(result.str(), "--", "");
}

void sentinel1calibrate::CalibrationRoutine::SetInputFilenames(const std::vector<std::string>& input_datasets,
                                                               const std::vector<std::string>&) {
    input_dataset_filenames_ = input_datasets;
}
void sentinel1calibrate::CalibrationRoutine::SetOutputFilename(const std::string& output_name) {
    output_path_ = output_name;
}
void CalibrationRoutine::ParseCalibrationType(std::string_view calibration_string) {
    if (boost::iequals(PARAMETER_VALUE_BETA, calibration_string)) {
        calibration_bands_.get_beta_lut = true;
    }
    if (boost::iequals(PARAMETER_VALUE_GAMMA, calibration_string)) {
        calibration_bands_.get_gamma_lut = true;
    }
    if (boost::iequals(PARAMETER_VALUE_SIGMA, calibration_string)) {
        calibration_bands_.get_sigma_lut = true;
    }
    if (boost::iequals(PARAMETER_VALUE_DN, calibration_string)) {
        calibration_bands_.get_dn_lut = true;
    }
}
void CalibrationRoutine::ValidateParameters() const {
    if (!srtm3_manager_ || !egm96_manager_) {
        throw std::invalid_argument("Missing parameter --dem");
    }
    ValidateSubSwath();
    ValidatePolarisation();
    ValidateCalibrationType();
}
bool CalibrationRoutine::DoesStringEqualAnyOf(std::string_view comparand, const std::vector<std::string>& string_list) {
    return std::any_of(string_list.begin(), string_list.end(),
                       [&comparand](std::string_view string) { return boost::iequals(comparand, string); });
}

void CalibrationRoutine::ValidateSubSwath() const {
    if (sub_swaths_.empty()) {
        throw std::invalid_argument("Missing parameter " + std::string(PARAMETER_SUB_SWATH));
    }
    for (const auto& swath : sub_swaths_) {
        if (!DoesStringEqualAnyOf(swath, ALLOWED_SUB_SWATHES)) {
            throw std::invalid_argument("Invalid parameter " + std::string(PARAMETER_SUB_SWATH) + ": " + swath +
                                        ". Value should be one of IW1, IW2, IW3");
        }
    }
}
void CalibrationRoutine::ValidatePolarisation() const {
    if (polarisations_.empty()) {
        throw std::invalid_argument("Missing parameter " + std::string(PARAMETER_POLARISATION));
    }
    {
        for (const auto& polarisation : polarisations_) {
            if (!DoesStringEqualAnyOf(polarisation, ALLOWED_POLARISATIONS)) {
                throw std::invalid_argument("Invalid parameter " + std::string(PARAMETER_POLARISATION) + ": " +
                                            polarisation + ". Value should be VV or VH");
            }
        }
    }
}
void CalibrationRoutine::ValidateCalibrationType() const {
    if (!(calibration_bands_.get_dn_lut || calibration_bands_.get_beta_lut || calibration_bands_.get_sigma_lut ||
          calibration_bands_.get_gamma_lut)) {
        throw std::invalid_argument("Missing parameter " + std::string(PARAMETER_CALIBRATION_TYPE));
    }
}

void CalibrationRoutine::SetSrtm3Manager(snapengine::Srtm3ElevationModel* manager) { srtm3_manager_ = manager; }

void CalibrationRoutine::SetEgm96Manager(const snapengine::EarthGravitationalModel96* manager) {
    egm96_manager_ = manager;
}

}  // namespace alus::sentinel1calibrate

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::sentinel1calibrate::CalibrationRoutine(); }  // NOSONAR

void DeleteAlgorithm(alus::AlgBond* instance) {
    delete instance;  // NOSONAR
}
}