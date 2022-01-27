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
#include "sentinel1_calibrate_executor.h"

#include <cstddef>
#include <memory>
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
#include "dataset.h"
#include "gdal_management.h"
#include "s1tbx-io/sentinel1/sentinel1_product_reader_plug_in.h"
#include "sentinel1_calibrate.h"
#include "snap-core/core/dataio/i_product_reader.h"
#include "snap-core/core/datamodel/product.h"
#include "topsar_split.h"

namespace {
const std::vector<std::string> ALLOWED_SUB_SWATHES{"IW1", "IW2", "IW3"};
const std::vector<std::string> ALLOWED_POLARISATIONS{"VV", "VH"};
const std::vector<std::string> ALLOWED_CALIBRATION_TYPES{"sigma", "beta", "gamma", "dn"};

constexpr std::string_view PARAMETER_SUB_SWATH{"subswath"};
constexpr std::string_view PARAMETER_POLARISATION{"polarisation"};
constexpr std::string_view PARAMETER_CALIBRATION_TYPE{"calibration_type"};
constexpr std::string_view PARAMETER_COMPLEX_OUTPUT{"complex"};
constexpr std::string_view PARAMETER_VALUE_BETA{"beta"};
constexpr std::string_view PARAMETER_VALUE_SIGMA{"sigma"};
constexpr std::string_view PARAMETER_VALUE_GAMMA{"gamma"};
constexpr std::string_view PARAMETER_VALUE_DN{"dn"};
}  // namespace

namespace alus::sentinel1calibrate {

void Sentinel1CalibrateExecutor::SetParameters(const app::AlgorithmParameters::Table& param_values) {
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
}

void Sentinel1CalibrateExecutor::SetTileSize(size_t width, size_t height) {
    tile_width_ = width;
    tile_height_ = height;
}

int Sentinel1CalibrateExecutor::Execute() {
    try {
        ValidateParameters();
        alus::gdalmanagement::SetCacheMax(4e9);  // GDAL Cache 4GB, enough for for whole swath input + output

        alus::topsarsplit::TopsarSplit split_op(input_dataset_filenames_.at(0), sub_swaths_.at(0),
                                                *polarisations_.begin());
        split_op.initialize();
        auto split_product = split_op.GetTargetProduct();
        Dataset<Iq16>* pixel_reader = split_op.GetPixelReader()->GetDataset();

        Sentinel1Calibrator calibrator{split_product,
                                       pixel_reader,
                                       sub_swaths_,
                                       polarisations_,
                                       calibration_bands_,
                                       output_dir_,
                                       false,
                                       static_cast<int>(tile_width_),
                                       static_cast<int>(tile_height_)};
        calibrator.Execute();
    } catch (const std::exception& e) {
        LOGE << "Exception caught while running Sentinel-1 Calibration - " << e.what();
        return 1;
    } catch (...) {
        LOGE << "Unknown exception caught while running Sentinel-1 Calibration.";
        return 2;
    }

    return 0;
}
std::string Sentinel1CalibrateExecutor::GetArgumentsHelp() const {
    boost::program_options::options_description options("Sentinel1 Calibration argument list");
    options.add_options()(PARAMETER_SUB_SWATH.data(),
                          "Subswath for which the calibration will be performed (IW1, IW2, IW3)")(
        PARAMETER_POLARISATION.data(), "Polarisation for which the calibration will be performed (VV or VH)")(
        PARAMETER_CALIBRATION_TYPE.data(),
        "Type of calibration (sigma, beta, "
        "gamma, dn)");

    std::stringstream result;
    result << options;
    return boost::algorithm::replace_all_copy(result.str(), "--", "");
}

void sentinel1calibrate::Sentinel1CalibrateExecutor::SetInputFilenames(const std::vector<std::string>& input_datasets,
                                                                       const std::vector<std::string>& metadata_paths) {
    input_dataset_filenames_ = input_datasets;
    metadata_paths_ = metadata_paths;
}
void sentinel1calibrate::Sentinel1CalibrateExecutor::SetOutputFilename(const std::string& output_name) {
    output_dir_ = output_name;
}
void Sentinel1CalibrateExecutor::ParseCalibrationType(std::string_view calibration_string) {
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
void Sentinel1CalibrateExecutor::ValidateParameters() const {
    ValidateSubSwath();
    ValidatePolarisation();
    ValidateCalibrationType();
}
bool Sentinel1CalibrateExecutor::DoesStringEqualAnyOf(std::string_view comparand,
                                                      const std::vector<std::string>& string_list) {
    return std::any_of(string_list.begin(), string_list.end(),
                       [&comparand](std::string_view string) { return boost::iequals(comparand, string); });
}

void Sentinel1CalibrateExecutor::ValidateSubSwath() const {
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
void Sentinel1CalibrateExecutor::ValidatePolarisation() const {
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
void Sentinel1CalibrateExecutor::ValidateCalibrationType() const {
    if (!(calibration_bands_.get_dn_lut || calibration_bands_.get_beta_lut || calibration_bands_.get_sigma_lut ||
          calibration_bands_.get_gamma_lut)) {
        throw std::invalid_argument("Missing parameter " + std::string(PARAMETER_CALIBRATION_TYPE));
    }
}
}  // namespace alus::sentinel1calibrate
