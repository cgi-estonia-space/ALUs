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
#include "sentinel1_calibrate.h"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>
#include <gdal_priv.h>
#include <boost/algorithm/string.hpp>

#include "abstract_metadata.h"
#include "calibration_info.h"
#include "calibration_type.h"
#include "ceres-core/null_progress_monitor.h"
#include "cuda_util.h"
#include "dataset.h"
#include "gdal_util.h"
#include "general_constants.h"
#include "general_utils.h"
#include "metadata_attribute.h"
#include "metadata_element.h"
#include "sentinel1_calibrate_kernel.h"
#include "sentinel1_utils.h"
#include "sentinrel1_calibrate_exception.h"
#include "shapes.h"
#include "shapes_util.h"
#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/product_data.h"
#include "snap-core/datamodel/pugixml_meta_data_reader.h"
#include "snap-core/datamodel/raster_data_node.h"
#include "snap-core/util/product_utils.h"
#include "snap-engine-utilities/datamodel/unit.h"
#include "snap-engine-utilities/gpf/input_product_validator.h"
#include "snap-engine-utilities/gpf/operator_utils.h"
#include "snap-engine-utilities/gpf/reader_utils.h"
#include "sentinel1_calibrate_safe_helper.h"

namespace alus::sentinel1calibrate {
Sentinel1Calibrator::Sentinel1Calibrator(std::shared_ptr<snapengine::Product> source_product,
                                         std::shared_ptr<Dataset<float>> source_dataset,
                                         std::vector<std::string> selected_sub_swaths,
                                         std::set<std::string, std::less<>> selected_polarisations,
                                         SelectedCalibrationBands selected_calibration_bands,
                                         std::string_view output_path, bool output_image_in_complex, int tile_width,
                                         int tile_height)
    : source_product_(source_product),
      source_dataset_(source_dataset),
      selected_sub_swaths_(std::move(selected_sub_swaths)),
      selected_polarisations_(std::move(selected_polarisations)),
      selected_calibration_bands_(selected_calibration_bands),
      output_image_in_complex_(output_image_in_complex),
      tile_width_(tile_width),
      tile_height_(tile_height),
      output_path_(output_path),
      safe_helper_(source_dataset->GetFilePath()) {
    Initialise();
}

void Sentinel1Calibrator::Execute() { SetTargetImages(); }

void Sentinel1Calibrator::ComputeTile([[maybe_unused]] std::shared_ptr<snapengine::Band> target_band,
                                      [[maybe_unused]] Rectangle target_rectangle, int band_index) {
    (void)band_index;  // TODO: it has to be somehow computed in order to differentiate single and multiband output
                       // (SNAPGPU-250)
    auto get_polarisation_and_sub_swath = [](std::string_view band_name) {
        if (std::count(band_name.begin(), band_name.end(), '_') == 1) {
            return std::string(band_name);  // Band is not complex, or there is no need to remove anything from its name
        }
        const auto first_separator_position = band_name.find('_');
        return std::string(band_name.substr(first_separator_position + 1));
    };
    auto get_sub_swath = [get_polarisation_and_sub_swath](std::string_view band_name) {
        const auto polarisation_and_sub_swath = get_polarisation_and_sub_swath(band_name);
        const auto first_separator_position = polarisation_and_sub_swath.find('_');
        return std::string(polarisation_and_sub_swath.substr(0, first_separator_position));
    };

    std::vector<void*> tile_device_vectors;
    const auto source_band_names = target_band_name_to_source_band_name_.at(target_band->GetName());
    const auto source_band_1 = source_product_->GetBand(source_band_names.at(0));
    const auto target_dataset = target_datasets_.at(get_sub_swath(target_band->GetName()));

    std::vector<float> source_data_1(0);
    source_data_1.reserve(target_rectangle.width * target_rectangle.height);
    std::vector<float> source_data_2(0);

    if (source_band_names.size() == 1) {
        const auto band_polarisation_and_sub_swath = get_polarisation_and_sub_swath(source_band_1->GetName());
        auto sub_dataset = safe_helper_.GetSubDatasetByPolarisationAndSubSwath(band_polarisation_and_sub_swath);

        CHECK_GDAL_ERROR(sub_dataset->GetRasterBand(1)->RasterIO(
            GF_Read, target_rectangle.x, target_rectangle.y, target_rectangle.width, target_rectangle.height,
            source_data_1.data(), target_rectangle.width, target_rectangle.height, GDT_Float32, 0, 0));
    }

    // If source_band_names contains several bands, then it is a complex band
    if (source_band_names.size() > 1) {
        const auto band_polarisation_and_sub_swath = get_polarisation_and_sub_swath(source_band_1->GetName());
        const auto& sub_dataset = safe_helper_.GetSubDatasetByPolarisationAndSubSwath(band_polarisation_and_sub_swath);

        source_data_2.reserve(target_rectangle.width * target_rectangle.height);

        std::vector<CInt16> complex_data(target_rectangle.width * target_rectangle.height);
        CHECK_GDAL_ERROR(sub_dataset->GetRasterBand(1)->RasterIO(
            GF_Read, target_rectangle.x, target_rectangle.y, target_rectangle.width, target_rectangle.height,
            complex_data.data(), target_rectangle.width, target_rectangle.height, GDT_CInt16, 0, 0));

        for (const auto value : complex_data) {
            source_data_1.push_back(static_cast<float>(value.i));
            source_data_2.push_back(value.q);
        }
    }

    const auto source_band_unit = snapengine::Unit::GetUnitType(source_band_1);
    const auto target_band_unit = snapengine::Unit::GetUnitType(target_band);

    const auto calibration_info = target_band_to_d_calibration_info_.at(target_band->GetName());
    const auto calibration_type = GetCalibrationType(target_band->GetName());

    // Copy source data to device
    cuda::KernelArray<float> d_source_data_1{nullptr, source_data_1.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_source_data_1.array, sizeof(float) * d_source_data_1.size));
    CHECK_CUDA_ERR(cudaMemcpy(d_source_data_1.array, source_data_1.data(), sizeof(float) * d_source_data_1.size,
                              cudaMemcpyHostToDevice));
    cuda_arrays_to_clean_.push_back(d_source_data_1.array);
    tile_device_vectors.push_back(d_source_data_1.array);

    cuda::KernelArray<float> d_source_data_2{nullptr, source_data_2.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_source_data_2.array, sizeof(float) * d_source_data_2.size));
    CHECK_CUDA_ERR(cudaMemcpy(d_source_data_2.array, source_data_2.data(), sizeof(float) * d_source_data_2.size,
                              cudaMemcpyHostToDevice));
    cuda_arrays_to_clean_.push_back(d_source_data_2.array);
    tile_device_vectors.push_back(d_source_data_2.array);

    CalibrationKernelArgs kernel_args{
        calibration_info, {}, target_rectangle, calibration_type, subset_offset_x_, subset_offset_y_, d_source_data_1,
        d_source_data_2};

    // Calculate parameters for each line. This kernel is synonymous with "for (int y = y0; y < maxY; ++y)" part of
    // SNAP's ComputeTile()
    CHECK_CUDA_ERR(cudaMalloc(&kernel_args.line_parameters_array.array,
                              sizeof(CalibrationLineParameters) * target_rectangle.height));
    kernel_args.line_parameters_array.size = target_rectangle.height;
    cuda_arrays_to_clean_.push_back(kernel_args.line_parameters_array.array);
    tile_device_vectors.push_back(kernel_args.line_parameters_array.array);

    LaunchSetupTileLinesKernel(kernel_args);

    PopulateLUTs(kernel_args.line_parameters_array, calibration_type, data_type_);

    // Calculate parameters for every tile pixel
    cuda::KernelArray<CalibrationPixelParameters> pixel_parameters{
        nullptr, static_cast<size_t>(target_rectangle.width * target_rectangle.height)};
    CHECK_CUDA_ERR(cudaMalloc(&pixel_parameters.array, pixel_parameters.size * sizeof(CalibrationPixelParameters)));
    cuda_arrays_to_clean_.push_back(pixel_parameters.array);
    tile_device_vectors.push_back(pixel_parameters.array);

    cuda::KernelArray<double> calibration_values{nullptr,
                                                 static_cast<size_t>(target_rectangle.width * target_rectangle.height)};
    CHECK_CUDA_ERR(cudaMalloc(&calibration_values.array, sizeof(double) * calibration_values.size));
    cuda_arrays_to_clean_.push_back(calibration_values.array);
    tile_device_vectors.push_back(calibration_values.array);

    LaunchCalculatePixelParamsKernel(kernel_args, pixel_parameters);

    const auto is_unit_amplitude = source_band_unit == snapengine::UnitType::AMPLITUDE;
    const auto is_unit_intensity = source_band_unit == snapengine::UnitType::INTENSITY;
    const auto is_unit_real = source_band_unit == snapengine::UnitType::REAL;
    const auto is_unit_intensity_db = source_band_unit == snapengine::UnitType::INTENSITY_DB;

    if (is_unit_amplitude) {
        LaunchAmplitudeKernel(kernel_args, pixel_parameters, calibration_values);
    } else if (is_unit_intensity) {
        if (data_type_ != CAL_TYPE::NONE) {
            LaunchIntensityWithRetroKernel(kernel_args, pixel_parameters, calibration_values);
        } else {
            LaunchIntensityWithoutRetroKernel(kernel_args, pixel_parameters, calibration_values);
        }
    } else if (is_unit_real) {
        if (target_band_unit == snapengine::UnitType::REAL) {
            LaunchRealKernel(kernel_args, pixel_parameters, calibration_values);
        } else if (target_band_unit == snapengine::UnitType::IMAGINARY) {
            LaunchImaginaryKernel(kernel_args, pixel_parameters, calibration_values);
        } else {
            LaunchComplexIntensityKernel(kernel_args, pixel_parameters, calibration_values);
        }
    } else if (is_unit_intensity_db) {
        LaunchIntensityDBKernel(kernel_args, pixel_parameters, calibration_values);
    } else {
        throw Sentinel1CalibrateException("unhandled unit.");
    }
    if (is_complex_ && output_image_in_complex_) {
        LaunchAdjustForCompexOutputKernel(kernel_args, pixel_parameters, calibration_values);
    }
    std::vector<double> h_calibration_values(calibration_values.size);
    CHECK_CUDA_ERR(cudaMemcpy(h_calibration_values.data(), calibration_values.array,
                              sizeof(double) * calibration_values.size, cudaMemcpyDeviceToHost));

    CHECK_GDAL_ERROR(target_dataset->GetGdalDataset()->GetRasterBand(1)->RasterIO(
        GF_Write, target_rectangle.x, target_rectangle.y, target_rectangle.width, target_rectangle.height,
        h_calibration_values.data(), target_rectangle.width, target_rectangle.height, GDT_Float64, 0, 0));
    auto tile_d_vectors_iterator = tile_device_vectors.begin();
    while (tile_d_vectors_iterator != tile_device_vectors.end()) {
        CHECK_CUDA_ERR(cudaFree(*tile_d_vectors_iterator));
        tile_d_vectors_iterator = tile_device_vectors.erase(tile_d_vectors_iterator);
    }
}
CAL_TYPE Sentinel1Calibrator::GetCalibrationType(std::string_view band_name) {
    auto name_contains = [&band_name](std::string_view text) { return band_name.find(text) != std::string::npos; };

    if (name_contains("Beta")) {
        return CAL_TYPE::BETA_0;
    }
    if (name_contains("Gamma")) {
        return CAL_TYPE::GAMMA;
    }
    if (name_contains("DN")) {
        return CAL_TYPE::DN;
    }
    return CAL_TYPE::SIGMA_0;
}
void Sentinel1Calibrator::CreateTargetProduct() {
    target_product_ = snapengine::Product::CreateProduct(
        source_product_->GetName() + std::string(PRODUCT_SUFFIX), source_product_->GetProductType(),
        source_product_->GetSceneRasterWidth(), source_product_->GetSceneRasterHeight());
    AddSelectedBands(source_band_names_);
    snapengine::ProductUtils::CopyProductNodes(source_product_, target_product_);
    target_product_->SetMetadataReader(std::make_shared<alus::snapengine::PugixmlMetaDataReader>());
}
void Sentinel1Calibrator::GetSampleType() {
    const auto sample_type = abstract_metadata_root_->GetAttributeString(snapengine::AbstractMetadata::SAMPLE_TYPE);
    if (sample_type == "COMPLEX") {
        is_complex_ = true;
    }
}
void Sentinel1Calibrator::GetSubsetOffset() {
    subset_offset_x_ = abstract_metadata_root_->GetAttributeInt(snapengine::AbstractMetadata::SUBSET_OFFSET_X);
    subset_offset_y_ = abstract_metadata_root_->GetAttributeInt(snapengine::AbstractMetadata::SUBSET_OFFSET_Y);
}

void Sentinel1Calibrator::GetVectors() {
    switch (data_type_) {
        case CAL_TYPE::NONE:
            break;
        case CAL_TYPE::SIGMA_0:
            selected_calibration_bands_.get_sigma_lut = true;
            break;
        case CAL_TYPE::BETA_0:
            selected_calibration_bands_.get_beta_lut = true;
            break;
        case CAL_TYPE::GAMMA:
            selected_calibration_bands_.get_gamma_lut = true;
            break;
        case CAL_TYPE::DN:
            selected_calibration_bands_.get_dn_lut = true;
            break;
        default:
            throw Sentinel1CalibrateException("unknown data type.");
    }

    calibration_info_list_ = GetCalibrationInfoList(
        source_product_->GetMetadataRoot()->GetElement(snapengine::AbstractMetadata::ORIGINAL_PRODUCT_METADATA),
        selected_polarisations_, selected_calibration_bands_);
}
void Sentinel1Calibrator::CreateTargetBandToCalibrationInfoMap() {
    auto contains = [](std::string_view band_name, std::string_view string) {
        return band_name.find(string) != std::string::npos;
    };

    const auto target_band_names = target_product_->GetBandNames();
    for (const auto& calibration : calibration_info_list_) {
        const auto& polarisation = calibration.polarisation;
        const auto& sub_swath = calibration.sub_swath;
        for (const auto& band_name : target_band_names) {
            if (is_multi_swath_) {
                if (contains(band_name, polarisation) && contains(band_name, sub_swath)) {
                    target_band_to_calibration_info_.try_emplace(band_name, calibration);
                }
            } else {
                if (contains(band_name, polarisation)) {
                    target_band_to_calibration_info_.try_emplace(band_name, calibration);
                }
            }
        }
    }
}

void Sentinel1Calibrator::Validate() {
    snapengine::InputProductValidator validator(source_product_);
    validator.CheckIfSentinel1Product();
    validator.CheckAcquisitionMode({"IW", "EW", "SM"});
    validator.CheckProductType({"SLC", "GRD"});

    // Need to check for isComplex because it is OK to calibrate GRD product which are always debursted.
    if (validator.IsComplex() && validator.IsTOPSARProduct() && validator.IsDebursted()) {
        throw Sentinel1CalibrateException("Calibration should be applied before deburst");
    }

    is_multi_swath_ = validator.IsMultiSwath();
}
void Sentinel1Calibrator::Initialise() {
    GDALSetCacheMax64(5e9);
    // General Calibrator initialisation
    snapengine::InputProductValidator input_product_validator(source_product_);
    input_product_validator.CheckIfSARProduct();

    if (output_image_in_complex_ && !input_product_validator.IsComplex()) {
        output_image_in_complex_ = false;
    }

    Validate();
    abstract_metadata_root_ = snapengine::AbstractMetadata::GetAbstractedMetadata(source_product_);
    CreateTargetProduct();

    // Initialisation specific to Sentinel1Calibrator
    GetSampleType();
    if (abstract_metadata_root_->GetAttribute(snapengine::AbstractMetadata::ABS_CALIBRATION_FLAG)
            ->GetData()
            ->GetElemBoolean()) {
        data_type_ = GetCalibrationType(source_product_->GetBandAt(0)->GetName());
    }
    GetSubsetOffset();
    GetVectors();
    CreateTargetBandToCalibrationInfoMap();
    UpdateTargetProductMetadata();

    CreateDatasetsFromProduct(target_product_, output_path_);
    CopyAllCalibrationInfoToDevice();
}
void Sentinel1Calibrator::UpdateTargetProductMetadata() const {
    auto abstract_root = snapengine::AbstractMetadata::GetAbstractedMetadata(target_product_);
    abstract_root->GetAttribute(snapengine::AbstractMetadata::ABS_CALIBRATION_FLAG)->GetData()->SetElemBoolean(true);

    const auto target_band_names = target_product_->GetBandNames();
    s1tbx::Sentinel1Utils::UpdateBandNames(abstract_root, selected_polarisations_, target_band_names);

    std::vector<std::shared_ptr<snapengine::MetadataElement>> band_metadata_list = snapengine::AbstractMetadata::GetBandAbsMetadataList(abstract_root);
    for (const auto& band_metadata : band_metadata_list) {
        bool pol_found{false};
        for (const auto& polarisation : selected_polarisations_) {
            if (band_metadata->GetName().find(polarisation) != std::string::npos) {
                pol_found = true;
                break;
            }
        }
        if (!pol_found) {
            abstract_root->RemoveElement(band_metadata);
        }
    }
}
void Sentinel1Calibrator::SetTargetImages() {
    auto string_contains = [](std::string_view string, std::string_view key) {
        return string.find(key) != std::string::npos;
    };

    const auto target_bands = target_product_->GetBands();
    int sub_dataset_id{1};  // TODO: currently this is unimplemented (SNAPGPU-250)
    for (const auto& band : target_bands) {
        if (string_contains(band->GetName(), selected_sub_swaths_.at(0))) {
            std::cout << "Processing band " << band->GetName() << std::endl;
            const auto tiles = CalculateTiles(band);
            for (const auto& tile : tiles) {
                ComputeTile(band, tile, sub_dataset_id);
            }
            std::cout << std::endl;
        }
        sub_dataset_id++;
    }
}

void Sentinel1Calibrator::AddSelectedBands(std::vector<std::string>& source_band_names) {
    if (output_image_in_complex_) {
        OutputInComplex(source_band_names);
    } else {
        OutputInIntensity(source_band_names);
    }
}
void Sentinel1Calibrator::OutputInComplex(std::vector<std::string>& source_band_names) {
    const auto source_bands = snapengine::OperatorUtils::GetSourceBands(source_product_, source_band_names, false);
    std::optional<std::string> next_unit;

    for (int i = 0; i < static_cast<int>(source_bands.size()); i += 2) {
        const auto source_band_i = source_bands.at(i);
        const auto unit = source_band_i->GetUnit();
        if (!unit.has_value()) {
            throw Sentinel1CalibrateException("Band" + source_band_i->GetName() + " requires a unit");
        }
        if (utils::general::DoesStringContain(unit->data(), snapengine::Unit::DB)) {
            throw Sentinel1CalibrateException("Calibration of bands in dB is not supported.");
        }
        if (utils::general::DoesStringContain(unit->data(), snapengine::Unit::IMAGINARY)) {
            throw Sentinel1CalibrateException("I and Q bands should be selected in pairs.");
        }
        if (utils::general::DoesStringContain(unit->data(), snapengine::Unit::REAL)) {
            if (i + 1 >= static_cast<int>(source_bands.size())) {
                throw Sentinel1CalibrateException("I and Q bands should be selected in pairs.");
            }
            next_unit = source_bands.at(i + 1)->GetUnit();
            if (!next_unit.has_value() ||
                !utils::general::DoesStringContain(next_unit->data(), snapengine::Unit::IMAGINARY)) {
                throw Sentinel1CalibrateException("I and Q bands should be selected in pairs.");
            }
        } else {
            throw Sentinel1CalibrateException("Please select I and Q bands in pairs only.");
        }
        const auto pol_position = source_band_i->GetName().rfind('_');
        if (const auto polarisation = source_band_i->GetName().substr(pol_position + 1);
            selected_polarisations_.find(polarisation) == selected_polarisations_.end()) {
            continue;
        }

        const auto source_band_q = source_bands.at(i + 1);
        std::vector<std::string> selected_source_band_names = {source_band_i->GetName(), source_band_q->GetName()};
        target_band_name_to_source_band_name_.try_emplace(selected_source_band_names.at(0), selected_source_band_names);
        const auto target_band_i =
            std::make_shared<snapengine::Band>(selected_source_band_names.at(0), snapengine::ProductData::TYPE_FLOAT32,
                                               source_band_i->GetRasterWidth(), source_band_i->GetRasterHeight());
        target_band_i->SetUnit(unit);
        target_band_i->SetNoDataValueUsed(true);
        target_band_i->SetNoDataValue(source_band_i->GetNoDataValue());
        target_product_->AddBand(target_band_i);

        target_band_name_to_source_band_name_.try_emplace(selected_source_band_names.at(1), source_band_names);
        const auto target_band_q =
            std::make_shared<snapengine::Band>(selected_source_band_names.at(1), snapengine::ProductData::TYPE_FLOAT32,
                                               source_band_q->GetRasterWidth(), source_band_q->GetRasterHeight());
        target_band_q->SetUnit(next_unit);
        target_band_q->SetNoDataValueUsed(true);
        target_band_q->SetNoDataValue(source_band_q->GetNoDataValue());
        target_product_->AddBand(target_band_q);

        std::string suffix("_");
        suffix += snapengine::OperatorUtils::GetSuffixFromBandName(source_band_i->GetName());
        snapengine::ReaderUtils::CreateVirtualIntensityBand(target_product_, target_band_i, target_band_q, suffix);
    }
}
void Sentinel1Calibrator::OutputInIntensity(const std::vector<std::string>& source_band_names) {
    const auto source_bands = snapengine::OperatorUtils::GetSourceBands(source_product_, source_band_names, false);

    for (size_t i = 0; i < source_bands.size(); i++) {
        const auto source_band = source_bands.at(i);
        const auto unit = source_band->GetUnit();
        if (!unit.has_value()) {
            throw Sentinel1CalibrateException("band" + source_band->GetName() + " requires a unit.");
        }

        if (!utils::general::DoesStringContain(unit->data(), snapengine::Unit::REAL) &&
            !utils::general::DoesStringContain(unit->data(), snapengine::Unit::AMPLITUDE) &&
            !utils::general::DoesStringContain(unit->data(), snapengine::Unit::INTENSITY)) {
            continue;
        }

        std::vector<std::string> selected_source_band_names;
        if (utils::general::DoesStringContain(unit->data(), snapengine::Unit::REAL)) {  // SLC
            if (i + 1 >= source_bands.size()) {
                throw Sentinel1CalibrateException("real and imaginary bands are not in pairs.");
            }
            if (const auto next_unit = source_bands.at(i + 1)->GetUnit();
                !next_unit.has_value() ||
                !utils::general::DoesStringContain(next_unit->data(), snapengine::Unit::IMAGINARY)) {
                throw Sentinel1CalibrateException("real and imaginary bands are not in pairs.");
            }

            selected_source_band_names = {source_band->GetName(), source_bands.at(i + 1)->GetName()};
            ++i;
        } else {  // GRD and Calibrated product
            selected_source_band_names = {source_band->GetName()};
        }

        const auto pol_location = selected_source_band_names.at(0).rfind('_') + 1;
        if (const auto polarisation = selected_source_band_names.at(0).substr(pol_location);
            selected_polarisations_.find(polarisation) == selected_polarisations_.end()) {
            continue;
        }

        const auto target_band_names = CreateTargetBandNames(selected_source_band_names.at(0));
        for (const auto& target_band_name : target_band_names) {
            if (!target_product_->GetBand(target_band_name)) {
                target_band_name_to_source_band_name_.try_emplace(target_band_name, selected_source_band_names);

                auto target_band =
                    std::make_shared<snapengine::Band>(target_band_name, snapengine::ProductData::TYPE_FLOAT32,
                                                       source_band->GetRasterWidth(), source_band->GetRasterHeight());

                target_band->SetUnit(snapengine::Unit::INTENSITY);
                target_band->SetDescription(source_band->GetDescription());
                target_band->SetNoDataValue(source_band->GetNoDataValue());
                target_band->SetNoDataValueUsed(source_band->IsNoDataValueUsed());
                target_product_->AddBand(target_band);
            }
        }
    }
}
std::vector<std::string> Sentinel1Calibrator::CreateTargetBandNames(std::string_view source_band_name) const {
    const auto count = static_cast<int>(selected_calibration_bands_.get_sigma_lut) +
                       static_cast<int>(selected_calibration_bands_.get_beta_lut) +
                       static_cast<int>(selected_calibration_bands_.get_dn_lut);
    std::vector<std::string> target_band_names;
    target_band_names.reserve(count);

    const auto pol_location = source_band_name.find('_');
    const auto polarisation = source_band_name.substr(pol_location);
    if (selected_calibration_bands_.get_sigma_lut) {
        target_band_names.push_back(std::string("Sigma0") + polarisation.data());
    }
    if (selected_calibration_bands_.get_gamma_lut) {
        target_band_names.push_back(std::string("Gamma0") + polarisation.data());
    }
    if (selected_calibration_bands_.get_beta_lut) {
        target_band_names.push_back(std::string("Beta0") + polarisation.data());
    }
    if (selected_calibration_bands_.get_dn_lut) {
        target_band_names.push_back(std::string("DN") + polarisation.data());
    }

    return target_band_names;
}

void Sentinel1Calibrator::CreateDatasetsFromProduct(std::shared_ptr<snapengine::Product> product,
                                                    std::string_view output_path) {
    auto get_band_sub_swath = [](std::string_view band_name) {
        auto delimiter_pos = band_name.find('_');
        const auto first_sub_string = band_name.substr(delimiter_pos + 1);
        delimiter_pos = first_sub_string.find('_');
        const auto sub_swath = first_sub_string.substr(0, delimiter_pos);
        return std::string(sub_swath);
    };

    auto get_polarisation_and_sub_swath = [](std::string_view band_name) {
        if (std::count(band_name.begin(), band_name.end(), '_') == 1) {
            return std::string(band_name);  // Band is not complex, os there is no need to remove anything from its name
        }
        const auto first_separator_position = band_name.find('_');
        return std::string(band_name.substr(first_separator_position + 1));
    };

    auto does_map_contain = [](std::map<std::string, std::shared_ptr<Dataset<float>>, std::less<>> map,
                               std::string_view key) { return map.find(key.data()) != map.end(); };

    GDALAllRegister();
    char** dataset_options = nullptr;
    const std::string_view format{"GTiff"};
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName(format.data());
    if (!driver) {
        throw Sentinel1CalibrateException("could not create GDAL driver for " + std::string(format.data()) + " format");
    }

    if (!CSLFetchBoolean(driver->GetMetadata(), GDAL_DCAP_CREATE, FALSE)) {
        throw Sentinel1CalibrateException("GDAL driver for " + std::string(format.data()) +
                                          " format does not support creating datasets.");
    }
    // End of placeholder

    const auto target_bands = target_product_->GetBands();
    for (int i = 1; i <= static_cast<int>(target_bands.size()); i++) {
        const auto band = target_bands.at(i - 1);
        const auto sub_swath = get_band_sub_swath(band->GetName());
        if (utils::general::DoesVectorContain(selected_sub_swaths_, sub_swath) &&
            !does_map_contain(target_datasets_, sub_swath)) {
            // Create dataset
            const auto output_file = output_path.data() + product->GetName() + "_" + sub_swath + ".tif";

            auto *const source_sub_dataset =
                safe_helper_.GetSubDatasetByPolarisationAndSubSwath(get_polarisation_and_sub_swath(band->GetName()));

            const auto band_count = GetCalibrationCount();

            GDALDataset* gdal_dataset =
                driver->Create(output_file.data(), source_sub_dataset->GetRasterXSize(),
                               source_sub_dataset->GetRasterYSize(), band_count, GDT_Float32, dataset_options);

            std::array<double, 6> geo_transform;
            source_sub_dataset->GetGeoTransform(geo_transform.data());
            gdal_dataset->SetGeoTransform(geo_transform.data());

            target_datasets_.try_emplace(sub_swath, std::make_shared<Dataset<float>>(*gdal_dataset));
        }
    }
}

// TODO: optimise this as there too many iterations here (SNAPGPU-251)
std::vector<Rectangle> Sentinel1Calibrator::CalculateTiles(std::shared_ptr<snapengine::Band> target_band) const {
    std::vector<Rectangle> output_tiles;
    int x_max = target_band->GetRasterWidth();
    int y_max = target_band->GetRasterHeight();
    int x_count = x_max / tile_width_ + 1;
    int y_count = y_max / tile_height_ + 1;

    for (int x_index = 0; x_index < x_count; ++x_index) {
        for (int y_index = 0; y_index < y_count; ++y_index) {
            Rectangle rectangle{x_index * tile_width_, y_index * tile_height_, tile_width_, tile_height_};
            if (rectangle.x > x_max || rectangle.y > y_max) {
                continue;
            }
            Rectangle intersection = shapeutils::GetIntersection(rectangle, {0, 0, x_max, y_max});
            if (intersection.width != 0 && intersection.height != 0) {
                output_tiles.push_back(intersection);
            }
        }
    }
    return output_tiles;
}

void Sentinel1Calibrator::CopyAllCalibrationInfoToDevice() {
    for (const auto& [band_name, calibration_info_element] : target_band_to_calibration_info_) {
        std::vector<int> calibration_lines;
        calibration_lines.reserve(calibration_info_element.calibration_vectors.size());
        std::vector<s1tbx::CalibrationVectorComputation> calibration_vectors;
        calibration_vectors.reserve(calibration_info_element.calibration_vectors.size());
        const auto calibration_type = GetCalibrationType(band_name);

        for (const auto& vector : calibration_info_element.calibration_vectors) {
            s1tbx::CalibrationVectorComputation vector_computation{
                vector.time_mjd, vector.line, nullptr, nullptr, nullptr, nullptr, nullptr, vector.array_size};

            const auto array_size = vector_computation.array_size * sizeof(float);

            CHECK_CUDA_ERR(cudaMalloc(&vector_computation.pixels, sizeof(int) * vector_computation.array_size));
            CHECK_CUDA_ERR(cudaMemcpyAsync(vector_computation.pixels, vector.pixels.data(),
                                           sizeof(int) * vector_computation.array_size, cudaMemcpyHostToDevice));
            cuda_arrays_to_clean_.push_back(vector_computation.pixels);

            switch (calibration_type) {
                case CAL_TYPE::SIGMA_0:
                    CHECK_CUDA_ERR(cudaMalloc(&vector_computation.sigma_nought, array_size));
                    CHECK_CUDA_ERR(cudaMemcpyAsync(vector_computation.sigma_nought, vector.sigma_nought.data(),
                                                   array_size, cudaMemcpyHostToDevice));
                    cuda_arrays_to_clean_.push_back(vector_computation.sigma_nought);
                    break;
                case CAL_TYPE::BETA_0:
                    CHECK_CUDA_ERR(cudaMalloc(&vector_computation.beta_nought, array_size));
                    CHECK_CUDA_ERR(cudaMemcpyAsync(vector_computation.beta_nought, vector.beta_nought.data(),
                                                   array_size, cudaMemcpyHostToDevice));
                    cuda_arrays_to_clean_.push_back(vector_computation.beta_nought);
                    break;
                case CAL_TYPE::GAMMA:
                    CHECK_CUDA_ERR(cudaMalloc(&vector_computation.gamma, array_size));
                    CHECK_CUDA_ERR(cudaMemcpyAsync(vector_computation.gamma, vector.gamma.data(), array_size,
                                                   cudaMemcpyHostToDevice));
                    cuda_arrays_to_clean_.push_back(vector_computation.gamma);
                    break;
                case CAL_TYPE::DN:
                    CHECK_CUDA_ERR(cudaMalloc(&vector_computation.dn, array_size));
                    CHECK_CUDA_ERR(
                        cudaMemcpyAsync(vector_computation.dn, vector.dn.data(), array_size, cudaMemcpyHostToDevice));
                    cuda_arrays_to_clean_.push_back(vector_computation.dn);
                    break;
                case CAL_TYPE::NONE:
                    // Nothing to do
                    break;
                default:
                    throw Sentinel1CalibrateException("unknown calibration type.");
            }

            switch (data_type_) {
                case CAL_TYPE::BETA_0:
                    if (!vector_computation.beta_nought) {
                        CHECK_CUDA_ERR(cudaMalloc(&vector_computation.beta_nought, array_size));
                        CHECK_CUDA_ERR(cudaMemcpyAsync(vector_computation.beta_nought, vector.beta_nought.data(),
                                                       array_size, cudaMemcpyHostToDevice));
                        cuda_arrays_to_clean_.push_back(vector_computation.sigma_nought);
                    }
                    break;
                case CAL_TYPE::SIGMA_0:
                    if (!vector_computation.sigma_nought) {
                        CHECK_CUDA_ERR(cudaMalloc(&vector_computation.sigma_nought, array_size));
                        CHECK_CUDA_ERR(cudaMemcpyAsync(vector_computation.sigma_nought, vector.sigma_nought.data(),
                                                       array_size, cudaMemcpyHostToDevice));
                        cuda_arrays_to_clean_.push_back(vector_computation.sigma_nought);
                    }
                    break;
                case CAL_TYPE::GAMMA:
                    if (!vector_computation.gamma) {
                        CHECK_CUDA_ERR(cudaMalloc(&vector_computation.gamma, array_size));
                        CHECK_CUDA_ERR(cudaMemcpyAsync(vector_computation.gamma, vector.gamma.data(), array_size,
                                                       cudaMemcpyHostToDevice));
                        cuda_arrays_to_clean_.push_back(vector_computation.gamma);
                    }
                    break;
                case CAL_TYPE::DN:
                    if (!vector_computation.dn) {
                        CHECK_CUDA_ERR(cudaMalloc(&vector_computation.dn, array_size));
                        CHECK_CUDA_ERR(cudaMemcpyAsync(vector_computation.dn, vector.dn.data(), array_size,
                                                       cudaMemcpyHostToDevice));
                        cuda_arrays_to_clean_.push_back(vector_computation.dn);
                    }
                    break;
                case CAL_TYPE::NONE:
                    // Nothing to do
                    break;
                default:
                    throw Sentinel1CalibrateException("unknown data type.");
            }

            calibration_vectors.push_back(vector_computation);
            calibration_lines.push_back(vector.line);
        }

        CalibrationInfoComputation info{calibration_info_element.first_line_time,
                                        calibration_info_element.last_line_time,
                                        calibration_info_element.line_time_interval,
                                        calibration_info_element.num_of_lines,
                                        calibration_info_element.count,
                                        {nullptr, calibration_lines.size()},
                                        {nullptr, calibration_vectors.size()}};
        CHECK_CUDA_ERR(cudaMalloc(&info.calibration_vectors.array,
                                  sizeof(s1tbx::CalibrationVectorComputation) * calibration_vectors.size()));
        cuda_arrays_to_clean_.push_back(info.calibration_vectors.array);
        CHECK_CUDA_ERR(cudaMemcpy(info.calibration_vectors.array, calibration_vectors.data(),
                                  sizeof(s1tbx::CalibrationVectorComputation) * calibration_vectors.size(),
                                  cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMalloc(&info.line_values.array, sizeof(int) * calibration_lines.size()));
        cuda_arrays_to_clean_.push_back(info.line_values.array);
        CHECK_CUDA_ERR(cudaMemcpy(info.line_values.array, calibration_lines.data(),
                                  sizeof(int) * calibration_lines.size(), cudaMemcpyHostToDevice));

        target_band_to_d_calibration_info_.try_emplace(band_name, info);
    }
}
int Sentinel1Calibrator::GetCalibrationCount() const {
    return static_cast<int>(selected_calibration_bands_.get_dn_lut) +
           static_cast<int>(selected_calibration_bands_.get_gamma_lut) +
           static_cast<int>(selected_calibration_bands_.get_beta_lut) +
           static_cast<int>(selected_calibration_bands_.get_sigma_lut) + 2 * static_cast<int>(output_image_in_complex_);
}

Sentinel1Calibrator::~Sentinel1Calibrator() {
    for (auto& array : cuda_arrays_to_clean_) {
        if (array) {
            cudaFree(array);
        }
    }
}

std::vector<CalibrationInfo> GetCalibrationInfoList(
    const std::shared_ptr<snapengine::MetadataElement>& original_product_metadata,
    std::set<std::string, std::less<>> selected_polarisations, SelectedCalibrationBands selected_calibration_bands) {
    std::vector<CalibrationInfo> calibration_info_list;

    const auto calibration_root_element =
        GetElement(original_product_metadata, snapengine::AbstractMetadata::CALIBRATION_ROOT);

    for (auto&& calibration_data_set_item : calibration_root_element->GetElements()) {
        const auto calibration_element =
            GetElement(calibration_data_set_item, snapengine::AbstractMetadata::CALIBRATION);

        const auto ads_header_element = GetElement(calibration_element, snapengine::AbstractMetadata::ADS_HEADER);

        const auto polarisation = ads_header_element->GetAttributeString(snapengine::AbstractMetadata::POLARISATION);
        if (selected_polarisations.find(polarisation) == selected_polarisations.end()) {
            continue;
        }

        const auto sub_swath = ads_header_element->GetAttributeString(snapengine::AbstractMetadata::swath);
        const auto first_line_time =
            s1tbx::Sentinel1Utils::GetTime(ads_header_element, snapengine::AbstractMetadata::START_TIME)->GetMjd();
        const auto last_line_time =
            s1tbx::Sentinel1Utils::GetTime(ads_header_element, snapengine::AbstractMetadata::STOP_TIME)->GetMjd();

        const auto num_of_lines = GetNumOfLines(original_product_metadata, polarisation, sub_swath);

        const auto line_time_interval = (last_line_time - first_line_time) / (num_of_lines - 1);

        const auto calibration_vector_list_element =
            GetElement(calibration_element, snapengine::AbstractMetadata::CALIBRATION_VECTOR_LIST);

        const auto count = calibration_vector_list_element->GetAttributeInt(snapengine::AbstractMetadata::COUNT);

        auto calibration_vectors = s1tbx::Sentinel1Utils::GetCalibrationVectors(
            calibration_vector_list_element, selected_calibration_bands.get_sigma_lut,
            selected_calibration_bands.get_beta_lut, selected_calibration_bands.get_gamma_lut,
            selected_calibration_bands.get_dn_lut);

        if (static_cast<size_t>(count) != calibration_vectors.size()) {
            throw std::runtime_error("Invalid amount of calibration vectors in " +
                                     calibration_data_set_item->GetName());
        }

        calibration_info_list.push_back({sub_swath, polarisation, first_line_time, last_line_time, line_time_interval,
                                         num_of_lines, count, calibration_vectors});
    }

    return calibration_info_list;
}
int GetNumOfLines(const std::shared_ptr<snapengine::MetadataElement>& original_product_root,
                  std::string_view polarisation, std::string_view swath) {
    const auto annotation_element = GetElement(original_product_root, snapengine::AbstractMetadata::ANNOTATION);
    for (auto&& annotation_data_set_item : annotation_element->GetElements()) {
        const auto element_name = annotation_data_set_item->GetName();
        if (boost::icontains(element_name, swath) && boost::icontains(element_name, polarisation)) {
            const auto product_element = GetElement(annotation_data_set_item, snapengine::AbstractMetadata::product);
            const auto image_annotation_element =
                GetElement(product_element, snapengine::AbstractMetadata::IMAGE_ANNOTATION);
            const auto image_information_element =
                GetElement(image_annotation_element, snapengine::AbstractMetadata::IMAGE_INFORMATION);
            return image_information_element->GetAttributeInt(snapengine::AbstractMetadata::NUMBER_OF_LINES);
        }
    }

    return snapengine::constants::INVALID_INDEX;
}

std::shared_ptr<snapengine::MetadataElement> GetElement(
    const std::shared_ptr<snapengine::MetadataElement>& parent_element, std::string_view element_name) {
    auto check_that_metadata_exists = [](const std::shared_ptr<snapengine::MetadataElement>& element,
                                         std::string_view parent_element, std::string_view element_name) {
        if (!element) {
            throw std::runtime_error(std::string(parent_element) + " is missing " + std::string(element_name) +
                                     " metadata element");
        }
    };

    auto element = parent_element->GetElement(element_name);
    check_that_metadata_exists(element, parent_element->GetName(), element_name);

    return element;
}
}  // namespace alus::sentinel1calibrate
