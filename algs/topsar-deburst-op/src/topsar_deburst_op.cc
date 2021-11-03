/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.sentinel1.gpf.TOPSARDeburstOp.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
 *
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
#include "topsar_deburst_op.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/tokenizer.hpp>

#include "ceres-core/core/i_progress_monitor.h"
#include "ceres-core/core/null_progress_monitor.h"
#include "custom/i_image_reader.h"
#include "custom/i_image_writer.h"
#include "custom/rectangle.h"
#include "i_meta_data_writer.h"  //todo move under custom, this is not ported
#include "pugixml_meta_data_reader.h"
#include "s1tbx-commons/sentinel1_utils.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/geo_pos.h"
#include "snap-core/core/datamodel/i_geo_coding.h"
#include "snap-core/core/datamodel/metadata_attribute.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/datamodel/pixel_pos.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/datamodel/tie_point_geo_coding.h"
#include "snap-core/core/datamodel/tie_point_grid.h"
#include "snap-core/core/datamodel/virtual_band.h"
#include "snap-core/core/util/product_utils.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/datamodel/unit.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"
#include "snap-engine-utilities/engine-utilities/gpf/input_product_validator.h"
#include "snap-engine-utilities/engine-utilities/gpf/operator_utils.h"
#include "snap-engine-utilities/engine-utilities/gpf/reader_utils.h"
#include "snap-engine-utilities/engine-utilities/gpf/tile_index.h"
#include "snap-gpf/gpf/i_tile.h"
#include "snap-gpf/gpf/internal/tile_impl.h"
#include "topsar_deburst_rectangle_generator.h"

namespace alus::s1tbx {

void TOPSARDeburstOp::GetProductType() {
    product_type_ = abs_root_->GetAttributeString(snapengine::AbstractMetadata::PRODUCT_TYPE);
}

void TOPSARDeburstOp::GetAcquisitionMode() {
    acquisition_mode_ = abs_root_->GetAttributeString(snapengine::AbstractMetadata::ACQUISITION_MODE);
}

void TOPSARDeburstOp::ComputeTargetStartEndTime() {
    target_first_line_time_ = su_->subswath_.at(0)->first_line_time_;
    target_last_line_time_ = su_->subswath_.at(0)->last_line_time_;
    for (int i = 1; i < num_of_sub_swath_; i++) {
        if (target_first_line_time_ > su_->subswath_.at(i)->first_line_time_) {
            target_first_line_time_ = su_->subswath_.at(i)->first_line_time_;
        }

        if (target_last_line_time_ < su_->subswath_.at(i)->last_line_time_) {
            target_last_line_time_ = su_->subswath_.at(i)->last_line_time_;
        }
    }
    target_line_time_interval_ = su_->subswath_.at(0)->azimuth_time_interval_;
}

void TOPSARDeburstOp::ComputeTargetSlantRangeTimeToFirstAndLastPixels() {
    target_slant_range_time_to_first_pixel_ = su_->subswath_.at(0)->slr_time_to_first_pixel_;
    target_slant_range_time_to_last_pixel_ = su_->subswath_.at(num_of_sub_swath_ - 1)->slr_time_to_last_pixel_;
    target_delta_slant_range_time_ = su_->subswath_.at(0)->range_pixel_spacing_ / snapengine::eo::constants::LIGHT_SPEED;
}

void TOPSARDeburstOp::ComputeTargetWidthAndHeight() {
    target_height_ = static_cast<int>((target_last_line_time_ - target_first_line_time_) / target_line_time_interval_);

    target_width_ =
        static_cast<int>((target_slant_range_time_to_last_pixel_ - target_slant_range_time_to_first_pixel_) /
                         target_delta_slant_range_time_);
}

void TOPSARDeburstOp::ComputeSubSwathEffectStartEndPixels() {
    sub_swath_effect_start_end_pixels_.resize(num_of_sub_swath_);
    for (int i = 0; i < num_of_sub_swath_; i++) {
        sub_swath_effect_start_end_pixels_.at(i) = SubSwathEffectStartEndPixels();
        if (i == 0) {
            sub_swath_effect_start_end_pixels_.at(i).x_min = 0;
        } else {
            double mid_time = (su_->subswath_.at(i - 1)->slr_time_to_last_valid_pixel_ +
                               su_->subswath_.at(i)->slr_time_to_first_valid_pixel_) /
                              2.0;

            sub_swath_effect_start_end_pixels_.at(i).x_min = static_cast<int>(std::round(
                (mid_time - su_->subswath_.at(i)->slr_time_to_first_pixel_) / target_delta_slant_range_time_));
        }

        if (i < num_of_sub_swath_ - 1) {
            double mid_time = (su_->subswath_.at(i)->slr_time_to_last_valid_pixel_ +
                               su_->subswath_.at(i + 1)->slr_time_to_first_valid_pixel_) /
                              2.0;

            sub_swath_effect_start_end_pixels_.at(i).x_max = static_cast<int>(std::round(
                (mid_time - su_->subswath_.at(i)->slr_time_to_first_pixel_) / target_delta_slant_range_time_));
        } else {
            sub_swath_effect_start_end_pixels_.at(i).x_max = static_cast<int>(std::round(
                (su_->subswath_.at(i)->slr_time_to_last_pixel_ - su_->subswath_.at(i)->slr_time_to_first_pixel_) /
                target_delta_slant_range_time_));
        }
    }
}

void TOPSARDeburstOp::CreateTargetProduct() {
    target_product_ =
        snapengine::Product::CreateProduct(source_product_->GetName() + std::string(PRODUCT_SUFFIX),
                                           source_product_->GetProductType(), target_width_, target_height_);

    std::vector<std::shared_ptr<snapengine::Band>> source_bands = source_product_->GetBands();

    // source band name is assumed in format: name_acquisitionModeAndSubSwathIndex_polarization_prefix
    // target band name is then in format: name_polarization_prefix
    bool has_virtual_phase_bands = false;
    for (const auto& src_band : source_bands) {
        auto src_band_name = src_band->GetName();
        boost::algorithm::to_lower(src_band_name);
        if (std::dynamic_pointer_cast<snapengine::VirtualBand>(src_band) &&
            boost::algorithm::contains(src_band_name, "phase")) {
            has_virtual_phase_bands = true;
            break;
        }
    }

    for (const auto& src_band : source_bands) {
        std::string src_band_name = src_band->GetName();
        if (!ContainSelectedPolarisations(src_band_name)) {
            continue;
        }

        if (std::dynamic_pointer_cast<snapengine::VirtualBand>(src_band)) {
            continue;
        }

        std::string tgt_band_name = GetTargetBandNameFromSourceBandName(src_band_name);
        if (!target_product_->ContainsBand(tgt_band_name)) {
            std::shared_ptr<snapengine::Band> trg_band =
                target_product_->AddBand(tgt_band_name, src_band->GetDataType());
            trg_band->SetUnit(src_band->GetUnit());
            if (src_band->IsNoDataValueUsed()) {
                trg_band->SetNoDataValueUsed(true);
                trg_band->SetNoDataValue(src_band->GetNoDataValue());
            }

            int i = target_product_->GetBandIndex(tgt_band_name);
            if (trg_band->GetUnit() == snapengine::Unit::IMAGINARY && i - 1 >= 0) {
                std::shared_ptr<snapengine::Band> i_band = target_product_->GetBandAt(i - 1);
                if (i_band->GetUnit() == snapengine::Unit::REAL) {
                    snapengine::ReaderUtils::CreateVirtualIntensityBand(target_product_, i_band, trg_band,
                                                                        '_' + GetPrefix(trg_band->GetName()));

                    if (has_virtual_phase_bands) {
                        snapengine::ReaderUtils::CreateVirtualPhaseBand(target_product_, i_band, trg_band,
                                                                        '_' + GetPrefix(trg_band->GetName()));
                    }
                }
            }
        }
    }

    snapengine::ProductUtils::CopyMetadata(source_product_, target_product_);
    snapengine::ProductUtils::CopyFlagCodings(source_product_, target_product_);
    //        snapengine::ProductUtils::CopyQuicklookBandName(source_product_, target_product_);
    target_product_->SetStartTime(
        std::make_shared<snapengine::Utc>(target_first_line_time_ / snapengine::eo::constants::SECONDS_IN_DAY));
    target_product_->SetEndTime(
        std::make_shared<snapengine::Utc>(target_last_line_time_ / snapengine::eo::constants::SECONDS_IN_DAY));
    target_product_->SetDescription(source_product_->GetDescription());

    CreateTiePointGrids();

    if (!source_product_->GetQuicklookBandName().empty()) {
        if (target_product_->GetBand(source_product_->GetQuicklookBandName()) != nullptr) {
            target_product_->SetQuicklookBandName(source_product_->GetQuicklookBandName());
        }
    }
    //    todo: naming logic is currently half baked on the fly changes (snap used different systems for that)
    target_product_->SetFileLocation(
        source_product_->GetFileLocation().parent_path().parent_path().generic_path().string() +
        boost::filesystem::path::preferred_separator + target_product_->GetName() +
        boost::filesystem::path::preferred_separator + target_product_->GetName() + ".tif");
}

std::string TOPSARDeburstOp::GetTargetBandNameFromSourceBandName(std::string_view src_band_name) const {
    if (num_of_sub_swath_ == 1) {
        return std::string(src_band_name);
    }

    auto first_separation_idx = src_band_name.find(acquisition_mode_);
    auto second_separation_idx = src_band_name.find_first_of('_', first_separation_idx + 1);
    //        todo:check if this works like expected
    return std::string(src_band_name)
        .substr(0, first_separation_idx)
        .append(src_band_name.substr(second_separation_idx + 1));
}

std::optional<std::string> TOPSARDeburstOp::GetSourceBandNameFromTargetBandName(std::string_view tgt_band_name,
                                                                                std::string_view acquisition_mode,
                                                                                std::string_view swath_index_str) {
    if (num_of_sub_swath_ == 1) {
        return std::string(tgt_band_name);
    }

    std::vector<std::string> src_band_names = source_product_->GetBandNames();
    for (const auto& src_band_name : src_band_names) {
        if ((src_band_name.find(std::string(acquisition_mode).append(swath_index_str)) != std::string::npos) &&
            GetTargetBandNameFromSourceBandName(src_band_name) == tgt_band_name) {
            return std::make_optional<std::string>(src_band_name);
        }
    }
    return std::nullopt;
}

std::string TOPSARDeburstOp::GetPrefix(std::string_view tgt_band_name) {
    auto first_separation_idx = tgt_band_name.find_first_of('_');
    return std::string(tgt_band_name.substr(first_separation_idx + 1));
}

bool TOPSARDeburstOp::ContainSelectedPolarisations(std::string_view band_name) const {
    if (std::string band_name_pol{snapengine::OperatorUtils::GetPolarizationFromBandName(band_name)};
        band_name_pol.empty()) {
        return true;
    }
    return std::any_of(selected_polarisations_.begin(), selected_polarisations_.end(),
                       [band_name](const auto& pol) { return band_name.find(pol) != std::string::npos; });
}

void TOPSARDeburstOp::CreateTiePointGrids() {
    int grid_width = 20;
    int grid_height = 5;

    int sub_sampling_x = target_width_ / grid_width;
    int sub_sampling_y = target_height_ / grid_height;

    int max_list = (grid_width + 1) * (grid_height + 1);
    std::vector<float> lat_list(max_list);
    std::vector<float> lon_list(max_list);
    std::vector<float> slrt_list(max_list);
    std::vector<float> inc_list(max_list);

    int k = 0;
    for (int i = 0; i <= grid_height; i++) {
        int y = i * sub_sampling_y;
        double az_time = target_first_line_time_ + y * target_line_time_interval_;
        for (int j = 0; j <= grid_width; j++) {
            int x = j * sub_sampling_x;
            double slr_time = target_slant_range_time_to_first_pixel_ + x * target_delta_slant_range_time_;
            //                (not from snap) and needs raw pointer for single sub_swath_
            lat_list.at(k) = static_cast<float>(su_->GetLatitude(az_time, slr_time, su_->subswath_.at(0).get()));
            lon_list.at(k) = static_cast<float>(su_->GetLongitude(az_time, slr_time, su_->subswath_.at(0).get()));
            slrt_list.at(k) = static_cast<float>(su_->GetSlantRangeTime(az_time, slr_time, su_->subswath_.at(0).get()) *
                                                 2 * snapengine::eo::constants::ONE_BILLION);  // 2-way ns
            inc_list.at(k) = static_cast<float>(su_->GetIncidenceAngle(az_time, slr_time, su_->subswath_.at(0).get()));
            k++;
        }
    }

    auto lat_grid =
        std::make_shared<snapengine::TiePointGrid>(snapengine::OperatorUtils::TPG_LATITUDE, grid_width + 1,
                                                   grid_height + 1, 0, 0, sub_sampling_x, sub_sampling_y, lat_list);

    auto lon_grid =
        std::make_shared<snapengine::TiePointGrid>(snapengine::OperatorUtils::TPG_LONGITUDE, grid_width + 1,
                                                   grid_height + 1, 0, 0, sub_sampling_x, sub_sampling_y, lon_list);

    auto slrt_grid =
        std::make_shared<snapengine::TiePointGrid>(snapengine::OperatorUtils::TPG_SLANT_RANGE_TIME, grid_width + 1,
                                                   grid_height + 1, 0, 0, sub_sampling_x, sub_sampling_y, slrt_list);

    auto inc_grid =
        std::make_shared<snapengine::TiePointGrid>(snapengine::OperatorUtils::TPG_INCIDENT_ANGLE, grid_width + 1,
                                                   grid_height + 1, 0, 0, sub_sampling_x, sub_sampling_y, inc_list);

    lat_grid->SetUnit(snapengine::Unit::DEGREES);
    lon_grid->SetUnit(snapengine::Unit::DEGREES);
    slrt_grid->SetUnit(snapengine::Unit::NANOSECONDS);
    inc_grid->SetUnit(snapengine::Unit::DEGREES);

    target_product_->AddTiePointGrid(lat_grid);
    target_product_->AddTiePointGrid(lon_grid);
    target_product_->AddTiePointGrid(slrt_grid);
    target_product_->AddTiePointGrid(inc_grid);

    auto tp_geo_coding = std::make_shared<snapengine::TiePointGeoCoding>(lat_grid, lon_grid);
    target_product_->SetSceneGeoCoding(tp_geo_coding);
}

void TOPSARDeburstOp::UpdateTargetProductMetadata() {
    UpdateAbstractMetadata();
    UpdateOriginalMetadata();
}

void TOPSARDeburstOp::UpdateAbstractMetadata() {
    std::shared_ptr<snapengine::MetadataElement> abs_tgt =
        snapengine::AbstractMetadata::GetAbstractedMetadata(target_product_);
    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::NUM_OUTPUT_LINES, target_height_);
    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE,
                                               target_width_);
    abs_tgt->SetAttributeUtc(
        snapengine::AbstractMetadata::FIRST_LINE_TIME,
        std::make_shared<snapengine::Utc>(target_first_line_time_ / snapengine::eo::constants::SECONDS_IN_DAY));

    abs_tgt->SetAttributeUtc(
        snapengine::AbstractMetadata::LAST_LINE_TIME,
        std::make_shared<snapengine::Utc>(target_last_line_time_ / snapengine::eo::constants::SECONDS_IN_DAY));
    abs_tgt->SetAttributeDouble(snapengine::AbstractMetadata::LINE_TIME_INTERVAL, target_line_time_interval_);

    std::shared_ptr<snapengine::TiePointGrid> lat_grid =
        target_product_->GetTiePointGrid(snapengine::OperatorUtils::TPG_LATITUDE);
    std::shared_ptr<snapengine::TiePointGrid> lon_grid =
        target_product_->GetTiePointGrid(snapengine::OperatorUtils::TPG_LONGITUDE);

    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::FIRST_NEAR_LAT,
                                               lat_grid->GetPixelFloat(0, 0));
    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::FIRST_NEAR_LONG,
                                               lon_grid->GetPixelFloat(0, 0));
    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::FIRST_FAR_LAT,
                                               lat_grid->GetPixelFloat(target_width_, 0));
    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::FIRST_FAR_LONG,
                                               lon_grid->GetPixelFloat(target_width_, 0));

    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::LAST_NEAR_LAT,
                                               lat_grid->GetPixelFloat(0, target_height_));
    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::LAST_NEAR_LONG,
                                               lon_grid->GetPixelFloat(0, target_height_));
    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::LAST_FAR_LAT,
                                               lat_grid->GetPixelFloat(target_width_, target_height_));
    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::LAST_FAR_LONG,
                                               lon_grid->GetPixelFloat(target_width_, target_height_));

    snapengine::AbstractMetadata::SetAttribute(
        abs_tgt, snapengine::AbstractMetadata::SLANT_RANGE_TO_FIRST_PIXEL,
        target_slant_range_time_to_first_pixel_ * snapengine::eo::constants::LIGHT_SPEED);

    AddBurstBoundary(abs_tgt);

    for (const auto& elem : abs_tgt->GetElements()) {
        if (boost::algorithm::starts_with(elem->GetName(), snapengine::AbstractMetadata::BAND_PREFIX)) {
            abs_tgt->RemoveElement(elem);
        }
    }

    if (num_of_sub_swath_ == 1) {
        abs_tgt->AddAttribute(
            std::make_shared<snapengine::MetadataAttribute>("firstValidPixel", snapengine::ProductData::TYPE_INT16));
        abs_tgt->SetAttributeInt("firstValidPixel", su_->subswath_.at(0)->first_valid_pixel_);
        abs_tgt->AddAttribute(
            std::make_shared<snapengine::MetadataAttribute>("lastValidPixel", snapengine::ProductData::TYPE_INT16));
        abs_tgt->SetAttributeInt("lastValidPixel", su_->subswath_.at(0)->last_valid_pixel_);
        abs_tgt->AddAttribute(std::make_shared<snapengine::MetadataAttribute>("slrTimeToFirstValidPixel",
                                                                              snapengine::ProductData::TYPE_FLOAT64));
        abs_tgt->SetAttributeDouble("slrTimeToFirstValidPixel", su_->subswath_.at(0)->slr_time_to_first_valid_pixel_);
        abs_tgt->AddAttribute(std::make_shared<snapengine::MetadataAttribute>("slrTimeToLastValidPixel",
                                                                              snapengine::ProductData::TYPE_FLOAT64));
        abs_tgt->SetAttributeDouble("slrTimeToLastValidPixel", su_->subswath_.at(0)->slr_time_to_last_valid_pixel_);
        abs_tgt->AddAttribute(std::make_shared<snapengine::MetadataAttribute>("firstValidLineTime",
                                                                              snapengine::ProductData::TYPE_FLOAT64));
        abs_tgt->SetAttributeDouble("firstValidLineTime", su_->subswath_.at(0)->first_valid_line_time_);
        abs_tgt->AddAttribute(std::make_shared<snapengine::MetadataAttribute>("lastValidLineTime",
                                                                              snapengine::ProductData::TYPE_FLOAT64));
        abs_tgt->SetAttributeDouble("lastValidLineTime", su_->subswath_.at(0)->last_valid_line_time_);
    }
}

void TOPSARDeburstOp::AddBurstBoundary(const std::shared_ptr<snapengine::MetadataElement>& abs_tgt) const {
    std::shared_ptr<snapengine::IGeoCoding> target_geo_coding = target_product_->GetSceneGeoCoding();

    std::vector<std::string> swath_list;
    for (const auto& elem : abs_tgt->GetElements()) {
        if (boost::algorithm::starts_with(elem->GetName(), snapengine::AbstractMetadata::BAND_PREFIX)) {
            std::string swath = elem->GetAttributeString("swath");
            if (!swath.empty() && (std::find(swath_list.begin(), swath_list.end(), swath) == swath_list.end())) {
                swath_list.emplace_back(swath);
            }
        }
    }

    double first_line_time = 0.0;
    double last_line_time = 0.0;
    double first_pixel_time = 0.0;
    double last_pixel_time = 0.0;
    auto burst_boundary = std::make_shared<snapengine::MetadataElement>("BurstBoundary");
    for (std::size_t i = 0; i < swath_list.size(); i++) {
        std::string sub_swath_name = swath_list.at(i);
        auto swath_elem = std::make_shared<snapengine::MetadataElement>(sub_swath_name);
        swath_elem->AddAttribute(
            std::make_shared<snapengine::MetadataAttribute>("count", snapengine::ProductData::TYPE_INT16));
        swath_elem->SetAttributeInt("count", su_->subswath_.at(i)->num_of_bursts_);

        for (int b = 0; b < su_->subswath_.at(i)->num_of_bursts_; b++) {
            auto burst_elem = std::make_shared<snapengine::MetadataElement>("Burst" + std::to_string(b));
            auto first_line_elem = std::make_shared<snapengine::MetadataElement>("FirstLineBoundaryPoints");
            auto last_line_elem = std::make_shared<snapengine::MetadataElement>("LastLineBoundaryPoints");

            if (b == 0) {
                first_line_time = su_->subswath_.at(i)->burst_first_line_time_.at(b);
            } else {
                first_line_time = (su_->subswath_.at(i)->burst_last_line_time_[b - 1] +
                                   su_->subswath_.at(i)->burst_first_line_time_.at(b)) /
                                  2.0;
            }

            if (b == su_->subswath_.at(i)->num_of_bursts_ - 1) {
                last_line_time = su_->subswath_.at(i)->burst_last_line_time_.at(b);
            } else {
                last_line_time = (su_->subswath_.at(i)->burst_last_line_time_.at(b) +
                                  su_->subswath_.at(i)->burst_first_line_time_.at(b + 1)) /
                                 2.0;
            }

            if (i == 0) {
                first_pixel_time = su_->subswath_.at(i)->slr_time_to_first_valid_pixel_;
            } else {
                first_pixel_time = (su_->subswath_.at(i - 1)->slr_time_to_last_valid_pixel_ +
                                    su_->subswath_.at(i)->slr_time_to_first_valid_pixel_) /
                                   2.0;
            }

            if (i == swath_list.size() - 1) {
                last_pixel_time = su_->subswath_.at(i)->slr_time_to_last_valid_pixel_;
            } else {
                last_pixel_time = (su_->subswath_.at(i)->slr_time_to_last_valid_pixel_ +
                                   su_->subswath_.at(i + 1)->slr_time_to_first_valid_pixel_) /
                                  2.0;
            }

            double delta_time = (last_pixel_time - first_pixel_time) / (NUM_OF_BOUNDARY_POINTS - 1);

            for (int p = 0; p < NUM_OF_BOUNDARY_POINTS; p++) {
                double slrt_to_point = first_pixel_time + p * delta_time;

                std::shared_ptr<snapengine::MetadataElement> first_line_point_elem =
                    CreatePointElement(first_line_time, slrt_to_point, target_geo_coding);

                first_line_elem->AddElement(first_line_point_elem);

                std::shared_ptr<snapengine::MetadataElement> last_line_point_elem =
                    CreatePointElement(last_line_time, slrt_to_point, target_geo_coding);

                last_line_elem->AddElement(last_line_point_elem);
            }

            burst_elem->SetAttributeDouble("FirstLineDeburst",
                                           (first_line_time - su_->subswath_.at(i)->burst_first_line_time_.at(0)) /
                                               su_->subswath_.at(i)->azimuth_time_interval_);
            burst_elem->SetAttributeDouble("LastLineDeburst",
                                           (last_line_time - su_->subswath_.at(i)->burst_first_line_time_.at(0)) /
                                               su_->subswath_.at(i)->azimuth_time_interval_);
            burst_elem->SetAttributeDouble("FirstLineTime", su_->subswath_.at(i)->burst_first_line_time_.at(b));
            burst_elem->SetAttributeDouble("LastLineTime", su_->subswath_.at(i)->burst_last_line_time_.at(b));
            burst_elem->SetAttributeDouble("FirstPixelTime", su_->subswath_.at(i)->slr_time_to_first_pixel_);
            burst_elem->SetAttributeDouble("LastPixelTime", su_->subswath_.at(i)->slr_time_to_last_pixel_);
            burst_elem->SetAttributeDouble("FirstValidPixelTime", su_->subswath_.at(i)->slr_time_to_first_valid_pixel_);
            burst_elem->SetAttributeDouble("LastValidPixelTime", su_->subswath_.at(i)->slr_time_to_last_valid_pixel_);

            burst_elem->AddElement(first_line_elem);
            burst_elem->AddElement(last_line_elem);
            swath_elem->AddElement(burst_elem);
        }
        burst_boundary->AddElement(swath_elem);
    }
    abs_tgt->AddElement(burst_boundary);
}

std::shared_ptr<snapengine::MetadataElement> TOPSARDeburstOp::CreatePointElement(
    double line_time, double pixel_time, const std::shared_ptr<snapengine::IGeoCoding>& target_geo_coding) const {
    auto point_elem = std::make_shared<snapengine::MetadataElement>("BoundaryPoint");

    auto x = static_cast<int>((pixel_time - target_slant_range_time_to_first_pixel_) / target_delta_slant_range_time_);
    auto y = static_cast<int>((line_time - target_first_line_time_) / target_line_time_interval_);

    auto geo_pos = std::make_shared<snapengine::GeoPos>();
    target_geo_coding->GetGeoPos(std::make_shared<snapengine::PixelPos>(x, y), geo_pos);

    point_elem->AddAttribute(
        std::make_shared<snapengine::MetadataAttribute>("lat", snapengine::ProductData::TYPE_FLOAT32));
    point_elem->SetAttributeDouble("lat", geo_pos->lat_);

    point_elem->AddAttribute(
        std::make_shared<snapengine::MetadataAttribute>("lon", snapengine::ProductData::TYPE_FLOAT32));
    point_elem->SetAttributeDouble("lon", geo_pos->lon_);

    return point_elem;
}

void TOPSARDeburstOp::UpdateOriginalMetadata() {
    UpdateSwathTiming();
    if (su_->GetNumOfSubSwath() > 1) {
        UpdateCalibrationVector();
        // updateNoiseVector(); //todo: not implemented yet (SNAP COMMENT)
    }
}

void TOPSARDeburstOp::UpdateSwathTiming() {
    std::shared_ptr<snapengine::MetadataElement> orig_prod_root =
        snapengine::AbstractMetadata::GetOriginalProductMetadata(target_product_);
    std::shared_ptr<snapengine::MetadataElement> annotation = orig_prod_root->GetElement("annotation");
    if (annotation == nullptr) {
        throw std::runtime_error("Annotation Metadata not found");
    }

    std::vector<std::shared_ptr<snapengine::MetadataElement>> elems = annotation->GetElements();
    for (const auto& elem : elems) {
        std::shared_ptr<snapengine::MetadataElement> product = elem->GetElement("product");
        std::shared_ptr<snapengine::MetadataElement> swath_timing = product->GetElement("swathTiming");
        swath_timing->SetAttributeString("linesPerBurst", "0");
        swath_timing->SetAttributeString("samplesPerBurst", "0");

        std::shared_ptr<snapengine::MetadataElement> burst_list = swath_timing->GetElement("burstList");
        burst_list->SetAttributeString("count", "0");
        std::vector<std::shared_ptr<snapengine::MetadataElement>> burst_list_elem = burst_list->GetElements();
        for (const auto& a_burst_list_elem : burst_list_elem) {
            burst_list->RemoveElement(a_burst_list_elem);
        }
    }
}

std::string TOPSARDeburstOp::GetMissionPrefix(const std::shared_ptr<snapengine::MetadataElement>& abs_root) {
    //        todo: check if works like expected
    std::string mission = abs_root->GetAttributeString(snapengine::AbstractMetadata::MISSION);
    return "S1" + mission.substr(mission.length() - 1, mission.length());
}

void TOPSARDeburstOp::UpdateCalibrationVector() {
    //        std::vector<std::string> selected_pols = Sentinel1Utils::GetProductPolarizations(abs_root_);
    //        todo: check if this is gets correct data (vs. snap)
    //       todo: selected polarizations should probably come from user, current version solves for all bands
    std::vector<std::string> selected_pols = su_->GetPolarizations();
    std::shared_ptr<snapengine::MetadataElement> orig_meta =
        snapengine::AbstractMetadata::GetOriginalProductMetadata(source_product_);
    if (orig_meta == nullptr) {
        throw std::runtime_error("Original product metadata not found");
    }
    std::shared_ptr<snapengine::MetadataElement> src_calibration = orig_meta->GetElement("calibration");
    if (src_calibration == nullptr) {
        throw std::runtime_error("Calibration element not found in Original product metadata");
    }
    std::shared_ptr<snapengine::MetadataElement> band_calibration =
        src_calibration->GetElementAt(0)->GetElement("calibration");

    std::string mission_prefix = GetMissionPrefix(abs_root_);
    boost::algorithm::to_lower(mission_prefix);

    std::shared_ptr<snapengine::MetadataElement> orig_prod_root =
        snapengine::AbstractMetadata::GetOriginalProductMetadata(target_product_);
    orig_prod_root->RemoveElement(orig_prod_root->GetElement("calibration"));
    auto calibration = std::make_shared<snapengine::MetadataElement>("calibration");
    for (const auto& pol : selected_pols) {
        std::string elem_name = mission_prefix + '-' + acquisition_mode_ + '-' + product_type_ + '-' + pol;
        auto elem = std::make_shared<snapengine::MetadataElement>(elem_name);
        std::shared_ptr<snapengine::MetadataElement> cal_elem = band_calibration->CreateDeepClone();
        std::shared_ptr<snapengine::MetadataElement> calibration_vector_list_elem =
            cal_elem->GetElement("calibrationVectorList");
        cal_elem->SetAttributeString("polarisation", pol);
        std::vector<std::shared_ptr<snapengine::MetadataElement>> list = calibration_vector_list_elem->GetElements();
        int vector_index = 0;
        std::string merged_pixel_str = GetMergedPixels(pol);
        //            todo: if used, check over if works the same as in snap
        boost::char_separator<char> sep{" "};
        boost::tokenizer<boost::char_separator<char>> tokenizer{merged_pixel_str, sep};
        std::size_t count = 0;
        for (auto it = tokenizer.begin(); it != tokenizer.end(); ++it) {
            count++;
        }

        for (const auto& calibration_vector_elem : list) {
            std::shared_ptr<snapengine::MetadataElement> pixel_elem = calibration_vector_elem->GetElement("pixel");
            pixel_elem->SetAttributeString("pixel", merged_pixel_str);
            pixel_elem->SetAttributeString("count", std::to_string(count));

            std::shared_ptr<snapengine::MetadataElement> sigma_nought_elem =
                calibration_vector_elem->GetElement("sigmaNought");
            std::string merged_sigma_nought_str = GetMergedVector("SigmaNought", pol, vector_index);
            sigma_nought_elem->SetAttributeString("sigmaNought", merged_sigma_nought_str);
            sigma_nought_elem->SetAttributeString("count", std::to_string(count));

            std::shared_ptr<snapengine::MetadataElement> beta_nought_elem =
                calibration_vector_elem->GetElement("betaNought");
            std::string merged_beta_nought_str = GetMergedVector("betaNought", pol, vector_index);
            beta_nought_elem->SetAttributeString("betaNought", merged_beta_nought_str);
            beta_nought_elem->SetAttributeString("count", std::to_string(count));

            std::shared_ptr<snapengine::MetadataElement> gamma_nought_elem =
                calibration_vector_elem->GetElement("gamma");
            std::string merged_gamma_nought_str = GetMergedVector("gamma", pol, vector_index);
            gamma_nought_elem->SetAttributeString("gamma", merged_gamma_nought_str);
            gamma_nought_elem->SetAttributeString("count", std::to_string(count));

            std::shared_ptr<snapengine::MetadataElement> dn_elem = calibration_vector_elem->GetElement("dn");
            std::string merged_d_n_str = GetMergedVector("dn", pol, vector_index);
            dn_elem->SetAttributeString("dn", merged_d_n_str);
            dn_elem->SetAttributeString("count", std::to_string(count));
            vector_index++;
        }
        elem->AddElement(cal_elem);
        calibration->AddElement(elem);
    }
    orig_prod_root->AddElement(calibration);
}

std::string TOPSARDeburstOp::GetMergedPixels(std::string_view pol) {
    std::stringstream merged_pixel_str;
    for (int s = 0; s < num_of_sub_swath_; s++) {
        std::vector<int> pixel_array = su_->GetCalibrationPixel(s + 1, pol, 0);
        for (int p : pixel_array) {
            if (p >= sub_swath_effect_start_end_pixels_.at(s).x_min &&
                p < sub_swath_effect_start_end_pixels_.at(s).x_max) {
                double slrt = su_->subswath_.at(s)->slr_time_to_first_pixel_ + p * target_delta_slant_range_time_;

                auto target_pixel_idx = static_cast<int>(
                    std::round((slrt - target_slant_range_time_to_first_pixel_) / target_delta_slant_range_time_));

                merged_pixel_str << target_pixel_idx << " ";
            }
        }
    }
    return merged_pixel_str.str();
}

std::string TOPSARDeburstOp::GetMergedVector(std::string_view vector_name, std::string_view pol, int vector_index) {
    std::stringstream merged_vector_str;
    for (int s = 0; s < num_of_sub_swath_; s++) {
        std::vector<int> pixel_array = su_->GetCalibrationPixel(s + 1, pol, vector_index);
        std::vector<float> vector_array = su_->GetCalibrationVector(s + 1, pol, vector_index, vector_name);
        for (std::size_t i = 0; i < pixel_array.size(); i++) {
            if (pixel_array.at(i) >= sub_swath_effect_start_end_pixels_.at(s).x_min &&
                pixel_array.at(i) < sub_swath_effect_start_end_pixels_.at(s).x_max) {
                merged_vector_str << vector_array.at(i) << ' ';
            }
        }
    }
    return merged_vector_str.str();
}

void TOPSARDeburstOp::ComputeTileInOneSwathFloat(int tx0, int ty0, int tx_max, int ty_max, int first_sub_swath_index,
                                                 const std::vector<snapengine::custom::Rectangle>& source_rectangle,
                                                 std::string_view tgt_band_name,
                                                 const std::shared_ptr<snapengine::ITile>& tgt_tile,
                                                 BurstInfo& burst_info) {
    int y_min = ComputeYMin(*su_->subswath_.at(first_sub_swath_index - 1));
    int y_max = ComputeYMax(*su_->subswath_.at(first_sub_swath_index - 1));
    int x_min = ComputeXMin(*su_->subswath_.at(first_sub_swath_index - 1));
    int x_max = ComputeXMax(*su_->subswath_.at(first_sub_swath_index - 1));

    int first_y = std::max(ty0, y_min);
    int last_y = std::min(ty_max, y_max + 1);
    int first_x = std::max(tx0, x_min);
    int last_x = std::min(tx_max, x_max + 1);

    if (first_y >= last_y || first_x >= last_x) {
        return;
    }
    std::string swath_index_str =
        num_of_sub_swath_ == 1 ? su_->GetSubSwathNames().at(0).substr(2) : std::to_string(first_sub_swath_index);

    std::shared_ptr<snapengine::Band> src_band = source_product_->GetBand(
        GetSourceBandNameFromTargetBandName(tgt_band_name, acquisition_mode_, swath_index_str).value_or(""));
    int src_band_indx = source_product_->GetBandIndex(src_band->GetName()) + 1;
    std::shared_ptr<snapengine::ITile> src_raster = GetSourceTile(src_band, source_rectangle.at(0), src_band_indx);
    auto src_tile_index = std::make_shared<snapengine::TileIndex>(src_raster);
    auto tgt_index = std::make_shared<snapengine::TileIndex>(tgt_tile);

    //        todo: at some point this used ProductData which was attachted to tiles (might want to bring this back from
    //        history if performance is ok)
    auto& src_array = src_raster->GetSimpleDataBuffer();
    auto& tgt_array = tgt_tile->GetSimpleDataBuffer();

    for (int y = first_y; y < last_y; y++) {
        if (!GetLineIndicesInSourceProduct(y, *su_->subswath_.at(first_sub_swath_index - 1), burst_info)) {
            continue;
        }

        int tgt_offset = tgt_index->CalculateStride(y);
        int offset;
        if (burst_info.sy1 != -1 && burst_info.target_time > burst_info.mid_time) {
            offset = src_tile_index->CalculateStride(burst_info.sy1);
        } else {
            offset = src_tile_index->CalculateStride(burst_info.sy0);
        }

        auto sx = static_cast<int>(
            std::round(((target_slant_range_time_to_first_pixel_ + first_x * target_delta_slant_range_time_) -
                        su_->subswath_.at(first_sub_swath_index - 1)->slr_time_to_first_pixel_) /
                       target_delta_slant_range_time_));
        std::copy(src_array.begin() + (sx - offset), (src_array.begin() + (sx - offset)) + (last_x - first_x),
                  tgt_array.begin() + (first_x - tgt_offset));
    }
}

snapengine::custom::Rectangle TOPSARDeburstOp::GetSourceRectangle(int tx0, int ty0, int tw, int th,
                                                                  int sub_swath_index) const {
    std::vector<std::unique_ptr<SubSwathInfo>> subswath_;

    int x0 = GetSampleIndexInSourceProduct(tx0, *su_->subswath_.at(sub_swath_index - 1));
    int x_max = GetSampleIndexInSourceProduct(tx0 + tw - 1, *su_->subswath_.at(sub_swath_index - 1));

    BurstInfo burst_times{};
    GetLineIndicesInSourceProduct(ty0, *su_->subswath_.at(sub_swath_index - 1), burst_times);
    int y0;
    if (burst_times.sy0 == -1 && burst_times.sy1 == -1) {
        y0 = 0;
    } else {
        y0 = burst_times.sy0;
    }

    GetLineIndicesInSourceProduct(ty0 + th - 1, *su_->subswath_.at(sub_swath_index - 1), burst_times);
    int y_max;
    if (burst_times.sy0 == -1 && burst_times.sy1 == -1) {
        y_max = su_->subswath_.at(sub_swath_index - 1)->num_of_lines_ - 1;
    } else {
        y_max = std::max(burst_times.sy0, burst_times.sy1);
    }

    int w = x_max - x0 + 1;
    int h = y_max - y0 + 1;

    return snapengine::custom::Rectangle(x0, y0, w, h);
}

int TOPSARDeburstOp::GetSampleIndexInSourceProduct(int tx, const SubSwathInfo& sub_swath) const {
    auto sx = static_cast<int>((((target_slant_range_time_to_first_pixel_ + tx * target_delta_slant_range_time_) -
                                 sub_swath.slr_time_to_first_pixel_) /
                                target_delta_slant_range_time_) +
                               0.5);
    return sx < 0 ? 0 : sx > sub_swath.num_of_samples_ - 1 ? sub_swath.num_of_samples_ - 1 : sx;
}

bool TOPSARDeburstOp::GetLineIndicesInSourceProduct(int ty, const SubSwathInfo& sub_swath,
                                                    BurstInfo& burst_times) const {
    double target_line_time = target_first_line_time_ + ty * target_line_time_interval_;
    burst_times.target_time = target_line_time;
    burst_times.sy0 = -1;
    burst_times.sy1 = -1;
    int k = 0;
    for (int i = 0; i < sub_swath.num_of_bursts_; i++) {
        if (target_line_time >= sub_swath.burst_first_line_time_.at(i) &&
            target_line_time < sub_swath.burst_last_line_time_.at(i)) {
            auto sy = i * sub_swath.lines_per_burst_ +
                      static_cast<int>(((target_line_time - sub_swath.burst_first_line_time_.at(i)) /
                                        sub_swath.azimuth_time_interval_) +
                                       0.5);
            if (k == 0) {
                burst_times.sy0 = sy;
                burst_times.burst_num0 = i;
            } else {
                burst_times.sy1 = sy;
                burst_times.burst_num1 = i;
                break;
            }
            ++k;
        }
    }

    if (burst_times.sy0 != -1 && burst_times.sy1 != -1) {
        // find time between bursts midTime
        // use first burst if targetLineTime is before midTime
        burst_times.mid_time = (sub_swath.burst_last_line_time_.at(burst_times.burst_num0) +
                                sub_swath.burst_first_line_time_.at(burst_times.burst_num1)) /
                               2.0;
    }
    return burst_times.sy0 != -1 || burst_times.sy1 != -1;
}

int TOPSARDeburstOp::ComputeYMin(const SubSwathInfo& sub_swath) const {
    return static_cast<int>((sub_swath.first_valid_line_time_ - target_first_line_time_) / target_line_time_interval_);
}

int TOPSARDeburstOp::ComputeYMax(const SubSwathInfo& sub_swath) const {
    return static_cast<int>((sub_swath.last_valid_line_time_ - target_first_line_time_) / target_line_time_interval_);
}

int TOPSARDeburstOp::ComputeXMin(const SubSwathInfo& sub_swath) const {
    return static_cast<int>((sub_swath.slr_time_to_first_valid_pixel_ - target_slant_range_time_to_first_pixel_) /
                            target_delta_slant_range_time_);
}

int TOPSARDeburstOp::ComputeXMax(const SubSwathInfo& sub_swath) const {
    return static_cast<int>((sub_swath.slr_time_to_last_valid_pixel_ - target_slant_range_time_to_first_pixel_) /
                            target_delta_slant_range_time_);
}

int TOPSARDeburstOp::GetSubSwathIndex(int tx, int ty, int first_sub_swath_index, int last_sub_swath_index,
                                      BurstInfo& burst_info) const {
    double target_sample_slr_time = target_slant_range_time_to_first_pixel_ + tx * target_delta_slant_range_time_;
    double target_line_time = target_first_line_time_ + ty * target_line_time_interval_;

    burst_info.swath0 = -1;
    burst_info.swath1 = -1;
    int cnt = 0;
    for (int i = first_sub_swath_index; i <= last_sub_swath_index; i++) {
        int i_1 = i - 1;
        if (target_line_time >= su_->subswath_.at(i_1)->first_valid_line_time_ &&
            target_line_time <= su_->subswath_.at(i_1)->last_valid_line_time_ &&
            target_sample_slr_time >= su_->subswath_.at(i_1)->slr_time_to_first_valid_pixel_ &&
            target_sample_slr_time <= su_->subswath_.at(i_1)->slr_time_to_last_valid_pixel_) {
            if (cnt == 0) {
                burst_info.swath0 = i;
            } else {
                burst_info.swath1 = i;
                break;
            }
            ++cnt;
        }
    }

    if (burst_info.swath1 != -1) {
        double middle_time = (su_->subswath_.at(burst_info.swath0 - 1)->slr_time_to_last_valid_pixel_ +
                              su_->subswath_.at(burst_info.swath1 - 1)->slr_time_to_first_valid_pixel_) /
                             2.0;

        if (target_sample_slr_time > middle_time) {
            return burst_info.swath1;
        }
    }
    return burst_info.swath0;
}

std::shared_ptr<TOPSARDeburstOp> TOPSARDeburstOp::CreateTOPSARDeburstOp(
    const std::shared_ptr<snapengine::Product>& product) {
    auto op = std::shared_ptr<TOPSARDeburstOp>(new TOPSARDeburstOp(product));
    op->Initialize();
    return op;
}

void TOPSARDeburstOp::Initialize() {
    try {
        // todo: following "if clause" is just for testing purposes and supports older custom formats we had while
        // developing (remove when support ends)
        if (source_product_->HasMetaDataReader()) {
            source_product_->GetMetadataRoot()->AddElement(
                source_product_->GetMetadataReader()->Read(alus::snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT));
            source_product_->GetMetadataRoot()->AddElement(source_product_->GetMetadataReader()->Read(
                alus::snapengine::AbstractMetadata::ORIGINAL_PRODUCT_METADATA));
            // test needs bands, this is ugly fix to a temporary problem
            auto bands_info =
                std::dynamic_pointer_cast<alus::snapengine::PugixmlMetaDataReader>(source_product_->GetMetadataReader())
                    ->ReadImageInterpretationTag();
            for (const auto& bi : bands_info) {
                auto band = std::make_shared<snapengine::Band>(bi.band_name, bi.product_data_type,
                                                               source_product_->GetSceneRasterWidth(),
                                                               source_product_->GetSceneRasterHeight());
                // set needed properties (this is just for future example)
                band->SetSpectralWavelength(bi.band_wavelength.value());
                source_product_->AddBand(band);
            }
        }

        auto validator = std::make_shared<snapengine::InputProductValidator>(source_product_);
        validator->CheckIfSARProduct();
        validator->CheckIfSentinel1Product();
        validator->CheckProductType({"SLC"});
        validator->CheckAcquisitionMode({"IW", "EW"});

        abs_root_ = snapengine::AbstractMetadata::GetAbstractedMetadata(source_product_);
        GetProductType();
        GetAcquisitionMode();
        su_ = std::make_unique<Sentinel1Utils>(source_product_);
        num_of_sub_swath_ = su_->GetNumOfSubSwath();

        // checkIfSplitProduct(); todo: might need in the future

        if (selected_polarisations_.empty()) {
            selected_polarisations_ = su_->GetPolarizations();
        }

        ComputeTargetStartEndTime();

        ComputeTargetSlantRangeTimeToFirstAndLastPixels();

        ComputeTargetWidthAndHeight();

        CreateTargetProduct();

        ComputeSubSwathEffectStartEndPixels();

        UpdateTargetProductMetadata();

        // deviation from snap
        target_rectangles_ =
            std::make_shared<TOPSARDeburstRectanglesGenerator>(target_product_->GetSceneRasterWidth(),
                                                               target_product_->GetSceneRasterHeight(), 624, 544)
                ->GetRectangles();

    } catch (const std::exception& e) {
        throw std::runtime_error("TOPSARDeburstOp exception: " + std::string(e.what()));
    }
}

void TOPSARDeburstOp::ComputeTileStack(
    std::unordered_map<std::shared_ptr<snapengine::Band>, std::shared_ptr<snapengine::ITile>>& target_tiles,
    const snapengine::custom::Rectangle& target_rectangle, const std::shared_ptr<ceres::IProgressMonitor>& pm) {
    try {
        int tx0 = target_rectangle.x;
        int ty0 = target_rectangle.y;
        int tw = target_rectangle.width;
        int th = target_rectangle.height;

        // determine subswaths covered by the tile
        double tile_slrt_to_first_pixel =
            target_slant_range_time_to_first_pixel_ + tx0 * target_delta_slant_range_time_;
        double tile_slrt_to_last_pixel =
            target_slant_range_time_to_first_pixel_ + (tx0 + tw - 1) * target_delta_slant_range_time_;
        double tile_first_line_time = target_first_line_time_ + ty0 * target_line_time_interval_;
        double tile_last_line_time = target_first_line_time_ + (ty0 + th - 1) * target_line_time_interval_;

        int first_sub_swath_index = -1;
        int last_sub_swath_index = -1;
        for (int i = 0; i < num_of_sub_swath_; i++) {
            if (tile_slrt_to_first_pixel >= su_->subswath_.at(i)->slr_time_to_first_valid_pixel_ &&
                tile_slrt_to_first_pixel <= su_->subswath_.at(i)->slr_time_to_last_valid_pixel_) {
                if ((tile_first_line_time >= su_->subswath_.at(i)->burst_first_valid_line_time_.at(0) &&
                     tile_first_line_time <
                         su_->subswath_.at(i)->burst_last_line_time_.at(su_->subswath_.at(i)->num_of_bursts_ - 1)) ||
                    (tile_last_line_time >= su_->subswath_.at(i)->burst_first_valid_line_time_.at(0) &&
                     tile_last_line_time <
                         su_->subswath_.at(i)->burst_last_line_time_.at(su_->subswath_.at(i)->num_of_bursts_ - 1))) {
                    first_sub_swath_index = i + 1;
                    break;
                }
            }
        }

        if (first_sub_swath_index == num_of_sub_swath_) {
            last_sub_swath_index = first_sub_swath_index;
        } else {
            for (int i = 0; i < num_of_sub_swath_; i++) {
                if (tile_slrt_to_last_pixel >= su_->subswath_.at(i)->slr_time_to_first_valid_pixel_ &&
                    tile_slrt_to_last_pixel <= su_->subswath_.at(i)->slr_time_to_last_valid_pixel_) {
                    if ((tile_first_line_time >= su_->subswath_.at(i)->burst_first_valid_line_time_.at(0) &&
                         tile_first_line_time < su_->subswath_.at(i)->burst_last_line_time_.at(
                                                    su_->subswath_.at(i)->num_of_bursts_ - 1)) ||
                        (tile_last_line_time >= su_->subswath_.at(i)->burst_first_valid_line_time_.at(0) &&
                         tile_last_line_time < su_->subswath_.at(i)->burst_last_line_time_.at(
                                                   su_->subswath_.at(i)->num_of_bursts_ - 1))) {
                        last_sub_swath_index = i + 1;
                    }
                }
            }
        }

        if (first_sub_swath_index == -1 && last_sub_swath_index == -1) {
            return;
        }

        if (first_sub_swath_index != -1 && last_sub_swath_index == -1) {
            last_sub_swath_index = first_sub_swath_index;
        }

        if (first_sub_swath_index == -1 && last_sub_swath_index != -1) {
            first_sub_swath_index = last_sub_swath_index;
        }

        int num_of_source_tiles = last_sub_swath_index - first_sub_swath_index + 1;
        bool tile_in_one_sub_swath = (num_of_source_tiles == 1);

        std::vector<snapengine::custom::Rectangle> source_rectangle(num_of_source_tiles);
        int k = 0;
        for (int i = first_sub_swath_index; i <= last_sub_swath_index; i++) {
            source_rectangle.at(k++) = GetSourceRectangle(tx0, ty0, tw, th, i);
        }

        BurstInfo burst_info{};
        int tx_max = tx0 + tw;
        int ty_max = ty0 + th;
        std::vector<std::shared_ptr<snapengine::Band>> tgt_bands = target_product_->GetBands();
        for (const auto& tgt_band : tgt_bands) {
            if (std::dynamic_pointer_cast<snapengine::VirtualBand>(tgt_band)) {
                continue;
            }

            std::string tgt_band_name = tgt_band->GetName();
            int data_type = tgt_band->GetDataType();
            auto tgt_tile = target_tiles.at(tgt_band);
            if (tile_in_one_sub_swath) {
                if (data_type == snapengine::ProductData::TYPE_INT16) {
                    throw std::runtime_error("Current support is for single swath float");
                    //                    todo: if need arises
                    //                        ComputeTileInOneSwathShort(tx0, ty0, tx_max, ty_max,
                    //                        first_sub_swath_index, source_rectangle,
                    //                                                   tgt_band_name, tgt_tile, burst_info);
                } else {
                    ComputeTileInOneSwathFloat(tx0, ty0, tx_max, ty_max, first_sub_swath_index, source_rectangle,
                                               tgt_band_name, tgt_tile, burst_info);
                }

            } else {
                if (data_type == snapengine::ProductData::TYPE_INT16) {
                    ComputeMultipleSubSwathsShort(tx0, ty0, tx_max, ty_max, first_sub_swath_index, last_sub_swath_index,
                                                  source_rectangle, tgt_band_name, tgt_tile, burst_info);
                } else {
                    ComputeMultipleSubSwathsFloat(tx0, ty0, tx_max, ty_max, first_sub_swath_index, last_sub_swath_index,
                                                  source_rectangle, tgt_band_name, tgt_tile, burst_info);
                }
            }

            // current writer implementation is for float single band! (important when supporting other bands)
            //            todo: if using multiple bands check if this works correctly (assuming it does atm)
            auto band_id = target_product_->GetBandIndex(tgt_band_name);
            target_product_->GetImageWriter()->WriteSubSampledData(target_rectangle, tgt_tile->GetSimpleDataBuffer(),
                                                                   band_id + 1);
        }
        pm->Done();
    } catch (const std::exception& e) {
        pm->Done();
        throw std::runtime_error("TOPSARDeburstOp exception: " + std::string(e.what()));
    }
}

std::shared_ptr<snapengine::ITile> TOPSARDeburstOp::GetSourceTile(
    const std::shared_ptr<snapengine::RasterDataNode>& raster_data_node, const snapengine::custom::Rectangle& region,
    int band_indx) {
    auto tile = std::make_shared<snapengine::TileImpl>(raster_data_node, region);
    std::vector<float>& buffer = tile->GetSimpleDataBuffer();
    source_product_->GetImageReader()->ReadSubSampledData(region, band_indx);
    buffer = source_product_->GetImageReader()->GetData();
    return tile;
}

void TOPSARDeburstOp::Compute() {
    std::unordered_map<std::shared_ptr<snapengine::Band>, std::shared_ptr<snapengine::ITile>> target_tiles;
    for (const auto& rect : target_rectangles_) {
        for (const auto& band : target_product_->GetBands()) {
            target_tiles[band] = std::make_shared<snapengine::TileImpl>(band, rect);
        }
        ComputeTileStack(target_tiles, rect, std::make_shared<ceres::NullProgressMonitor>());
    }
}

const std::shared_ptr<snapengine::Product>& TOPSARDeburstOp::GetTargetProduct() const { return target_product_; }

}  // namespace alus::s1tbx