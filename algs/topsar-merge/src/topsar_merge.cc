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
#include "topsar_merge.h"

#include <boost/algorithm/string/predicate.hpp>

#include <algorithm>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

#include "algorithm_exception.h"
#include "custom/gdal_image_writer.h"
#include "custom/i_image_reader.h"
#include "general_constants.h"
#include "general_utils.h"
#include "s1tbx-commons/sentinel1_utils.h"
#include "shapes_util.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/product_data_utc.h"
#include "snap-core/core/datamodel/pugixml_meta_data_reader.h"
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
#include "snap-engine-utilities/engine-utilities/gpf/stack_utils.h"
#include "snap-gpf/gpf/internal/tile_impl.h"

namespace {
constexpr std::string_view ALGORITHM_NAME{"TOPSAR Merge"};
}  // namespace

namespace alus::topsarmerge {

void TopsarMergeOperator::Initialise() {
    if (source_products_.size() < MINIMAL_AMOUNT_OF_PRODUCTS) {
        THROW_ALGORITHM_EXCEPTION(ALGORITHM_NAME, "Please select at least two products.");
    }

    PrepareBeamDimapSourceProducts();
    CheckSourceProductsValidity();
    UpdateSubSwathParameters();
    ComputeTargetStartEndTime(parameters_, sub_swath_merge_info_);
    ComputeTargetSlantRangeTimeToFirstAndLastPixels(parameters_, sub_swath_merge_info_);
    ComputeTargetWidthAndHeight(parameters_);
    CreateTargetProduct();
    UpdateTargetProductMetadata();

    target_rectangles_ = shapeutils::GenerateRectanglesForRaster(
        target_product_->GetSceneRasterWidth(), target_product_->GetSceneRasterHeight(), tile_width_, tile_height_);
}
void TopsarMergeOperator::CheckSourceProductsValidity() {
    for (const auto& source_product : source_products_) {
        snapengine::InputProductValidator validator(source_product);
        validator.CheckIfSARProduct();
        validator.CheckIfSentinel1Product();
        if (!validator.IsTOPSARProduct()) {
            THROW_ALGORITHM_EXCEPTION(ALGORITHM_NAME, "Source product should be a TOPSAR product.");
        }
        validator.CheckIfTOPSARBurstProduct(false);
        // TODO: check that map is not projected. Currently not implemented due to fact that CrsGeocoding and
        // MapGeoCoding are not completely ported to ALUS
    }
    CheckAreSubSwathsFromSameProduct();
}
void TopsarMergeOperator::CheckAreSubSwathsFromSameProduct() {
    const auto source_product = source_products_.at(0);
    const auto abstract_root_0 = snapengine::AbstractMetadata::GetAbstractedMetadata(source_product);
    parameters_.number_of_subswaths = source_products_.size();

    const auto number_of_bands_0 = source_product->GetNumBands();
    std::vector<size_t> subswath_index_vector(parameters_.number_of_subswaths);

    const auto product_0_name = abstract_root_0->GetAttributeString(snapengine::AbstractMetadata::PRODUCT);
    parameters_.acquisition_mode = abstract_root_0->GetAttributeString(snapengine::AbstractMetadata::ACQUISITION_MODE);
    parameters_.product_type = abstract_root_0->GetAttributeString(snapengine::AbstractMetadata::PRODUCT_TYPE);

    const auto subswath_name_0 = abstract_root_0->GetAttributeString(snapengine::AbstractMetadata::swath);
    if (subswath_name_0.empty()) {
        THROW_ALGORITHM_EXCEPTION(ALGORITHM_NAME, std::string("Cannot get \"swath\" information from source product") +
                                                      product_0_name + " abstracted metadata.");
    }
    subswath_index_vector.at(0) = GetSubSwathIndexFromName(subswath_name_0);

    using bimap_position = boost::bimap<size_t, size_t>::value_type;
    source_product_index_to_sub_swath_index_map_.insert(bimap_position(0, subswath_index_vector.at(0)));

    for (size_t p = 1; p < parameters_.number_of_subswaths; p++) {
        const auto abstract_root = snapengine::AbstractMetadata::GetAbstractedMetadata(source_products_.at(p));
        const auto product_name = abstract_root->GetAttributeString(snapengine::AbstractMetadata::PRODUCT);
        if (product_name != product_0_name) {
            THROW_ALGORITHM_EXCEPTION(ALGORITHM_NAME, "Source products are not from the same Sentinel-1 product.");
        }

        if (const auto acquisition_mode =
                abstract_root->GetAttributeString(snapengine::AbstractMetadata::ACQUISITION_MODE);
            acquisition_mode != parameters_.acquisition_mode) {
            THROW_ALGORITHM_EXCEPTION(ALGORITHM_NAME, "Source products do not have the same acquisition mode.");
        }

        if (number_of_bands_0 != source_products_.at(p)->GetNumBands()) {
            THROW_ALGORITHM_EXCEPTION(ALGORITHM_NAME, "Source products do not have the same number of bands.");
        }

        const auto subswath_name = abstract_root->GetAttributeString(snapengine::AbstractMetadata::swath);
        if (subswath_name.empty()) {
            THROW_ALGORITHM_EXCEPTION(
                ALGORITHM_NAME, std::string("Cannot get \"swath\" information from source product") + product_name +
                                    "abstracted metadata.");
        }
        subswath_index_vector.at(p) = GetSubSwathIndexFromName(subswath_name);
        source_product_index_to_sub_swath_index_map_.insert(bimap_position(p, subswath_index_vector.at(p)));
    }

    std::sort(subswath_index_vector.begin(),
              subswath_index_vector.end());  // TODO: add parallel_unsequenced_policy if minimal supported gcc version
                                             //       is changed to 9.1+ (Ubuntu 21.04 already uses 10+)
    parameters_.reference_sub_swath_index = subswath_index_vector.at(0);
    for (size_t s = 0; s < parameters_.number_of_subswaths - 1; ++s) {
        if (subswath_index_vector.at(s + 1) - subswath_index_vector.at(s) != 1) {
            THROW_ALGORITHM_EXCEPTION(ALGORITHM_NAME, "Isolate sub-swath detected in source products.");
        }
    }
}

void TopsarMergeOperator::UpdateSubSwathParameters() {
    try {
        for (size_t p = 0; p < parameters_.number_of_subswaths; p++) {
            const auto s =
                source_product_index_to_sub_swath_index_map_.left.at(p) - parameters_.reference_sub_swath_index;
            auto sentinel_1_utils = std::make_shared<s1tbx::Sentinel1Utils>(source_products_.at(p));
            sentinel_utils_.try_emplace(s, sentinel_1_utils);
            sub_swath_info_.try_emplace(s, sentinel_1_utils->GetSubSwath().at(0));
            if (selected_polarisations_.empty()) {
                selected_polarisations_ = sentinel_1_utils->GetPolarizations();
            }

            const auto abstract_root = snapengine::AbstractMetadata::GetAbstractedMetadata(source_products_.at(p));
            const auto& sub_swath_info = sub_swath_info_.at(s);
            sub_swath_info->first_valid_pixel_ = snapengine::AbstractMetadata::GetAttributeInt(
                abstract_root, snapengine::AbstractMetadata::FIRST_VALID_PIXEL);
            sub_swath_info->last_valid_pixel_ = snapengine::AbstractMetadata::GetAttributeInt(
                abstract_root, snapengine::AbstractMetadata::LAST_VALID_PIXEL);
            sub_swath_info->slr_time_to_first_valid_pixel_ = snapengine::AbstractMetadata::GetAttributeDouble(
                abstract_root, snapengine::AbstractMetadata::SLR_TIME_TO_FIRST_VALID_PIXEL);
            sub_swath_info->slr_time_to_last_valid_pixel_ = snapengine::AbstractMetadata::GetAttributeDouble(
                abstract_root, snapengine::AbstractMetadata::SLR_TIME_TO_LAST_VALID_PIXEL);
            sub_swath_info->first_valid_line_time_ = snapengine::AbstractMetadata::GetAttributeDouble(
                abstract_root, snapengine::AbstractMetadata::FIRST_VALID_LINE_TIME);
            sub_swath_info->last_valid_line_time_ = snapengine::AbstractMetadata::GetAttributeDouble(
                abstract_root, snapengine::AbstractMetadata::LAST_VALID_LINE_TIME);

            sub_swath_merge_info_.try_emplace(
                s, GetSubSwathMergeInfoFromSentinel1SubSwathInfo(*sentinel_1_utils->GetSubSwath().at(0)));
        }
    } catch (const std::exception& e) {
        THROW_ALGORITHM_EXCEPTION(ALGORITHM_NAME, e.what());
    }
}
void TopsarMergeOperator::CreateTargetProduct() {
    const auto product_index =
        source_product_index_to_sub_swath_index_map_.right.at(parameters_.reference_sub_swath_index);

    target_product_ = snapengine::Product::CreateProduct(
        source_products_.at(product_index)->GetName() + PRODUCT_SUFFIX.data(), parameters_.product_type,
        parameters_.target_width, parameters_.target_height);

    // Source band name is assumed to be in format: name_acquisitionModeAndSubSwathIndex_polarisation_prefix
    // Target band is then in format: name_polarisation_prefix
    PopulateSourceBandNameVectors(static_cast<int>(product_index));

    snapengine::ProductUtils::CopyMetadata(source_products_.at(product_index), target_product_);
    snapengine::ProductUtils::CopyFlagCodings(source_products_.at(product_index), target_product_);
    target_product_->SetStartTime(std::make_shared<snapengine::Utc>(parameters_.target_first_line_time /
                                                                    snapengine::eo::constants::SECONDS_IN_DAY));
    target_product_->SetEndTime(std::make_shared<snapengine::Utc>(parameters_.target_last_line_time /
                                                                  snapengine::eo::constants::SECONDS_IN_DAY));
    target_product_->SetDescription(source_products_.at(product_index)->GetDescription());

    CreateTargetTiePointGrids();

    if (!boost::ends_with(output_path_, gdal::constants::GDAL_GTIFF_FILE_EXTENSION)) {
        output_path_ += gdal::constants::GDAL_GTIFF_FILE_EXTENSION;
    }
    target_product_->SetFileLocation(boost::filesystem::path(output_path_));
}
void TopsarMergeOperator::CreateTargetTiePointGrids() {
    const auto sub_sampling_x = parameters_.target_width / TIE_POINT_GRID_WIDTH;
    const auto sub_sampling_y = parameters_.target_height / TIE_POINT_GRID_HEIGHT;

    const auto grid_size = TIE_POINT_GRID_WIDTH * TIE_POINT_GRID_HEIGHT;

    std::vector<float> latitude_vector(grid_size);
    std::vector<float> longitude_vector(grid_size);
    std::vector<float> slant_range_time_vector(grid_size);
    std::vector<float> incident_angle_vector(grid_size);

    size_t k{0};

    for (int i = 0; i < TIE_POINT_GRID_HEIGHT; i++) {
        const auto y = i * sub_sampling_y;
        const auto azimuth_time = parameters_.target_first_line_time + y * parameters_.target_line_time_interval;
        for (int j = 0; j < TIE_POINT_GRID_WIDTH; j++, k++) {
            const auto x = j * sub_sampling_x;
            const auto slant_range_time =
                parameters_.target_slant_range_time_to_first_pixel + x * parameters_.target_delta_slant_range_time;
            const auto s = GetSubSwathIndexBySlrTime(slant_range_time, parameters_, sub_swath_merge_info_);
            latitude_vector.at(k) =
                static_cast<float>(sentinel_utils_.at(s)->GetLatitude(azimuth_time, slant_range_time));
            longitude_vector.at(k) =
                static_cast<float>(sentinel_utils_.at(s)->GetLongitude(azimuth_time, slant_range_time));
            slant_range_time_vector.at(k) =
                static_cast<float>(sentinel_utils_.at(s)->GetSlantRangeTime(azimuth_time, slant_range_time) * 2 *
                                   snapengine::eo::constants::ONE_BILLION);  // 2-way ns
            incident_angle_vector.at(k) =
                static_cast<float>(sentinel_utils_.at(s)->GetIncidenceAngle(azimuth_time, slant_range_time));
        }
    }

    const auto latitude_grid = std::make_shared<snapengine::TiePointGrid>(
        snapengine::OperatorUtils::TPG_LATITUDE, TIE_POINT_GRID_WIDTH, TIE_POINT_GRID_HEIGHT, TIE_POINT_GRID_OFFSET_X,
        TIE_POINT_GRID_OFFSET_Y, static_cast<double>(sub_sampling_x), static_cast<double>(sub_sampling_y),
        latitude_vector);

    const auto longitude_grid = std::make_shared<snapengine::TiePointGrid>(
        snapengine::OperatorUtils::TPG_LONGITUDE, TIE_POINT_GRID_WIDTH, TIE_POINT_GRID_HEIGHT, TIE_POINT_GRID_OFFSET_X,
        TIE_POINT_GRID_OFFSET_Y, static_cast<double>(sub_sampling_x), static_cast<double>(sub_sampling_y),
        longitude_vector);

    const auto slant_range_time_grid = std::make_shared<snapengine::TiePointGrid>(
        snapengine::OperatorUtils::TPG_SLANT_RANGE_TIME, TIE_POINT_GRID_WIDTH, TIE_POINT_GRID_HEIGHT,
        TIE_POINT_GRID_OFFSET_X, TIE_POINT_GRID_OFFSET_Y, static_cast<double>(sub_sampling_x),
        static_cast<double>(sub_sampling_y), slant_range_time_vector);

    const auto incident_angle_grid = std::make_shared<snapengine::TiePointGrid>(
        snapengine::OperatorUtils::TPG_INCIDENT_ANGLE, TIE_POINT_GRID_WIDTH, TIE_POINT_GRID_HEIGHT,
        TIE_POINT_GRID_OFFSET_X, TIE_POINT_GRID_OFFSET_Y, static_cast<double>(sub_sampling_x),
        static_cast<double>(sub_sampling_y), incident_angle_vector);

    latitude_grid->SetUnit(snapengine::Unit::DEGREES);
    longitude_grid->SetUnit(snapengine::Unit::DEGREES);
    slant_range_time_grid->SetUnit(snapengine::Unit::NANOSECONDS);
    incident_angle_grid->SetUnit(snapengine::Unit::DEGREES);

    target_product_->AddTiePointGrid(latitude_grid);
    target_product_->AddTiePointGrid(longitude_grid);
    target_product_->AddTiePointGrid(slant_range_time_grid);
    target_product_->AddTiePointGrid(incident_angle_grid);

    const auto tie_point_geocoding = std::make_shared<snapengine::TiePointGeoCoding>(latitude_grid, longitude_grid);

    target_product_->SetSceneGeoCoding(tie_point_geocoding);
}

void TopsarMergeOperator::UpdateTargetProductMetadata() const { UpdateTargetProductAbstractedMetadata(); }

void TopsarMergeOperator::UpdateTargetProductAbstractedMetadata() const {
    const auto target_abstracted_metadata = snapengine::AbstractMetadata::GetAbstractedMetadata(target_product_);
    snapengine::AbstractMetadata::SetAttribute(
        target_abstracted_metadata, snapengine::AbstractMetadata::NUM_OUTPUT_LINES, parameters_.target_height);
    snapengine::AbstractMetadata::SetAttribute(
        target_abstracted_metadata, snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE, parameters_.target_width);
    target_abstracted_metadata->SetAttributeUtc(
        snapengine::AbstractMetadata::FIRST_LINE_TIME,
        std::make_shared<snapengine::Utc>(parameters_.target_first_line_time /
                                          snapengine::eo::constants::SECONDS_IN_DAY));
    target_abstracted_metadata->SetAttributeUtc(
        snapengine::AbstractMetadata::LAST_LINE_TIME,
        std::make_shared<snapengine::Utc>(parameters_.target_last_line_time /
                                          snapengine::eo::constants::SECONDS_IN_DAY));
    target_abstracted_metadata->SetAttributeDouble(snapengine::AbstractMetadata::LINE_TIME_INTERVAL,
                                                   parameters_.target_line_time_interval);
    target_abstracted_metadata->SetAttributeDouble(
        snapengine::AbstractMetadata::SLANT_RANGE_TO_FIRST_PIXEL,
        parameters_.target_slant_range_time_to_first_pixel * snapengine::eo::constants::LIGHT_SPEED);

    const auto latitude_grid = target_product_->GetTiePointGrid(snapengine::OperatorUtils::TPG_LATITUDE);
    const auto longitude_grid = target_product_->GetTiePointGrid(snapengine::OperatorUtils::TPG_LONGITUDE);

    snapengine::AbstractMetadata::SetAttribute(target_abstracted_metadata, snapengine::AbstractMetadata::FIRST_NEAR_LAT,
                                               latitude_grid->GetPixelFloat(0, 0));
    snapengine::AbstractMetadata::SetAttribute(
        target_abstracted_metadata, snapengine::AbstractMetadata::FIRST_NEAR_LONG, longitude_grid->GetPixelFloat(0, 0));
    snapengine::AbstractMetadata::SetAttribute(target_abstracted_metadata, snapengine::AbstractMetadata::FIRST_FAR_LAT,
                                               latitude_grid->GetPixelFloat(parameters_.target_width, 0));
    snapengine::AbstractMetadata::SetAttribute(target_abstracted_metadata, snapengine::AbstractMetadata::FIRST_FAR_LONG,
                                               longitude_grid->GetPixelFloat(parameters_.target_width, 0));

    snapengine::AbstractMetadata::SetAttribute(target_abstracted_metadata, snapengine::AbstractMetadata::LAST_NEAR_LAT,
                                               latitude_grid->GetPixelFloat(0, parameters_.target_height));
    snapengine::AbstractMetadata::SetAttribute(target_abstracted_metadata, snapengine::AbstractMetadata::LAST_NEAR_LONG,
                                               longitude_grid->GetPixelFloat(0, parameters_.target_height));
    snapengine::AbstractMetadata::SetAttribute(
        target_abstracted_metadata, snapengine::AbstractMetadata::LAST_FAR_LAT,
        latitude_grid->GetPixelFloat(parameters_.target_width, parameters_.target_height));
    snapengine::AbstractMetadata::SetAttribute(
        target_abstracted_metadata, snapengine::AbstractMetadata::LAST_FAR_LONG,
        longitude_grid->GetPixelFloat(parameters_.target_width, parameters_.target_height));

    const auto incidence_near = snapengine::OperatorUtils::GetIncidenceAngle(target_product_)
                                    ->GetPixelDouble(0, target_product_->GetSceneRasterHeight() / 2);

    snapengine::AbstractMetadata::SetAttribute(target_abstracted_metadata, snapengine::AbstractMetadata::INCIDENCE_NEAR,
                                               incidence_near);

    const auto incidence_far =
        snapengine::OperatorUtils::GetIncidenceAngle(target_product_)
            ->GetPixelDouble(target_product_->GetSceneRasterWidth() - 1, target_product_->GetSceneRasterHeight() / 2);

    snapengine::AbstractMetadata::SetAttribute(target_abstracted_metadata, snapengine::AbstractMetadata::INCIDENCE_FAR,
                                               incidence_far);

    target_abstracted_metadata->RemoveAttribute(
        target_abstracted_metadata->GetAttribute(snapengine::AbstractMetadata::FIRST_VALID_PIXEL));
    target_abstracted_metadata->RemoveAttribute(
        target_abstracted_metadata->GetAttribute(snapengine::AbstractMetadata::LAST_VALID_PIXEL));
    target_abstracted_metadata->RemoveAttribute(
        target_abstracted_metadata->GetAttribute(snapengine::AbstractMetadata::SLR_TIME_TO_FIRST_VALID_PIXEL));
    target_abstracted_metadata->RemoveAttribute(
        target_abstracted_metadata->GetAttribute(snapengine::AbstractMetadata::SLR_TIME_TO_LAST_VALID_PIXEL));
    target_abstracted_metadata->RemoveAttribute(
        target_abstracted_metadata->GetAttribute(snapengine::AbstractMetadata::FIRST_VALID_LINE_TIME));
    target_abstracted_metadata->RemoveAttribute(
        target_abstracted_metadata->GetAttribute(snapengine::AbstractMetadata::LAST_VALID_LINE_TIME));

    target_abstracted_metadata->RemoveElement(
        target_abstracted_metadata->GetElement(snapengine::AbstractMetadata::BURST_BOUNDARY));
    target_abstracted_metadata->RemoveElement(
        target_abstracted_metadata->GetElement(snapengine::AbstractMetadata::ESD_MEASUREMENT));

    const auto burst_boundary_target =
        std::make_shared<snapengine::MetadataElement>(snapengine::AbstractMetadata::BURST_BOUNDARY);
    const auto esd_measurement_target =
        std::make_shared<snapengine::MetadataElement>(snapengine::AbstractMetadata::ESD_MEASUREMENT);

    for (const auto& source_product : source_products_) {
        const auto abs_root = snapengine::AbstractMetadata::GetAbstractedMetadata(source_product);
        if (const auto burst_boundary_source = abs_root->GetElement(snapengine::AbstractMetadata::BURST_BOUNDARY);
            burst_boundary_source && burst_boundary_source->GetNumElements() > 0) {
            const auto element = burst_boundary_source->GetElementAt(0);
            if (element) {
                burst_boundary_target->AddElement(element->CreateDeepClone());
            }
        }

        if (const auto esd_measurement_source = abs_root->GetElement(snapengine::AbstractMetadata::ESD_MEASUREMENT);
            esd_measurement_source) {
            const auto element = esd_measurement_source->GetElementAt(0);
            if (element) {
                esd_measurement_target->AddElement(element->CreateDeepClone());
            }
        }
    }
    target_abstracted_metadata->AddElement(burst_boundary_target);
    target_abstracted_metadata->AddElement(esd_measurement_target);
}

void TopsarMergeOperator::ComputeTileStack(
    std::unordered_map<std::shared_ptr<snapengine::Band>, std::shared_ptr<snapengine::ITile>>& target_tiles,
    const Rectangle& target_rectangle) {
    // Determine subswaths covered by the tile
    int first_sub_swath_index{-1};
    int last_sub_swath_index{-1};
    FindFirstAndLastSubSwathIndices(first_sub_swath_index, last_sub_swath_index, target_rectangle, parameters_,
                                    sub_swath_merge_info_);

    if (first_sub_swath_index == -1 && last_sub_swath_index == -1) {
        return;
    }

    const auto number_of_source_tiles = last_sub_swath_index - first_sub_swath_index + 1;
    const bool is_tile_in_one_sub_swath = number_of_source_tiles == 1;

    const auto source_rectangles =
        GetSourceRectangles(number_of_source_tiles, first_sub_swath_index, last_sub_swath_index, target_rectangle,
                            parameters_, sub_swath_merge_info_);

    for (const auto& target_band : target_product_->GetBands()) {
        const auto target_band_name = target_band->GetName();
        const auto data_type = target_band->GetDataType();
        const auto target_tile = target_tiles.at(target_band);
        if (is_tile_in_one_sub_swath) {
            if (data_type == snapengine::ProductData::TYPE_INT16) {
                ComputeTileInOneSwathShort(target_rectangle, first_sub_swath_index, source_rectangles.at(0),
                                           target_band_name, target_tile.get());
            } else {
                ComputeTileInOneSwathFloat(target_rectangle, first_sub_swath_index, source_rectangles.at(0),
                                           target_band_name, target_tile.get());
            }
        } else {
            if (data_type == snapengine::ProductData::TYPE_INT16) {
                THROW_ALGORITHM_EXCEPTION(ALGORITHM_NAME, "Short data type support is not yet implemented.");
            } else {
                ComputeMultipleSubSwathFloat(target_rectangle, first_sub_swath_index, last_sub_swath_index,
                                             source_rectangles, target_band_name, target_tile.get());
            }
        }

        target_product_->GetImageWriter()->WriteSubSampledData(target_rectangle, target_tile->GetSimpleDataBuffer(), 1);
    }
}

void TopsarMergeOperator::Compute() {
    std::unordered_map<std::shared_ptr<snapengine::Band>, std::shared_ptr<snapengine::ITile>> target_tiles;
    for (const auto& rectangle : target_rectangles_) {
        for (const auto& band : target_product_->GetBands()) {
            target_tiles[band] = std::make_shared<snapengine::TileImpl>(band, rectangle);
        }
        ComputeTileStack(target_tiles, rectangle);
    }
}
void TopsarMergeOperator::ComputeTileInOneSwathFloat(Rectangle target_rectangle, int first_sub_swath_index,
                                                     Rectangle source_rectangle, std::string_view target_band_name,
                                                     snapengine::ITile* target_tile) {
    const auto sub_swath = sub_swath_info_.at(first_sub_swath_index);
    const auto& sub_swath_merge_info = sub_swath_merge_info_.at(first_sub_swath_index);

    const auto y_min = ComputeYMin(sub_swath_merge_info, parameters_);
    const auto y_max = ComputeYMax(sub_swath_merge_info, parameters_);
    const auto x_min = ComputeXMin(sub_swath_merge_info, parameters_);
    const auto x_max = ComputeXMax(sub_swath_merge_info, parameters_);

    const auto first_y = std::max(target_rectangle.y, y_min);
    const auto last_y = std::min(target_rectangle.y + target_rectangle.height, y_max + 1);
    const auto first_x = std::max(target_rectangle.x, x_min);
    const auto last_x = std::min(target_rectangle.x + target_rectangle.width, x_max + 1);

    if (first_y >= last_y || first_x >= last_x) {
        return;
    }

    const auto swath_index = GetSubSwathIndexFromName(sub_swath->subswath_name_);
    const auto swath_index_str = std::to_string(swath_index);
    const auto source_band =
        GetSourceBandFromTargetBandName(target_band_name, parameters_.acquisition_mode, swath_index_str);
    auto source_raster =
        GetSourceTile(source_band, source_rectangle, 1,
                      static_cast<int>(source_product_index_to_sub_swath_index_map_.right.at(swath_index)));
    auto source_tile_index = std::make_shared<snapengine::TileIndex>(source_raster);
    auto target_tile_index = std::make_shared<snapengine::TileIndex>(target_tile);

    auto& source_data_vector = source_raster->GetSimpleDataBuffer();
    auto& target_data_vector = target_tile->GetSimpleDataBuffer();

    for (int y = first_y; y < last_y; ++y) {
        const auto source_y_0 = GetLineIndexInSourceProduct(y, sub_swath_merge_info, parameters_);
        const auto target_offset = target_tile_index->CalculateStride(y);
        const auto source_offset = source_tile_index->CalculateStride(source_y_0);

        const auto source_x_0 = static_cast<int>((std::round(((parameters_.target_slant_range_time_to_first_pixel +
                                                               first_x * parameters_.target_delta_slant_range_time) -
                                                              sub_swath->slr_time_to_first_pixel_) /
                                                             parameters_.target_delta_slant_range_time)));

        std::copy(source_data_vector.begin() + (source_x_0 - source_offset),
                  source_data_vector.begin() + (source_x_0 - source_offset) + (last_x - first_x),
                  target_data_vector.begin() + (first_x - target_offset));
    }
}

void TopsarMergeOperator::ComputeMultipleSubSwathFloat(Rectangle target_rectangle, int first_sub_swath_index,
                                                       int last_sub_swath_index,
                                                       const std::vector<Rectangle>& source_rectangles,
                                                       std::string_view target_band_name,
                                                       snapengine::ITile* target_tile) {
    const auto number_of_source_tiles = last_sub_swath_index - first_sub_swath_index + 1;
    const auto target_index = std::make_shared<snapengine::TileIndex>(target_tile);
    std::vector<std::shared_ptr<snapengine::ITile>> source_tiles;

    std::vector<const std::vector<float>*> source_data_vector(number_of_source_tiles);
    auto& target_data_vector = target_tile->GetSimpleDataBuffer();

    int k = 0;
    for (int i = first_sub_swath_index; i <= last_sub_swath_index; ++i) {
        const auto swath_index = GetSubSwathIndexFromName(sub_swath_info_.at(i)->subswath_name_);
        const auto swath_index_string = std::to_string(swath_index);
        const auto source_band =
            GetSourceBandFromTargetBandName(target_band_name, parameters_.acquisition_mode, swath_index_string);
        const auto source_raster =
            GetSourceTile(source_band, source_rectangles.at(k), 1,
                          static_cast<int>(source_product_index_to_sub_swath_index_map_.right.at(swath_index)));
        source_tiles.emplace_back(source_raster);
        source_data_vector.at(k) = &source_raster->GetSimpleDataBuffer();
        k++;
    }

    for (int y = target_rectangle.y; y < target_rectangle.y + target_rectangle.height; ++y) {
        const auto target_offset = target_index->CalculateStride(y);

        for (int x = target_rectangle.x; x < target_rectangle.x + target_rectangle.width; ++x) {
            const auto sub_swath_index =
                GetSubSwathIndex(x, y, first_sub_swath_index, last_sub_swath_index, parameters_, sub_swath_merge_info_);
            if (sub_swath_index == -1) {
                continue;
            }
            const auto source_y =
                GetLineIndexInSourceProduct(y, sub_swath_merge_info_.at(sub_swath_index), parameters_);
            const auto source_x = GetSampleIndexInSourceProduct(
                x, sub_swath_info_.at(sub_swath_index)->num_of_samples_,
                sub_swath_info_.at(sub_swath_index)->slr_time_to_first_pixel_, parameters_);
            float val{0};
            k = sub_swath_index - first_sub_swath_index;

            if (const auto index = source_tiles.at(k)->GetDataBufferIndex(source_x, source_y); index >= 0) {
                val = source_data_vector.at(k)->at(index);
            }

            target_data_vector.at(x - target_offset) = val;
        }
    }
}
std::shared_ptr<snapengine::Band> TopsarMergeOperator::GetSourceBandFromTargetBandName(
    std::string_view target_band_name, std::string_view acquisition_mode, std::string_view swath_index_string) {
    for (size_t s = 0; s < parameters_.number_of_subswaths; s++) {
        const auto source_band_names = source_products_.at(s)->GetBandNames();
        for (const auto& source_band_name : source_band_names) {
            if (utils::general::DoesStringContain(source_band_name,
                                                  std::string(acquisition_mode) + std::string(swath_index_string)) &&
                GetTargetBandNameFromSourceBandName(source_band_name, parameters_.acquisition_mode) ==
                    target_band_name) {
                return source_products_.at(s)->GetBand(source_band_name);
            }
        }
    }
    return {};
}
std::shared_ptr<snapengine::ITile> TopsarMergeOperator::GetSourceTile(
    std::shared_ptr<snapengine::RasterDataNode> raster_data_node, Rectangle region, int band_index,
    int source_product_index) {
    auto tile = std::make_shared<snapengine::TileImpl>(raster_data_node, region);
    std::vector<float>& buffer = tile->GetSimpleDataBuffer();
    source_products_.at(source_product_index)->GetImageReader()->ReadSubSampledData(region, band_index);
    buffer = source_products_.at(source_product_index)->GetImageReader()->GetData();

    return tile;
}

TopsarMergeOperator::TopsarMergeOperator(std::vector<std::shared_ptr<snapengine::Product>> source_products,
                                         std::vector<std::string> selected_polarisations, int tile_width,
                                         int tile_height, std::string_view output_path)
    : tile_width_(tile_width),
      tile_height_(tile_height),
      source_products_(std::move(source_products)),
      selected_polarisations_(std::move(selected_polarisations)),
      output_path_(output_path) {
    Initialise();
}
void TopsarMergeOperator::PrepareBeamDimapSourceProducts() const {
    for (auto source_product : source_products_) {
        if (source_product->HasMetaDataReader()) {
            source_product->GetMetadataRoot()->AddElement(
                source_product->GetMetadataReader()->Read(snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT));
            source_product->GetMetadataRoot()->AddElement(
                source_product->GetMetadataReader()->Read(snapengine::AbstractMetadata::ORIGINAL_PRODUCT_METADATA));
            auto bands_info =
                std::dynamic_pointer_cast<snapengine::PugixmlMetaDataReader>(source_product->GetMetadataReader())
                    ->ReadImageInterpretationTag();
            for (const auto& info : bands_info) {
                auto band = std::make_shared<snapengine::Band>(info.band_name, info.product_data_type,
                                                               source_product->GetSceneRasterWidth(),
                                                               source_product->GetSceneRasterHeight());
                band->SetSpectralWavelength(static_cast<float>(info.band_wavelength.value()));
                source_product->AddBand(band);
            }
        }
    }
}
void TopsarMergeOperator::ComputeTileInOneSwathShort([[maybe_unused]] Rectangle target_rectangle,
                                                     [[maybe_unused]] int first_sub_swath_index,
                                                     [[maybe_unused]] Rectangle source_rectangle,
                                                     [[maybe_unused]] std::string_view target_band_name,
                                                     [[maybe_unused]] const snapengine::ITile* target_tile) const {
    THROW_ALGORITHM_EXCEPTION(ALGORITHM_NAME, "Short data type support is not yet implemented.");
}
std::shared_ptr<snapengine::Product> TopsarMergeOperator::GetTargetProduct() const { return target_product_; }

void TopsarMergeOperator::PopulateSourceBandNameVectors(int product_index) const {
    auto contains_selected_polarisations = [this](std::string_view band_name) {
        return std::any_of(
            selected_polarisations_.begin(), selected_polarisations_.end(),
            [&band_name](auto& polarisation) { return band_name.find(polarisation) != std::string::npos; });
    };

    for (const auto& source_band : source_products_.at(product_index)->GetBands()) {
        const auto source_band_name = source_band->GetName();

        if (!contains_selected_polarisations(source_band_name)) {
            continue;
        }

        const auto target_band_name =
            GetTargetBandNameFromSourceBandName(source_band_name, parameters_.acquisition_mode);
        if (!target_product_->ContainsBand(target_band_name)) {
            const auto target_band = target_product_->AddBand(target_band_name, source_band->GetDataType());
            target_band->SetUnit(source_band->GetUnit());
            if (source_band->IsNoDataValueSet()) {
                target_band->SetNoDataValueUsed(true);
                target_band->SetNoDataValue(source_band->GetNoDataValue());
            }
        }
    }
}
}  // namespace alus::topsarmerge