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
#include "terrain_correction_metadata.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>

#include "dataset.h"
#include "pugixml_meta_data_reader.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/tie_point_grid.h"
#include "snap-engine-utilities/engine-utilities//datamodel/metadata/abstract_metadata.h"

namespace alus::terraincorrection {

Metadata::Metadata(std::string_view dim_metadata_file, std::string_view lat_tie_points_file,
                   std::string_view lon_tie_points_file) {
    FillDimMetadata(dim_metadata_file);
    FetchTiePointGrids(dim_metadata_file, lat_tie_point_grid_, lon_tie_point_grid_);
    snapengine::PugixmlMetaDataReader xml_reader{dim_metadata_file};
    auto tie_point_grids = xml_reader.ReadTiePointGridsTag();

    lat_tie_point_grid_ = std::move(tie_point_grids.at(std::string(LATITUDE_TIE_POINT_GRID)));
    lon_tie_point_grid_ = std::move(tie_point_grids.at(std::string(LONGITUDE_TIE_POINT_GRID)));

    FetchTiePoints(lat_tie_points_file, lat_tie_point_grid_, lat_tie_points_buffer_);
    FetchTiePoints(lon_tie_points_file, lon_tie_point_grid_, lon_tie_points_buffer_);
}

Metadata::Metadata(std::shared_ptr<snapengine::Product> product) {
    metadata_fields_.product = product->GetProductType();
    if (metadata_fields_.product == "SLC") {
        metadata_fields_.product_type = metadata::ProductType::SLC;
    } else if (metadata_fields_.product == "GRD") {
        metadata_fields_.product_type = metadata::ProductType::GRD;
    } else {
        throw std::invalid_argument("Expected 'GRD' or 'SLC' product type - '" + metadata_fields_.product +
                                    "' not supported. Please check the input dataset.");
    }

    auto root = snapengine::AbstractMetadata::GetAbstractedMetadata(product);
    FillMetadataFrom(root);
    if (!product->ContainsTiePointGrid(LATITUDE_TIE_POINT_GRID) ||
        !product->ContainsTiePointGrid(LONGITUDE_TIE_POINT_GRID)) {
        throw std::runtime_error("Terrain correction metadata requires tie point grid information from the product.");
    }
    lat_tie_point_grid_ = product->GetTiePointGrid(LATITUDE_TIE_POINT_GRID);
    lon_tie_point_grid_ = product->GetTiePointGrid(LONGITUDE_TIE_POINT_GRID);

    const int n_bands = product->GetNumBands();
    for (int i = 0; i < n_bands; i++) {
        auto band = product->GetBandAt(i);
        snapengine::SpectralBandInfo band_info = {};
        band_info.band_index = i;
        band_info.band_name = band->GetName();
        band_info.product_data_type = band->GetDataType();
        band_info.log_10_scaled = band->IsLog10Scaled();
        band_info.no_data_value_used = band->IsNoDataValueUsed();
        band_info.no_data_value = band->GetNoDataValue();
        // todo add optional values?

        metadata_fields_.band_info.push_back(std::move(band_info));
    }

    if (metadata_fields_.product_type == metadata::ProductType::GRD) {
        metadata_fields_.srgr_coefficients = ParseSrgrCoefficients(root);
    }
}

void Metadata::FetchTiePointGrids(std::string_view dim_metadata_file,
                                  std::shared_ptr<snapengine::TiePointGrid>& lat_tie_point_grid,
                                  std::shared_ptr<snapengine::TiePointGrid>& lon_tie_point_grid) {
    snapengine::PugixmlMetaDataReader xml_reader{dim_metadata_file};
    auto tie_point_grids = xml_reader.ReadTiePointGridsTag();

    lat_tie_point_grid = std::move(tie_point_grids.at(std::string(LATITUDE_TIE_POINT_GRID)));
    lon_tie_point_grid = std::move(tie_point_grids.at(std::string(LONGITUDE_TIE_POINT_GRID)));
}

void Metadata::FetchTiePoints(std::string_view tie_points_file, std::shared_ptr<snapengine::TiePointGrid>& tie_points,
                              std::vector<float>& tie_points_buffer) {
    Dataset<float> ds(tie_points_file);  // *.img is of type float
    ds.LoadRasterBand(1);
    const bool width_ok = ds.GetRasterSizeX() == tie_points->GetGridWidth();
    const bool height_ok = ds.GetRasterSizeY() == tie_points->GetGridHeight();
    if (!width_ok || !height_ok) {
        throw std::runtime_error(std::string(tie_points_file) + " dimensions mismatch!\n");
    }
    tie_points_buffer = std::vector<float>(ds.GetHostDataBuffer());
    tie_points = std::make_shared<snapengine::TiePointGrid>(
        tie_points->GetName(), tie_points->GetGridWidth(), tie_points->GetGridHeight(), tie_points->GetOffsetX(),
        tie_points->GetOffsetY(), tie_points->GetSubSamplingX(), tie_points->GetSubSamplingY(), tie_points_buffer);
}

void Metadata::FillDimMetadata(std::string_view dim_metadata_file) {
    alus::snapengine::PugixmlMetaDataReader xml_reader{dim_metadata_file};
    auto master_root = xml_reader.Read(alus::snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);
    FillMetadataFrom(master_root);
    metadata_fields_.band_info = xml_reader.ReadImageInterpretationTag();
}

void Metadata::FillMetadataFrom(std::shared_ptr<snapengine::MetadataElement> master_root) {
    metadata_fields_.mission = master_root->GetAttributeString(snapengine::AbstractMetadata::MISSION);
    metadata_fields_.radar_frequency = master_root->GetAttributeDouble(snapengine::AbstractMetadata::RADAR_FREQUENCY);
    metadata_fields_.range_spacing =
        snapengine::AbstractMetadata::GetAttributeDouble(master_root, snapengine::AbstractMetadata::RANGE_SPACING);
    if (metadata_fields_.range_spacing <= 0.0) {
        throw std::runtime_error("Invalid input for range pixel spacing: " +
                                 std::to_string(metadata_fields_.range_spacing));
    }
    metadata_fields_.first_line_time = snapengine::AbstractMetadata::ParseUtc(
        master_root->GetAttributeString(snapengine::AbstractMetadata::FIRST_LINE_TIME));

    metadata_fields_.last_line_time = snapengine::AbstractMetadata::ParseUtc(
        master_root->GetAttributeString(snapengine::AbstractMetadata::LAST_LINE_TIME));

    metadata_fields_.line_time_interval =
        master_root->GetAttributeDouble(snapengine::AbstractMetadata::LINE_TIME_INTERVAL);
    if (metadata_fields_.line_time_interval <= 0.0) {
        throw std::runtime_error("Invalid input for Line Time Interval: " +
                                 std::to_string(metadata_fields_.line_time_interval));
    }

    metadata_fields_.orbit_state_vectors2 = snapengine::AbstractMetadata::GetOrbitStateVectors(master_root);
    if (metadata_fields_.orbit_state_vectors2.empty()) {
        throw std::runtime_error("Invalid Orbit State Vectors");
    }

    metadata_fields_.slant_range_to_first_pixel = snapengine::AbstractMetadata::GetAttributeDouble(
        master_root, snapengine::AbstractMetadata::SLANT_RANGE_TO_FIRST_PIXEL);

    metadata_fields_.avg_scene_height =
        snapengine::AbstractMetadata::GetAttributeDouble(master_root, snapengine::AbstractMetadata::AVG_SCENE_HEIGHT);

    metadata_fields_.first_near_lat = master_root->GetAttributeDouble(snapengine::AbstractMetadata::FIRST_NEAR_LAT);
    metadata_fields_.first_near_long = master_root->GetAttributeDouble(snapengine::AbstractMetadata::FIRST_NEAR_LONG);
    metadata_fields_.first_far_lat = master_root->GetAttributeDouble(snapengine::AbstractMetadata::FIRST_FAR_LAT);
    metadata_fields_.first_far_long = master_root->GetAttributeDouble(snapengine::AbstractMetadata::FIRST_FAR_LONG);
    metadata_fields_.last_near_lat = master_root->GetAttributeDouble(snapengine::AbstractMetadata::LAST_NEAR_LAT);
    metadata_fields_.last_near_long = master_root->GetAttributeDouble(snapengine::AbstractMetadata::LAST_NEAR_LONG);
    metadata_fields_.last_far_lat = master_root->GetAttributeDouble(snapengine::AbstractMetadata::LAST_FAR_LAT);
    metadata_fields_.last_far_long = master_root->GetAttributeDouble(snapengine::AbstractMetadata::LAST_FAR_LONG);

    metadata_fields_.azimuth_spacing = master_root->GetAttributeDouble(snapengine::AbstractMetadata::AZIMUTH_SPACING);
}

std::vector<SrgrCoefficients> Metadata::ParseSrgrCoefficients(std::shared_ptr<snapengine::MetadataElement> root) {
    auto elem_root = root->GetElement(snapengine::AbstractMetadata::SRGR_COEFFICIENTS);
    if (elem_root == nullptr) {
        throw std::runtime_error("The element '" + std::string(snapengine::AbstractMetadata::SRGR_COEFFICIENTS) +
                                 "' is missing from metadata. Needed for GRD dataset terrain correction.");
    }

    auto srgr_coef_list_elements = elem_root->GetElements();
    const auto srgr_coef_list_length = srgr_coef_list_elements.size();
    if (srgr_coef_list_length == 0) {
        throw std::runtime_error("The metadata section '" +
                                 std::string(snapengine::AbstractMetadata::SRGR_COEFFICIENTS) +
                                 "' does not consist of any elements. Needed for GRD dataset terrain correction.");
    }

    std::vector<SrgrCoefficients> srgr_entries(srgr_coef_list_length);
    for (size_t list_index{0}; list_index < srgr_coef_list_length; list_index++) {
        const auto& input_element = srgr_coef_list_elements.at(list_index);
        auto& output_entry = srgr_entries.at(list_index);
        output_entry.time_mjd = input_element->GetAttributeUtc(snapengine::AbstractMetadata::SRGR_COEF_TIME)->GetMjd();
        output_entry.ground_range_origin =
            input_element->GetAttributeDouble(snapengine::AbstractMetadata::GROUND_RANGE_ORIGIN);
        const auto coef_value_count = input_element->GetNumElements();
        if (coef_value_count == 0) {
            throw std::runtime_error("The metadata section '" +
                                     std::string(snapengine::AbstractMetadata::SRGR_COEFFICIENTS) + "' element no " +
                                     std::to_string(list_index + 1) + " does not have any coefficient values.");
        }
        output_entry.coefficient.resize(coef_value_count);
        for (int i{0}; i < coef_value_count; i++) {
            const auto& elem = input_element->GetElementAt(i);
            if (elem == nullptr) {
                throw std::runtime_error(
                    "Implementation error detected while parsing elements of the metadata section '" +
                    std::string(snapengine::AbstractMetadata::SRGR_COEFFICIENTS) + "' - there is reported " +
                    std::to_string(coef_value_count) + " coefficient elements present, but querying element no. " +
                    std::to_string(i) + " resulted in 'nullptr'");
            }
            output_entry.coefficient.at(i) = elem->GetAttributeDouble(snapengine::AbstractMetadata::SRGR_COEF, 0.0);
        }
    }

    return srgr_entries;
}

}  // namespace alus::terraincorrection
