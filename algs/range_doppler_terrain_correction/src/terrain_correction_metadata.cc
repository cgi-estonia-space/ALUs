#include "terrain_correction_metadata.h"

#include <algorithm>

#include "dataset.h"
#include "pugixml_meta_data_reader.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"

namespace alus::terraincorrection {

Metadata::Metadata(std::string_view dim_metadata_file,
                   std::string_view lat_tie_points_file,
                   std::string_view lon_tie_points_file) {
    FillDimMetadata(dim_metadata_file);
    FetchTiePoints(lat_tie_points_file, lat_tie_points_);
    FetchTiePoints(lon_tie_points_file, lon_tie_points_);
}

void Metadata::FetchTiePoints(std::string_view tie_points_file, TiePoints& tie_points) {
    Dataset<double> ds(tie_points_file);
    ds.LoadRasterBand(1);
    tie_points.grid_width = ds.GetRasterSizeX();
    tie_points.grid_height = ds.GetRasterSizeY();
    const auto& db = ds.GetHostDataBuffer();
    std::transform(db.cbegin(), db.cend(), std::back_inserter(tie_points.values), [](double v) -> float {
        return static_cast<float>(v);
    });
}

void Metadata::FillDimMetadata(std::string_view dim_metadata_file) {
    alus::snapengine::PugixmlMetaDataReader xml_reader{dim_metadata_file};
    auto master_root = xml_reader.Read(alus::snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);
    metadata_fields_.mission = master_root->GetAttributeString(snapengine::AbstractMetadata::MISSION);
    ;
    metadata_fields_.radar_frequency = master_root->GetAttributeDouble(snapengine::AbstractMetadata::RADAR_FREQUENCY);
    metadata_fields_.range_spacing =
        snapengine::AbstractMetadata::GetAttributeDouble(master_root, snapengine::AbstractMetadata::RANGE_SPACING);
    if (metadata_fields_.range_spacing <= 0.0) {
        throw std::runtime_error("Invalid input for range pixel spacing: " +
                                 std::to_string(metadata_fields_.range_spacing));
    }
    metadata_fields_.first_line_time = *snapengine::AbstractMetadata::ParseUtc(
        master_root->GetAttributeString(snapengine::AbstractMetadata::FIRST_LINE_TIME));

    metadata_fields_.last_line_time = *snapengine::AbstractMetadata::ParseUtc(
        master_root->GetAttributeString(snapengine::AbstractMetadata::LAST_LINE_TIME));

    metadata_fields_.line_time_interval =
        master_root->GetAttributeDouble(snapengine::AbstractMetadata::LINE_TIME_INTERVAL);
    if (metadata_fields_.line_time_interval <= 0.0) {
        throw std::runtime_error("Invalid input for Line Time Interval: " +
                                 std::to_string(metadata_fields_.line_time_interval));
    }

    metadata_fields_.orbit_state_vectors2 = snapengine::AbstractMetadata::GetOrbitStateVectors(master_root);
    if (metadata_fields_.orbit_state_vectors2.empty()) {
        throw std::runtime_error("Invalid Obit State Vectors");
    }

    metadata_fields_.slant_range_to_first_pixel = snapengine::AbstractMetadata::GetAttributeDouble(
        master_root, snapengine::AbstractMetadata::SLANT_RANGE_TO_FIRST_PIXEL);

    metadata_fields_.avg_scene_height =
        snapengine::AbstractMetadata::GetAttributeDouble(master_root, snapengine::AbstractMetadata::AVG_SCENE_HEIGHT);

    metadata_fields_.first_valid_pixel = master_root->GetAttributeInt(snapengine::AbstractMetadata::FIRST_VALID_PIXEL);
    metadata_fields_.last_valid_pixel = master_root->GetAttributeInt(snapengine::AbstractMetadata::LAST_VALID_PIXEL);

    metadata_fields_.first_near_lat = master_root->GetAttributeDouble(snapengine::AbstractMetadata::FIRST_NEAR_LAT);
    metadata_fields_.first_near_long = master_root->GetAttributeDouble(snapengine::AbstractMetadata::FIRST_NEAR_LONG);
    metadata_fields_.first_far_lat = master_root->GetAttributeDouble(snapengine::AbstractMetadata::FIRST_FAR_LAT);
    metadata_fields_.first_far_long = master_root->GetAttributeDouble(snapengine::AbstractMetadata::FIRST_FAR_LONG);
    metadata_fields_.last_near_lat = master_root->GetAttributeDouble(snapengine::AbstractMetadata::LAST_NEAR_LAT);
    metadata_fields_.last_near_long = master_root->GetAttributeDouble(snapengine::AbstractMetadata::LAST_NEAR_LONG);
    metadata_fields_.last_far_lat = master_root->GetAttributeDouble(snapengine::AbstractMetadata::LAST_FAR_LAT);
    metadata_fields_.last_far_long = master_root->GetAttributeDouble(snapengine::AbstractMetadata::LAST_FAR_LONG);
    metadata_fields_.first_valid_line_time =
        master_root->GetAttributeDouble(snapengine::AbstractMetadata::FIRST_VALID_LINE_TIME);
    metadata_fields_.last_valid_line_time =
        master_root->GetAttributeDouble(snapengine::AbstractMetadata::LAST_VALID_LINE_TIME);

    metadata_fields_.azimuth_spacing = master_root->GetAttributeDouble(snapengine::AbstractMetadata::AZIMUTH_SPACING);

    metadata_fields_.band_info = xml_reader.ReadImageInterpretationTag();
}

}  // namespace alus::terraincorrection