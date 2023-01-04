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

#include "result_export.h"

#include <sstream>
#include <string_view>

#include <gdal_priv.h>
#include <ogr_geometry.h>

#include "algorithm_exception.h"
#include "constants.h"
#include "gdal_management.h"
#include "gdal_util.h"

namespace {
constexpr std::string_view ATTRIBUTE_EXTRACTED_FEATURE_NAME{"extractedFeature"};
constexpr std::string_view ATTRIBUTE_FEATURE_VECTOR_NAME{"featureVector"};
constexpr std::string_view ATTRIBUTE_INPUT_PARAMETERS_NAME{"inputParameters"};
constexpr std::string_view ATTRIBUTE_CLASSIFIER_NAME{"classifier"};
constexpr std::string_view ATTRIBUTE_CLASS_NAME{"class"};
}  // namespace

namespace alus::featurextractiongabor {

ResultExport::ResultExport(std::string_view filename, OGRSpatialReference* srs,
                           GeoTransformParameters patches_corner_gt, size_t orientation_count, size_t frequency_count)
    : patches_corner_gt_{patches_corner_gt},
      input_parameters_attribute_value_("Or=" + std::to_string(orientation_count) +
                                        ";Sc=" + std::to_string(frequency_count)) {
    auto* driver = GetGdalSqliteDriver();
    CHECK_GDAL_PTR(driver);
    std::string filename_w_ext = std::string(filename) + ".sqlite";
    CPLSetConfigOption("SQLITE_USE_OGR_VFS", "YES");
    CPLSetConfigOption("OGR_SQLITE_SYNCHRONOUS", "OFF");
    CPLSetConfigOption("OGR_SQLITE_CACHE", "512MB");
    char** ds_options{};
    ds_options = CSLSetNameValue(ds_options, "SPATIALITE", "YES");
    feature_ds_ = driver->Create(filename_w_ext.c_str(), 0, 0, 0, GDT_Unknown, ds_options);
    CSLDestroy(ds_options);
    CHECK_GDAL_PTR(feature_ds_);
    char** layer_options{};
    layer_options = CSLSetNameValue(layer_options, "FORMAT", "SPATIALITE");
    layer_options = CSLSetNameValue(layer_options, "OVERWRITE", "YES");
    layer_ = feature_ds_->CreateLayer("patches", srs, wkbPolygon, layer_options);
    CSLDestroy(layer_options);
    CHECK_GDAL_PTR(layer_);
    SetupAttributeTable();
}

void ResultExport::SetupAttributeTable() {
    auto hidden_gdal_error_callback = [](std::string_view err) {
        THROW_ALGORITHM_EXCEPTION(
            ALG_NAME,
            "GDAL error occurred, this might disrupt intended shapefile construction. Error - " + std::string(err));
    };
    const auto gdal_error_cb_guard = gdalmanagement::SetErrorHandle(hidden_gdal_error_callback);

    OGRFieldDefn extracted_feature_field(ATTRIBUTE_EXTRACTED_FEATURE_NAME.data(), OFTString);
    OGRFieldDefn input_parameters_field(ATTRIBUTE_INPUT_PARAMETERS_NAME.data(), OFTString);
    OGRFieldDefn feature_vectors_field(ATTRIBUTE_FEATURE_VECTOR_NAME.data(), OFTString);
    OGRFieldDefn classifier_field(ATTRIBUTE_CLASSIFIER_NAME.data(), OFTString);
    OGRFieldDefn class_field(ATTRIBUTE_CLASS_NAME.data(), OFTString);
    CHECK_OGR_ERROR(layer_->CreateField(&extracted_feature_field));
    CHECK_OGR_ERROR(layer_->CreateField(&input_parameters_field));
    CHECK_OGR_ERROR(layer_->CreateField(&feature_vectors_field));
    CHECK_OGR_ERROR(layer_->CreateField(&classifier_field));
    CHECK_OGR_ERROR(layer_->CreateField(&class_field));
}

void ResultExport::Add(const PatchResult& result) {
    features_x_y_[std::make_pair(result.x, result.y)].emplace_back(result.mean, result.std_dev);
}

void ResultExport::StoreFeatures() {
    const auto origin_lon = patches_corner_gt_.originLon;
    const auto pixel_size_lon = patches_corner_gt_.pixelSizeLon;
    const auto origin_lat = patches_corner_gt_.originLat;
    const auto pixel_size_lat = patches_corner_gt_.pixelSizeLat;

    // Do not commit on every feature added. This renders the process VERY slow.
    CHECK_OGR_ERROR(OGR_L_StartTransaction(layer_));

    for (const auto& feat_res : features_x_y_) {
        OGRFeature* feature = OGRFeature::CreateFeature(layer_->GetLayerDefn());
        CHECK_GDAL_PTR(feature);
        feature->SetField(ATTRIBUTE_EXTRACTED_FEATURE_NAME.data(), "GaborFeatures");
        feature->SetField(ATTRIBUTE_INPUT_PARAMETERS_NAME.data(), input_parameters_attribute_value_.data());

        const auto x = feat_res.first.first;   // 0 index based.
        const auto y = feat_res.first.second;  // 0 index based.
        OGRLinearRing boundaries;
        boundaries.addPoint(origin_lon + x * pixel_size_lon, origin_lat + y * pixel_size_lat);
        boundaries.addPoint(origin_lon + static_cast<double>(x + 1) * pixel_size_lon, origin_lat + y * pixel_size_lat);
        boundaries.addPoint(origin_lon + static_cast<double>(x + 1) * pixel_size_lon,
                            origin_lat + static_cast<double>(y + 1) * pixel_size_lat);
        boundaries.addPoint(origin_lon + x * pixel_size_lon, origin_lat + static_cast<double>(y + 1) * pixel_size_lat);
        boundaries.addPoint(origin_lon + x * pixel_size_lon, origin_lat + y * pixel_size_lat);
        OGRPolygon boundary;
        CHECK_OGR_ERROR(boundary.addRing(&boundaries));
        CHECK_OGR_ERROR(feature->SetGeometry(&boundary));

        std::stringstream feature_vector_attribute_value;
        for (const auto& res : feat_res.second) {
            feature_vector_attribute_value << res.first << "," << res.second << ", ";
        }
        auto feature_vector_attribute_value_str = feature_vector_attribute_value.str();
        feature_vector_attribute_value_str.pop_back();
        feature_vector_attribute_value_str.pop_back();
        feature->SetField(ATTRIBUTE_FEATURE_VECTOR_NAME.data(), feature_vector_attribute_value_str.c_str());

        CHECK_OGR_ERROR(layer_->CreateFeature(feature));
        OGRFeature::DestroyFeature(feature);
    }

    CHECK_OGR_ERROR(OGR_L_CommitTransaction(layer_));
}

ResultExport::~ResultExport() { GDALClose(feature_ds_); }
}  // namespace alus::featurextractiongabor