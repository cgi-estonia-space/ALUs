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

#include "projection.h"

#include <array>
#include <stdexcept>

#include <ogr_spatialref.h>

#include "algorithm_exception.h"
#include "alus_log.h"
#include "gdal_util.h"
#include "transform_constants.h"

namespace {
void TryAssignSrs(OGRSpatialReference* srs, std::string_view projection) {
    const auto srs_res = srs->SetFromUserInput(projection.data());
    if (srs_res != OGRERR_NONE) {
        throw std::invalid_argument(
            std::string(projection) +
            " is not supported, see SetFromUserInput() documentation from GDAL library. OGR error - " +
            std::to_string(srs_res));
    }
}

OGRCoordinateTransformation* TryCreateCoordinateTransform(const OGRSpatialReference* source,
                                                          const OGRSpatialReference* target) {
    auto* coord_trans = OGRCreateCoordinateTransformation(source, target);
    CHECK_GDAL_PTR(coord_trans);
    if (source->GetAxisMappingStrategy() != target->GetAxisMappingStrategy()) {
        LOGI << "Beware - axis mapping strategies do not align between input and specified projection (e.g. LON and "
                "LAT placement sequence)";
    }

    return coord_trans;
}

void TryTransformingCoordinates(OGRCoordinateTransformation* transformer, GDALDataset* from, GDALDataset* to) {
    std::array<double, alus::transform::GEOTRANSFORM_ARRAY_LENGTH> gt;
    CHECK_GDAL_ERROR(from->GetGeoTransform(gt.data()));

    double* x = &gt[alus::transform::TRANSFORM_LON_ORIGIN_INDEX];
    double* y = &gt[alus::transform::TRANSFORM_LAT_ORIGIN_INDEX];
    if (transformer->Transform(1, x, y) != 1) {
        THROW_ALGORITHM_EXCEPTION(
            APP_NAME, "Transforming coordinates failed - no more details provided by underlying(GDAL) library.");
    }

    CHECK_GDAL_ERROR(to->SetGeoTransform(gt.data()));
}

void TryTransformingCoordinates(OGRCoordinateTransformation* transformer, double* convert_gt) {
    double* x = &convert_gt[alus::transform::TRANSFORM_LON_ORIGIN_INDEX];
    double* y = &convert_gt[alus::transform::TRANSFORM_LAT_ORIGIN_INDEX];
    if (transformer->Transform(1, x, y) != 1) {
        THROW_ALGORITHM_EXCEPTION(
            APP_NAME, "Transforming coordinates failed - no more details provided by underlying(GDAL) library.");
    }
}

void TryCalculatingPixelSize(GDALDataset* from, GDALDataset* to, double longitude_factor, double latitude_factor) {
    std::array<double, alus::transform::GEOTRANSFORM_ARRAY_LENGTH> gt;
    CHECK_GDAL_ERROR(from->GetGeoTransform(gt.data()));
    if (!std::isnan(longitude_factor)) {
        gt.at(alus::transform::TRANSFORM_PIXEL_X_SIZE_INDEX) =
            gt.at(alus::transform::TRANSFORM_PIXEL_X_SIZE_INDEX) * longitude_factor;
    }

    if (!std::isnan(latitude_factor)) {
        gt.at(alus::transform::TRANSFORM_PIXEL_Y_SIZE_INDEX) =
            gt.at(alus::transform::TRANSFORM_PIXEL_Y_SIZE_INDEX) * latitude_factor;
    }

    CHECK_GDAL_ERROR(to->SetGeoTransform(gt.data()));
}

}  // namespace

namespace alus::resample {
void Reprojection(GDALDataset* from, GDALDataset* reprojected, std::string_view projection, double longitude_factor,
                  double latitude_factor) {
    const auto* in_srs = from->GetGCPSpatialRef();
    CHECK_GDAL_PTR(in_srs);
    OGRSpatialReference out_srs;
    TryAssignSrs(&out_srs, projection);
    CHECK_GDAL_ERROR(reprojected->SetSpatialRef(&out_srs));
    auto* transformer = TryCreateCoordinateTransform(in_srs, &out_srs);
    TryTransformingCoordinates(transformer, from, reprojected);
    TryCalculatingPixelSize(from, reprojected, longitude_factor, latitude_factor);
}

void Reprojection(const OGRSpatialReference* source, OGRSpatialReference* dest_srs, double* dest_gt,
                  std::string_view projection) {
    TryAssignSrs(dest_srs, projection);
    auto* transformer = TryCreateCoordinateTransform(source, dest_srs);
    TryTransformingCoordinates(transformer, dest_gt);
}

}  // namespace alus::resample