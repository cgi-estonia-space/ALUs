/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.TiePointGeoCoding.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
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
#include "snap-core/datamodel/tie_point_geo_coding.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "alus_log.h"
#include "snap-core/dataio/product_subset_def.h"
#include "snap-core/datamodel/i_scene.h"
#include "snap-core/datamodel/tie_point_grid.h"
#include "snap-core/util/guardian.h"
#include "snap-core/util/math/f_x_y_sum.h"
#include "snap-core/util/math/f_x_y_sum_bi_cubic.h"
#include "snap-core/util/math/f_x_y_sum_bi_linear.h"
#include "snap-core/util/math/f_x_y_sum_bi_quadric.h"
#include "snap-core/util/math/f_x_y_sum_cubic.h"
#include "snap-core/util/math/f_x_y_sum_linear.h"
#include "snap-core/util/math/f_x_y_sum_quadric.h"
#include "snap-core/util/math/math_utils.h"
#include "custom/dimension.h"
#include "custom/rectangle.h"

namespace alus {
namespace snapengine {

// todo: provide actual CRS WGS84 when this gets used
TiePointGeoCoding::TiePointGeoCoding(const std::shared_ptr<TiePointGrid>& lat_grid,
                                     const std::shared_ptr<TiePointGrid>& lon_grid)
    : TiePointGeoCoding(lat_grid, lon_grid, "DefaultGeographicCRS.WGS84") {}

TiePointGeoCoding::TiePointGeoCoding(const std::shared_ptr<TiePointGrid>& lat_grid,
                                     const std::shared_ptr<TiePointGrid>& lon_grid,
                                     const CoordinateReferenceSystemWKT& geo_c_r_s)
    : AbstractGeoCoding(geo_c_r_s) {
    //    todo: find a solution for guardians
    //    Guardian::AssertNotNull("latGrid", lat_grid);
    //    Guardian::AssertNotNull("lonGrid", lon_grid);
    //    Guardian::AssertNotNull("geoCRS", geo_c_r_s);
    if (lat_grid->GetGridWidth() != lon_grid->GetGridWidth() ||
        lat_grid->GetGridHeight() != lon_grid->GetGridHeight() || lat_grid->GetOffsetX() != lon_grid->GetOffsetX() ||
        lat_grid->GetOffsetY() != lon_grid->GetOffsetY() ||
        lat_grid->GetSubSamplingX() != lon_grid->GetSubSamplingX() ||
        lat_grid->GetSubSamplingY() != lon_grid->GetSubSamplingY()) {
        throw std::invalid_argument("latGrid is not compatible with lonGrid");
    }
    lat_grid_ = lat_grid;
    lon_grid_ = lon_grid;
    //     datum_ = "Datum.WGS_84"; // todo delete me! I'm not used   (THIS IS FROM ESA SNAP, AND IT WAS NOT DELETED!)
    approximations_computed_ = false;
}

// TiePointGeoCoding::TiePointGeoCoding(const std::shared_ptr<TiePointGrid>& lat_grid,
//                                     const std::shared_ptr<TiePointGrid>& lon_grid,
//                                     const std::shared_ptr<Datum>& datum) {}

void TiePointGeoCoding::ComputeApproximations() {
    if (!approximations_computed_) {
        std::shared_ptr<TiePointGrid> normalized_lon_grid = InitNormalizedLonGrid();
        InitLatLonMinMax(normalized_lon_grid);
        approximations_ = InitApproximations(normalized_lon_grid);
        approximations_computed_ = true;
    }
}
bool TiePointGeoCoding::IsValidGeoPos(double lat, double lon) { return !std::isnan(lat) && !std::isnan(lon); }

std::shared_ptr<TiePointGrid> TiePointGeoCoding::InitNormalizedLonGrid() {
    const int w = lon_grid_->GetGridWidth();
    const int h = lon_grid_->GetGridHeight();

    double p1;
    double p2;
    double lon_delta;
    bool west_normalized = false;
    bool east_normalized = false;

    const std::vector<float> longitudes = lon_grid_->GetTiePoints();
    const int num_values = longitudes.size();
    std::vector<float> normalized_longitudes(num_values);
    //        System.arraycopy(longitudes, 0, normalized_longitudes, 0, num_values);
    std::copy(longitudes.begin(), longitudes.end(), normalized_longitudes.begin());
    double lon_delta_max = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {  // Normalise line-wise, by detecting longituidal discontinuities. lonDelta is
            // the difference between a base point and the current point
            const int index = x + y * w;
            if (x == 0 && y == 0) {  // first point in grid: base point is un-normalised
                p1 = normalized_longitudes.at(index);
            } else if (x == 0) {  // first point in line: base point is the (possibly) normalised lon. of first
                // point of last line
                p1 = normalized_longitudes.at(x + (y - 1) * w);
            } else {  // other points in line: base point is the (possibly) normalised lon. of last point in line
                p1 = normalized_longitudes.at(index - 1);
            }
            p2 = normalized_longitudes.at(index);  // the current, un-normalised point
            lon_delta = p2 - p1;                   // difference = current point minus base point

            if (lon_delta > 180.0) {
                p2 -= 360.0;             // place new point in the west (with a lon. < -180)
                west_normalized = true;  // mark what we've done
                normalized_longitudes.at(index) = static_cast<float>(p2);
            } else if (lon_delta < -180.0) {
                p2 += 360.0;             // place new point in the east (with a lon. > +180)
                east_normalized = true;  // mark what we've done
                normalized_longitudes.at(index) = static_cast<float>(p2);
            } else {
                lon_delta_max = std::max(lon_delta_max, std::abs(lon_delta));
            }
        }
    }

    // West-normalisation can result in longitudes down to -540 degrees
    if (west_normalized) {
        // This ensures that the all longitude points are >= -180 degree
        for (int i = 0; i < num_values; i++) {
            normalized_longitudes.at(i) += 360;
        }
    }

    normalized_ = west_normalized || east_normalized;

    std::shared_ptr<TiePointGrid> normalized_lon_grid;
    if (normalized_) {
        normalized_lon_grid = std::make_shared<TiePointGrid>(
            lon_grid_->GetName(), lon_grid_->GetGridWidth(), lon_grid_->GetGridHeight(), lon_grid_->GetOffsetX(),
            lon_grid_->GetOffsetY(), lon_grid_->GetSubSamplingX(), lon_grid_->GetSubSamplingY(), normalized_longitudes,
            lon_grid_->GetDiscontinuity());
    } else {
        normalized_lon_grid = lon_grid_;
    }

    LOGV << "TiePointGeoCoding.westNormalized = " << west_normalized;
    LOGV << "TiePointGeoCoding.eastNormalized = " << east_normalized;
    LOGV << "TiePointGeoCoding.normalized = " << normalized_;
    LOGV << "TiePointGeoCoding.lonDeltaMax = " << lon_delta_max;

    return normalized_lon_grid;
}
void TiePointGeoCoding::InitLatLonMinMax(const std::shared_ptr<TiePointGrid>& normalized_lon_grid) {
    const std::vector<float> lat_points = GetLatGrid()->GetTiePoints();
    const std::vector<float> lon_points = normalized_lon_grid->GetTiePoints();
    normalized_lon_min_ = +std::numeric_limits<double>::max();
    normalized_lon_max_ = -std::numeric_limits<double>::max();
    lat_min_ = +std::numeric_limits<double>::max();
    lat_max_ = -std::numeric_limits<double>::max();
    for (double lon_point : lon_points) {
        normalized_lon_min_ = std::min(normalized_lon_min_, lon_point);
        normalized_lon_max_ = std::max(normalized_lon_max_, lon_point);
    }
    for (double lat_point : lat_points) {
        lat_min_ = std::min(lat_min_, lat_point);
        lat_max_ = std::max(lat_max_, lat_point);
    }

    overlap_start_ = normalized_lon_min_;
    if (overlap_start_ < -180) {
        overlap_start_ += 360;
    }
    overlap_end_ = normalized_lon_max_;
    if (overlap_end_ > 180) {
        overlap_end_ -= 360;
    }

    LOGV << "TiePointGeoCoding.normalizedLonMin = " << normalized_lon_min_;
    LOGV << "TiePointGeoCoding.normalizedLonMax = " << normalized_lon_max_;
    LOGV << "TiePointGeoCoding.latMin = " << lat_min_;
    LOGV << "TiePointGeoCoding.latMax = " << lat_max_;
    LOGV << "TiePointGeoCoding.overlapRange = " << overlap_start_ << " - " << overlap_end_;
}

std::vector<std::shared_ptr<Approximation>> TiePointGeoCoding::InitApproximations(
    const std::shared_ptr<TiePointGrid>& normalized_lon_grid) {
    const int num_points = lat_grid_->GetGridData()->GetNumElems();
    const int w = lat_grid_->GetGridWidth();
    const int h = lat_grid_->GetGridHeight();
    const double sub_sampling_x = lat_grid_->GetSubSamplingX();
    const double sub_sampling_y = lat_grid_->GetSubSamplingY();

    // 10 points are at least required for a quadratic polynomial
    // start with some appropriate tile number
    int num_tiles = static_cast<int>(ceil(num_points / 10.0));
    num_tiles = std::min(std::max(1, num_tiles), 300);
    int num_tiles_i = 1;
    int num_tiles_j = 1;
    while (num_tiles > 1) {
        std::shared_ptr<custom::Dimension> tile_dim =
            MathUtils::FitDimension(num_tiles, w * sub_sampling_x, h * sub_sampling_y);
        int new_num_tiles_i = tile_dim->width;
        int new_num_tiles_j = tile_dim->height;
        int new_num_tiles = new_num_tiles_i * new_num_tiles_j;
        // 10 points are at least required for a quadratic polynomial
        if (num_points / new_num_tiles >= 10) {
            num_tiles = new_num_tiles;
            num_tiles_i = new_num_tiles_i;
            num_tiles_j = new_num_tiles_j;
            break;
        }
        num_tiles--;
    }

    LOGV << "TiePointGeoCoding.numTiles =  " << num_tiles;
    LOGV << "TiePointGeoCoding.numTilesI = " << num_tiles_i;
    LOGV << "TiePointGeoCoding.numTilesJ = " << num_tiles_j;

    // Compute actual approximations for all tiles
    std::vector<std::shared_ptr<Approximation>> approximations(num_tiles);
    std::vector<std::shared_ptr<custom::Rectangle>> rectangles =
        MathUtils::SubdivideRectangle(w, h, num_tiles_i, num_tiles_j, 1);
    for (std::size_t i = 0; i < rectangles.size(); i++) {
        std::shared_ptr<Approximation> approximation = CreateApproximation(normalized_lon_grid, rectangles.at(i));
        if (approximation == nullptr) {
            // did not spend a lot of time digging here, if this is wrong place to throw it will come up somewhere
            throw std::runtime_error("unable to create approximation");
        }
        approximations.at(i) = approximation;
    }
    return approximations;
}

std::shared_ptr<FXYSum> TiePointGeoCoding::GetBestPolynomial(std::vector<std::vector<double>> data,
                                                             std::vector<int> indices) {
    // These are the potential polynomials which we will check
    //        todo: check this over, not sure it works like that (some tests?)
    const std::vector<std::shared_ptr<FXYSum>> potential_polynomials{
        std::make_shared<Linear>(),
        std::make_shared<BiLinear>(),
        std::make_shared<Quadric>(),
        std::make_shared<BiQuadric>(),
        std::make_shared<Cubic>(),
        std::make_shared<BiCubic>(),
        std::make_shared<FXYSum>(FXYSum::FXY_4TH, 4),
        std::make_shared<FXYSum>(FXYSum::FXY_BI_4TH, 4 + 4)};

    // Find the polynomial which best fitts the warp points
    double rmse_min = std::numeric_limits<double>::max();
    int index = -1;
    for (std::size_t i = 0; i < potential_polynomials.size(); i++) {
        std::shared_ptr<FXYSum> potential_polynomial = potential_polynomials.at(i);
        const int order = potential_polynomial->GetOrder();
        std::size_t num_points_required;
        if (order >= 0) {
            num_points_required = (order + 2) * (order + 1) / 2;
        } else {
            num_points_required = 2 * potential_polynomial->GetNumTerms();
        }
        if (data.size() >= num_points_required) {
            try {
                potential_polynomial->Approximate(data, indices);
                double rmse = potential_polynomial->GetRootMeanSquareError();
                double max_error = potential_polynomial->GetMaxError();
                if (rmse < rmse_min) {
                    index = i;
                    rmse_min = rmse;
                }
                if (max_error < ABS_ERROR_LIMIT) {  // this accuracy is sufficient
                    index = i;
                    break;
                }
            } catch (const std::exception& e) {
                LOGW << "Polynomial cannot be constructed due to a numerically singular or degenerate matrix:"
                     << e.what();
            }
        }
    }
    return index >= 0 ? potential_polynomials.at(index) : nullptr;
}

std::vector<std::vector<double>> TiePointGeoCoding::CreateWarpPoints(
    const std::shared_ptr<TiePointGrid>& lon_grid, const std::shared_ptr<custom::Rectangle>& subset_rect) {
    const std::shared_ptr<TiePointGrid> lat_grid = GetLatGrid();
    const int w = lat_grid->GetGridWidth();
    const int sw = subset_rect->width;
    const int sh = subset_rect->height;
    const int i1 = subset_rect->x;
    const int i2 = i1 + sw - 1;
    const int j1 = subset_rect->y;
    const int j2 = j1 + sh - 1;

    LOGV << "Selecting warp points for X/Y approximations";
    LOGV << "  subset rectangle (in tie point coordinates): " << subset_rect;
    LOGV << "  index i: " << i1 << " to " << i2;
    LOGV << "  index j: " << j1 << " to " << j2;

    std::vector<int> warp_parameters = DetermineWarpParameters(sw, sh);
    int num_u = warp_parameters.at(0);
    int num_v = warp_parameters.at(1);
    int step_i = warp_parameters.at(2);
    int step_j = warp_parameters.at(3);

    // Collect numU * numV warp points
    const int m = num_u * num_v;
    std::vector<std::vector<double>> data(4, std::vector<double>(m));
    double lat;
    double lon;
    double x;
    double y;
    int i;
    int j;
    int k = 0;
    for (int v = 0; v < num_v; v++) {
        j = j1 + v * step_j;
        // Adjust bottom border
        if (j > j2) {
            j = j2;
        }
        for (int u = 0; u < num_u; u++) {
            i = i1 + u * step_i;
            // Adjust right border
            if (i > i2) {
                i = i2;
            }
            lat = lat_grid->GetGridData()->GetElemDoubleAt(j * w + i);
            lon = lon_grid->GetGridData()->GetElemDoubleAt(j * w + i);
            x = lat_grid->GetOffsetX() + i * lat_grid->GetSubSamplingX();
            y = lat_grid->GetOffsetY() + j * lat_grid->GetSubSamplingY();
            data.at(k).at(0) = lat;
            data.at(k).at(1) = lon;
            data.at(k).at(2) = x;
            data.at(k).at(3) = y;
            k++;
        }
    }

    //        todo: think this "assertrue" through (currently ignore)
    //        Debug::AssertTrue(k == m);
    LOGV << "TiePointGeoCoding: numU=" << num_u << ", stepI=" << step_i;
    LOGV << "TiePointGeoCoding: numV=" << num_v << ", stepJ=" << step_j;

    return data;
}

std::vector<int> TiePointGeoCoding::DetermineWarpParameters(int sw, int sh) {
    // Determine stepI and stepJ so that maximum number of warp points is not exceeded,
    // numU * numV shall be less than _MAX_NUM_POINTS_PER_TILE.
    //
    int num_u = sw;
    int num_v = sh;
    int step_i = 1;
    int step_j = 1;

    // Adjust number of hor/ver (numU,numV) tie-points to be considered
    // so that a maximum of circa numPointsMax points is not exceeded
    bool adjust_step_i = num_u >= num_v;
    while (num_u * num_v > MAX_NUM_POINTS_PER_TILE) {
        if (adjust_step_i) {
            step_i++;
            num_u = sw / step_i;
            while (num_u * step_i < sw) {
                num_u++;
            }
        } else {
            step_j++;
            num_v = sh / step_j;
            while (num_v * step_j < sh) {
                num_v++;
            }
        }
        adjust_step_i = num_u >= num_v;
    }
    return std::vector<int>{num_u, num_v, step_i, step_j};
}

std::shared_ptr<Approximation> TiePointGeoCoding::CreateApproximation(
    const std::shared_ptr<TiePointGrid>& normalized_lon_grid, const std::shared_ptr<custom::Rectangle>& subset_rect) {
    std::vector<std::vector<double>> data = CreateWarpPoints(normalized_lon_grid, subset_rect);

    double sum_lat = 0.0;
    double sum_lon = 0.0;
    for (const auto& point : data) {
        sum_lat += point.at(0);
        sum_lon += point.at(1);
    }
    double center_lon = sum_lon / data.size();
    double center_lat = sum_lat / data.size();
    const double max_square_distance = GetMaxSquareDistance(data, center_lat, center_lon);

    for (std::size_t i = 0; i < data.size(); i++) {
        data.at(i).at(0) = RescaleLatitude(data.at(i).at(0));
        data.at(i).at(1) = RescaleLongitude(data.at(i).at(1), center_lon);
    }

    std::vector<int> x_indices{0, 1, 2};
    std::vector<int> y_indices{0, 1, 3};

    const std::shared_ptr<FXYSum> f_x = GetBestPolynomial(data, x_indices);
    const std::shared_ptr<FXYSum> f_y = GetBestPolynomial(data, y_indices);
    if (f_x == nullptr || f_y == nullptr) {
        return nullptr;
    }

    const double rmse_x = f_x->GetRootMeanSquareError();
    const double rmse_y = f_y->GetRootMeanSquareError();

    const double max_error_x = f_x->GetMaxError();
    const double max_error_y = f_y->GetMaxError();

    LOGV << "TiePointGeoCoding: RMSE X      = " << rmse_x << ", " << (rmse_x < ABS_ERROR_LIMIT ? "OK" : "too large");
    LOGV << "TiePointGeoCoding: RMSE Y      = " << rmse_y << ", " << (rmse_y < ABS_ERROR_LIMIT ? "OK" : "too large");
    LOGV << "TiePointGeoCoding: Max.error X = " << max_error_x << ", "
         << (max_error_x < ABS_ERROR_LIMIT ? "OK" : "too large");
    LOGV << "TiePointGeoCoding: Max.error Y = " << max_error_y << ", "
         << (max_error_y < ABS_ERROR_LIMIT ? "OK" : "too large");

    return std::make_shared<Approximation>(f_x, f_y, center_lat, center_lon, max_square_distance * 1.1);
}

double TiePointGeoCoding::GetMaxSquareDistance(const std::vector<std::vector<double>>& data, double center_lat,
                                               double center_lon) {
    double max_square_distance = 0.0;
    for (const auto& point : data) {
        const double d_lat = point.at(0) - center_lat;
        const double d_lon = point.at(1) - center_lon;
        const double square_distance = d_lat * d_lat + d_lon * d_lon;
        if (square_distance > max_square_distance) {
            max_square_distance = square_distance;
        }
    }
    return max_square_distance;
}

std::shared_ptr<Approximation> TiePointGeoCoding::GetBestApproximation(
    const std::vector<std::shared_ptr<Approximation>>& approximations, double lat, double lon) {
    std::shared_ptr<Approximation> approximation = nullptr;
    if (approximations.size() == 1) {
        auto a = approximations.at(0);
        const double square_distance = a->GetSquareDistance(lat, lon);
        if (square_distance < a->GetMinSquareDistance()) {
            approximation = a;
        }
    } else {
        double min_square_distance = std::numeric_limits<double>::max();
        for (const auto& a : approximations) {
            const double square_distance = a->GetSquareDistance(lat, lon);
            if (square_distance < min_square_distance && square_distance < a->GetMinSquareDistance()) {
                min_square_distance = square_distance;
                approximation = a;
            }
        }
    }

    return approximation;
}

std::shared_ptr<Approximation> TiePointGeoCoding::FindRenormalizedApproximation(double lat, double renormalized_lon,
                                                                                double distance) {
    std::shared_ptr<Approximation> renormalized_approximation =
        GetBestApproximation(approximations_, lat, renormalized_lon);
    if (renormalized_approximation) {
        double renormalized_distance = renormalized_approximation->GetSquareDistance(lat, renormalized_lon);
        if (renormalized_distance < distance) {
            return renormalized_approximation;
        }
    }
    return nullptr;
}

bool TiePointGeoCoding::CanGetPixelPos() {
    if (!approximations_computed_) {
        ComputeApproximations();
    }
    return !approximations_.empty();
}

std::shared_ptr<GeoPos> TiePointGeoCoding::GetGeoPos(const std::shared_ptr<PixelPos>& pixel_pos,
                                                     std::shared_ptr<GeoPos>& geo_pos) {
    if (geo_pos == nullptr) {
        geo_pos = std::make_shared<GeoPos>();
    }
    if (pixel_pos->x_ < 0 || pixel_pos->x_ > lat_grid_->GetRasterWidth() || pixel_pos->y_ < 0 ||
        pixel_pos->y_ > lat_grid_->GetRasterHeight()) {
        geo_pos->SetInvalid();
    } else {
        geo_pos->lat_ = lat_grid_->GetPixelDouble(pixel_pos->x_, pixel_pos->y_);
        geo_pos->lon_ = lon_grid_->GetPixelDouble(pixel_pos->x_, pixel_pos->y_);
    }
    return geo_pos;
}

std::shared_ptr<PixelPos> TiePointGeoCoding::GetPixelPos(const std::shared_ptr<GeoPos>& geo_pos,
                                                         std::shared_ptr<PixelPos>& pixel_pos) {
    if (!approximations_computed_) {
        ComputeApproximations();
    }
    if (!approximations_.empty()) {
        double lat = NormalizeLat(geo_pos->lat_);
        double lon = NormalizeLon(geo_pos->lon_);
        // ensure that pixel is out of image (= no source position)
        if (pixel_pos == nullptr) {
            pixel_pos = std::make_shared<PixelPos>();
        }

        if (IsValidGeoPos(lat, lon)) {
            std::shared_ptr<Approximation> approximation = GetBestApproximation(approximations_, lat, lon);
            // retry with pixel in overlap range, re-normalise
            // solves the problem with overlapping normalized and unnormalized orbit areas (AATSR)
            if (lon >= overlap_start_ && lon <= overlap_end_) {
                double square_distance;
                if (approximation) {
                    square_distance = approximation->GetSquareDistance(lat, lon);
                } else {
                    square_distance = std::numeric_limits<double>::max();
                }
                double temp_lon = lon + 360;
                std::shared_ptr<Approximation> renormalized_approximation =
                    FindRenormalizedApproximation(lat, temp_lon, square_distance);
                if (renormalized_approximation) {
                    approximation = renormalized_approximation;
                    lon = temp_lon;
                }
            }
            if (approximation) {
                lat = RescaleLatitude(lat);
                lon = RescaleLongitude(lon, approximation->GetCenterLon());
                pixel_pos->x_ = approximation->GetFX()->ComputeZ(lat, lon);
                pixel_pos->y_ = approximation->GetFY()->ComputeZ(lat, lon);
            } else {
                pixel_pos->SetInvalid();
            }
        } else {
            pixel_pos->SetInvalid();
        }
    }
    return pixel_pos;
}

double TiePointGeoCoding::NormalizeLat(double lat) {
    if (lat < -90 || lat > 90) {
        return std::nan("NormalizeLat");
    }
    return lat;
}

double TiePointGeoCoding::NormalizeLon(double lon) {
    if (lon < -180 || lon > 180) {
        return std::nan("NormalizeLon");
    }
    double normalized_lon = lon;
    if (normalized_lon < normalized_lon_min_) {
        normalized_lon += 360;
    }
    if (normalized_lon < normalized_lon_min_ || normalized_lon > normalized_lon_max_) {
        return std::nan("NormalizeLon");
    }
    return normalized_lon;
}

bool TiePointGeoCoding::TransferGeoCoding([[maybe_unused]] const std::shared_ptr<IScene>& src_scene,
                                          const std::shared_ptr<IScene>& dest_scene,
                                          const std::shared_ptr<ProductSubsetDef>& subset_def) {
    std::string lat_grid_name = GetLatGrid()->GetName();
    std::string lon_grid_name = GetLonGrid()->GetName();
    std::shared_ptr<Product> dest_product = dest_scene->GetProduct();
    auto lat_grid = dest_product->GetTiePointGrid(lat_grid_name);
    if (lat_grid == nullptr) {
        lat_grid = TiePointGrid::CreateSubset(GetLatGrid(), subset_def);
        dest_product->AddTiePointGrid(lat_grid);
    }
    auto lon_grid = dest_product->GetTiePointGrid(lon_grid_name);
    if (lon_grid == nullptr) {
        lon_grid = TiePointGrid::CreateSubset(GetLonGrid(), subset_def);
        dest_product->AddTiePointGrid(lon_grid);
    }
    if (lat_grid && lon_grid) {
        dest_scene->SetGeoCoding(std::make_shared<TiePointGeoCoding>(lat_grid, lon_grid));
        return true;
    }
    return false;
}

void TiePointGeoCoding::Dispose() {
    // esa snap has it empty like that
}

}  // namespace snapengine
}  // namespace alus
