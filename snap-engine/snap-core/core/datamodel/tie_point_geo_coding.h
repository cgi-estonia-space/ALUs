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
#pragma once

#include <memory>
#include <vector>

#include "geo_pos.h"
#include "pixel_pos.h"

#include "snap-core/core/datamodel/abstract_geo_coding.h"

namespace alus {
namespace snapengine {
namespace custom {
// pre-declare
struct Rectangle;
}  // namespace custom
class FXYSum;
class TiePointGrid;
class ProductSubsetDef;
class IScene;
// todo: port? it is deprecated, but might be used at some places
class Datum;

/////////////////////////////////////////////////////////////////////////
// Inner Classes

class Approximation final {
private:
    //    todo: provide FCYSUM
    const std::shared_ptr<FXYSum> _f_x_;
    const std::shared_ptr<FXYSum> _f_y_;
    const double _center_lat_;
    const double _center_lon_;
    const double _min_square_distance_;

public:
    Approximation(const std::shared_ptr<FXYSum>& f_x, const std::shared_ptr<FXYSum>& f_y, double center_lat,
                  double center_lon, double min_square_distance)
        : _f_x_(f_x),
          _f_y_(f_y),
          _center_lat_(center_lat),
          _center_lon_(center_lon),
          _min_square_distance_(min_square_distance){};

    std::shared_ptr<FXYSum> GetFX() { return _f_x_; }

    std::shared_ptr<FXYSum> GetFY() { return _f_y_; }

    double GetCenterLat() { return _center_lat_; }

    double GetCenterLon() { return _center_lon_; }

    double GetMinSquareDistance() { return _min_square_distance_; }

    /**
     * Computes the square distance to the given geographical coordinate.
     *
     * @param lat the latitude value
     * @param lon the longitude value
     * @return the square distance
     */
    double GetSquareDistance(double lat, double lon) {
        const double dx = lon - _center_lon_;
        const double dy = lat - _center_lat_;
        return dx * dx + dy * dy;
    }
};

/**
 * A geo-coding based on two tie-point grids. One grid stores the latitude tie-points, the other stores the longitude
 * tie-points.
 */
class TiePointGeoCoding : public virtual AbstractGeoCoding {
private:
    static constexpr double ABS_ERROR_LIMIT{0.5};  // pixels
    static constexpr int MAX_NUM_POINTS_PER_TILE{1000};

    std::shared_ptr<TiePointGrid> lat_grid_;
    std::shared_ptr<TiePointGrid> lon_grid_;
    std::shared_ptr<Datum> datum_;

    bool approximations_computed_;

    bool normalized_;
    double normalized_lon_min_;
    double normalized_lon_max_;
    double lat_min_;
    double lat_max_;
    double overlap_start_;
    double overlap_end_;
    std::vector<std::shared_ptr<Approximation>> approximations_;

    // java version used synchronized !
    void ComputeApproximations();

    bool IsValidGeoPos(double lat, double lon);

    /////////////////////////////////////////////////////////////////////////
    // Private stuff
    std::shared_ptr<TiePointGrid> InitNormalizedLonGrid();

    void InitLatLonMinMax(const std::shared_ptr<TiePointGrid>& normalized_lon_grid);

    std::vector<std::shared_ptr<Approximation>> InitApproximations(
        const std::shared_ptr<TiePointGrid>& normalized_lon_grid);

    static std::shared_ptr<FXYSum> GetBestPolynomial(std::vector<std::vector<double>> data, std::vector<int> indices);

    std::vector<std::vector<double>> CreateWarpPoints(const std::shared_ptr<TiePointGrid>& lon_grid,
                                                      const std::shared_ptr<custom::Rectangle>& subset_rect);

    // package local for testing
    // maybe use this method later
    static std::vector<int> DetermineWarpParameters(int sw, int sh);

    std::shared_ptr<Approximation> CreateApproximation(const std::shared_ptr<TiePointGrid>& normalized_lon_grid,
                                                       const std::shared_ptr<custom::Rectangle>& subset_rect);

    static double GetMaxSquareDistance(const std::vector<std::vector<double>>& data, double center_lat,
                                       double center_lon);

    static std::shared_ptr<Approximation> GetBestApproximation(
        const std::vector<std::shared_ptr<Approximation>>& approximations, double lat, double lon);

    std::shared_ptr<Approximation> FindRenormalizedApproximation(double lat, double renormalized_lon, double distance);

    double GetNormalizedLonMin() { return normalized_lon_min_; }

    static double RescaleLongitude(double lon, double center_lon) { return (lon - center_lon) / 90.0; }

    static double RescaleLatitude(double lat) { return lat / 90.0; }

public:
    /**
     * Constructs geo-coding based on two given tie-point grids based on the WGS-84 CRS.
     *
     * @param latGrid the latitude grid
     * @param lonGrid the longitude grid
     */
    TiePointGeoCoding(const std::shared_ptr<TiePointGrid>& lat_grid, const std::shared_ptr<TiePointGrid>& lon_grid);

    /**
     * Constructs geo-coding based on two given tie-point grids.
     *
     * @param latGrid The latitude grid
     * @param lonGrid The longitude grid
     * @param geoCRS  The CRS to be used as both the geographic CRS and map CRS.
     */
    TiePointGeoCoding(const std::shared_ptr<TiePointGrid>& lat_grid, const std::shared_ptr<TiePointGrid>& lon_grid,
                      const CoordinateReferenceSystemWKT& geo_c_r_s);
    /**
     * Constructs geo-coding based on two given tie-point grids.
     *
     * @param latGrid the latitude grid
     * @param lonGrid the longitude grid
     * @param datum   the geodetic datum
     * @deprecated since SNAP 1.0, use {@link #TiePointGeoCoding(TiePointGrid, TiePointGrid, CoordinateReferenceSystem)}
     */
    [[deprecated]] TiePointGeoCoding(const std::shared_ptr<TiePointGrid>& lat_grid,
                                     const std::shared_ptr<TiePointGrid>& lon_grid,
                                     const std::shared_ptr<Datum>& datum);

    /**
     * Gets the datum, the reference point or surface against which {@link GeoPos} measurements are made.
     *
     * @return the datum
     */
    std::shared_ptr<Datum> GetDatum() override { return datum_; }

    /**
     * Gets the flag indicating that the geographic boundary of the tie-points in this geo-coding
     * intersects the 180 degree meridian.
     *
     * @return true if so
     */
    bool IsCrossingMeridianAt180() override { return normalized_; }

    void Dispose() override;

    /**
     * Checks whether this geo-coding can determine the geodetic position from a pixel position.
     *
     * @return <code>true</code>, if so
     */
    bool CanGetGeoPos() override { return true; }

    /**
     * Checks whether this geo-coding can determine the pixel position from a geodetic position.
     *
     * @return <code>true</code>, if so
     */
    bool CanGetPixelPos() override;

    /**
     * @return the latitude grid, never <code>null</code>.
     */
    std::shared_ptr<TiePointGrid> GetLatGrid() { return lat_grid_; }

    /**
     * @return the longitude grid, never <code>null</code>.
     */
    std::shared_ptr<TiePointGrid> GetLonGrid() { return lon_grid_; }

    /**
     * Returns the latitude and longitude value for a given pixel co-ordinate.
     *
     * @param pixelPos the pixel's co-ordinates given as x,y
     * @param geoPos   an instance of <code>GeoPos</code> to be used as retun value. If this parameter is
     *                 <code>null</code>, the method creates a new instance which it then returns.
     * @return the geographical position as lat/lon.
     */
    std::shared_ptr<GeoPos> GetGeoPos(const std::shared_ptr<PixelPos>& pixel_pos,
                                      std::shared_ptr<GeoPos>& geo_pos) override;

    /**
     * Returns the pixel co-ordinates as x/y for a given geographical position given as lat/lon.
     *
     * @param geoPos   the geographical position as lat/lon.
     * @param pixelPos an instance of <code>Point</code> to be used as retun value. If this parameter is
     *                 <code>null</code>, the method creates a new instance which it then returns.
     * @return the pixel co-ordinates as x/y
     */
    std::shared_ptr<PixelPos> GetPixelPos(const std::shared_ptr<GeoPos>& geo_pos,
                                          std::shared_ptr<PixelPos>& pixel_pos) override;

    /**
     * Gets the normalized latitude value.
     * The method returns <code>Double.NaN</code> if the given latitude value is out of bounds.
     *
     * @param lat the raw latitude value in the range -90 to +90 degrees
     * @return the normalized latitude value, <code>Double.NaN</code> else
     */
    static double NormalizeLat(double lat);

    /**
     * Gets the normalized longitude value.
     * The method returns <code>Double.NaN</code> if the given longitude value is out of bounds
     * or if it's normalized value is not in the value range of this geo-coding's normalized longitude grid..
     *
     * @param lon the raw longitude value in the range -180 to +180 degrees
     * @return the normalized longitude value, <code>Double.NaN</code> else
     */
    double NormalizeLon(double lon);

    //    bool Equals(const std::any& o) override {
    //        if (this == o) {
    //            return true;
    //        }
    //        if (o == null || getClass() != o.getClass()) {
    //            return false;
    //        }
    //
    //        TiePointGeoCoding that = (TiePointGeoCoding) o;
    //
    //        if (!latGrid.equals(that.latGrid)) {
    //            return false;
    //        }
    //        if (!lonGrid.equals(that.lonGrid)) {
    //            return false;
    //        }
    //
    //        return true;
    //    }

    //    @Override
    // public int hashCode() {
    //        int result = latGrid.hashCode();
    //        result = 31 * result + lonGrid.hashCode();
    //        return result;
    //    }
    //
    //    @Override
    // public void dispose() {
    //    }

    /**
     * Transfers the geo-coding of the {@link Scene srcScene} to the {@link Scene destScene} with respect to the given
     * {@link ProductSubsetDef subsetDef}.
     *
     * @param srcScene  the source scene
     * @param destScene the destination scene
     * @param subsetDef the definition of the subset, may be <code>null</code
     *                  >
     * @return true, if the geo-coding could be transferred.
     */
    bool TransferGeoCoding(const std::shared_ptr<IScene>& src_scene, const std::shared_ptr<IScene>& dest_scene,
                           const std::shared_ptr<ProductSubsetDef>& subset_def) override;
};

}  // namespace snapengine
}  // namespace alus
