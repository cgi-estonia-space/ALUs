#pragma once

#include <vector>

#include "dataset.hpp"
#include "dem.hpp"
#include "product.h"
#include "resampling.h"
#include "tc_tile.h"
#include "terrain_correction_metadata.h"
#include "tie_point_grid.h"

namespace alus {

class TerrainCorrection {
   public:
    TerrainCorrection(alus::Dataset coh_ds, /*alus::Dataset metadata,*/ alus::Dataset dem);
    alus::terraincorrection::RangeDopplerTerrainMetadata metadata_;

    void DoWork();
    void LocalDemCuda();
    void LocalDemCuda(alus::Dataset *target_dataset);

    alus::snapengine::Product CreateTargetProduct(const alus::snapengine::geocoding::Geocoding *geocoding,
                                                  snapengine::geocoding::Geocoding *&target_geocoding,
                                                  int source_width,
                                                  int source_height, const char* output_file_name) const;  // TODO: move to private after testing

    std::vector<double> GetElevations() { return coh_ds_elevations_; }

    /**
     * Stub for the main terrain correction algorithm
     *
     * @return Returns the new image data after applying Range-Doppler Terrain Correction algorithm
     * @todo It should return a product, not a vector
     */
    std::vector<double> ExecuteTerrainCorrection(const char* output_file_name, size_t tile_width, size_t tile_height);

    ~TerrainCorrection();

   private:
    void LocalDemCpu();

    alus::Dataset coh_ds_;
    //    alus::Dataset metadata_ds_;
    alus::Dem dem_ds_;
    std::vector<double> coh_ds_elevations_{};
    std::vector<float> lat_tie_points_{};
    std::vector<float> lon_tie_points_{};

    void FillMetadata();
    //    Dataset CreateTargetProduct(const double azimuth_pixel_spacing, const double range_pixel_spacing);

    /**
     * Computes target image boundary by creating a rectangle around the source image. The source image should be
     * TiePoint Geocoded. Boundary is in degree coordinates.
     *
     * @param src_lat_tie_point_grid Grid of latitude tie points.
     * @param src_lon_tie_point_grid Grid of longitude tie points.
     * @param source_width Source image width.
     * @param source_height Source image height.
     * @return Vector containing four target image boundaries: minimum latitude, maximum latitude, minimum longitude,
     * maximum longitude.
     */
    static std::vector<double> ComputeImageBoundary(const alus::snapengine::geocoding::Geocoding *geocoding,
                                             int source_width,
                                             int source_height);

    double *d_local_dem_{nullptr};

    /**
     * Method for splitting the input image into several tiles.
     *
     * @param base_image The input image represented as a simple large tile.
     * @param dest_bounds Destination image bounds.
     */
    std::vector<alus::TcTile> CalculateTiles(alus::snapengine::resampling::Tile &base_image,
                                              alus::Rectangle dest_bounds, int tile_width, int tile_height);

    /**
     * Fills the TerrainCorrection tile with DEM coordinates.
     *
     * @param tile TerrainCorrection tile. Tile coordinates should not be missing.
     */
    static void FillDemCoordinates(alus::TcTile &tile,
                            alus::snapengine::geocoding::Geocoding *target_geocoding,
                            alus::snapengine::geocoding::Geocoding *dem_geocoding);
};
}  // namespace alus