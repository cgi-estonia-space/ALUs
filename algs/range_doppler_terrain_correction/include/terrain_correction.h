#pragma once

#include <vector>

#include "computation_metadata.h"
#include "dataset.h"
#include "dem.hpp"
#include "product.h"
#include "resampling.h"
#include "tc_tile.h"
#include "terrain_correction_metadata.h"
#include "tie_point_grid.h"

namespace alus::terraincorrection {

class TerrainCorrection {
   public:
    TerrainCorrection(alus::Dataset coh_ds, /*alus::Dataset metadata,*/ alus::Dataset dem);

    explicit TerrainCorrection(alus::Dataset coh_ds,
                               RangeDopplerTerrainMetadata metadata,
                               const Metadata::TiePoints& lat_tie_points,
                               const Metadata::TiePoints& lon_tie_points);

    static alus::snapengine::Product CreateTargetProduct(
        const alus::snapengine::geocoding::Geocoding* geocoding,
        snapengine::geocoding::Geocoding*& target_geocoding,
        int source_width,
        int source_height,
        double azimuth_spacing,
        const std::string& output_filename);  // TODO: move to private after testing

    //    std::vector<double> GetElevations() { return coh_ds_elevations_; }

    void ExecuteTerrainCorrection(const std::string& output_file_name, size_t tile_width, size_t tile_height);

    ~TerrainCorrection();

   private:

    Dataset coh_ds_;
    RangeDopplerTerrainMetadata metadata_;
    //Dem dem_ds_;
//    std::vector<double> coh_ds_elevations_{};
    const Metadata::TiePoints& lat_tie_points_;
    const Metadata::TiePoints& lon_tie_points_;

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

    /**
     * Method for splitting the input image into several tiles.
     *
     * @param base_image The input image represented as a simple large tile.
     * @param dest_bounds Destination image bounds.
     */
    std::vector<alus::TcTile> CalculateTiles(alus::snapengine::resampling::Tile &base_image,
                                              alus::Rectangle dest_bounds, int tile_width, int tile_height);

    ComputationMetadata CreateComputationMetadata();

    std::vector<void*> cuda_arrays_to_clean_{};
    void FreeCudaArrays();
};
}  // namespace alus