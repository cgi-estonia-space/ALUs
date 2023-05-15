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
#pragma once

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

#include "gdal_priv.h"

#include "computation_metadata.h"
#include "dataset.h"
#include "dem_property.h"
#include "dem_type.h"
#include "get_position.h"
#include "metadata_record.h"
#include "pointer_holders.h"
#include "product_old.h"
#include "resampling.h"
#include "snap-core/core/datamodel/tie_point_grid.h"
#include "srgr_coefficients.h"
#include "tc_tile.h"
#include "terrain_correction_kernel.h"
#include "terrain_correction_metadata.h"

namespace alus::terraincorrection {

class TerrainCorrection {
public:
    explicit TerrainCorrection(GDALDataset* input_dataset, const RangeDopplerTerrainMetadata& metadata,
                               std::shared_ptr<snapengine::TiePointGrid> lat_tie_point_grid,
                               std::shared_ptr<snapengine::TiePointGrid> lon_tie_point_grid,
                               const PointerHolder* dem_tiles, size_t dem_tiles_length,
                               const dem::Property* dem_property, const dem::Type dem_type,
                               const std::vector<dem::Property>& dem_property_value, int selected_band_id = 1,
                               bool use_average_scene_height = false);

    snapengine::old::Product CreateTargetProduct(const snapengine::geocoding::Geocoding* geocoding,
                                                 std::string_view output_filename);

    void RegisterMetadata(common::metadata::Container md) { metadata_output_ = md; }
    void ExecuteTerrainCorrection(std::string_view output_file_name, size_t tile_width, size_t tile_height);

    TerrainCorrection(const TerrainCorrection&) = delete;
    TerrainCorrection(TerrainCorrection&&) = delete;
    TerrainCorrection& operator=(const TerrainCorrection&) = delete;
    TerrainCorrection& operator=(TerrainCorrection&&) = delete;

    ~TerrainCorrection();

    std::pair<std::string, std::shared_ptr<GDALDataset>> GetOutputDataset() const;

private:
    GDALDataset* input_ds_;
    const RangeDopplerTerrainMetadata& metadata_;
    [[maybe_unused]] snapengine::geocoding::Geocoding* target_geocoding_{};
    const PointerHolder* d_dem_tiles_;
    const size_t d_dem_tiles_length_;
    const dem::Property* dem_property_;
    const dem::Type dem_type_;
    const std::vector<dem::Property> dem_property_value_;
    std::vector<void*> cuda_arrays_to_clean_{};
    const int selected_band_id_;
    std::shared_ptr<snapengine::TiePointGrid> lat_tie_point_grid_;
    std::shared_ptr<snapengine::TiePointGrid> lon_tie_point_grid_;
    const bool use_average_scene_height_{false};
    GetPositionMetadata d_get_position_metadata_;
    std::vector<snapengine::OrbitStateVectorComputation> h_orbit_state_vectors_;
    std::pair<std::string, std::shared_ptr<GDALDataset>> output_;
    common::metadata::Container metadata_output_;
    cuda::KernelArray<SrgrCoefficientsDevice> d_srgr_coefficients_{nullptr, 0};

    /**
     * Computes target image boundary by creating a rectangle around the source image. The source image should be
     * TiePoint Geocoded. Boundary is in degree coordinates.
     *
     * @param src_lat_tie_point_grid Grid of latitude tie points.
     * @param src_lon_tie_point_grid Grid of longitude tie points.
     * @param source_width Source image width.
     * @param source_height Source image height.
     * @return Vector containing four target image boundaries: minimum latitude, maximum latitude, minimum
     * longitude, maximum longitude.
     */
    static std::vector<double> ComputeImageBoundary(const snapengine::geocoding::Geocoding* geocoding, int source_width,
                                                    int source_height);

    /**
     * Method for splitting the input image into several tiles.
     *
     * @param base_image The input image represented as a simple large tile.
     * @param dest_bounds Destination image bounds.
     */
    std::vector<TcTileCoordinates> CalculateTiles(const snapengine::resampling::Tile& base_image, Rectangle dest_bounds,
                                                  int tile_width, int tile_height) const;

    void CreateHostMetadata(double line_time_interval_in_days);
    void CreateGetPositionDeviceArrays(int y_size, double line_time_interval_in_days);

    /**
     * Copy device arrays to host, for debugging only.
     */
    void DebugDeviceArrays() const;

    void DebugSrgrEntries() const;

    void FreeCudaArrays();

    struct SharedThreadData;
    static void CalculateTile(TcTileCoordinates tile, SharedThreadData* tc_data, PerThreadData* c);
    static void TileLoop(SharedThreadData* tc_data, PerThreadData* ctx);

    void CreateSrgrCoefficientsOnDevice();
};
}  // namespace alus::terraincorrection
