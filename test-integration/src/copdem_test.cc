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

#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "copdem_cog_30m.h"
#include "dem_property.h"

double GetElevationWrapper(double lon, double lat, alus::PointerArray p_array, const alus::dem::Property* dem_prop);

namespace {

using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::SizeIs;

TEST(CopDemCog30m, LoadsTilesCorrectly) {
    const std::vector<std::string> input_files{"./goods/Copernicus_DSM_COG_10_N49_00_E004_00_DEM.tif",
                                               "./goods/Copernicus_DSM_COG_10_N51_00_E006_00_DEM.tif"};

    auto manager = alus::dem::CopDemCog30m(std::move(input_files));
    manager.LoadTiles();
    const auto& dem_properties = manager.GetPropertiesValue();
    ASSERT_THAT(dem_properties, SizeIs(2));

    alus::dem::Property n49_e004_prop{};
    n49_e004_prop.tile_pixel_count_inverted_x = 1 / 3600.0;
    n49_e004_prop.tile_pixel_count_inverted_y = 1 / 3600.0;
    n49_e004_prop.tile_pixel_count_x = 3600;
    n49_e004_prop.tile_pixel_count_y = 3600;
    n49_e004_prop.grid_total_width_pixels = 3600 * 360;
    n49_e004_prop.tile_pixel_size_deg_x = 0.000277777777778;
    n49_e004_prop.tile_pixel_size_deg_inverted_x = 3600;
    n49_e004_prop.tile_lat_origin = 50.000138888888891;
    n49_e004_prop.tile_lon_origin = 3.999861111111111;
    n49_e004_prop.tile_lat_extent = 49.0001389;
    n49_e004_prop.tile_lon_extent = 4.9998611;

    alus::dem::Property n51_e006_prop{};
    n51_e006_prop.tile_pixel_count_inverted_x = 1 / 2400.0;
    n51_e006_prop.tile_pixel_count_inverted_y = 1 / 3600.0;
    n51_e006_prop.tile_pixel_count_x = 2400;
    n51_e006_prop.tile_pixel_count_y = 3600;
    n51_e006_prop.grid_total_width_pixels = 2400 * 360;
    n51_e006_prop.tile_pixel_size_deg_x = 0.000416666666667;
    n51_e006_prop.tile_pixel_size_deg_inverted_x = 2400;
    n51_e006_prop.tile_lat_origin = 52.000138888888891;
    n51_e006_prop.tile_lon_origin = 5.999791666666667;
    n51_e006_prop.tile_lat_extent = 51.0001389;
    n51_e006_prop.tile_lon_extent = 6.9997917;

    std::vector<alus::dem::Property> expectedProperties{n49_e004_prop, n51_e006_prop};
    for (size_t i = 0; i < dem_properties.size(); i++) {
        const auto& p = dem_properties.at(i);
        const auto& ep = expectedProperties.at(i);
        ASSERT_THAT(p.tile_pixel_count_inverted_x, DoubleEq(ep.tile_pixel_count_inverted_x));
        ASSERT_THAT(p.tile_pixel_count_inverted_y, DoubleEq(ep.tile_pixel_count_inverted_y));
        ASSERT_THAT(p.tile_pixel_count_x, Eq(ep.tile_pixel_count_x));
        ASSERT_THAT(p.tile_pixel_count_y, Eq(ep.tile_pixel_count_y));
        ASSERT_THAT(p.grid_tile_count_x, Eq(360));
        ASSERT_THAT(p.grid_tile_count_y, Eq(180));
        ASSERT_THAT(p.grid_total_width_pixels, Eq(ep.grid_total_width_pixels));
        ASSERT_THAT(p.grid_total_height_pixels, Eq(180 * 3600));
        ASSERT_THAT(p.no_data_value, DoubleEq(0.0));
        ASSERT_THAT(p.tile_pixel_size_deg_x, DoubleNear(ep.tile_pixel_size_deg_x, 1e-12));
        ASSERT_THAT(p.tile_pixel_size_deg_y, DoubleNear(0.000277777777778, 1e-12));
        ASSERT_THAT(p.tile_pixel_size_deg_inverted_x, DoubleNear(ep.tile_pixel_size_deg_inverted_x, 1e-12));
        ASSERT_THAT(p.tile_pixel_size_deg_inverted_y, DoubleNear(3600, 1e-12));
        ASSERT_THAT(p.grid_max_lat, Eq(90));
        ASSERT_THAT(p.grid_max_lon, Eq(180));
        ASSERT_THAT(p.tile_lat_origin, DoubleNear(ep.tile_lat_origin, 1e-12));
        ASSERT_THAT(p.tile_lat_extent, DoubleNear(ep.tile_lat_extent, 1e-7));
        ASSERT_THAT(p.tile_lon_origin, DoubleNear(ep.tile_lon_origin, 1e-12));
        ASSERT_THAT(p.tile_lon_extent, DoubleNear(ep.tile_lon_extent, 1e-7));
    }

    manager.TransferToDevice();
    const auto count = manager.GetTileCount();
    ASSERT_THAT(count, Eq(2));
    ASSERT_THAT(manager.GetProperties(), Ne(nullptr));
    ASSERT_THAT(manager.GetBuffers(), Ne(nullptr));

    alus::PointerArray dem_buffers{const_cast<alus::PointerHolder*>(manager.GetBuffers()), manager.GetTileCount()};
    double result = GetElevationWrapper(4.0008, 49.9999, dem_buffers, manager.GetProperties());
    ASSERT_THAT(190.081, DoubleNear(result, 1e-3));
    result = GetElevationWrapper(6.4808, 51.0772, dem_buffers, manager.GetProperties());
    ASSERT_THAT(-76.41619, DoubleNear(result, 0.3));
}

}  // namespace