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

namespace {

using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Eq;
using ::testing::SizeIs;

TEST(CopDemCog30m, LoadsTilesCorrectly) {
    const std::vector<std::string> input_files{"./goods/Copernicus_DSM_COG_10_N49_00_E004_00_DEM.tif",
                                               "./goods/Copernicus_DSM_COG_10_N51_00_E006_00_DEM.tif"};

    auto manager = alus::dem::CopDemCog30m(std::move(input_files));
    manager.LoadTiles();
    const auto& dem_properties = manager.GetPropertiesValue();
    ASSERT_THAT(dem_properties, SizeIs(2));

    alus::dem::Property n49_e004_prop{};
    n49_e004_prop.pixels_per_tile_inverted_x_axis = 1 / 3600.0;
    n49_e004_prop.pixels_per_tile_inverted_y_axis = 1 / 3600.0;
    n49_e004_prop.pixels_per_tile_x_axis = 3600;
    n49_e004_prop.pixels_per_tile_y_axis = 3600;
    n49_e004_prop.raster_width = 3600 * 360;
    n49_e004_prop.pixel_size_degrees_x_axis = 0.000277777777778;
    n49_e004_prop.pixel_size_degrees_inverted_x_axis = 3600;
    n49_e004_prop.lat_origin = 50.000138888888891;
    n49_e004_prop.lon_origin = 3.999861111111111;
    n49_e004_prop.lat_extent = 49.0001389;
    n49_e004_prop.lon_extent = 4.9998611;

    alus::dem::Property n51_e006_prop{};
    n51_e006_prop.pixels_per_tile_inverted_x_axis = 1 / 2400.0;
    n51_e006_prop.pixels_per_tile_inverted_y_axis = 1 / 3600.0;
    n51_e006_prop.pixels_per_tile_x_axis = 2400;
    n51_e006_prop.pixels_per_tile_y_axis = 3600;
    n51_e006_prop.raster_width = 2400 * 360;
    n51_e006_prop.pixel_size_degrees_x_axis = 0.000416666666667;
    n51_e006_prop.pixel_size_degrees_inverted_x_axis = 2400;
    n51_e006_prop.lat_origin = 52.000138888888891;
    n51_e006_prop.lon_origin = 5.999791666666667;
    n51_e006_prop.lat_extent = 51.0001389;
    n51_e006_prop.lon_extent = 6.9997917;


    std::vector<alus::dem::Property> expectedProperties{n49_e004_prop, n51_e006_prop};
    for (size_t i = 0; i < dem_properties.size(); i++) {
        const auto& p = dem_properties.at(i);
        const auto& ep = expectedProperties.at(i);
        ASSERT_THAT(p.pixels_per_tile_inverted_x_axis, DoubleEq(ep.pixels_per_tile_inverted_x_axis));
        ASSERT_THAT(p.pixels_per_tile_inverted_y_axis, DoubleEq(ep.pixels_per_tile_inverted_y_axis));
        ASSERT_THAT(p.pixels_per_tile_x_axis, Eq(ep.pixels_per_tile_x_axis));
        ASSERT_THAT(p.pixels_per_tile_y_axis, Eq(ep.pixels_per_tile_y_axis));
        ASSERT_THAT(p.tiles_x_axis, Eq(360));
        ASSERT_THAT(p.tiles_y_axis, Eq(180));
        ASSERT_THAT(p.raster_width, Eq(ep.raster_width));
        ASSERT_THAT(p.raster_height, Eq(180 * 3600));
        ASSERT_THAT(p.no_data_value, DoubleEq(0.0));
        ASSERT_THAT(p.pixel_size_degrees_x_axis, DoubleNear(ep.pixel_size_degrees_x_axis, 1e-12));
        ASSERT_THAT(p.pixel_size_degrees_y_axis, DoubleNear(0.000277777777778, 1e-12));
        ASSERT_THAT(p.pixel_size_degrees_inverted_x_axis, DoubleNear(ep.pixel_size_degrees_inverted_x_axis, 1e-12));
        ASSERT_THAT(p.pixel_size_degrees_inverted_y_axis, DoubleNear(3600, 1e-12));
        ASSERT_THAT(p.lat_coverage, Eq(90));
        ASSERT_THAT(p.lon_coverage, Eq(180));
        ASSERT_THAT(p.lat_origin, DoubleNear(ep.lat_origin, 1e-12));
        ASSERT_THAT(p.lat_extent, DoubleNear(ep.lat_extent, 1e-7));
        ASSERT_THAT(p.lon_origin, DoubleNear(ep.lon_origin, 1e-12));
        ASSERT_THAT(p.lon_extent, DoubleNear(ep.lon_extent, 1e-7));
    }
}

}  // namespace