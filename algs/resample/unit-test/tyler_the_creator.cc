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

#include "gmock/gmock.h"

#include <stdexcept>

#include "tyler_the_creator.h"

namespace {

using ::testing::DoubleEq;
using ::testing::Eq;

class TylerTheCreatorTest : public ::testing::Test {
public:
    const alus::resample::TileConstruct simple_tiles{{100, 100}, {50, 50, 0.1, -0.1}, {10, 10}, 0};
    const alus::resample::TileConstruct sophisticated_tiles{
        {10980, 10980}, {499980.0, 6500040.0, 10.0, -10.0}, {512, 512}, 32};
    const alus::resample::TileConstruct single_tile{{123, 123}, {50, 50, 0.1, -0.1}, {123, 123}, 0};
};

// NOLINTBEGIN(readability-magic-numbers)
TEST_F(TylerTheCreatorTest, ThrowsIfInvalidTileConstructGiven) {
    auto invalid_tiles = simple_tiles;
    invalid_tiles.tile_dimension = {0, 0};
    EXPECT_THROW(alus::resample::CreateTiles(invalid_tiles), std::invalid_argument);
    invalid_tiles.tile_dimension = {1, 0};
    EXPECT_THROW(alus::resample::CreateTiles(invalid_tiles), std::invalid_argument);
    invalid_tiles.tile_dimension = {0, 1};
    EXPECT_THROW(alus::resample::CreateTiles(invalid_tiles), std::invalid_argument);
    invalid_tiles.tile_dimension = {50, 50};
    invalid_tiles.overlap = 26;
    EXPECT_THROW(alus::resample::CreateTiles(invalid_tiles), std::invalid_argument);
    invalid_tiles.image_dimension = {4, 6};
    invalid_tiles.tile_dimension = {5, 5};
    invalid_tiles.overlap = 0;
    EXPECT_THROW(alus::resample::CreateTiles(invalid_tiles), std::invalid_argument);
    invalid_tiles.image_dimension = {10, 10};
    invalid_tiles.tile_dimension = {3, 3};
    invalid_tiles.overlap = 2;
    EXPECT_THROW(alus::resample::CreateTiles(invalid_tiles), std::invalid_argument);
}

TEST_F(TylerTheCreatorTest, ReturnsSingleTile) {
    const auto tiles = alus::resample::CreateTiles(single_tile);
    ASSERT_THAT(tiles.size(), Eq(1));
    EXPECT_THAT(tiles.front().dimension.columnsX, Eq(single_tile.image_dimension.columnsX));
    EXPECT_THAT(tiles.front().dimension.rowsY, Eq(single_tile.image_dimension.rowsY));
    EXPECT_THAT(tiles.front().tile_no_x, Eq(1));
    EXPECT_THAT(tiles.front().tile_no_y, Eq(1));
    EXPECT_THAT(tiles.front().offset.x, Eq(0));
    EXPECT_THAT(tiles.front().offset.y, Eq(0));
    EXPECT_THAT(tiles.front().gt.originLon, DoubleEq(single_tile.image_gt.originLon));
    EXPECT_THAT(tiles.front().gt.originLat, DoubleEq(single_tile.image_gt.originLat));
    EXPECT_THAT(tiles.front().gt.pixelSizeLon, DoubleEq(single_tile.image_gt.pixelSizeLon));
    EXPECT_THAT(tiles.front().gt.pixelSizeLat, DoubleEq(single_tile.image_gt.pixelSizeLat));
}

TEST_F(TylerTheCreatorTest, ReturnsCorrectTiles) {
    {
        const auto tiles = alus::resample::CreateTiles(simple_tiles);
        ASSERT_THAT(tiles.size(), Eq(100));

        size_t i{};
        for (const auto& t : tiles) {
            ASSERT_THAT(t.gt.originLon,
                        DoubleEq(simple_tiles.image_gt.originLon +
                                 ((i % simple_tiles.tile_dimension.columnsX) * simple_tiles.tile_dimension.columnsX *
                                  simple_tiles.image_gt.pixelSizeLon)));
            // NOLINTBEGIN(bugprone-integer-division)
            ASSERT_THAT(t.gt.originLat,
                        DoubleEq(simple_tiles.image_gt.originLat +
                                 ((i / simple_tiles.tile_dimension.columnsX) * simple_tiles.tile_dimension.rowsY *
                                  simple_tiles.image_gt.pixelSizeLat)));
            // NOLINTEND(bugprone-integer-division)
            ASSERT_THAT(t.dimension.rowsY, Eq(simple_tiles.tile_dimension.rowsY));
            ASSERT_THAT(t.dimension.columnsX, Eq(simple_tiles.tile_dimension.columnsX));
            ASSERT_THAT(t.gt.pixelSizeLon, DoubleEq(simple_tiles.image_gt.pixelSizeLon));
            ASSERT_THAT(t.gt.pixelSizeLat, DoubleEq(simple_tiles.image_gt.pixelSizeLat));
            ASSERT_THAT(t.tile_no_x, Eq((i % simple_tiles.tile_dimension.columnsX) + 1));
            ASSERT_THAT(t.tile_no_y, Eq((i / simple_tiles.tile_dimension.columnsX) + 1));
            i++;
        }
        EXPECT_THAT(tiles.back().offset.x, Eq(90));
        EXPECT_THAT(tiles.back().offset.y, Eq(90));
        EXPECT_THAT(tiles.back().gt.originLat, DoubleEq(41));
        EXPECT_THAT(tiles.back().gt.originLon, DoubleEq(59));
        EXPECT_THAT(tiles.back().tile_no_x, Eq(10));
        EXPECT_THAT(tiles.back().tile_no_y, Eq(10));
    }

    {
        const auto tiles = alus::resample::CreateTiles(sophisticated_tiles);
        ASSERT_THAT(tiles.size(), Eq(529));

        for (const auto& t : tiles) {
            ASSERT_THAT(t.gt.pixelSizeLon, DoubleEq(sophisticated_tiles.image_gt.pixelSizeLon));
            ASSERT_THAT(t.gt.pixelSizeLat, DoubleEq(sophisticated_tiles.image_gt.pixelSizeLat));
        }

        EXPECT_THAT(tiles.front().dimension.columnsX, Eq(sophisticated_tiles.tile_dimension.columnsX));
        EXPECT_THAT(tiles.front().dimension.rowsY, Eq(sophisticated_tiles.tile_dimension.rowsY));
        EXPECT_THAT(tiles.front().offset.x, Eq(0));
        EXPECT_THAT(tiles.front().offset.y, Eq(0));
        EXPECT_THAT(tiles.front().gt.originLon, DoubleEq(sophisticated_tiles.image_gt.originLon));
        EXPECT_THAT(tiles.front().gt.originLat, DoubleEq(sophisticated_tiles.image_gt.originLat));

        EXPECT_THAT(tiles.at(1).tile_no_x, Eq(2));
        EXPECT_THAT(tiles.at(1).tile_no_y, Eq(1));
        EXPECT_THAT(tiles.at(1).dimension.columnsX, Eq(sophisticated_tiles.tile_dimension.columnsX));
        EXPECT_THAT(tiles.at(1).dimension.rowsY, Eq(sophisticated_tiles.tile_dimension.rowsY));
        EXPECT_THAT(tiles.at(1).offset.x,
                    Eq(sophisticated_tiles.tile_dimension.columnsX - sophisticated_tiles.overlap));
        EXPECT_THAT(tiles.at(1).offset.y, Eq(0));

        EXPECT_THAT(tiles.back().dimension.columnsX, Eq(420));
        EXPECT_THAT(tiles.back().dimension.rowsY, Eq(420));
        EXPECT_THAT(tiles.back().offset.x, Eq(10980 - 420));
        EXPECT_THAT(tiles.back().offset.y, Eq(10980 - 420));
        EXPECT_THAT(tiles.back().gt.originLon, DoubleEq(605580.0));
        EXPECT_THAT(tiles.back().gt.originLat, DoubleEq(6394440.0));
        EXPECT_THAT(tiles.back().tile_no_x, Eq(23));
        EXPECT_THAT(tiles.back().tile_no_y, Eq(23));
    }
}
// NOLINTEND(readability-magic-numbers)

}  // namespace