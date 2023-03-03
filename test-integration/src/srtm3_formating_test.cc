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
#include <fstream>
#include <string>
#include <vector>

#include "gmock/gmock.h"

#include "comparators.h"
#include "cuda_friendly_object.h"
#include "cuda_util.h"
#include "pointer_holders.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96.h"
#include "srtm3_elevation_model.h"

namespace {

class SRTM3TileTester {
private:
    std::string test_file_name_;

public:
    std::vector<int> xs_;
    std::vector<int> ys_;
    std::vector<float> results_;

    size_t size_;

    explicit SRTM3TileTester(std::string_view test_file_name) { this->test_file_name_ = test_file_name; }
    ~SRTM3TileTester() = default;

    void ReadTestData() {
        std::ifstream test_data_reader(this->test_file_name_);
        if (!test_data_reader.is_open()) {
            throw std::ios::failure("srtm3 tile test data file not open.");
        }
        test_data_reader >> this->size_;
        this->xs_.resize(this->size_);
        this->ys_.resize(this->size_);
        this->results_.resize(this->size_);

        for (size_t i = 0; i < this->size_; i++) {
            test_data_reader >> this->xs_.at(i) >> this->ys_.at(i) >> this->results_.at(i);
        }

        test_data_reader.close();
    }
};

TEST(SRTM3, tileFormating) {
    SRTM3TileTester tester("./goods/tileFormatTestData.txt");
    tester.ReadTestData();

    std::shared_ptr<alus::snapengine::EarthGravitationalModel96> egm_96 =
        std::make_shared<alus::snapengine::EarthGravitationalModel96>();
    egm_96->HostToDevice();

    std::vector<std::string> files{"./goods/srtm_41_01.tif", "./goods/srtm_42_01.tif"};
    alus::snapengine::Srtm3ElevationModel srtm_3_dem(files, egm_96);
    srtm_3_dem.LoadTiles();
    srtm_3_dem.TransferToDevice();

    std::vector<float> end_tile;
    std::vector<float> end_results;
    end_results.resize(tester.size_);
    std::vector<alus::PointerHolder> tiles;
    tiles.resize(2);
    const int chosen_tile = 0;
    CHECK_CUDA_ERR(cudaMemcpy(tiles.data(), srtm_3_dem.GetBuffers(), 2 * sizeof(alus::PointerHolder),
                              cudaMemcpyDeviceToHost));
    int tile_x_size = tiles.at(chosen_tile).x;
    int tile_y_size = tiles.at(chosen_tile).y;
    int tile_size = tile_x_size * tile_y_size;
    end_tile.resize(tile_size);

    CHECK_CUDA_ERR(
        cudaMemcpy(end_tile.data(), tiles.at(chosen_tile).pointer, tile_size * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < tester.size_; i++) {
        end_results.at(i) = end_tile.at(tester.xs_.at(i) + tile_x_size * tester.ys_.at(i));
    }

    const float error_margin{0.0000001};
    size_t count = alus::EqualsArrays(end_results.data(), tester.results_.data(), tester.size_, error_margin);
    EXPECT_EQ(count, 0) << "SRTM3 tiling test results do not match. Mismatches: " << count << '\n';
}

}  // namespace
