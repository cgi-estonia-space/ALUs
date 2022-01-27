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
#include <memory>

#include <gmock/gmock.h>

#include "comparators.h"
#include "coregistration_controller.h"
#include "dataset.h"
#include "dem_assistant.h"
#include "gdal_util.h"

namespace {

class CoregTester {
public:
    std::vector<double> land_strip_;
    std::vector<double> coast_strip_;
    std::vector<double> sea_strip_;

    explicit CoregTester(std::string_view strip_file) { ReadTestData(strip_file); }

    void ReadTestData(std::string_view strip_file) {
        const int test_file_amount{3};
        int test_sets;
        std::ifstream input_file(strip_file.data());
        if (!input_file.is_open()) {
            throw std::runtime_error(std::string("Coregistration test can not open test file: ").append(strip_file));
        }
        input_file >> test_sets;
        if (test_sets != test_file_amount) {
            throw std::runtime_error("Coregistration tester is supposed to have 3 test sets");
        }
        alus::Rectangle rectangle;
        int size;

        input_file >> rectangle.x >> rectangle.y >> rectangle.width >> rectangle.height;
        size = rectangle.width * rectangle.height;
        land_strip_.resize(size);
        for (int i = 0; i < size; i++) {
            input_file >> land_strip_.at(i);
        }

        input_file >> rectangle.x >> rectangle.y >> rectangle.width >> rectangle.height;
        size = rectangle.width * rectangle.height;
        coast_strip_.resize(size);
        for (int i = 0; i < size; i++) {
            input_file >> coast_strip_.at(i);
        }

        input_file >> rectangle.x >> rectangle.y >> rectangle.width >> rectangle.height;
        size = rectangle.width * rectangle.height;
        sea_strip_.resize(size);
        for (int i = 0; i < size; i++) {
            input_file >> sea_strip_.at(i);
        }

        input_file.close();
    }
};

// TODO at very least full3 and split cut should test both slave bands  // NOLINT

TEST(coregistration, full3) {
    {
        std::vector<std::string> srtm3_files{"./goods/srtm_43_06.tif", "./goods/srtm_44_06.tif"};
        std::shared_ptr<alus::app::DemAssistant> dem_assistant =
            alus::app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(std::move(srtm3_files));
        dem_assistant->GetSrtm3Manager()->HostToDevice();

        const std::string_view output_file{"./goods/beirut_images/coregistration_test.tif"};

        std::unique_ptr<alus::coregistration::Coregistration> cor =
            std::make_unique<alus::coregistration::Coregistration>("./goods/apply_orbit_file_op/orbit-files/");
        cor->Initialize(
            "./goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD.SAFE",
            "./goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE",
            output_file.data(), "IW1", "VV");
        dem_assistant->GetSrtm3Manager()->HostToDevice();
        cor->DoWork(dem_assistant->GetEgm96Manager()->GetDeviceValues(),
                    {dem_assistant->GetSrtm3Manager()->GetSrtmBuffersInfo(),
                     dem_assistant->GetSrtm3Manager()->GetDeviceSrtm3TilesCount()});

        auto target_datasets = cor->GetTargetDataset()->GetDataset();

        CoregTester tester("./goods/coregistration_strips.txt");

        const std::string_view output_slave_i{"./goods/beirut_images/coregistration_test_slave_I.tif"};
        const std::string_view output_slave_q{"./goods/beirut_images/coregistration_test_slave_Q.tif"};
        alus::GeoTiffWriteFile(target_datasets.at(2), output_slave_i);
        alus::GeoTiffWriteFile(target_datasets.at(3), output_slave_q);  // NOLINT

        alus::Dataset<double> test_set(output_slave_i);
        alus::Rectangle rectangle{1000, 10000, 1, 100};   // NOLINT
        alus::Rectangle rectangle2{7436, 6293, 1, 100};   // NOLINT
        alus::Rectangle rectangle3{15576, 4440, 1, 100};  // NOLINT
        std::vector<double> data_buf(100);                // NOLINT
        test_set.ReadRectangle(rectangle, 1, data_buf.data());

        size_t count_land = alus::EqualsArraysd(data_buf.data(), tester.land_strip_.data(), 100, 0.001);  // NOLINT
        EXPECT_EQ(count_land, 0) << "Land results I do not match. Mismatches: " << count_land << '\n';

        test_set.ReadRectangle(rectangle2, 1, data_buf.data());

        size_t count_coast = alus::EqualsArraysd(data_buf.data(), tester.coast_strip_.data(), 100, 0.001);  // NOLINT
        EXPECT_EQ(count_coast, 0) << "Land results I do not match. Mismatches: " << count_coast << '\n';

        test_set.ReadRectangle(rectangle3, 1, data_buf.data());

        size_t count_sea = alus::EqualsArraysd(data_buf.data(), tester.sea_strip_.data(), 100, 0.001);  // NOLINT
        EXPECT_EQ(count_sea, 0) << "Land results I do not match. Mismatches: " << count_sea << '\n';
    }
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaDeviceReset());  // for cuda-memcheck --leak-check full
}

TEST(coregistration, splitCut) {
    {
        std::vector<std::string> srtm3_files{"./goods/srtm_43_06.tif", "./goods/srtm_44_06.tif"};
        std::shared_ptr<alus::app::DemAssistant> dem_assistant =
            alus::app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(std::move(srtm3_files));
        dem_assistant->GetSrtm3Manager()->HostToDevice();
        const std::string_view output_file_cut{"./goods/beirut_images/coregistration_test_cut.tif"};

        std::unique_ptr<alus::coregistration::Coregistration> cor =
            std::make_unique<alus::coregistration::Coregistration>("goods/apply_orbit_file_op/orbit-files/");

        cor->Initialize("goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD.SAFE",
                        "goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE",
                        output_file_cut.data(), "IW1", "VV", 4, 6);  // NOLINT

        cor->DoWork(dem_assistant->GetEgm96Manager()->GetDeviceValues(),
                    {dem_assistant->GetSrtm3Manager()->GetSrtmBuffersInfo(),
                     dem_assistant->GetSrtm3Manager()->GetDeviceSrtm3TilesCount()});

        auto target_datasets = cor->GetTargetDataset()->GetDataset();
        const std::string_view output_slave_i{"./goods/beirut_images/coregistration_test_slave_I_cut.tif"};
        const std::string_view output_slave_q{"./goods/beirut_images/coregistration_test_slave_Q_cut.tif"};
        alus::GeoTiffWriteFile(target_datasets.at(2), output_slave_i);
        alus::GeoTiffWriteFile(target_datasets.at(3), output_slave_q);  // NOLINT

        CoregTester tester("./goods/coregistration_strips_cut.txt");
        alus::Dataset<double> test_set(output_slave_i);
        alus::Rectangle rectangle{3991, 1620, 1, 100};   // NOLINT
        alus::Rectangle rectangle2{5350, 1531, 100, 1};  // NOLINT
        alus::Rectangle rectangle3{5457, 1602, 100, 1};  // NOLINT
        std::vector<double> data_buf(100);               // NOLINT
        test_set.ReadRectangle(rectangle, 1, data_buf.data());

        size_t count_land = alus::EqualsArraysd(data_buf.data(), tester.land_strip_.data(), 100, 0.001);  // NOLINT
        EXPECT_EQ(count_land, 0) << "Land results I do not match. Mismatches: " << count_land << '\n';

        test_set.ReadRectangle(rectangle2, 1, data_buf.data());

        size_t count_coast = alus::EqualsArraysd(data_buf.data(), tester.coast_strip_.data(), 100, 0.001);  // NOLINT
        EXPECT_EQ(count_coast, 0) << "Land results I do not match. Mismatches: " << count_coast << '\n';

        test_set.ReadRectangle(rectangle3, 1, data_buf.data());

        size_t count_sea = alus::EqualsArraysd(data_buf.data(), tester.sea_strip_.data(), 100, 0.001);  // NOLINT
        EXPECT_EQ(count_sea, 0) << "Land results I do not match. Mismatches: " << count_sea << '\n';
    }
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaDeviceReset());  // for cuda-memcheck --leak-check full
}

}  // namespace
