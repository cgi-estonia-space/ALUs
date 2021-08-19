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

#include <algorithm>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

#include <boost/filesystem.hpp>
#include <gdal_priv.h>

#include "gmock/gmock.h"

#include "gdal_util.h"
#include "test_utils.h"
#include "general_constants.h"

namespace {
using namespace alus;

class GeoTiffWriterTest : public ::testing::Test {
public:
    GeoTiffWriterTest() {
        GenerateRandomVector();
        CreateInMemoryDataset();
    }

protected:
    std::shared_ptr<GDALDataset> mem_dataset_;
    const double error_margin_{1e-10};

private:
    const int width_{12000};
    const int height_{8300};
    const double min_range_{-10.0};
    const double max_range_{10.0};

    std::vector<double> in_memory_data_;

    void GenerateRandomVector() {
        std::random_device random_device;
        std::mt19937_64 generator(random_device());
        std::uniform_real_distribution<double> distribution(min_range_, max_range_);

        in_memory_data_.resize(width_ * height_);
        std::generate(in_memory_data_.begin(), in_memory_data_.end(),
                      [&distribution, &generator] { return distribution(generator); });
    }

    void CreateInMemoryDataset() {
        GDALDataset* temp = GetGDALDriverManager()
                                ->GetDriverByName(utils::constants::GDAL_MEM_DRIVER)
                                ->Create("/dev/null", width_, height_, 1, GDT_Float32, nullptr);
        mem_dataset_.reset(temp, [](auto dataset) { GDALClose(dataset); });
        CHECK_GDAL_ERROR(mem_dataset_->GetRasterBand(1)->RasterIO(
            GF_Write, 0, 0, width_, height_, in_memory_data_.data(), width_, height_, GDT_Float32, 0, 0));
    }
};

TEST_F(GeoTiffWriterTest, WriteCorrectOutput) {
    const std::string_view output_file_name{"/tmp/copy"};
    GeoTiffWriteFile(mem_dataset_.get(), output_file_name);
    const std::string_view expected_output_file{"/tmp/copy.tif"};

    ASSERT_THAT(boost::filesystem::exists(expected_output_file.data()), ::testing::IsTrue());

    auto* comparand = static_cast<GDALDataset*>(GDALOpen(expected_output_file.data(), GA_ReadOnly));
    std::shared_ptr<GDALDataset> comparand_dataset(comparand, [](GDALDataset* dataset) { GDALClose(dataset); });

    ASSERT_THAT(utils::test::AreDatasetsEqual(comparand_dataset, mem_dataset_, error_margin_), ::testing::IsTrue());
    ASSERT_THAT(boost::filesystem::remove(expected_output_file.data()), ::testing::IsTrue());
}
}  // namespace