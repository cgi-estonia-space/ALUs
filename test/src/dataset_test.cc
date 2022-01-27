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
#include "dataset.h"
#include "target_dataset.h"

#include "gmock/gmock.h"

#include "gdal_management.h"
#include "tests_common.h"

using alus::tests::TIF_PATH_1;

namespace {

class DatasetTest : public ::testing::Test {
public:
    DatasetTest() = default;

    alus::gdalmanagement::ErrorCallback silentGdalErrorHandle_ = [](std::string_view) {};
    alus::gdalmanagement::ErrorCallbackGuard gdalErrorHandleGuard_{
        alus::gdalmanagement::SetErrorHandle(silentGdalErrorHandle_)};

    ~DatasetTest() override = default;
};

TEST_F(DatasetTest, onInvalidFilenameThrows) {
    std::string_view filename = "non_existent_unittest_file";
    ASSERT_THROW(auto a = alus::Dataset<double>(filename), alus::DatasetError);
}

TEST_F(DatasetTest, loadsValidTifFile) {
    auto ds = alus::Dataset<double>(TIF_PATH_1);
    ds.LoadRasterBand(1);
    ASSERT_EQ(100, ds.GetRasterSizeX());
    ASSERT_EQ(100, ds.GetRasterSizeY());
    ASSERT_EQ(ds.GetRasterSizeX() * ds.GetRasterSizeY(), ds.GetHostDataBuffer().size())
        << "Loaded band 1 buffer does not contain exact data from dataset.";
}

TEST_F(DatasetTest, returnsCorrectCoordinatesForEdgeIndexes) {
    auto ds = alus::Dataset<double>(TIF_PATH_1);
    ds.LoadRasterBand(1);
    auto const zero = ds.GetPixelCoordinatesFromIndex(0, 0);
    auto const zero_one = ds.GetPixelCoordinatesFromIndex(0, 99);
    auto const one_zero = ds.GetPixelCoordinatesFromIndex(99, 0);
    auto const one_one = ds.GetPixelCoordinatesFromIndex(99, 99);
    EXPECT_DOUBLE_EQ(22.2362770, std::get<0>(zero));  // GetOriginLon()
    EXPECT_DOUBLE_EQ(58.3731210, std::get<1>(zero));  // GetOriginLat()
    EXPECT_DOUBLE_EQ(22.238828045110708, std::get<0>(one_zero));
    EXPECT_DOUBLE_EQ(58.360151848948476, std::get<1>(zero_one));
    EXPECT_DOUBLE_EQ(22.238828045110708, std::get<0>(one_one));
    EXPECT_DOUBLE_EQ(58.360151848948476, std::get<1>(one_one));
}

TEST_F(DatasetTest, returnsCorrectIndexesForCoordinates) {
    auto ds = alus::Dataset<double>(TIF_PATH_1);
    ds.LoadRasterBand(1);

    auto const zero = ds.GetPixelIndexFromCoordinates(22.236277, 58.373121);
    EXPECT_EQ(0, std::get<0>(zero));
    EXPECT_EQ(0, std::get<1>(zero));
    auto const lon0_lat99 = ds.GetPixelIndexFromCoordinates(22.236277, 58.360151);
    EXPECT_EQ(0, std::get<0>(lon0_lat99));
    EXPECT_EQ(99, std::get<1>(lon0_lat99));
    auto const lon99_lat0 = ds.GetPixelIndexFromCoordinates(22.238828045110708, 58.373121);
    EXPECT_EQ(99, std::get<0>(lon99_lat0));
    EXPECT_EQ(0, std::get<1>(lon99_lat0));
    auto const lon99_lat99 = ds.GetPixelIndexFromCoordinates(22.238828045110708, 58.360151);
    EXPECT_EQ(99, std::get<0>(lon99_lat99));
    EXPECT_EQ(99, std::get<1>(lon99_lat99));
    auto const lon50_lat50 = ds.GetPixelIndexFromCoordinates(22.2375654, 58.3665709);  // Values from gdalinfo.
    EXPECT_EQ(49, std::get<0>(lon50_lat50));
    EXPECT_EQ(50, std::get<1>(lon50_lat50));
}

TEST_F(DatasetTest, createsTargetDataset) {
    auto ds = alus::Dataset<double>(TIF_PATH_1);
    ds.LoadRasterBand(1);
    std::vector<double> from(ds.GetRasterSizeY() * ds.GetRasterSizeX());
    {
        alus::TargetDatasetParams params;
        params.filename = "/tmp/test.tif";
        params.band_count = 1;
        params.driver = ds.GetGdalDataset()->GetDriver();
        params.dimension = ds.GetRasterDimensions();
        params.transform = ds.GetTransform();
        params.projectionRef = ds.GetGdalDataset()->GetProjectionRef();
        auto tgt = alus::TargetDataset<double>(params);
        ASSERT_EQ(tgt.GetSize(), from.size());
        std::fill(from.begin(), from.end(), 15.6734);  // NOLINT
        tgt.WriteRectangle(from.data(), {0, 0, ds.GetRasterSizeX(), ds.GetRasterSizeY()}, 1);
    }

    auto check_ds = alus::Dataset<double>("/tmp/test.tif");
    check_ds.LoadRasterBand(1);
    auto const& check_data = check_ds.GetHostDataBuffer();
    ASSERT_EQ(check_data.size(), from.size());
    ASSERT_TRUE(std::equal(check_data.begin(), check_data.end(), from.begin()));
}

TEST_F(DatasetTest, throwsWhenWritingInvalidSizes) {
    auto ds = alus::Dataset<double>(TIF_PATH_1);
    ds.LoadRasterBand(1);

    alus::TargetDatasetParams params;
    params.filename = "/tmp/test.tif";
    params.band_count = 1;
    params.driver = ds.GetGdalDataset()->GetDriver();
    params.dimension = ds.GetRasterDimensions();
    params.transform = ds.GetTransform();
    params.projectionRef = ds.GetGdalDataset()->GetProjectionRef();
    auto tgt = alus::TargetDataset<double>(params);
    auto dims = tgt.GetDimensions();
    dims = dims + 1;

    std::vector<double> dummy_size_small(5);  // NOLINT
    ASSERT_THROW(tgt.WriteRectangle(dummy_size_small.data(), {0, 0, 0, 0}, 1), std::invalid_argument);
    std::vector<double> dummy_full_size(tgt.GetSize());
    ASSERT_THROW(tgt.WriteRectangle(dummy_full_size.data(), {1, 1, 0, 0}, 1), std::invalid_argument);
}

TEST_F(DatasetTest, writesToTargetDatasetWithOffsets) {
    alus::RasterDimension dim{};
    std::vector<double> fill1;
    std::vector<double> fill2;
    alus::RasterDimension fill_dim{10, 1};  // NOLINT
    {
        auto ds = alus::Dataset<double>(TIF_PATH_1);
        alus::TargetDatasetParams params;
        params.filename = "/tmp/test.tif";
        params.band_count = 1;
        params.driver = ds.GetGdalDataset()->GetDriver();
        params.dimension = ds.GetRasterDimensions();
        params.transform = ds.GetTransform();
        params.projectionRef = ds.GetGdalDataset()->GetProjectionRef();
        auto tgt = alus::TargetDataset<double>(params);

        fill1.resize(fill_dim.GetSize());
        fill2.resize(fill_dim.GetSize());
        std::fill(fill1.begin(), fill1.end(), 34.98);  // NOLINT
        std::fill(fill2.begin(), fill2.end(), 1.075);  // NOLINT

        dim = tgt.GetDimensions();
        tgt.WriteRectangle(fill1.data(), {0, 0, fill_dim.columnsX, fill_dim.rowsY}, 1);
        tgt.WriteRectangle(fill2.data(), {0, 50, fill_dim.columnsX, fill_dim.rowsY}, 1);  // NOLINT
    }

    auto check = alus::Dataset<double>("/tmp/test.tif");
    check.LoadRasterBand(1);
    ASSERT_EQ(check.GetRasterDimensions(), dim);
    auto check_data = check.GetHostDataBuffer();
    ASSERT_EQ(check_data.size(), dim.GetSize());
    EXPECT_TRUE(std::equal(check_data.begin(), check_data.begin() + fill_dim.GetSize(), fill1.begin()));
    EXPECT_TRUE(
        std::equal(check_data.begin() + 50 * 100, check_data.begin() + 50 * 100 + fill_dim.GetSize(), fill2.begin()));
}
}  // namespace
