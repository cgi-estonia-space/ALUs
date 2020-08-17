#include "dataset.hpp"
#include "target_dataset.hpp"

#include "gmock/gmock.h"

#include "gdal_util.hpp"
#include "tests_common.hpp"

using namespace alus::tests;
using ::testing::ContainerEq;
using ::testing::ElementsAre;
using ::testing::SizeIs;

namespace {

class DatasetTest : public ::testing::Test {
   public:
    DatasetTest() { CPLPushErrorHandler(silentGdalErrorHandler); }

    ~DatasetTest() override { CPLPopErrorHandler(); }
};

TEST_F(DatasetTest, onInvalidFilenameThrows) {
    std::string f{"non_existent_unittest_file"};
    ASSERT_THROW(alus::Dataset({f}), alus::DatasetError);
}

TEST_F(DatasetTest, loadsValidTifFile) {
    auto ds = alus::Dataset(TIF_PATH_1);
    ds.LoadRasterBand(1);
    ASSERT_EQ(100, ds.GetXSize());
    ASSERT_EQ(100, ds.GetYSize());
    ASSERT_EQ(ds.GetXSize() * ds.GetYSize(), ds.GetDataBuffer().size())
        << "Loaded band 1 buffer does not contain exact data from dataset.";
}

TEST_F(DatasetTest, returnsCorrectCoordinatesForEdgeIndexes) {
    auto ds = alus::Dataset(TIF_PATH_1);
    ds.LoadRasterBand(1);
    auto const zero = ds.GetPixelCoordinatesFromIndex(0, 0);
    auto const zeroOne = ds.GetPixelCoordinatesFromIndex(0, 99);
    auto const oneZero = ds.GetPixelCoordinatesFromIndex(99, 0);
    auto const oneOne = ds.GetPixelCoordinatesFromIndex(99, 99);
    EXPECT_DOUBLE_EQ(22.2362770, std::get<0>(zero));  // GetOriginLon()
    EXPECT_DOUBLE_EQ(58.3731210, std::get<1>(zero));  // GetOriginLat()
    EXPECT_DOUBLE_EQ(22.238828045110708, std::get<0>(oneZero));
    EXPECT_DOUBLE_EQ(58.360151848948476, std::get<1>(zeroOne));
    EXPECT_DOUBLE_EQ(22.238828045110708, std::get<0>(oneOne));
    EXPECT_DOUBLE_EQ(58.360151848948476, std::get<1>(oneOne));
}

TEST_F(DatasetTest, returnsCorrectIndexesForCoordinates) {
    auto ds = alus::Dataset(TIF_PATH_1);
    ds.LoadRasterBand(1);

    auto const zero = ds.GetPixelIndexFromCoordinates(22.236277, 58.373121);
    EXPECT_EQ(0, std::get<0>(zero));
    EXPECT_EQ(0, std::get<1>(zero));
    auto const lon0Lat99 = ds.GetPixelIndexFromCoordinates(22.236277, 58.360151);
    EXPECT_EQ(0, std::get<0>(lon0Lat99));
    EXPECT_EQ(99, std::get<1>(lon0Lat99));
    auto const lon99Lat0 = ds.GetPixelIndexFromCoordinates(22.238828045110708, 58.373121);
    EXPECT_EQ(99, std::get<0>(lon99Lat0));
    EXPECT_EQ(0, std::get<1>(lon99Lat0));
    auto const lon99Lat99 = ds.GetPixelIndexFromCoordinates(22.238828045110708, 58.360151);
    EXPECT_EQ(99, std::get<0>(lon99Lat99));
    EXPECT_EQ(99, std::get<1>(lon99Lat99));
    auto const lon50lat50 = ds.GetPixelIndexFromCoordinates(22.2375654, 58.3665709);  // Values from gdalinfo.
    EXPECT_EQ(49, std::get<0>(lon50lat50));
    EXPECT_EQ(50, std::get<1>(lon50lat50));
}

TEST_F(DatasetTest, createsTargetDataset) {
    auto ds = alus::Dataset(TIF_PATH_1);
    ds.LoadRasterBand(1);
    std::vector<double> from(ds.GetRasterSizeY() * ds.GetRasterSizeX());
    {
        auto tgt = alus::TargetDataset(ds, "/tmp/test.tif");
        ASSERT_EQ(tgt.getSize(), from.size());
        std::fill(from.begin(), from.end(), 15.6734);
        tgt.write(from);
    }

    auto checkDs = alus::Dataset("/tmp/test.tif");
    checkDs.LoadRasterBand(1);
    auto const& checkData = checkDs.GetDataBuffer();
    ASSERT_EQ(checkData.size(), from.size());
    ASSERT_TRUE(std::equal(checkData.begin(), checkData.end(), from.begin()));
}

TEST_F(DatasetTest, throwsWhenWritingInvalidSizes) {
    auto ds = alus::Dataset(TIF_PATH_1);
    ds.LoadRasterBand(1);
    auto tgt = alus::TargetDataset(ds, "/tmp/test.tif");
    auto const dims = tgt.getDimensions();

    std::vector<double> dummySizeSmall(5);
    ASSERT_THROW(tgt.write(dummySizeSmall), std::invalid_argument);
    std::vector<double> dummyFullSize(tgt.getSize());
    ASSERT_THROW(tgt.write(dummyFullSize, {1, 1}), alus::GdalErrorException);
    ASSERT_THROW(tgt.write(dummyFullSize, {0, 0}, dims + 1), std::invalid_argument);
}

TEST_F(DatasetTest, writesToTargetDatasetWithOffsets) {
    alus::RasterDimension dim{};
    std::vector<double> fill1;
    std::vector<double> fill2;
    alus::RasterDimension fillDim{10, 1};
    {
        auto ds = alus::Dataset(TIF_PATH_1);
        auto tgt = alus::TargetDataset(ds, "/tmp/test.tif");

        fill1.resize(fillDim.getSize());
        fill2.resize(fillDim.getSize());
        std::fill(fill1.begin(), fill1.end(), 34.98);
        std::fill(fill2.begin(), fill2.end(), 1.075);

        dim = tgt.getDimensions();
        tgt.write(fill1, {0, 0}, fillDim);
        tgt.write(fill2, {0, 50}, fillDim);
    }

    auto check = alus::Dataset("/tmp/test.tif");
    check.LoadRasterBand(1);
    ASSERT_EQ(check.GetRasterDimensions(), dim);
    auto checkData = check.GetDataBuffer();
    ASSERT_EQ(checkData.size(), dim.getSize());
    EXPECT_TRUE(std::equal(checkData.begin(), checkData.begin() + fillDim.getSize(), fill1.begin()));
    EXPECT_TRUE(
        std::equal(checkData.begin() + 50 * 100, checkData.begin() + 50 * 100 + fillDim.getSize(), fill2.begin()));
}
}  // namespace
