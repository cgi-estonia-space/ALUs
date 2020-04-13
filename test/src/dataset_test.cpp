#include "dataset.hpp"
#include "target_dataset.hpp"

#include "gmock/gmock.h"

#include "gdal_util.hpp"
#include "tests_common.hpp"

using namespace slap::tests;
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
    ASSERT_THROW(slap::Dataset({f}), slap::DatasetError);
}

TEST_F(DatasetTest, loadsValidTifFile) {
    auto ds = slap::Dataset(TIF_PATH_1);
    ASSERT_EQ(100, ds.getBand1Xsize());
    ASSERT_EQ(100, ds.getBand1Ysize());
    ASSERT_EQ(ds.getBand1Xsize() * ds.getBand1Ysize(), ds.getBand1Data().size())
        << "Loaded band 1 buffer does not contain exact data from dataset.";
}

TEST_F(DatasetTest, returnsCorrectCoordinatesForEdgeIndexes) {
    auto ds = slap::Dataset(TIF_PATH_1);
    auto const zero = ds.getPixelCoordinatesFromIndex(0, 0);
    auto const zeroOne = ds.getPixelCoordinatesFromIndex(0, 99);
    auto const oneZero = ds.getPixelCoordinatesFromIndex(99, 0);
    auto const oneOne = ds.getPixelCoordinatesFromIndex(99, 99);
    EXPECT_DOUBLE_EQ(22.2362770, std::get<0>(zero));  // getOriginLon()
    EXPECT_DOUBLE_EQ(58.3731210, std::get<1>(zero));  // getOriginLat()
    EXPECT_DOUBLE_EQ(22.238828045110708, std::get<0>(oneZero));
    EXPECT_DOUBLE_EQ(58.360151848948476, std::get<1>(zeroOne));
    EXPECT_DOUBLE_EQ(22.238828045110708, std::get<0>(oneOne));
    EXPECT_DOUBLE_EQ(58.360151848948476, std::get<1>(oneOne));
}

TEST_F(DatasetTest, returnsCorrectIndexesForCoordinates) {
    auto ds = slap::Dataset(TIF_PATH_1);

    auto const zero = ds.getPixelIndexFromCoordinates(22.236277, 58.373121);
    EXPECT_EQ(0, std::get<0>(zero));
    EXPECT_EQ(0, std::get<1>(zero));
    auto const lon0Lat99 = ds.getPixelIndexFromCoordinates(22.236277, 58.360151);
    EXPECT_EQ(0, std::get<0>(lon0Lat99));
    EXPECT_EQ(99, std::get<1>(lon0Lat99));
    auto const lon99Lat0 = ds.getPixelIndexFromCoordinates(22.238828045110708, 58.373121);
    EXPECT_EQ(99, std::get<0>(lon99Lat0));
    EXPECT_EQ(0, std::get<1>(lon99Lat0));
    auto const lon99Lat99 = ds.getPixelIndexFromCoordinates(22.238828045110708, 58.360151);
    EXPECT_EQ(99, std::get<0>(lon99Lat99));
    EXPECT_EQ(99, std::get<1>(lon99Lat99));
    auto const lon50lat50 = ds.getPixelIndexFromCoordinates(22.2375654, 58.3665709);  // Values from gdalinfo.
    EXPECT_EQ(49, std::get<0>(lon50lat50));
    EXPECT_EQ(50, std::get<1>(lon50lat50));
}

TEST_F(DatasetTest, createsTargetDataset) {
    auto ds = slap::Dataset(TIF_PATH_1);
    std::vector<double> from(ds.getRasterSizeY() * ds.getRasterSizeX());
    {
        auto tgt = slap::TargetDataset(ds, "/tmp/test.tif");
        ASSERT_EQ(tgt.getSize(), from.size());
        std::fill(from.begin(), from.end(), 15.6734);
        tgt.write(from);
    }

    auto checkDs = slap::Dataset("/tmp/test.tif");
    auto const& checkData = checkDs.getBand1Data();
    ASSERT_EQ(checkData.size(), from.size());
    ASSERT_TRUE(std::equal(checkData.begin(), checkData.end(), from.begin()));
}

TEST_F(DatasetTest, throwsWhenWritingInvalidSizes) {
    auto ds = slap::Dataset(TIF_PATH_1);
    auto tgt = slap::TargetDataset(ds, "/tmp/test.tif");
    auto const dims = tgt.getDimensions();

    std::vector<double> dummySizeSmall(5);
    ASSERT_THROW(tgt.write(dummySizeSmall), std::invalid_argument);
    std::vector<double> dummyFullSize(tgt.getSize());
    ASSERT_THROW(tgt.write(dummyFullSize, {1, 1}), slap::GdalErrorException);
    ASSERT_THROW(tgt.write(dummyFullSize, {0, 0}, dims + 1), std::invalid_argument);
}

TEST_F(DatasetTest, writesToTargetDatasetWithOffsets) {
    slap::RasterDimension dim{};
    std::vector<double> fill1;
    std::vector<double> fill2;
    slap::RasterDimension fillDim{10, 1};
    {
        auto ds = slap::Dataset(TIF_PATH_1);
        auto tgt = slap::TargetDataset(ds, "/tmp/test.tif");

        fill1.resize(fillDim.getSize());
        fill2.resize(fillDim.getSize());
        std::fill(fill1.begin(), fill1.end(), 34.98);
        std::fill(fill2.begin(), fill2.end(), 1.075);

        dim = tgt.getDimensions();
        tgt.write(fill1, {0, 0}, fillDim);
        tgt.write(fill2, {0, 50}, fillDim);
    }

    auto check = slap::Dataset("/tmp/test.tif");
    ASSERT_EQ(check.getRasterDimensions(), dim);
    auto checkData = check.getBand1Data();
    ASSERT_EQ(checkData.size(), dim.getSize());
    EXPECT_TRUE(std::equal(checkData.begin(), checkData.begin() + fillDim.getSize(), fill1.begin()));
    EXPECT_TRUE(
        std::equal(checkData.begin() + 50 * 100, checkData.begin() + 50 * 100 + fillDim.getSize(), fill2.begin()));
}
}  // namespace
