#include "dataset.hpp"

#include "gmock/gmock.h"

namespace {

void gdalErrorHandler(CPLErr, CPLErrorNum, const char*) {
    // Not doing anything, simply silencing GDAL error log in our console by
    // supplying dummy handler.
}

/**
 * Output from gdalinfo.
 * Upper Left  (  22.2362770,  58.3731210) ( 22d14'10.60"E, 58d22'23.24"N)
 * Lower Left  (  22.2362770,  58.3600208) ( 22d14'10.60"E, 58d21'36.08"N)
 * Upper Right (  22.2388538,  58.3731210) ( 22d14'19.87"E, 58d22'23.24"N)
 * Lower Right (  22.2388538,  58.3600208) ( 22d14'19.87"E, 58d21'36.08"N)
 * Center      (  22.2375654,  58.3665709) ( 22d14'15.24"E, 58d21'59.66"N)
 */
std::string const TEST_TIF{"./goods/karujarve_kallas.tif"};

class DatasetTest : public ::testing::Test {
   public:
    DatasetTest() {
        CPLPushErrorHandler(gdalErrorHandler);
    }

    ~DatasetTest() {
        CPLPopErrorHandler();
    }
};

TEST_F(DatasetTest, onInvalidFilenameThrows) {
    std::string f{"filename"};
    ASSERT_THROW(slap::Dataset({f}), slap::DatasetError);
}

TEST_F(DatasetTest, loadsValidTifFile) {
    auto ds = slap::Dataset(TEST_TIF);
    ASSERT_EQ(100, ds.getGDALDataset()->GetRasterXSize());
    ASSERT_EQ(100, ds.getGDALDataset()->GetRasterYSize());
}

TEST_F(DatasetTest, returnsCorrectCoordinatesForEdgeIndexes)
{
    auto ds = slap::Dataset(TEST_TIF);
    auto const zero = ds.getPixelCoordinatesFromIndex(0, 0);
    auto const zeroOne = ds.getPixelCoordinatesFromIndex(0, 99);
    auto const oneZero = ds.getPixelCoordinatesFromIndex(99, 0);
    auto const oneOne = ds.getPixelCoordinatesFromIndex(99, 99);
    EXPECT_DOUBLE_EQ(22.2362770, std::get<0>(zero)); // getOriginLon()
    EXPECT_DOUBLE_EQ(58.3731210, std::get<1>(zero)); // getOriginLat()
    EXPECT_DOUBLE_EQ(22.238828045110708, std::get<0>(oneZero));
    EXPECT_DOUBLE_EQ(58.360151848948476, std::get<1>(zeroOne));
    EXPECT_DOUBLE_EQ(22.238828045110708, std::get<0>(oneOne));
    EXPECT_DOUBLE_EQ(58.360151848948476, std::get<1>(oneOne));
}

TEST_F(DatasetTest, returnsCorrectIndexesForCoordinates)
{
     auto ds = slap::Dataset(TEST_TIF);

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
    auto const lon50lat50 = ds.getPixelIndexFromCoordinates(22.2375654, 58.3665709); // Values from gdalinfo.
    EXPECT_EQ(49, std::get<0>(lon50lat50));
    EXPECT_EQ(50, std::get<1>(lon50lat50));
}
}  // namespace
