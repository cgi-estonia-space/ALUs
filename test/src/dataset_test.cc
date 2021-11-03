#include "dataset.h"
#include "target_dataset.h"

#include "gmock/gmock.h"

#include "gdal_management.h"
#include "tests_common.h"

using namespace alus::tests;
using ::testing::ContainerEq;
using ::testing::ElementsAre;
using ::testing::SizeIs;

namespace {

class DatasetTest : public ::testing::Test {
public:
    DatasetTest() : gdalErrorHandleGuard_{alus::gdalmanagement::SetErrorHandle(silentGdalErrorHandle_)} {}

    alus::gdalmanagement::ErrorCallback silentGdalErrorHandle_ = [](std::string_view) {};
    alus::gdalmanagement::ErrorCallbackGuard gdalErrorHandleGuard_;

    ~DatasetTest() override { }
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
    auto ds = alus::Dataset<double>(TIF_PATH_1);
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
        ASSERT_EQ(tgt.getSize(), from.size());
        std::fill(from.begin(), from.end(), 15.6734);
        tgt.WriteRectangle(from.data(), {0, 0, ds.GetRasterSizeX(), ds.GetRasterSizeY()}, 1);
    }

    auto checkDs = alus::Dataset<double>("/tmp/test.tif");
    checkDs.LoadRasterBand(1);
    auto const& checkData = checkDs.GetHostDataBuffer();
    ASSERT_EQ(checkData.size(), from.size());
    ASSERT_TRUE(std::equal(checkData.begin(), checkData.end(), from.begin()));
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
    auto dims = tgt.getDimensions();
    dims = dims + 1;

    std::vector<double> dummySizeSmall(5);
    ASSERT_THROW(tgt.WriteRectangle(dummySizeSmall.data(), {0, 0, 0, 0}, 1), std::invalid_argument);
    std::vector<double> dummyFullSize(tgt.getSize());
    ASSERT_THROW(tgt.WriteRectangle(dummyFullSize.data(), {1, 1, 0, 0}, 1), std::invalid_argument);
}

TEST_F(DatasetTest, writesToTargetDatasetWithOffsets) {
    alus::RasterDimension dim{};
    std::vector<double> fill1;
    std::vector<double> fill2;
    alus::RasterDimension fillDim{10, 1};
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

        fill1.resize(fillDim.getSize());
        fill2.resize(fillDim.getSize());
        std::fill(fill1.begin(), fill1.end(), 34.98);
        std::fill(fill2.begin(), fill2.end(), 1.075);

        dim = tgt.getDimensions();
        tgt.WriteRectangle(fill1.data(), {0, 0, fillDim.columnsX, fillDim.rowsY}, 1);
        tgt.WriteRectangle(fill2.data(), {0, 50, fillDim.columnsX, fillDim.rowsY}, 1);
    }

    auto check = alus::Dataset<double>("/tmp/test.tif");
    check.LoadRasterBand(1);
    ASSERT_EQ(check.GetRasterDimensions(), dim);
    auto checkData = check.GetHostDataBuffer();
    ASSERT_EQ(checkData.size(), dim.getSize());
    EXPECT_TRUE(std::equal(checkData.begin(), checkData.begin() + fillDim.getSize(), fill1.begin()));
    EXPECT_TRUE(
        std::equal(checkData.begin() + 50 * 100, checkData.begin() + 50 * 100 + fillDim.getSize(), fill2.begin()));
}
}  // namespace
