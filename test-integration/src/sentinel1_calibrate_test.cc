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
#include "sentinel1_calibrate.h"
#include "gmock/gmock.h"
#include "sentinel1_calibrate_kernel.h"

#include <cstdio>
#include <memory>

#include <boost/filesystem.hpp>

#include "sentinel1_product_reader_plug_in.h"
#include "test_utils.h"
#include "topsar_split.h"

namespace {

using alus::GeoTiffWriteFile;
using alus::s1tbx::Sentinel1ProductReaderPlugIn;
using alus::sentinel1calibrate::ComplexIntensityData;
using alus::sentinel1calibrate::Sentinel1Calibrator;
using alus::utils::test::HashFromBand;

class Sentinel1CalibrateTest : public ::testing::Test {
protected:
    boost::filesystem::path input_file_{
        "./goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE"};

    const std::shared_ptr<Sentinel1ProductReaderPlugIn> reader_plug_in_ =
        std::make_shared<Sentinel1ProductReaderPlugIn>();
};

GDALDataset* CreateIntensityDataset(GDALDataset* src) {
    const int w = src->GetRasterXSize();
    const int h = src->GetRasterYSize();
    auto* float_ds = alus::GetGdalMemDriver()->Create("dummy", w, h, 1, GDT_Float32, nullptr);
    std::vector<ComplexIntensityData> buffer(w * h);
    CHECK_GDAL_ERROR(src->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, w, h, buffer.data(), w, h, GDT_CInt16, 0, 0));
    for (auto& el : buffer) {
        float i = el.iq16.i;
        float q = el.iq16.q;
        el.float32 = i * i + q * q;
    }
    CHECK_GDAL_ERROR(
        float_ds->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, w, h, buffer.data(), w, h, GDT_Float32, 0, 0));
    return float_ds;
}

TEST_F(Sentinel1CalibrateTest, Virumaa) {
    // Scope for forcing destruction of Sentinel1Calibrator

    const std::string result_file{
        "/tmp/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_Cal_IW1.tif"};
    const std::string result_file_float{
        "/tmp/alus_cal_float/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_Cal_IW1.tif"};
    std::remove(result_file.data());
    {
        ASSERT_THAT(boost::filesystem::exists(input_file_), ::testing::IsTrue());

        alus::topsarsplit::TopsarSplit split_op(input_file_.string(), "IW1", "VV");
        split_op.Initialize();
        split_op.OpenPixelReader(input_file_.string());
        auto split_product = split_op.GetTargetProduct();
        auto* pixel_reader = split_op.GetPixelReader()->GetDataset()->GetGdalDataset();

        {
            Sentinel1Calibrator calibrator{split_product, pixel_reader, {"IW1"}, {"VV"}, {true, false, false, false},
                                           "/tmp/",       false,        2000,    2000};  // NOLINT
            calibrator.Execute();

            const auto outputs = calibrator.GetOutputDatasets();
            for (const auto& output : outputs) {
                GeoTiffWriteFile(output.second.get(), output.first);
            }
        }

        {
            boost::filesystem::create_directory("/tmp/alus_cal_float/");
            GDALDataset* float_ds = CreateIntensityDataset(pixel_reader);
            Sentinel1Calibrator calibrator{
                split_product,          float_ds, {"IW1"}, {"VV"}, {true, false, false, false},
                "/tmp/alus_cal_float/", false,    2000,    2000};  // NOLINT
            calibrator.Execute();

            GDALClose(float_ds);
            const auto outputs = calibrator.GetOutputDatasets();
            for (const auto& output : outputs) {
                GeoTiffWriteFile(output.second.get(), output.first);
            }
        }
    }

    ASSERT_THAT(boost::filesystem::exists(result_file), ::testing::IsTrue());
    const std::string expected_md5{"d389aa9b7cefb448"};
    ASSERT_THAT(HashFromBand(result_file), ::testing::Eq(expected_md5));

    ASSERT_THAT(boost::filesystem::exists(result_file_float), ::testing::IsTrue());
    const std::string expected_md5_float{"1de26d3bdf680ef5"};
    ASSERT_THAT(HashFromBand(result_file_float), ::testing::Eq(expected_md5_float));

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaDeviceReset());  // for cuda-memcheck --leak-check full
}
}  // namespace
