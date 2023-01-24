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

#include <boost/filesystem/path.hpp>

#include "dataset.h"
#include "gdal_util.h"
#include "sentinel1_product_reader_plug_in.h"
#include "test_utils.h"
#include "thermal_noise_remover.h"
#include "topsar_split.h"

using alus::Dataset;
using alus::GeoTiffWriteFile;
using alus::Iq16;
using alus::s1tbx::Sentinel1ProductReaderPlugIn;
using alus::tnr::ThermalNoiseRemover;
using alus::utils::test::HashFromBand;

namespace {
class ThermalNoiseRemovalTest : public ::testing::Test {
protected:
    boost::filesystem::path input_file_{
        "./goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6.SAFE"};

    const std::shared_ptr<Sentinel1ProductReaderPlugIn> reader_plug_in_ =
        std::make_shared<Sentinel1ProductReaderPlugIn>();
};

TEST_F(ThermalNoiseRemovalTest, fullSubswathProcessing) {
    const std::string result_file{
        "/tmp/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_tnr_IW1.tif"};
    std::remove(result_file.data());

    {
        ASSERT_THAT(boost::filesystem::exists(input_file_), ::testing::IsTrue());

        alus::topsarsplit::TopsarSplit split_op(input_file_.string(), "IW1", "VV");
        split_op.Initialize();
        split_op.OpenPixelReader(input_file_.string());
        auto split_product = split_op.GetTargetProduct();
        auto* pixel_reader = split_op.GetPixelReader()->GetDataset();

        ThermalNoiseRemover tnr(split_product, pixel_reader, "IW1", "VV", "/tmp/", 6000, 6000);

        tnr.Execute();

        const auto output = tnr.GetOutputDataset();
        GeoTiffWriteFile(output.first.get(), output.second);
    }

    ASSERT_THAT(boost::filesystem::exists(result_file), ::testing::IsTrue());
    const std::string expected_md5{"6051889bfce84560"};
    ASSERT_THAT(HashFromBand(result_file), ::testing::Eq(expected_md5));
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaDeviceReset());  // for cuda-memcheck --leak-check full
}

TEST_F(ThermalNoiseRemovalTest, partialSubswathProcessing) {
    const std::string result_file{
        "/tmp/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_tnr_IW1.tif"};
    std::remove(result_file.data());

    {
        ASSERT_THAT(boost::filesystem::exists(input_file_), ::testing::IsTrue());

        alus::topsarsplit::TopsarSplit split_op(input_file_.string(), "IW1", "VH", 4, 6);
        split_op.Initialize();
        split_op.OpenPixelReader(input_file_.string());
        auto split_product = split_op.GetTargetProduct();
        auto* pixel_reader = split_op.GetPixelReader()->GetDataset();

        ThermalNoiseRemover tnr(split_product, pixel_reader, "IW1", "VH", "/tmp/", 6000, 6000);

        tnr.Execute();

        const auto output = tnr.GetOutputDataset();
        GeoTiffWriteFile(output.first.get(), output.second);
    }

    ASSERT_THAT(boost::filesystem::exists(result_file), ::testing::IsTrue());
    const std::string expected_md5{"3c1f54d83b9ea5e6"};
    ASSERT_THAT(HashFromBand(result_file), ::testing::Eq(expected_md5));
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaDeviceReset());  // for cuda-memcheck --leak-check full
}
}  // namespace