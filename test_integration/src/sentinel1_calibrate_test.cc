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

#include <cstdio>
#include <memory>

#include <boost/filesystem.hpp>

#include "i_product_reader.h"
#include "product.h"
#include "sentinel1_product_reader_plug_in.h"
#include "test_utils.h"
#include "topsar_split.h"

namespace {

using namespace alus;
using namespace alus::sentinel1calibrate;

class Sentinel1CalibrateTest : public ::testing::Test {
protected:
    boost::filesystem::path input_file_{
        "./goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_thin.SAFE"};

    const std::shared_ptr<s1tbx::Sentinel1ProductReaderPlugIn> reader_plug_in_ =
        std::make_shared<s1tbx::Sentinel1ProductReaderPlugIn>();
};

TEST_F(Sentinel1CalibrateTest, Virumaa) {
    // Scope for forcing destruction of Sentinel1Calibrator

    const std::string result_file{
        "/tmp/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_THIN_Cal_IW1.tif"};
    std::remove(result_file.data());
    {
        ASSERT_THAT(boost::filesystem::exists(input_file_), ::testing::IsTrue());

        alus::topsarsplit::TopsarSplit split_op(input_file_.string(), "IW1", "VV");
        split_op.initialize();
        auto split_product = split_op.GetTargetProduct();
        Dataset<Iq16>* pixel_reader = split_op.GetPixelReader()->GetDataset();


        Sentinel1Calibrator calibrator{split_product, pixel_reader, {"IW1"}, {"VV"}, {true, false, false, false},
                                       "/tmp/",       false,       2000,    2000};
        calibrator.Execute();

        const auto outputs = calibrator.GetOutputDatasets();
        for (const auto& output : outputs) {
            GeoTiffWriteFile(output.second.get(), output.first);
        }
    }

    ASSERT_THAT(boost::filesystem::exists(result_file), ::testing::IsTrue());
    const std::string expected_md5{"d389aa9b7cefb448"};
    ASSERT_THAT(utils::test::HashFromBand(result_file), ::testing::Eq(expected_md5));
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaDeviceReset());  // for cuda-memcheck --leak-check full
}
}  // namespace
