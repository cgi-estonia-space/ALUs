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

#include <memory>

#include <boost/filesystem.hpp>

#include "i_product_reader.h"
#include "product.h"
#include "sentinel1_product_reader_plug_in.h"
#include "test_utils.h"

namespace {

using namespace alus;
using namespace alus::sentinel1calibrate;

class Sentinel1CalibrateTest : public ::testing::Test {
protected:
    boost::filesystem::path input_file_{
        "./goods/sentinel1_calibrate/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563.SAFE"};

    const std::shared_ptr<s1tbx::Sentinel1ProductReaderPlugIn> reader_plug_in_ =
        std::make_shared<s1tbx::Sentinel1ProductReaderPlugIn>();
};

TEST_F(Sentinel1CalibrateTest, Virumaa) {
    ASSERT_THAT(boost::filesystem::exists(input_file_), ::testing::IsTrue());
    const std::shared_ptr<snapengine::IProductReader> product_reader = reader_plug_in_->CreateReaderInstance();
    std::shared_ptr<snapengine::Product> input_product =
        product_reader->ReadProductNodes(boost::filesystem::canonical(input_file_), nullptr);
    const auto source_dataset = std::make_shared<Dataset<float>>(boost::filesystem::canonical(input_file_).string());

    Sentinel1Calibrator calibrator{
        input_product, source_dataset, {"IW1"}, {"VV"}, {false, true, false, false}, "/tmp/", false, 5000, 5000,
    };
    calibrator.Execute();
    const std::string result_file{
        "/tmp/S1A_IW_SLC__1SDV_20180815T154813_20180815T154840_023259_028747_4563_Cal_IW1.tif"};
    ASSERT_THAT(boost::filesystem::exists(result_file), ::testing::IsTrue());
    const std::string expected_md5{"c48e18d4ab4c9760"};
    ASSERT_THAT(utils::test::HashFromBand(result_file), ::testing::Eq(expected_md5));
}

}  // namespace
