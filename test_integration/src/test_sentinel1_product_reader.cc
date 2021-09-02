/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.sentinel1.TestSentinel1ProductReader.java
 * ported for native code.
 * Copied from(https://github.com/senbox-org/s1tbx). It was originally stated:
 *
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
#include <memory>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "gtest/gtest.h"

#include "s1tbx-commons/test/reader_test.h"

namespace {

/**
 * Test Product Reader.
 *
 * original snap java version author lveci
 */
class TestSentinel1ProductReader : public ::testing::Test, public alus::s1tbx::ReaderTest {
protected:
    boost::filesystem::path input_s1_safe_{
        "./goods/beirut_images/S1A_IW_SLC__1SDV_20200805T034334_20200805T034401_033766_03E9F9_52F6_thin.SAFE"};
    boost::filesystem::path input_s1_safe_zip_{
        "./goods/beirut_images/S1B_IW_SLC__1SDV_20200730T034254_20200730T034321_022695_02B131_E8DD_thin.SAFE.zip"};

public:
    TestSentinel1ProductReader() : ReaderTest(std::make_shared<alus::s1tbx::Sentinel1ProductReaderPlugIn>()) {}
};

TEST_F(TestSentinel1ProductReader, testOpeningFile) {
    {  // ARTIFICAL SCOPE TO FORCE DESTRUCTORS
        ASSERT_TRUE(boost::filesystem::exists(input_s1_safe_));

        std::shared_ptr<alus::snapengine::Product> prod =
            TestReader(boost::filesystem::canonical("manifest.safe", input_s1_safe_));

        ValidateProduct(prod);
        ValidateMetadata(prod);
        ValidateBands(prod, std::vector<std::string>{"i_IW1_VV", "q_IW1_VV", "Intensity_IW1_VV", "i_IW2_VH", "q_IW2_VH",
                                                     "Intensity_IW2_VH"});
    }
}

TEST_F(TestSentinel1ProductReader, testOpeningZip) {
    {  // ARTIFICAL SCOPE TO FORCE DESTRUCTORS
        ASSERT_TRUE(boost::filesystem::exists(input_s1_safe_zip_));

        std::shared_ptr<alus::snapengine::Product> prod = TestReader(boost::filesystem::canonical(input_s1_safe_zip_));

        ValidateProduct(prod);
        ValidateMetadata(prod);
        ValidateBands(prod, std::vector<std::string>{"i_IW1_VV", "q_IW1_VV", "Intensity_IW1_VV", "i_IW2_VH", "q_IW2_VH",
                                                     "Intensity_IW2_VH"});
    }
}

}  // namespace
