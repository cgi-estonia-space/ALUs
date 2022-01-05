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
#include <string_view>
#include <vector>

#include "gmock/gmock.h"

#include "zip_util.h"

namespace {

using namespace alus;

class ZipTest : public ::testing::Test {
protected:
    static constexpr std::string_view ARCHIVE_PATH{"./goods/srtm_41_01.zip"};
};

TEST_F(ZipTest, GetArchiveContentsTest) {
    const std::vector<std::string> expected_archive_contents{"readme.txt", "srtm_41_01.hdr", "srtm_41_01.tfw",
                                                             "srtm_41_01.tif"};
    const auto archive_contents = common::zip::GetZipContents(ARCHIVE_PATH);

    ASSERT_THAT(archive_contents, ::testing::ElementsAreArray(expected_archive_contents));
}
}  // namespace
