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

#include <string>

#include "product_name.h"

namespace {

const std::string DIR_THAT_IS_ALWAYS_AVAILABLE{"/tmp"};

using ::testing::Eq;
using ::testing::IsFalse;
using ::testing::IsTrue;

TEST(ProductName, ConstructingWithFinalPathDoesReturnInitialFilename) {
    const auto p1 = alus::common::ProductName(DIR_THAT_IS_ALWAYS_AVAILABLE + "/filename");
    ASSERT_THAT(p1.IsFinal(), IsTrue());
    EXPECT_THAT(p1.Construct(), Eq(DIR_THAT_IS_ALWAYS_AVAILABLE + "/filename"));
}

TEST(ProductName, IsFinalReturnsFalseWhenConstructingWithDirectory) {
    const auto p1 = alus::common::ProductName(DIR_THAT_IS_ALWAYS_AVAILABLE);
    EXPECT_THAT(p1.IsFinal(), IsFalse());
}

TEST(ProductName, AddDoesConstructCorrectFilename) {
    {
        auto p1 = alus::common::ProductName(DIR_THAT_IS_ALWAYS_AVAILABLE);
        p1.Add("S1A");
        p1.Add("IW");
        p1.Add("SLC");
        EXPECT_THAT(p1.Construct(), Eq(DIR_THAT_IS_ALWAYS_AVAILABLE + "/" + "S1A_IW_SLC"));
    }

    {
        auto p1 = alus::common::ProductName(DIR_THAT_IS_ALWAYS_AVAILABLE, '-');
        p1.Add("S1A");
        p1.Add("IW");
        p1.Add("SLC");
        EXPECT_THAT(p1.Construct(), Eq(DIR_THAT_IS_ALWAYS_AVAILABLE + "/" + "S1A-IW-SLC"));
    }
}

TEST(ProductName, ConstructingWithExtension) {
    auto p1 = alus::common::ProductName(DIR_THAT_IS_ALWAYS_AVAILABLE);
    p1.Add("S2B_JP_LLR");
    EXPECT_THAT(p1.Construct(".jp2"), Eq(DIR_THAT_IS_ALWAYS_AVAILABLE + "/" + "S2B_JP_LLR.jp2"));
}

}  // namespace
