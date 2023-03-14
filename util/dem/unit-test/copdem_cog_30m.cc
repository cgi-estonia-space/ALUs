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

#include "copdem_cog_30m.h"

namespace {

using ::testing::Eq;

TEST(CopDemCog30m, ComputeIdReturnsCorrectResult) {
    // Bottom left points.
    ASSERT_THAT(alus::dem::CopDemCog30m::ComputeId(-180, -90), Eq(180));
    ASSERT_THAT(alus::dem::CopDemCog30m::ComputeId(179, 89), Eq(359 * 1000 + 1));
    ASSERT_THAT(alus::dem::CopDemCog30m::ComputeId(4.0, 49.0), Eq((180 + 4) * 1000 + 41));
}

}  // namespace