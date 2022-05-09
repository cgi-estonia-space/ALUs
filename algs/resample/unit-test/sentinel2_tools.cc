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

#include <array>
#include <string_view>

#include "gmock/gmock.h"

#include "sentinel2_tools.h"

namespace {

using ::testing::Eq;

TEST(Resample, ThrowsWhenNotSentinel2FormatProduct) {
    EXPECT_THROW(alus::resample::TryParseS2Fields(""), alus::resample::Sentinel2DatasetNameParseException);
    EXPECT_THROW(alus::resample::TryParseS2Fields("S2C_MSIL1C_20170105T013442_N0204_R031_T53NMJ_20170105T013443"),
                 alus::resample::Sentinel2DatasetNameParseException);
    EXPECT_THROW(alus::resample::TryParseS2Fields("S2A_MSIL1B_20170105T013442_N0204_R031_T53NMJ_20170105T013443"),
                 alus::resample::Sentinel2DatasetNameParseException);
    EXPECT_THROW(alus::resample::TryParseS2Fields("S2A_MSIL1C_20170105T013442_N0204_R031__20170105T013443"),
                 alus::resample::Sentinel2DatasetNameParseException);
}

TEST(Resample, ParsesCorrectlySentinel2FormatFieldsFromFilename) {
    constexpr std::array<std::string_view, 3> FILENAMES_S2A{
        "S2A_MSIL1C_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE",
        "S2A_MSIL1C_20170105T013442_N0204_R031_T53NMJ_20170105T013443",
        "S2A_MSIL1C_20170105T013442_N0204_R031_T53NMJ_20170105T013443.zip"};

    for (const auto& f : FILENAMES_S2A) {
        const auto& s2_fields = alus::resample::TryParseS2Fields(f);
        EXPECT_THAT(s2_fields.mission, Eq("S2A"));
        EXPECT_THAT(s2_fields.level, Eq("MSIL1C"));
        EXPECT_THAT(s2_fields.sensing_start, Eq("20170105T013442"));
        EXPECT_THAT(s2_fields.processing_baseline, Eq("N0204"));
        EXPECT_THAT(s2_fields.orbit, Eq("R031"));
        EXPECT_THAT(s2_fields.tile, Eq("T53NMJ"));
        EXPECT_THAT(s2_fields.discriminator, Eq("20170105T013443"));
    }

    constexpr std::array<std::string_view, 2> FILENAMES_S2B{
        "S2B_MSIL1C_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE",
        "S2B_MSIL1C_20170105T013442_N0204_R031_T53NMJ_20170105T013443"};

    for (const auto& f : FILENAMES_S2B) {
        const auto& s2_fields = alus::resample::TryParseS2Fields(f);
        EXPECT_THAT(s2_fields.mission, Eq("S2B"));
        EXPECT_THAT(s2_fields.level, Eq("MSIL1C"));
        EXPECT_THAT(s2_fields.sensing_start, Eq("20170105T013442"));
        EXPECT_THAT(s2_fields.processing_baseline, Eq("N0204"));
        EXPECT_THAT(s2_fields.orbit, Eq("R031"));
        EXPECT_THAT(s2_fields.tile, Eq("T53NMJ"));
        EXPECT_THAT(s2_fields.discriminator, Eq("20170105T013443"));
    }

    constexpr std::array<std::string_view, 2> FILENAMES_L2A{
        "S2A_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE",
        "S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.zip"};

    for (const auto& f : FILENAMES_L2A) {
        const auto& s2_fields = alus::resample::TryParseS2Fields(f);
        EXPECT_THAT(s2_fields.level, Eq("MSIL2A"));
        EXPECT_THAT(s2_fields.sensing_start, Eq("20170105T013442"));
        EXPECT_THAT(s2_fields.processing_baseline, Eq("N0204"));
        EXPECT_THAT(s2_fields.orbit, Eq("R031"));
        EXPECT_THAT(s2_fields.tile, Eq("T53NMJ"));
        EXPECT_THAT(s2_fields.discriminator, Eq("20170105T013443"));
    }
}

TEST(Resample, CreatesSentinel2GdalDatasetName) {
    EXPECT_THAT(alus::resample::TryCreateSentinel2GdalDatasetName(
                    "S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.zip"),
                Eq("S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.zip"));
    EXPECT_THAT(alus::resample::TryCreateSentinel2GdalDatasetName(
                    "/home/foo/S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.zip"),
                Eq("/home/foo/S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.zip"));
    EXPECT_THAT(alus::resample::TryCreateSentinel2GdalDatasetName(
                    "./S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.zip"),
                Eq("./S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.zip"));
    EXPECT_THAT(alus::resample::TryCreateSentinel2GdalDatasetName(
                    "S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE"),
                Eq("S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE/MTD_MSIL2A.xml"));
    EXPECT_THAT(alus::resample::TryCreateSentinel2GdalDatasetName(
                    "/home/foo/S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE"),
                Eq("/home/foo/S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE/MTD_MSIL2A.xml"));
    EXPECT_THAT(alus::resample::TryCreateSentinel2GdalDatasetName(
                    "S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE/MTD_MSIL2A.xml"),
                Eq("S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE/MTD_MSIL2A.xml"));
    EXPECT_THAT(alus::resample::TryCreateSentinel2GdalDatasetName(
                    "/home/foo/S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE/MTD_MSIL2A.xml"),
                Eq("/home/foo/S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443.SAFE/MTD_MSIL2A.xml"));
    EXPECT_THAT(alus::resample::TryCreateSentinel2GdalDatasetName(
                    "S2A_MSIL1C_20170105T013442_N0204_R031_T53NMJ_20170105T013443"),
                Eq("S2A_MSIL1C_20170105T013442_N0204_R031_T53NMJ_20170105T013443/MTD_MSIL1C.xml"));
    EXPECT_THAT(alus::resample::TryCreateSentinel2GdalDatasetName(
                    "/home/foo/S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443"),
                Eq("/home/foo/S2B_MSIL2A_20170105T013442_N0204_R031_T53NMJ_20170105T013443/MTD_MSIL2A.xml"));
}
}  // namespace