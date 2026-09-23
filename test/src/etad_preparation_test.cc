/** ETAD burst association tests. SPDX-License-Identifier: GPL-3.0-or-later */
#include "etad_preparation.h"

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace {
namespace etad = alus::s1tbx::etad;
namespace slc = alus::s1tbx::slc;

slc::Metadata SlcMetadata() {
    slc::Metadata metadata;
    metadata.product_name = "SLC_20000102T000000_20000102T000010_012345";
    metadata.swath_id = "IW1";
    metadata.polarisation = "VV";
    metadata.radar_frequency_hz = 5.405E9;
    metadata.raster = {0.1, 0.006, 0.0001, 2, 2, 4, 2};
    metadata.bursts = {{86401.0}, {86402.0}};
    return metadata;
}

etad::Metadata EtadMetadata() {
    etad::Metadata metadata;
    metadata.acquisitions.push_back(
        {"S1A_IW_SLC__1ADV_20000102T000000_20000102T000010_012345_DIFFERENT.SAFE",
         7,
         86400.0,
         86410.0,
         {{"IW1", 1, {4, 1}}}});
    metadata.bursts.push_back({4, 7, 1, "IW1", {86400.95, 86402.1, 0.005, 0.007}, 0, 0, 1, 1, 2, 2, {}});
    metadata.bursts.push_back({1, 7, 1, "IW1", {86401.95, 86402.3, 0.005, 0.007}, 0, 0, 1, 1, 2, 2, {}});
    return metadata;
}

TEST(EtadPreparation, AssociatesBurstStartsInsteadOfFirstOverlappingCoverage) {
    const auto association = etad::AssociateBursts(SlcMetadata(), {0, 2}, EtadMetadata());
    EXPECT_EQ(association.etad_p_index, 7);
    ASSERT_EQ(association.bursts.size(), 2U);
    EXPECT_EQ(association.bursts[0].slc_burst_index, 0U);
    EXPECT_EQ(association.bursts[0].split_burst_index, 0U);
    EXPECT_EQ(association.bursts[0].first_output_line, 0U);
    EXPECT_EQ(association.bursts[0].etad_b_index, 4);
    EXPECT_EQ(association.bursts[1].slc_burst_index, 1U);
    EXPECT_EQ(association.bursts[1].split_burst_index, 1U);
    EXPECT_EQ(association.bursts[1].first_output_line, 2U);
    EXPECT_EQ(association.bursts[1].etad_b_index, 1);

    const auto subset = etad::AssociateBursts(SlcMetadata(), {1, 1}, EtadMetadata());
    ASSERT_EQ(subset.bursts.size(), 1U);
    EXPECT_EQ(subset.bursts[0].slc_burst_index, 1U);
    EXPECT_EQ(subset.bursts[0].split_burst_index, 0U);
    EXPECT_EQ(subset.bursts[0].first_output_line, 0U);
    EXPECT_EQ(subset.bursts[0].etad_b_index, 1);
}

TEST(EtadPreparation, FallsBackToNativeMetadataForBurstProductNames) {
    auto slc_metadata = SlcMetadata();
    slc_metadata.product_name = "S1A_SLC_20000102T000001_343935_IW1_VV";
    slc_metadata.mission = "S1A";
    slc_metadata.absolute_orbit = 12345;
    slc_metadata.start_time = 86400.5;
    // Native annotation timestamps retain fractions while ETAD input-product bounds can be whole-second values.
    slc_metadata.stop_time = 86410.656183;
    const auto association = etad::AssociateBursts(slc_metadata, {0, 1}, EtadMetadata());
    ASSERT_EQ(association.bursts.size(), 1U);
    EXPECT_EQ(association.etad_p_index, 7);
    EXPECT_EQ(association.bursts[0].etad_b_index, 4);
}

TEST(EtadPreparation, RejectsInvalidSelectionAndAssociation) {
    const auto slc_metadata = SlcMetadata();
    EXPECT_THROW(etad::AssociateBursts(slc_metadata, {0, 0}, EtadMetadata()), std::runtime_error);
    EXPECT_THROW(etad::AssociateBursts(slc_metadata, {2, 1}, EtadMetadata()), std::runtime_error);

    auto no_match = SlcMetadata();
    no_match.bursts[0].first_line_time = 86403.0;
    EXPECT_THROW(etad::AssociateBursts(no_match, {0, 1}, EtadMetadata()), std::runtime_error);

    auto incomplete_coverage = EtadMetadata();
    incomplete_coverage.bursts[0].coverage.azimuth_max = 86401.05;
    EXPECT_THROW(etad::AssociateBursts(slc_metadata, {0, 1}, incomplete_coverage), std::runtime_error);

    auto ambiguous = EtadMetadata();
    ambiguous.acquisitions[0].swaths[0].burst_indices.push_back(9);
    ambiguous.bursts.push_back({9, 7, 1, "IW1", {86400.96, 86401.3, 0.005, 0.007}, 0, 0, 1, 1, 2, 2, {}});
    EXPECT_THROW(etad::AssociateBursts(slc_metadata, {0, 1}, ambiguous), std::runtime_error);
}

}  // namespace
