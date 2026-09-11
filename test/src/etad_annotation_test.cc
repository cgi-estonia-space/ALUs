/** ETAD annotation parsing and input-structure tests. SPDX-License-Identifier: GPL-3.0-or-later */
#include "sentinel1_etad_product.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <pugixml.hpp>

namespace {
namespace etad = alus::s1tbx::etad;

/** Native annotation with non-contiguous indices, overlapping bursts and variable grid sizes. */
std::string Annotation() {
    pugi::xml_document xml;
    xml.load_string(R"(<etadProduct>
      <etadHeader><missionId>S1A</missionId><mode>IW</mode><productType>SLC</productType></etadHeader>
      <productCoverage><temporalCoverage>
        <azimuthTimeMin>2000-01-02T00:00:00.125000</azimuthTimeMin>
        <azimuthTimeMax>2000-01-02T00:00:10.125000</azimuthTimeMax>
        <rangeTimeMin>0.005</rangeTimeMin><rangeTimeMax>0.007</rangeTimeMax>
      </temporalCoverage></productCoverage>
      <productInformation><carrierFrequency>5405000454.334349</carrierFrequency></productInformation>
      <productComponents><numberOfSwaths>2</numberOfSwaths><numberOfBursts>3</numberOfBursts>
        <numberOfInputProducts>1</numberOfInputProducts><inputProductList count="1"><inputProduct pIndex="7">
          <productID>S1A_IW_SLC__1ADV_20000102T000000_20000102T000010_012345_DIFFERENT.SAFE</productID>
          <startTime>2000-01-02T00:00:00</startTime><stopTime>2000-01-02T00:00:10</stopTime>
          <swathList count="2"><swath sIndex="1"><swathID>IW1</swathID>
            <bIndexList count="2"><bIndex>4</bIndex><bIndex>1</bIndex></bIndexList></swath>
            <swath sIndex="2"><swathID>IW2</swathID><bIndexList count="1"><bIndex>2</bIndex></bIndexList></swath>
          </swathList></inputProduct></inputProductList>
      </productComponents>
      <processingInformation><auxInputData><auxSetap><instrumentTimingCalibrationReference>
        <rangeCalibration>7.4e-10</rangeCalibration><azimuthCalibration>6.3e-6</azimuthCalibration>
      </instrumentTimingCalibrationReference><instrumentTimingCalibrationOffsetList count="2">
        <instrumentTimingCalibrationOffset><swath>IW1</swath><polarisation>VV</polarisation>
          <rangeOffset>1e-10</rangeOffset><azimuthOffset>2e-6</azimuthOffset></instrumentTimingCalibrationOffset>
        <instrumentTimingCalibrationOffset><swath>IW2</swath><polarisation>VV</polarisation>
          <rangeOffset>2e-10</rangeOffset><azimuthOffset>3e-6</azimuthOffset></instrumentTimingCalibrationOffset>
      </instrumentTimingCalibrationOffsetList></auxSetap></auxInputData></processingInformation>
      <etadBurstList count="3"/></etadProduct>)");
    auto root = xml.document_element();
    for (const int index : {1, 2, 4}) {
        auto burst = root.child("etadBurstList").append_child("etadBurst");
        auto data = burst.append_child("burstData");
        data.append_attribute("bIndex") = index;
        data.append_attribute("pIndex") = 7;
        data.append_attribute("sIndex") = index == 2 ? 2 : 1;
        data.append_child("swathID").text() = index == 2 ? "IW2" : "IW1";
        auto coverage = burst.append_child("burstCoverage");
        coverage.append_copy(root.child("productCoverage").child("temporalCoverage"));
        auto spatial = coverage.append_child("spatialCoverage");
        for (const auto* name :
             {"EarlyAzimuthNearRange", "EarlyAzimuthFarRange", "LateAzimuthNearRange", "LateAzimuthFarRange"}) {
            auto corner = spatial.append_child("coordinates");
            corner.append_attribute("corner") = name;
            corner.append_child("latitude").text() = 45.0;
            corner.append_child("longitude").text() = 4.0;
        }
        auto grid = burst.append_child("gridInformation");
        grid.append_child("gridStartAzimuthTime").text() = 2.0;
        grid.append_child("gridStartRangeTime").text() = 0.001;
        auto dimensions = grid.append_child("gridDimensions");
        dimensions.append_child("azimuthExtent").text() = index == 4 ? 3 : 2;
        dimensions.append_child("rangeExtent").text() = 3;
        auto sampling = grid.append_child("gridSampling");
        sampling.append_child("azimuth").text() = 1.0;
        sampling.append_child("range").text() = 0.001;
    }
    std::ostringstream output;
    xml.save(output);
    return output.str();
}

std::string Replace(std::string xml, const std::string& from, const std::string& to) {
    const auto position = xml.find(from);
    if (position == std::string::npos) {
        throw std::runtime_error("fixture replacement not found: " + from);
    }
    xml.replace(position, from.size(), to);
    return xml;
}

TEST(EtadAnnotation, NativeXmlPreservesIdentityDimensionsAndEpoch) {
    const auto metadata = etad::ParseAnnotation(Annotation());
    ASSERT_EQ(metadata.acquisitions.size(), 1U);
    EXPECT_EQ(metadata.acquisitions[0].p_index, 7);
    EXPECT_EQ(metadata.acquisitions[0].swaths[0].burst_indices, (std::vector<int>{4, 1}));
    EXPECT_DOUBLE_EQ(metadata.coverage.azimuth_min, 86400.125);
    EXPECT_EQ(etad::GetBurst(metadata, 4).rows, 3U);
    EXPECT_EQ(etad::GetBurst(metadata, 1).rows, 2U);
    EXPECT_EQ(etad::GetBurst(metadata, 2).swath_id, "IW2");
    EXPECT_THROW(etad::GetBurst(metadata, 99), std::runtime_error);
}

TEST(EtadAnnotation, RejectsMalformedMetadata) {
    const auto xml = Annotation();
    EXPECT_THROW(etad::ParseAnnotation("<broken>"), std::runtime_error);
    EXPECT_THROW(etad::ParseAnnotation(Replace(xml, "<mode>IW</mode>", "<mode>EW</mode>")), std::runtime_error);
    EXPECT_THROW(etad::ParseAnnotation(Replace(xml, "2000-01-02T00:00:00.125000", "2000-01-02T25:00:00.125000")),
                 std::runtime_error);
    EXPECT_THROW(etad::ParseAnnotation(Replace(xml, "<bIndex>4</bIndex>", "<bIndex>99</bIndex>")), std::runtime_error);
    EXPECT_THROW(etad::ParseAnnotation(Replace(xml, "<bIndex>4</bIndex>", "<bIndex>1</bIndex>")), std::runtime_error);
    EXPECT_THROW(etad::ParseAnnotation(Replace(xml, "<azimuth>1</azimuth>", "<azimuth>0</azimuth>")),
                 std::runtime_error);
    EXPECT_THROW(
        etad::ParseAnnotation(Replace(xml, "<azimuthExtent>2</azimuthExtent>", "<azimuthExtent>1</azimuthExtent>")),
        std::runtime_error);
    EXPECT_THROW(
        etad::ParseAnnotation(Replace(xml, "<numberOfBursts>3</numberOfBursts>", "<numberOfBursts>2</numberOfBursts>")),
        std::runtime_error);
}

TEST(EtadAnnotation, ReferenceAndChannelOffsetsAreRetainedSeparately) {
    const auto metadata = etad::ParseAnnotation(Annotation());
    ASSERT_EQ(metadata.calibrations.size(), 2U);
    EXPECT_DOUBLE_EQ(metadata.calibrations[0].range_seconds, 7.4E-10);
    EXPECT_DOUBLE_EQ(metadata.calibrations[0].range_offset_seconds, 1E-10);
    EXPECT_DOUBLE_EQ(metadata.calibrations[1].range_offset_seconds, 2E-10);
    EXPECT_DOUBLE_EQ(metadata.calibrations[0].azimuth_offset_seconds, 2E-6);
    EXPECT_DOUBLE_EQ(metadata.calibrations[1].azimuth_offset_seconds, 3E-6);
}

TEST(EtadAnnotation, DirectCalibrationValuesRetainSwathAndPolarisation) {
    pugi::xml_document xml;
    xml.load_string(Annotation().c_str());
    auto aux = xml.document_element().child("processingInformation").child("auxInputData").child("auxSetap");
    while (aux.first_child()) {
        aux.remove_child(aux.first_child());
    }
    auto list = aux.append_child("instrumentTimingCalibrationList");
    list.append_attribute("count") = 2;
    for (const auto* pol : {"VV", "VH"}) {
        auto entry = list.append_child("instrumentTimingCalibration");
        entry.append_child("swath").text() = "IW1";
        entry.append_child("polarisation").text() = pol;
        entry.append_child("rangeCalibration").text() = std::string(pol) == "VV" ? 1E-9 : 2E-9;
        entry.append_child("azimuthCalibration").text() = 3E-6;
    }
    std::ostringstream text;
    xml.save(text);
    const auto metadata = etad::ParseAnnotation(text.str());
    ASSERT_EQ(metadata.calibrations.size(), 2U);
    EXPECT_EQ(metadata.calibrations[1].polarisation, "VH");
    EXPECT_DOUBLE_EQ(metadata.calibrations[0].range_seconds, 1E-9);
    EXPECT_DOUBLE_EQ(metadata.calibrations[1].range_seconds, 2E-9);
    EXPECT_DOUBLE_EQ(metadata.calibrations[1].range_offset_seconds, 0);
}

TEST(EtadSelection, SensingTokensSurviveSplitNamesAndDifferentProductIdentifiers) {
    auto metadata = etad::ParseAnnotation(Annotation());
    EXPECT_EQ(etad::MatchAcquisition(metadata, "SLC_20000102T000000_20000102T000010_012345_split").p_index, 7);
    EXPECT_EQ(etad::MatchAcquisition(metadata, "20000102T000000_20000102T000010").p_index, 7);
    EXPECT_THROW(etad::MatchAcquisition(metadata, "renamed"), std::runtime_error);
    EXPECT_THROW(etad::MatchAcquisition(metadata, "20000102T000000_20000102T000010_012346"), std::runtime_error);
    metadata.acquisitions.push_back(metadata.acquisitions[0]);
    EXPECT_THROW(etad::MatchAcquisition(metadata, "20000102T000000_20000102T000010"), std::runtime_error);
}

TEST(EtadSelection, InclusiveAcquisitionStrictBurstAndRangeFirstListedOverlap) {
    const auto metadata = etad::ParseAnnotation(Annotation());
    EXPECT_NE(etad::FindAcquisition(metadata, 86400), nullptr);
    EXPECT_NE(etad::FindAcquisition(metadata, 86410), nullptr);
    EXPECT_EQ(etad::FindAcquisition(metadata, 86410.001), nullptr);
    const auto* burst = etad::FindBurst(metadata, 7, "IW1", 86401, 0.006);
    ASSERT_NE(burst, nullptr);
    EXPECT_EQ(burst->b_index, 4);
    EXPECT_EQ(etad::FindBurst(metadata, 1, "IW1", 86401, 0.006), nullptr);
    EXPECT_EQ(etad::FindBurst(metadata, 7, "IW3", 86401, 0.006), nullptr);
    EXPECT_EQ(etad::FindBurst(metadata, 7, "IW1", 86400.125, 0.006), nullptr);
    EXPECT_EQ(etad::FindBurst(metadata, 7, "IW1", 86410.125, 0.006), nullptr);
    EXPECT_EQ(etad::FindBurst(metadata, 7, "IW1", 86401, 0.005), nullptr);
    EXPECT_EQ(etad::FindBurst(metadata, 7, "IW1", 86401, 0.007), nullptr);
}
}  // namespace
