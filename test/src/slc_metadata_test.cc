/** Native SLC annotation parsing tests. SPDX-License-Identifier: GPL-3.0-or-later */
#include "slc_metadata.h"

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace {
namespace slc = alus::s1tbx::slc;

std::string Annotation() {
    return R"(<product>
      <adsHeader>
        <missionId>s1a</missionId><productType>SLC</productType><polarisation>vv</polarisation>
        <mode>iw</mode><swath>iw1</swath>
        <startTime>2000-01-02T00:00:00.125000</startTime>
        <stopTime>2000-01-02T00:00:10.125000Z</stopTime>
        <absoluteOrbitNumber>12345</absoluteOrbitNumber>
      </adsHeader>
      <generalAnnotation><productInformation><radarFrequency>5405000454.334349</radarFrequency>
      </productInformation></generalAnnotation>
      <imageAnnotation><imageInformation>
        <slantRangeTime>0.006</slantRangeTime><rangePixelSpacing>149896.229</rangePixelSpacing>
        <azimuthTimeInterval>0.5</azimuthTimeInterval>
        <numberOfSamples>3</numberOfSamples><numberOfLines>4</numberOfLines>
      </imageInformation></imageAnnotation>
      <swathTiming><linesPerBurst>2</linesPerBurst><samplesPerBurst>3</samplesPerBurst>
        <burstList count="2"><burst><azimuthTime>2000-01-02T00:00:01.250000</azimuthTime></burst>
          <burst><azimuthTime>2000-01-02T00:00:03.250000</azimuthTime></burst></burstList>
      </swathTiming>
    </product>)";
}

std::string Replace(std::string xml, const std::string& from, const std::string& to) {
    const auto position = xml.find(from);
    if (position == std::string::npos) {
        throw std::runtime_error("fixture replacement not found: " + from);
    }
    xml.replace(position, from.size(), to);
    return xml;
}

TEST(SlcMetadata, NativeAnnotationPreservesIdentityUnitsAndBurstOrder) {
    const auto metadata = slc::ParseAnnotation(Annotation(), "S1A_TEST_PRODUCT");
    EXPECT_EQ(metadata.product_name, "S1A_TEST_PRODUCT");
    EXPECT_EQ(metadata.mission, "S1A");
    EXPECT_EQ(metadata.mode, "IW");
    EXPECT_EQ(metadata.swath_id, "IW1");
    EXPECT_EQ(metadata.polarisation, "VV");
    EXPECT_EQ(metadata.absolute_orbit, 12345);
    EXPECT_DOUBLE_EQ(metadata.start_time, 86400.125);
    EXPECT_DOUBLE_EQ(metadata.stop_time, 86410.125);
    EXPECT_DOUBLE_EQ(metadata.radar_frequency_hz, 5405000454.334349);
    EXPECT_DOUBLE_EQ(metadata.raster.first_range_time, 0.006);
    EXPECT_DOUBLE_EQ(metadata.raster.range_time_interval, 0.001);
    EXPECT_EQ(metadata.raster.lines_per_burst, 2U);
    EXPECT_EQ(metadata.raster.samples_per_burst, 3U);
    ASSERT_EQ(metadata.bursts.size(), 2U);
    EXPECT_DOUBLE_EQ(metadata.bursts[0].first_line_time, 86401.25);
    EXPECT_DOUBLE_EQ(metadata.bursts[1].first_line_time, 86403.25);
}

TEST(SlcMetadata, RejectsMalformedAndInconsistentAnnotation) {
    const auto xml = Annotation();
    EXPECT_THROW(slc::ParseAnnotation("<broken>", "product"), std::runtime_error);
    EXPECT_THROW(slc::ParseAnnotation(xml, ""), std::runtime_error);
    EXPECT_THROW(slc::ParseAnnotation(Replace(xml, "<mode>iw</mode>", "<mode>EW</mode>"), "product"),
                 std::runtime_error);
    EXPECT_THROW(slc::ParseAnnotation(Replace(xml, "00:00:10.125000Z", "25:00:10.125000Z"), "product"),
                 std::runtime_error);
    EXPECT_THROW(slc::ParseAnnotation(Replace(xml, "burstList count=\"2\"", "burstList count=\"3\""), "product"),
                 std::runtime_error);
    EXPECT_THROW(slc::ParseAnnotation(Replace(xml, "<numberOfLines>4", "<numberOfLines>5"), "product"),
                 std::runtime_error);
    EXPECT_THROW(slc::ParseAnnotation(Replace(xml, "<numberOfSamples>3", "<numberOfSamples>4"), "product"),
                 std::runtime_error);
    EXPECT_THROW(slc::ParseAnnotation(Replace(xml, "<numberOfSamples>3", "<numberOfSamples>-1"), "product"),
                 std::runtime_error);
    EXPECT_THROW(slc::ParseAnnotation(Replace(xml, "00:00:03.250000", "00:00:00.250000"), "product"),
                 std::runtime_error);
}
}  // namespace
