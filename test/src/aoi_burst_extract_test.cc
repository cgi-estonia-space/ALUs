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

#include <vector>

#include <boost/geometry.hpp>

#include "aoi_burst_extract.h"
#include "raster_properties.h"

namespace {

/*
 * Geometries used in this test can be visualized by
 * GEOMETRYCOLLECTION(
 * POLYGON((36.03317558581512 34.71570741931848,35.03914088219177 34.86867135255156,35.00213872465311
 * 34.70240346523518,35.99552328254985 34.54919208767427,36.03317558581512 34.71570741931848)),
 * POLYGON((35.99552328254985 34.54919208767427,35.00213872465311 34.70240346523518,34.96530740657919 34.53662231052934,
 * 35.93678417920061 34.386525070748064,35.99552328254985 34.54919208767427)),
 * POLYGON((35.25741948459887 34.78398978827197,35.29522567116086 34.7115067010492,35.38659062201899 34.69855660076446,
 * 35.25741948459887 34.78398978827197)),
 * POLYGON((34.94551844546249 34.99589539381856,34.78799266812088 34.470303376275396,
 * 36.28133703731933 34.21016499223519,36.423110236926775 34.778814534583205,34.94551844546249 34.99589539381856)),
 * POLYGON((35.865468985137475 34.78398978827197,36.07025249568157 34.55078227649045,
 * 36.10805868224356 34.750344836683894,35.865468985137475 34.78398978827197)),
 * POLYGON((34.74703596601206 35.428332683155986,36.01039270029175 35.119683856498895,
 * 36.10805868224356 34.695966337460874,34.81319679249553 35.29730065144389,34.74703596601206 35.428332683155986)))
 * at https://clydedacruz.github.io/openstreetmap-wkt-playground/
 */
class AoiBurstExtract : public ::testing::Test {
public:
    AoiBurstExtract() {
        boost::geometry::read<boost::geometry::format_wkt>(burst1_expected_box_, BURST1_EXPECTED_BOX_WKT.data());
        boost::geometry::read<boost::geometry::format_wkt>(burst2_expected_box_, BURST2_EXPECTED_BOX_WKT.data());
        boost::geometry::read<boost::geometry::format_wkt>(burst1_inner_aoi_, BURST1_INNER_AOI.data());
        boost::geometry::read<boost::geometry::format_wkt>(burst1_outer_aoi_, BURST1_OUTER_AOI.data());
        boost::geometry::read<boost::geometry::format_wkt>(burst1_intersecting_aoi_, BURST1_INTERSECTING_AOI.data());
        boost::geometry::read<boost::geometry::format_wkt>(burst1_not_containing_aoi_,
                                                           BURST1_NOT_CONTAINING_AOI.data());
    }

    const std::vector<alus::Coordinates> beirut_burst_line_1_{
        {36.03317558581512, 34.71570741931848}, {35.98053459523368, 34.72402648536686},
        {35.92710924936979, 34.73244348560416}, {35.87432262850503, 34.74073531201949},
        {35.82190975883752, 34.74894415822732}, {35.76997534648072, 34.75705419567446},
        {35.71850182606551, 34.76506894437733}, {35.66747257284508, 34.77299149070796},
        {35.61687192167196, 34.78082491364173}, {35.56668506152945, 34.78857214201961},
        {35.51689792744521, 34.79623576798969}, {35.46749723921216, 34.80381847418609},
        {35.41847034459694, 34.81132258493338}, {35.36980527597931, 34.81875052938061},
        {35.32149060283389, 34.82610439103574}, {35.27351548704647, 34.8333863315581},
        {35.22586960160898, 34.84059839907307}, {35.17854306256166, 34.84774237848912},
        {35.13152649229082, 34.85482021355166}, {35.08481087781347, 34.86183350620586},
        {35.03914088219177, 34.86867135255156}};

    const std::vector<alus::Coordinates> beirut_burst_line_2_{
        {35.99552328254985, 34.54919208767427}, {35.94167267732444, 34.55771918566346},
        {35.88835894572777, 34.56613620673014}, {35.83556102292847, 34.5744472762801},
        {35.78325908153008, 34.58265610261994}, {35.73143452375636, 34.59076641343774},
        {35.68006976979432, 34.59878146786055}, {35.62914828082054, 34.60670453632973},
        {35.57865443106048, 34.61453869302515}, {35.52857340429129, 34.62228667782449},
        {35.47889122593212, 34.62995131986704}, {35.42959459576794, 34.63753506605251},
        {35.38067093062786, 34.64504040249913}, {35.33210828261728, 34.65246971226034},
        {35.28389524597442, 34.65982506131306}, {35.23602102289761, 34.66710866965237},
        {35.18847527312045, 34.67432241215389}, {35.14124817494741, 34.68146824116039},
        {35.09433035955234, 34.68854803625642}, {35.04771283875525, 34.69556341102302},
        {35.00213872465311, 34.70240346523518}};

    const std::vector<alus::Coordinates> beirut_burst_line_3_{
        {35.93678417920061, 34.38652507074804}, {35.9004061535996, 34.39230709903136},
        {35.84723770322268, 34.40071748036523}, {35.79420191953901, 34.40908191360079},
        {35.74479743587158, 34.41685406854229}, {35.6930814915122, 34.42496458881858},
        {35.64182433277611, 34.43298013394249}, {35.59100941975777, 34.44090378482134},
        {35.54062116584437, 34.44873861089978}, {35.4906448338953, 34.4564875320806},
        {35.44106642869064, 34.4641531320967},  {35.39187273563474, 34.47173808568988},
        {35.34305116357618, 34.4792447049979},  {35.29458979280381, 34.4866753648602},
        {35.2464772869379, 34.49403231242209},  {35.19870281877035, 34.50131751861224},
        {35.15125612830631, 34.50853310130866}, {35.1041273759338, 34.5156808255979},
        {35.05730721639107, 34.52276257315103}, {35.01078672087253, 34.52978012529903},
        {34.96530740657919, 34.53662231052932}};

    static constexpr std::string_view BURST1_EXPECTED_BOX_WKT{
        "POLYGON ((36.03317558581512 34.71570741931848, 35.03914088219177 34.86867135255156, 35.00213872465311 "
        "34.70240346523518, 35.99552328254985 34.54919208767427, 36.03317558581512 34.71570741931848))"};
    static constexpr std::string_view BURST2_EXPECTED_BOX_WKT{
        "POLYGON ((35.99552328254985 34.54919208767427, 35.00213872465311 34.70240346523518, 34.96530740657919 "
        "34.53662231052932, 35.93678417920061 34.38652507074804, 35.99552328254985 34.54919208767427))"};

    alus::topsarsplit::BurstBox burst1_expected_box_;
    alus::topsarsplit::BurstBox burst2_expected_box_;

    static constexpr std::string_view BURST1_INNER_AOI{
        "POLYGON((35.25741948459887 34.78398978827197,35.29522567116086 34.7115067010492,35.38659062201899 "
        "34.69855660076446,35.25741948459887 34.78398978827197))"};
    static constexpr std::string_view BURST1_OUTER_AOI{
        "POLYGON((34.94551844546249 34.99589539381856,34.78799266812088 34.470303376275396,36.28133703731933 "
        "34.21016499223519,36.423110236926775 34.778814534583205,34.94551844546249 34.99589539381856))"};
    static constexpr std::string_view BURST1_INTERSECTING_AOI{
        "POLYGON((35.865468985137475 34.78398978827197,36.07025249568157 34.55078227649045,36.10805868224356 "
        "34.750344836683894,35.865468985137475 34.78398978827197))"};
    static constexpr std::string_view BURST1_NOT_CONTAINING_AOI{
        "POLYGON((34.74703596601206 35.428332683155986,36.01039270029175 35.119683856498895,36.10805868224356 "
        "34.695966337460874,34.81319679249553 35.29730065144389,34.74703596601206 35.428332683155986))"};

    alus::topsarsplit::Aoi burst1_inner_aoi_;
    alus::topsarsplit::Aoi burst1_outer_aoi_;
    alus::topsarsplit::Aoi burst1_intersecting_aoi_;
    alus::topsarsplit::Aoi burst1_not_containing_aoi_;
};

TEST_F(AoiBurstExtract, GetBurstBoxFromReturnsCorrectBox) {
    const auto result = alus::topsarsplit::GetBurstBoxFrom(beirut_burst_line_1_, beirut_burst_line_2_);
    const auto result2 = alus::topsarsplit::GetBurstBoxFrom(beirut_burst_line_2_, beirut_burst_line_3_);
    EXPECT_TRUE(boost::geometry::equals(burst1_expected_box_, result));
    EXPECT_TRUE(boost::geometry::equals(burst2_expected_box_, result2));
}

TEST_F(AoiBurstExtract, IsCoveredReturnsTrue) {
    EXPECT_TRUE(alus::topsarsplit::IsCovered(burst1_inner_aoi_, burst1_expected_box_));
    EXPECT_TRUE(alus::topsarsplit::IsCovered(burst1_outer_aoi_, burst1_expected_box_));
    EXPECT_TRUE(alus::topsarsplit::IsCovered(burst1_intersecting_aoi_, burst1_expected_box_));
}

TEST_F(AoiBurstExtract, IsCoveredReturnsFalse) {
    EXPECT_FALSE(alus::topsarsplit::IsCovered(burst1_not_containing_aoi_, burst1_expected_box_));
    EXPECT_FALSE(alus::topsarsplit::IsCovered(burst1_inner_aoi_, burst2_expected_box_));
    EXPECT_FALSE(alus::topsarsplit::IsCovered(burst1_intersecting_aoi_, burst2_expected_box_));
}
}  // namespace
