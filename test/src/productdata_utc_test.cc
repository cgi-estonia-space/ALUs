/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductDataUTCTest.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
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
#include <fstream>
#include "gmock/gmock.h"
#include "product_data_utc.h"

namespace {

class ProductDataUtcTest {};

TEST(ProductDataUtc, testFormat) {
    const int days{4};
    const int seconds{6};
    const int microseconds{7};
    alus::snapengine::Utc utc = alus::snapengine::Utc(days, seconds, microseconds);
    ASSERT_EQ(days, utc.GetElemIntAt(0));
    ASSERT_EQ(seconds, utc.GetElemIntAt(1));
    ASSERT_EQ(microseconds, utc.GetElemIntAt(2));
    ASSERT_EQ(days, utc.GetDaysFraction());
    ASSERT_EQ(seconds, utc.GetSecondsFraction());
    ASSERT_EQ(microseconds, utc.GetMicroSecondsFraction());
    ASSERT_EQ("05-JAN-2000 00:00:06.000007", utc.Format());
}

TEST(ProductDataUtc, testParse) {
    std::shared_ptr<alus::snapengine::Utc> utc = alus::snapengine::Utc::Parse("05-Jan-2000 00:00:06.000007");
    ASSERT_EQ(4, utc->GetElemIntAt(0));
    ASSERT_EQ(6, utc->GetElemIntAt(1));
    ASSERT_EQ(7, utc->GetElemIntAt(2));
}

TEST(ProductDataUtc, testMjdToUTCConversion) {
    // these dates represent 3 consecutive scanline-times of a MERIS RR orbit
    alus::snapengine::Utc utc1 = alus::snapengine::Utc(2923.999998208953);   // NOLINT
    alus::snapengine::Utc utc2 = alus::snapengine::Utc(2924.000000245851);   // NOLINT
    alus::snapengine::Utc utc3 = alus::snapengine::Utc(2924.0000022827494);  // NOLINT

    ASSERT_EQ(2923, utc1.GetDaysFraction());
    ASSERT_EQ(2924, utc2.GetDaysFraction());
    ASSERT_EQ(2924, utc3.GetDaysFraction());
    ASSERT_EQ(86399, utc1.GetSecondsFraction());
    ASSERT_EQ(0, utc2.GetSecondsFraction());
    ASSERT_EQ(0, utc3.GetSecondsFraction());
    ASSERT_EQ(845253, utc1.GetMicroSecondsFraction());
    ASSERT_EQ(21241, utc2.GetMicroSecondsFraction());
    ASSERT_EQ(197229, utc3.GetMicroSecondsFraction());
}

TEST(ProductDataUtc, testMjdAfter2000) {
    std::shared_ptr<alus::snapengine::Utc> utc =
        alus::snapengine::Utc::Parse("02 Jul 2001 13:10:11", "%d %b %Y %H:%M:%S");
    double mjd = utc->GetMjd();
    alus::snapengine::Utc utc1 = alus::snapengine::Utc(mjd);
    // note: a bit different from java implementation
    ASSERT_EQ(utc1.GetAsDate(), utc->GetAsDate());
}

TEST(ProductDataUtc, testMjdBefore2000) {
    std::shared_ptr<alus::snapengine::Utc> utc =
        alus::snapengine::Utc::Parse("02 Jul 1999 13:10:11", "%d %b %Y %H:%M:%S");
    double mjd = utc->GetMjd();
    alus::snapengine::Utc utc1 = alus::snapengine::Utc(mjd);
    ASSERT_EQ(utc1.GetAsDate(), utc->GetAsDate());
}

TEST(ProductDataUtc, testParseAndFormat) {
    std::string expected = "23-DEC-2004 22:16:43.556677";
    std::shared_ptr<alus::snapengine::Utc> utc = alus::snapengine::Utc::Parse(expected);
    ASSERT_EQ(expected, utc->Format());
}

TEST(ProductDataUtc, testDoubleConstructor) {
    std::string expected = "23-DEC-2004 22:16:43.556677";
    std::shared_ptr<alus::snapengine::Utc> utc = alus::snapengine::Utc::Parse(expected);
    double mjd = utc->GetMjd();
    alus::snapengine::Utc utc2 = alus::snapengine::Utc(mjd);
    ASSERT_EQ(expected, utc2.Format());
}

TEST(ProductDataUtc, testDateParsingEmptyString) {
    EXPECT_THROW(alus::snapengine::Utc::Parse(""), std::invalid_argument);
}

//ported just to provide example (timezone UTZ only atm and calendar logic is different)
TEST(ProductDataUtc, testMerisDateParsing){
    std::string jan("03-JAN-2003 01:02:03.3456");
    std::string feb("05-FEB-2002 02:03:04.67890");
    std::string mar("06-MAR-2002 02:03:04.67890");
    std::string apr("07-APR-2004 04:06:22.32311");
    std::string mai("08-MAY-2005 12:33:57.32311");
    std::string dec("23-DEC-2004 22:16:43.556677");

    auto date = alus::snapengine::Utc::Parse(jan)->GetAsDate();

    ASSERT_EQ(3, date.date().day());
    ASSERT_EQ(1, date.date().month().as_enum());
    ASSERT_EQ(2003, date.date().year());
    ASSERT_EQ(1, date.time_of_day().hours());
    ASSERT_EQ(2, date.time_of_day().minutes());
    ASSERT_EQ(3, date.time_of_day().seconds());
    // not sure why they don't have milliseconds()
    ASSERT_EQ(346, date.time_of_day().total_milliseconds() -
                       ((date.time_of_day().hours() * 3600 + date.time_of_day().minutes() * 60 +
                         date.time_of_day().seconds()) *
                        1000));

    date = alus::snapengine::Utc::Parse(feb)->GetAsDate();
    ASSERT_EQ(5, date.date().day());
    ASSERT_EQ(2, date.date().month().as_enum());
    ASSERT_EQ(2002, date.date().year());
    ASSERT_EQ(2, date.time_of_day().hours());
    ASSERT_EQ(3, date.time_of_day().minutes());
    ASSERT_EQ(4, date.time_of_day().seconds());
    ASSERT_EQ(679, date.time_of_day().total_milliseconds() -
                       ((date.time_of_day().hours() * 3600 + date.time_of_day().minutes() * 60 +
                         date.time_of_day().seconds()) *
                        1000));

    date = alus::snapengine::Utc::Parse(mar)->GetAsDate();

    ASSERT_EQ(6, date.date().day());
    ASSERT_EQ(3, date.date().month().as_enum());
    ASSERT_EQ(2002, date.date().year());
    ASSERT_EQ(2, date.time_of_day().hours());
    ASSERT_EQ(3, date.time_of_day().minutes());
    ASSERT_EQ(4, date.time_of_day().seconds());
    ASSERT_EQ(679, date.time_of_day().total_milliseconds() -
                       ((date.time_of_day().hours() * 3600 + date.time_of_day().minutes() * 60 +
                         date.time_of_day().seconds()) *
                        1000));

    date = alus::snapengine::Utc::Parse(apr)->GetAsDate();

    ASSERT_EQ(7, date.date().day());
    ASSERT_EQ(4, date.date().month().as_enum());
    ASSERT_EQ(2004, date.date().year());
    ASSERT_EQ(4, date.time_of_day().hours());
    ASSERT_EQ(6, date.time_of_day().minutes());
    ASSERT_EQ(22, date.time_of_day().seconds());
    ASSERT_EQ(323, date.time_of_day().total_milliseconds() -
                       ((date.time_of_day().hours() * 3600 + date.time_of_day().minutes() * 60 +
                         date.time_of_day().seconds()) *
                        1000));

    date = alus::snapengine::Utc::Parse(mai)->GetAsDate();

    ASSERT_EQ(8, date.date().day());
    ASSERT_EQ(5, date.date().month().as_enum());
    ASSERT_EQ(2005, date.date().year());
    ASSERT_EQ(12, date.time_of_day().hours());
    ASSERT_EQ(33, date.time_of_day().minutes());
    ASSERT_EQ(57, date.time_of_day().seconds());
    ASSERT_EQ(323, date.time_of_day().total_milliseconds() -
                       ((date.time_of_day().hours() * 3600 + date.time_of_day().minutes() * 60 +
                         date.time_of_day().seconds()) *
                        1000));

    date = alus::snapengine::Utc::Parse(dec)->GetAsDate();

    ASSERT_EQ(23, date.date().day());
    ASSERT_EQ(12, date.date().month().as_enum());
    ASSERT_EQ(2004, date.date().year());
    ASSERT_EQ(22, date.time_of_day().hours());
    ASSERT_EQ(16, date.time_of_day().minutes());
    ASSERT_EQ(43, date.time_of_day().seconds());
    ASSERT_EQ(557, date.time_of_day().total_milliseconds() -
                       ((date.time_of_day().hours() * 3600 + date.time_of_day().minutes() * 60 +
                         date.time_of_day().seconds()) *
                        1000));
}

// a bit different from original java
TEST(ProductDataUtc, testGetAsDate) {
    auto date = alus::snapengine::Utc::Parse("23-DEC-2004 22:16:43.556677")->GetAsDate();
    ASSERT_EQ(23, date.date().day());
    ASSERT_EQ(12, date.date().month().as_enum());
    ASSERT_EQ(2004, date.date().year());
    ASSERT_EQ(22, date.time_of_day().hours());
    ASSERT_EQ(16, date.time_of_day().minutes());
    ASSERT_EQ(43, date.time_of_day().seconds());
    ASSERT_EQ(557, date.time_of_day().total_milliseconds() -
                       ((date.time_of_day().hours() * 3600 + date.time_of_day().minutes() * 60 +
                         date.time_of_day().seconds()) *
                        1000));
}

// a bit different from original java
TEST(ProductDataUtc, testGetCalendar) {
    // java calendar default start
    boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));  // NOLINT
    // snap calendar default start
    boost::posix_time::ptime calendar(alus::snapengine::Utc::CreateCalendar());
    ASSERT_EQ(946684800000L, (calendar - epoch).total_milliseconds());
}

}  // namespace