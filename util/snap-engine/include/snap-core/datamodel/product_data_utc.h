/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class UTC which is inside org.esa.snap.core.datamodel.ProductData.java
 * ported for native code. Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally
 * stated to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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
#pragma once

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "product_data_uint.h"

namespace alus {
namespace snapengine {
class ProductData;
/**
 * The {@code ProductData.UTC} class is a {@code ProductData.UInt} specialisation for UTC date/time
 * values.
 * <p> Internally, data is stored in an {@code int[3]} array which represents a Modified Julian Day 2000
 * ({@link ProductData.UTC#getMJD() MJD}) as a {@link
 * ProductData.UTC#getDaysFraction() days}, a {@link
 * ProductData.UTC#getSecondsFraction() seconds} and a {@link
 * ProductData.UTC#getMicroSecondsFraction() micro-seconds} fraction.
 *
 * @see ProductData.UTC#getMJD()
 * @see ProductData.UTC#getDaysFraction()
 * @see ProductData.UTC#getSecondsFraction()
 * @see ProductData.UTC#getMicroSecondsFraction()
 */
class Utc : public UInt {
   private:
    static constexpr double SECONDS_PER_DAY{86400.0};
    static constexpr double SECONDS_TO_DAYS{1.0 / SECONDS_PER_DAY};
    static constexpr double MICROS_PER_SECOND{1000000.0};
    static constexpr double MICROS_TO_SECONDS{1.0 / MICROS_PER_SECOND};

   public:
    //    todo:we already have some solution for format, check that before doing
    /**
     * The default pattern used to format date strings.
     */
    //    static constexpr std::string_view DATE_FORMAT_PATTERN{"dd-MMM-yyyy HH:mm:ss"};
    //    static constexpr std::string_view DATE_FORMAT_PATTERN{"%d-%b-%Y %T"};
    static constexpr std::string_view DATE_FORMAT_PATTERN{"%d-%b-%Y %H:%M:%S"};
    /**
     * Creates a date format using the given pattern. The date format returned, will use the
     * english locale ('en') and a calendar returned by the {@link #createCalendar()} method.
     *
     * @param pattern the data format pattern
     *
     * @return a date format
     *
     * @see java.text.SimpleDateFormat
     */
    static boost::posix_time::time_facet* CreateDateFormatOut(std::string_view pattern);
    static boost::posix_time::time_input_facet* CreateDateFormatIn(std::string_view pattern);

    /**
     * Gets the MJD 2000 calendar on which this UTC date/time is based. The date is initially set the 1st January
     * 2000, 0:00.
     *
     * @return the MJD 2000 calendar
     *
     * @see #getAsCalendar()
     */
    // todo: might just be enough to create a date in boost which is 2000, 1, 1
    // replaces CreateCalendar from java version
    //    static boost::posix_time::ptime CreateCalendar();
    static boost::gregorian::date CreateCalendar();

    /**
     * Parses a UTC value given as text in MJD 2000 format.
     * The method returns {@link #parse(String, String)} using {@link #DATE_FORMAT_PATTERN} as pattern.
     *
     * @param text a UTC value given as text
     *
     * @return the UTC value represented by the given text
     *
     * @throws ParseException thrown if the text could not be parsed
     * @see #createCalendar
     * @see #createDateFormat
     */
    static std::shared_ptr<Utc> Parse(std::string_view text);

    /**
     * Parses a UTC value given as text. The method also considers an optional
     * mircoseconds fraction at the end of the text string. The mircoseconds fraction
     * is a dot '.' followed by a maximum of 6 digits.
     *
     * @param text    a UTC value given as text
     * @param pattern the date/time pattern
     *
     * @return the UTC value represented by the given text
     *
     * @throws ParseException thrown if the text could not be parsed
     * @see #createCalendar
     * @see #createDateFormat
     */
    static std::shared_ptr<Utc> Parse(std::string_view text, std::string_view pattern);

    /**
     * Parses a UTC value given as text. The method also considers an optional
     * mircoseconds fraction at the end of the text string. The mircoseconds fraction
     * is a dot '.' followed by a maximum of 6 digits.
     *
     * @param text    a UTC value given as text
     * @param dateFormat the date/time pattern
     *
     * @return the UTC value represented by the given text
     *
     * @throws ParseException thrown if the text could not be parsed
     * @see #createCalendar
     * @see #createDateFormat
     */
    static std::shared_ptr<Utc> Parse(std::string_view text, boost::posix_time::time_input_facet*);

    /**
     * Creates a new UTC instance based on the given time and microseconds fraction.
     *
     * @param date   the UTC time
     * @param micros the microseconds fraction
     *
     * @return a new UTC instance
     */
    //    todo: might need datetime, but implementation should be much simpler
    static std::shared_ptr<Utc> Create(boost::posix_time::ptime date, long micros);

    /**
     * Retuns a "deep" copy of this product data.
     *
     * @return a copy of this product data
     */
   protected:
    [[nodiscard]] std::shared_ptr<ProductData> CreateDeepClone() const override;
   public:
    /**
     * Constructs a new {@code UTC} value.
     */
    Utc();
    /**
     * Constructs a MJD2000 date instance.
     *
     * @param elems an array containg at least the three elements {@code {days, seconds, microSeconds}}
     */
    explicit Utc(std::vector<uint32_t> elems);
    /**
     * Constructs a MJD2000 date instance.
     *
     * @param days         the number of days since 2000-01-01 00:00
     * @param seconds      the seconds fraction of the number of days
     * @param microSeconds the microseconds fraction of the number of days
     */
    Utc(int days, int seconds, int microseconds);

    /**
     * Constructs a MJD2000 date instance.
     *
     * @param mjd the Modified Julian Day 2000 (MJD2000) as double value
     *
     * @see #getMJD()
     */
    explicit Utc(double mjd);

    /**
     * Returns the Modified Julian Day 2000 (MJD2000) represented by this UTC value as double value.
     *
     * @return this UTC value computed as days
     */
    [[nodiscard]] double GetMjd() const;
    /**
     * Returns the days fraction of the Modified Julian Day (MJD) as a signed integer (the 1st element of the
     * internal data array).
     *
     * @return This UTC's days fraction.
     *
     * @see #getMJD()
     */
    [[nodiscard]] int GetDaysFraction() const { return this->GetElemIntAt(0); }
    /**
     * Returns the seconds fraction of the Modified Julian Day (MJD) as a signed integer (the 2nd element of the
     * internal data array).
     *
     * @return This UTC's seconds fraction.
     *
     * @see #getMJD()
     */
    [[nodiscard]] int GetSecondsFraction() const { return this->GetElemIntAt(1); }
    /**
     * Returns the micro-seconds fraction of the Modified Julian Day (MJD) as a signed integer (the 3rd element
     * of the internal data array).
     *
     * @return This UTC's micro-seconds fraction.
     *
     * @see #getMJD()
     */
    [[nodiscard]] int GetMicroSecondsFraction() const { return this->GetElemIntAt(2); }

    /**
     * Formats this UTC date/time value as a string using the format {@link #DATE_FORMAT_PATTERN} and the default
     * MJD 2000 calendar.
     *
     * @return a formated UTC date/time string
     *
     * @see #createCalendar
     * @see #createDateFormat
     */
    std::string Format() const;
    /**
     * Gets the MJD 2000 calendar on which this UTC date/time is based.
     * The date of the calendar is set to this UTC value.
     *
     * @return the MJD 2000 calendar
     *
     * @see #createCalendar()
     * @see #getAsDate()
     */
    // this should be good enough to derive whatever was used in java
    boost::posix_time::ptime GetAsCalendar() const;

    /**
     * Returns this UTC date/time value as a Date. The method interpretes this UTC value as a MJD 2000 date
     * (Modified Julian Day where the  first day is the 01.01.2000).
     *
     * @see #getAsCalendar()
     */
    boost::posix_time::ptime GetAsDate();

    /**
     * Returns this UTC date/time value as a string using the format {@link #DATE_FORMAT_PATTERN}. Simply calls
     * {@link #format()}.
     */

    [[nodiscard]] std::string GetElemString() override;

    /**
     * Returns this value's data type String.
     */
    std::string GetTypeString() override { return ProductData::GetTypeString(TYPE_UTC); }

};

}  // namespace snapengine
}  // namespace alus