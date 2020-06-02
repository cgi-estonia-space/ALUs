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

#pragma once

namespace alus {
namespace snapengine {

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
class Utc {
   public:
    int days_{};
    int seconds_{};
    int microseconds_{};

    Utc() = default;

    explicit Utc(int days, int seconds, int microseconds);

    /**
     * Returns the Modified Julian Day 2000 (MJD2000) represented by this UTC value as double value.
     *
     * @return this UTC value computed as days
     */
    double getMjd() const;
    int getDaysFraction() const { return days_; }
    int getSecondsFraction() const { return seconds_; }
    int getMicroSecondsFraction() const { return microseconds_; }

    ~Utc() = default;

   private:
    static constexpr double SECONDS_PER_DAY{86400.0};
    static constexpr double SECONDS_TO_DAYS{1.0 / SECONDS_PER_DAY};
    static constexpr double MICROS_PER_SECOND{1000000.0};
    static constexpr double MICROS_TO_SECONDS{1.0 / MICROS_PER_SECOND};
};

}  // namespace snapengine
}  // namespace alus