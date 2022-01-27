/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.GeoPos.java
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
#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace alus::snapengine {
/**
 * The <code>GeoPos</code> class represents a geographical position measured in longitudes and latitudes.
 *
 * @author Norman Fomferra
 * @author Daniel Knowles
 * @version $Revision$ $Date$
 * @
 */
// SEP2018 - Daniel Knowles - Modified to support multiple format options for string representation of the latitude and
// longitude values.
//                            Compass Format: indicate North, South, West, East either as "+/-" or with the suffixes
//                            "N,S,W,E" Decimal Format: display value either as a decimal number or in the format
//                            DDD°[MM'[SS"]]
class GeoPos {
private:
    static constexpr double MIN_PER_DEG{60.0};
    static constexpr double SEC_PER_DEG{MIN_PER_DEG * 60.0};

    /**
     * Creates a string representation of the given decimal degree value. The string returned has the format
     * DDD?[MM'[SS.sssss"]] [N|S|W|E].
     */
    // SEP2018 - Daniel Knowles - Modified to support compass_format and decimal_format options
    static std::string GetDegreeString(double value, bool longitudial, bool compass_format, bool decimal_format);

    static bool IsLatValid(double lat);

    static bool IsLonValid(double lon);

    static int FloorInt(double value);

public:
    /**
     * The geographical latitude in decimal degree, valid range is -90 to +90.
     */
    double lat_;
    /**
     * The geographical longitude in decimal degree, valid range is -180 to +180.
     */
    double lon_;

    /**
     * Constructs a new geo-position with latitude and longitude set to zero.
     */
    GeoPos();

    /**
     * Constructs a new geo-position with latitude and longitude set to that of the given geo-position.
     *
     * @param geo_pos the  geo-position providing the latitude and longitude, must not be <code>null</code>
     */
    GeoPos(const GeoPos& geo_pos);

    /**
     * Constructs a new geo-position with the given latitude and longitude values.
     *
     * @param lat the geographical latitude in decimal degree, valid range is -90 to +90
     * @param lon the geographical longitude in decimal degree, valid range is -180 to +180
     */
    GeoPos(double lat, double lon);

    /**
     * Gets the latitude value.
     *
     * @return the geographical latitude in decimal degree
     */
    [[nodiscard]] double GetLat() const { return lat_; }

    /**
     * Gets the longitude value.
     *
     * @return the geographical longitude in decimal degree
     */
    [[nodiscard]] double GetLon() const { return lon_; }

    /**
     * Sets the geographical location of this point.
     *
     * @param lat the geographical latitude in decimal degree, valid range is -90 to +90
     * @param lon the geographical longitude in decimal degree, valid range is -180 to +180
     */
    void SetLocation(double lat, double lon);

    /**
     * Tests whether or not this geo-position is valid.
     *
     * @return true, if so
     */
    [[nodiscard]] bool IsValid() const;

    /**
     * Tests whether or not all given geo-positions are valid.
     *
     * @return true, if so
     */
    static bool AreValid(const std::vector<GeoPos>& a) {
        return std::all_of(a.cbegin(), a.cend(), [](const GeoPos& gp) { return gp.IsValid(); });
    };

    /**
     * Sets the lat/lon fields so that {@link #isValid()} will return false.
     */
    void SetInvalid();

    /**
     * Indicates whether some other object is "equal to" this one.
     *
     * @param obj the reference object with which to compare.
     *
     * @return <code>true</code> if this object is the same as the obj argument; <code>false</code> otherwise.
     */
    // todo::only support if needed
    //   bool Equals(Object obj);
    //   {
    //        if (super.equals(obj)) {
    //            return true;
    //        }
    //        if (!(obj instanceof GeoPos)) {
    //            return false;
    //        }
    //        GeoPos other = (GeoPos) obj;
    //        return other.lat == lat && other.lon == lon;
    //    }

    /**
     * Returns a hash code value for the object.
     *
     * @return a hash code value for this object.
     */
    //     todo:only support if needed
    //    @Override
    //   public int hashCode() {
    //        return (int) (Double.doubleToLongBits(lat) + Double.doubleToLongBits(lon));
    //    }

    /**
     * Returns a string representation of the object. In general, the <code>toString</code> method returns a string that
     * "textually represents" this object.
     *
     * @return a string representation of the object.
     */
    [[nodiscard]] std::string ToString() const;

    /**
     * Normalizes this position so that its longitude is in the range -180 to +180 degree.
     */
    void Normalize();

    /**
     * Normalizes the given longitude so that it is in the range -180 to +180 degree and returns it.
     * Note that -180 will remain as is, although -180 is equivalent to +180 degrees.
     *
     * @param lon the longitude in degree
     *
     * @return the normalized longitude in the range
     */
    static double NormalizeLon(double lon);

    /**
     * Returns a string representation of the latitude value.
     *
     * @return a string of the form DDD?[MM'[SS"]] [N|S].
     */
    // SEP2018 - Daniel Knowles - Modified to call getLatString(lat, true, false)
    [[nodiscard]] std::string GetLatString() const;

    /**
     * Returns a string representation of the latitude value.
     *
     * @return a string of the form DDD?[MM'[SS"]] [W|E].
     */
    // SEP2018 - Daniel Knowles - Modified to call getLonString(lon, true, false)
    [[nodiscard]] std::string GetLonString() const;

    /**
     * Returns a string representation of the given longitude value.
     *
     *
     * @param lat the geographical latitude in decimal degree
     *
     * @return a string of the form DDD?[MM'[SS"]] [N|S].
     */
    // SEP2018 - Daniel Knowles - Modified to call getLatString(lon, true, false)
    static std::string GetLatString(double lat);

    /**
     * Returns a string representation of the given longitude value.
     *
     *
     *
     *
     * @param lon the geographical longitude in decimal degree
     *
     * @return a string of the form DDD?[MM'[SS"]] [W|E].
     */
    // SEP2018 - Daniel Knowles - Modified to call getLonString(lon, true, false)
    static std::string GetLonString(double lon);

    /**
     * Returns a string representation of the given longitude value.
     *
     * @param lon
     * @param compassFormat true: "[degrees] [N|S]"
     *                      false: "+/- [degrees]"
     * @param decimalFormat true: "degrees in decimal"
     *                      false: "degrees in DDD°[MM'[SS"]]"
     * @return a formatted string representing latitude
     * @author Daniel Knowles
     * @since Sept 2018
     */
    static std::string GetLonString(double lon, bool compass_format, bool decimal_format);

    /**
     * Returns a string representation of the given latitude value.
     *
     * @param lat
     * @param compassFormat true: "[degrees] [N|S]"
     *                      false: "+/- [degrees]"
     * @param decimalFormat true: "degrees in decimal"
     *                      false: "degrees in DDD°[MM'[SS"]]"
     * @return a formatted string representing latitude
     * @author Daniel Knowles
     * @since Sept 2018
     */
    static std::string GetLatString(double lat, bool compass_format, bool decimal_format);

    /**
     * Returns a string representation of the latitude value.
     *
     * @param compassFormat true: "[degrees] [N|S]"
     *                      false: "+/- [degrees]"
     * @param decimalFormat true: "degrees in decimal"
     *                      false: "degrees in DDD°[MM'[SS"]]"
     * @return a formatted string representing latitude
     * @author Daniel Knowles
     * @since Sept 2018
     */
    [[nodiscard]] std::string GetLatString(bool compass_format, bool decimal_format) const;

    /**
     * Returns a string representation of the longitude value.
     *
     * @param compassFormat true: "[degrees] [N|S]"
     *                      false: "+/- [degrees]"
     * @param decimalFormat true: "degrees in decimal"
     *                      false: "degrees in DDD°[MM'[SS"]]"
     * @return a formatted string representing longitude
     * @author Daniel Knowles
     * @since Sept 2018
     */
    [[nodiscard]] std::string GetLonString(bool compass_format, bool decimal_format) const;
};
}  // namespace alus::snapengine
