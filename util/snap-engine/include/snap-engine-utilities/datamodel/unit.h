/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.engine_utilities.datamodel.Unit.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated:
 * Copyright (C) 2014 by Array Systems Computing Inc. http://www.array.ca
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

#include <memory>

namespace alus {
namespace snapengine {

class Band;

enum class UnitType {
    AMPLITUDE,
    INTENSITY,
    REAL,
    IMAGINARY,
    PHASE,
    ABS_PHASE,
    COHERENCE,
    AMPLITUDE_DB,
    INTENSITY_DB,
    METERS,
    CENTIMETERS,
    METERS_PER_DAY,
    CLASS,
    SOIL_MOISTURE,
    DEGREES,
    NANOSECONDS,
    UNKNOWN
};

class Unit {
public:
    static constexpr std::string_view AMPLITUDE = "amplitude";
    static constexpr std::string_view INTENSITY = "intensity";
    static constexpr std::string_view PHASE = "phase";
    static constexpr std::string_view ABS_PHASE = "abs_phase";
    static constexpr std::string_view COHERENCE = "coherence";
    static constexpr std::string_view REAL = "real";
    static constexpr std::string_view IMAGINARY = "imaginary";
    static constexpr std::string_view DB = "db";
    static constexpr std::string_view AMPLITUDE_DB = "amplitude_db";
    static constexpr std::string_view INTENSITY_DB = "intensity_db";
    static constexpr std::string_view METERS = "meters";
    static constexpr std::string_view CENTIMETERS = "centimeters";
    static constexpr std::string_view METERS_PER_DAY = "m/day";
    static constexpr std::string_view CLASS = "class";
    static constexpr std::string_view SOIL_MOISTURE = "m^3water/m^3soil";
    //        tiepoint grid units
    static constexpr std::string_view DEGREES = "deg";
    static constexpr std::string_view RADIANS = "radians";
    static constexpr std::string_view NANOSECONDS = "ns";
    //        temporary unit, should be removed later and use bit mask
    static constexpr std::string_view BIT = "bit";

    static UnitType GetUnitType(const std::shared_ptr<Band>& source_band);
};

}  // namespace snapengine
}  // namespace alus
