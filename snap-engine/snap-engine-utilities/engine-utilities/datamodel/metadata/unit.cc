/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.datamodel.Unit.java
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
#include "snap-engine-utilities/engine-utilities/datamodel/unit.h"

#include <boost/algorithm/string.hpp>

#include "snap-core/core/datamodel/band.h"

namespace alus::snapengine {

UnitType Unit::GetUnitType(const std::shared_ptr<Band>& source_band) {
    if (!source_band->GetUnit().has_value()) {
        return UnitType::UNKNOWN;
    }
    auto unit = source_band->GetUnit().value();
    boost::algorithm::to_lower(unit);

    if (unit.find(AMPLITUDE) != std::string::npos) {
        if (unit.find(DB) != std::string::npos) {
            return UnitType::AMPLITUDE_DB;
        }
        return UnitType::AMPLITUDE;
    }
    if (unit.find(INTENSITY) != std::string::npos) {
        if (unit.find(DB) != std::string::npos) {
            return UnitType::INTENSITY_DB;
        }
        return UnitType::INTENSITY;
    }
    if (unit.find(PHASE) != std::string::npos) {
        return UnitType::PHASE;
    }
    if (unit.find(ABS_PHASE) != std::string::npos) {
        return UnitType::ABS_PHASE;
    }
    if (unit.find(REAL) != std::string::npos) {
        return UnitType::REAL;
    }
    if (unit.find(IMAGINARY) != std::string::npos) {
        return UnitType::IMAGINARY;
    }
    if (unit.find(METERS) != std::string::npos) {
        return UnitType::METERS;
    }
    if (unit.find(CENTIMETERS) != std::string::npos) {
        return UnitType::CENTIMETERS;
    }
    if (unit.find(METERS_PER_DAY) != std::string::npos) {
        return UnitType::METERS_PER_DAY;
    }
    if (unit.find(COHERENCE) != std::string::npos) {
        return UnitType::COHERENCE;
    }
    if (unit.find(CLASS) != std::string::npos) {
        return UnitType::CLASS;
    }
    if (unit.find(SOIL_MOISTURE) != std::string::npos) {
        return UnitType::SOIL_MOISTURE;
    }
    if (unit.find(DEGREES) != std::string::npos) {
        return UnitType::DEGREES;
    }
    if (unit.find(NANOSECONDS) != std::string::npos) {
        return UnitType::NANOSECONDS;
    }
    return UnitType::UNKNOWN;
}

}  // namespace alus::snapengine
