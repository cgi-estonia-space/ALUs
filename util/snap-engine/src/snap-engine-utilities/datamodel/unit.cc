#include "snap-engine-utilities/datamodel/unit.h"

#include <boost/algorithm/string.hpp>

#include "snap-core/datamodel/band.h"

namespace alus {
namespace snapengine {

UnitType Unit::GetUnitType(const std::shared_ptr<Band>& source_band) {
    if (!source_band->GetUnit().has_value()) {
        return UnitType::UNKNOWN;
    }
    auto unit = source_band->GetUnit().value();
    boost::algorithm::to_lower(unit);

    if (unit.find(AMPLITUDE) != std::string::npos) {
        if (unit.find(DB) != std::string::npos) {
            return UnitType::AMPLITUDE_DB;
        } else {
            return UnitType::AMPLITUDE;
        }
    } else if (unit.find(INTENSITY) != std::string::npos) {
        if (unit.find(DB) != std::string::npos) {
            return UnitType::INTENSITY_DB;
        } else {
            return UnitType::INTENSITY;
        }
    } else if (unit.find(PHASE) != std::string::npos) {
        return UnitType::PHASE;
    } else if (unit.find(ABS_PHASE) != std::string::npos) {
        return UnitType::ABS_PHASE;
    } else if (unit.find(REAL) != std::string::npos) {
        return UnitType::REAL;
    } else if (unit.find(IMAGINARY) != std::string::npos) {
        return UnitType::IMAGINARY;
    } else if (unit.find(METERS) != std::string::npos) {
        return UnitType::METERS;
    } else if (unit.find(CENTIMETERS) != std::string::npos) {
        return UnitType::CENTIMETERS;
    } else if (unit.find(METERS_PER_DAY) != std::string::npos) {
        return UnitType::METERS_PER_DAY;
    } else if (unit.find(COHERENCE) != std::string::npos) {
        return UnitType::COHERENCE;
    } else if (unit.find(CLASS) != std::string::npos) {
        return UnitType::CLASS;
    } else if (unit.find(SOIL_MOISTURE) != std::string::npos) {
        return UnitType::SOIL_MOISTURE;
    } else if (unit.find(DEGREES) != std::string::npos) {
        return UnitType::DEGREES;
    } else if (unit.find(NANOSECONDS) != std::string::npos) {
        return UnitType::NANOSECONDS;
    } else {
        return UnitType::UNKNOWN;
    }
}

}  // namespace snapengine
}  // namespace alus
