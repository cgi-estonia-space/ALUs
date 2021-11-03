/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.SARUtils.java
 * ported for native code.
 * Copied from(https://github.com/senbox-org/s1tbx). It was originally stated:
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
#include "s1tbx-commons/sar_utils.h"

#include <stdexcept>

#include "general_constants.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"

namespace alus::s1tbx {

double SarUtils::GetRadarFrequency(const std::shared_ptr<snapengine::MetadataElement>& abs_root) {
    double radar_freq = snapengine::AbstractMetadata::GetAttributeDouble(
                            abs_root, alus::snapengine::AbstractMetadata::RADAR_FREQUENCY) *
                        snapengine::eo::constants::ONE_MILLION;  // Hz
    if (radar_freq <= 0.0) {
        throw std::runtime_error("Invalid radar frequency: " + std::to_string(radar_freq));
    }
    return snapengine::eo::constants::LIGHT_SPEED / radar_freq;
}

double SarUtils::GetRadarWavelength(const std::shared_ptr<snapengine::MetadataElement>& abs_root) {
    double radar_freq = snapengine::AbstractMetadata::GetAttributeDouble(
                            abs_root, alus::snapengine::AbstractMetadata::RADAR_FREQUENCY) *
                        snapengine::eo::constants::ONE_MILLION;  // Hz
    if (radar_freq <= 0.0) {
        throw std::runtime_error("Invalid radar frequency: " + std::to_string(radar_freq));
    }
    return snapengine::eo::constants::LIGHT_SPEED / radar_freq;
}

}  // namespace alus::s1tbx
