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
#pragma once

#include <memory>

#include "snap-core/datamodel/metadata_element.h"

namespace alus::s1tbx {

/**
 * SAR specific common functions
 */
class SarUtils {
public:
    /**
     * Get radar frequency from the abstracted metadata (in Hz).
     *
     * @param absRoot the AbstractMetadata
     * @return wavelength
     * @throws Exception The exceptions.
     */
    [[deprecated]] static double GetRadarFrequency(const std::shared_ptr<snapengine::MetadataElement>& abs_root);
    /**
     * Get radar wavelength from the abstracted metadata (in nm).
     *
     * @param absRoot the AbstractMetadata
     * @return wavelength
     * @throws Exception The exceptions.
     */
    static double GetRadarWavelength(const std::shared_ptr<snapengine::MetadataElement>& abs_root);
};

}  // namespace alus::s1tbx
