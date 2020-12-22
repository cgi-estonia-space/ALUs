/* This file is a filtered duplicate of a SNAP's
 * static nested class DOUBLE which is inside org.esa.s1tbx.commons.SARUtils.java
 * ported for native code. Copied from a s1tbx (https://github.com/senbox-org/s1tbx) repository originally stated:
 * Copyright (C) 2016 by Array Systems Computing Inc. http://www.array.ca
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

#include "metadata_element.h"

namespace alus {
namespace s1tbx {

/**
 * SAR specific common functions
 */
class SarUtils {
    /**
     * Get radar frequency from the abstracted metadata (in Hz).
     *
     * @param absRoot the AbstractMetadata
     * @return wavelength
     * @throws Exception The exceptions.
     */
   public:
    static double GetRadarFrequency(std::shared_ptr<snapengine::MetadataElement> abs_root);
};

}  // namespace s1tbx
}  // namespace alus
