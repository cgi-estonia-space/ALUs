/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.dataio.DecodeQualification.java
 * ported for native code.
 * Copied from a snap-engine (https://github.com/senbox-org/snap-engine) repository originally stated:
 *
 * Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)
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

namespace alus::snapengine {

/**
 * The qualification of a product reader for decoding a given input.
 */
enum class DecodeQualification {
    /**
     * The reader is intended to decode a given input.
     */
    INTENDED,
    /**
     * The reader is suitable to decode a given input, but has not specifically been designed for it.
     */
    SUITABLE,
    /**
     * The reader is unable to decode a given input.
     */
    UNABLE
};

}  // namespace alus::snapengine
