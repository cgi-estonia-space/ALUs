/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.Scaling.java
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

namespace alus {
namespace snapengine {
/**
 * The scaling method used for geophysical value transformation in a {@link Band}.
 *
 * @author Norman Fomferra
 * @version $Revision$ $Date$
 */
class Scaling {
public:
    Scaling() = default;
    Scaling(const Scaling&) = delete;
    Scaling& operator=(const Scaling&) = delete;
    virtual ~Scaling() = default;
    /**
     * The forward scaling method.
     * @param value the value to be scaled
     * @return the transformed value
     */
    virtual double Scale(double value) = 0;

    /**
     * The inverse scaling method.
     * @param value the value to be inverse-scaled
     * @return the transformed value
     */
    virtual double ScaleInverse(double value) = 0;
};
}  // namespace snapengine
}  // namespace alus
