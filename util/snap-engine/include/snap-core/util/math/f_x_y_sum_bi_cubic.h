/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.util.math.FXYSum.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General  License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General  License for
 * more details.
 *
 * You should have received a copy of the GNU General  License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#pragma once

#include "f_x_y_sum.h"

namespace alus {
namespace snapengine {

/**
 * Provides an optimized <code>computeZ</code> method for linear polynomials (order = 1).
 *
 * @see #FXY_LINEAR
 */
class BiCubic final : public virtual FXYSum {
public:
    BiCubic();

    explicit BiCubic(std::vector<double> coefficients);

    double ComputeZ(double x, double y) override;
};

}  // namespace snapengine
}  // namespace alus
