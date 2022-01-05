/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.math.FXY.java
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
#include "snap-core/core/util/math/i_f_x_y.h"

#include "snap-core/core/util/math/functions.h"

namespace alus::snapengine {

const std::reference_wrapper<IFXY> IFXY::X4Y4 = *new functions::FxyX4Y4();
const std::reference_wrapper<IFXY> IFXY::X4Y3 = *new functions::FxyX4Y3();
const std::reference_wrapper<IFXY> IFXY::X3Y4 = *new functions::FxyX3Y4();
const std::reference_wrapper<IFXY> IFXY::X4Y2 = *new functions::FxyX4Y2();
const std::reference_wrapper<IFXY> IFXY::X2Y4 = *new functions::FxyX2Y4();
const std::reference_wrapper<IFXY> IFXY::X4Y = *new functions::FxyX4Y();
const std::reference_wrapper<IFXY> IFXY::XY4 = *new functions::FxyXY4();
const std::reference_wrapper<IFXY> IFXY::X4 = *new functions::FxyX4();
const std::reference_wrapper<IFXY> IFXY::Y4 = *new functions::FxyY4();
const std::reference_wrapper<IFXY> IFXY::X3Y3 = *new functions::FxyX3Y3();
const std::reference_wrapper<IFXY> IFXY::X3Y2 = *new functions::FxyX3Y2();
const std::reference_wrapper<IFXY> IFXY::X2Y3 = *new functions::FxyX2Y3();
const std::reference_wrapper<IFXY> IFXY::X3Y = *new functions::FxyX3Y();
const std::reference_wrapper<IFXY> IFXY::X2Y2 = *new functions::FxyX2Y2();
const std::reference_wrapper<IFXY> IFXY::XY3 = *new functions::FxyXY3();
const std::reference_wrapper<IFXY> IFXY::X3 = *new functions::FxyX3();
const std::reference_wrapper<IFXY> IFXY::X2Y = *new functions::FxyX2Y();
const std::reference_wrapper<IFXY> IFXY::XY2 = *new functions::FxyXY2();
const std::reference_wrapper<IFXY> IFXY::Y3 = *new functions::FxyY3();
const std::reference_wrapper<IFXY> IFXY::X2 = *new functions::FxyX2();
const std::reference_wrapper<IFXY> IFXY::XY = *new functions::FxyXY();
const std::reference_wrapper<IFXY> IFXY::Y2 = *new functions::FxyY2();
const std::reference_wrapper<IFXY> IFXY::X = *new functions::FxyX();
const std::reference_wrapper<IFXY> IFXY::Y = *new functions::FxyY();
const std::reference_wrapper<IFXY> IFXY::ONE = *new functions::Fxy1();

IFXY::~IFXY() = default;

}  // namespace alus::snapengine