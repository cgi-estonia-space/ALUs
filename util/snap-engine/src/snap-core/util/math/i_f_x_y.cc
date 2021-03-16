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
#include "snap-core/util/math/i_f_x_y.h"

#include "snap-core/util/math/functions.h"

namespace alus {
namespace snapengine {

const std::reference_wrapper<IFXY> IFXY::X4Y4 = *new functions::FXY_X4Y4();
const std::reference_wrapper<IFXY> IFXY::X4Y3 = *new functions::FXY_X4Y3();
const std::reference_wrapper<IFXY> IFXY::X3Y4 = *new functions::FXY_X3Y4();
const std::reference_wrapper<IFXY> IFXY::X4Y2 = *new functions::FXY_X4Y2();
const std::reference_wrapper<IFXY> IFXY::X2Y4 = *new functions::FXY_X2Y4();
const std::reference_wrapper<IFXY> IFXY::X4Y = *new functions::FXY_X4Y();
const std::reference_wrapper<IFXY> IFXY::XY4 = *new functions::FXY_XY4();
const std::reference_wrapper<IFXY> IFXY::X4 = *new functions::FXY_X4();
const std::reference_wrapper<IFXY> IFXY::Y4 = *new functions::FXY_Y4();
const std::reference_wrapper<IFXY> IFXY::X3Y3 = *new functions::FXY_X3Y3();
const std::reference_wrapper<IFXY> IFXY::X3Y2 = *new functions::FXY_X3Y2();
const std::reference_wrapper<IFXY> IFXY::X2Y3 = *new functions::FXY_X2Y3();
const std::reference_wrapper<IFXY> IFXY::X3Y = *new functions::FXY_X3Y();
const std::reference_wrapper<IFXY> IFXY::X2Y2 = *new functions::FXY_X2Y2();
const std::reference_wrapper<IFXY> IFXY::XY3 = *new functions::FXY_XY3();
const std::reference_wrapper<IFXY> IFXY::X3 = *new functions::FXY_X3();
const std::reference_wrapper<IFXY> IFXY::X2Y = *new functions::FXY_X2Y();
const std::reference_wrapper<IFXY> IFXY::XY2 = *new functions::FXY_XY2();
const std::reference_wrapper<IFXY> IFXY::Y3 = *new functions::FXY_Y3();
const std::reference_wrapper<IFXY> IFXY::X2 = *new functions::FXY_X2();
const std::reference_wrapper<IFXY> IFXY::XY = *new functions::FXY_XY();
const std::reference_wrapper<IFXY> IFXY::Y2 = *new functions::FXY_Y2();
const std::reference_wrapper<IFXY> IFXY::X = *new functions::FXY_X();
const std::reference_wrapper<IFXY> IFXY::Y = *new functions::FXY_Y();
const std::reference_wrapper<IFXY> IFXY::ONE = *new functions::FXY_1();

IFXY::~IFXY() {}

}  // namespace snapengine
}  // namespace alus
