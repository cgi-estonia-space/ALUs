/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.math.FX.java
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
#include "snap-core/core/util/math/i_f_x.h"

#include "snap-core/core/util/math/functions.h"

namespace alus::snapengine {

const std::reference_wrapper<IFX> IFX::XXXX = *new functions::FxX4();  // todo object slicing might break something
                                                                       // here
const std::reference_wrapper<IFX> IFX::XXX = *new functions::FxX3();
const std::reference_wrapper<IFX> IFX::XX = *new functions::FxX2();
const std::reference_wrapper<IFX> IFX::X = *new functions::FxX();
const std::reference_wrapper<IFX> IFX::ONE = *new functions::Fx1();

IFX::~IFX() = default;

}  // namespace alus::snapengine