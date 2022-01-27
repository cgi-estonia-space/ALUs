/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.VirtualBand.java
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
#include "virtual_band.h"

namespace alus::snapengine {

VirtualBand::VirtualBand(std::string_view name, int data_type, int width, int height, std::string_view expression)
    : Band(name, data_type, width, height) {
    SetSpectralBandIndex(-1);
    SetSynthetic(true);
    expression_ = expression;
}
void VirtualBand::SetExpression(std::string_view expression) {
    if (!expression.empty() && expression_ != expression) {
        expression_ = expression;
        //        if (IsSourceImageSet()) {
        //            SetSourceImage(nullptr);
        //        }
        //        these are mostly used for visuals
        //        ResetValidMask();
        //        SetStx(nullptr);
        //        SetImageInfo(nullptr);
        SetModified(true);
        // not supporting/porting events
        //        fireProductNodeChanged(PROPERTY_NAME_EXPRESSION);
        //        fireProductNodeChanged(PROPERTY_NAME_DATA);
        //        fireProductNodeDataChanged();
    }
}

}  // namespace alus::snapengine