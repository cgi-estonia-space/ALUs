/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.Mask.java
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
#include "snap-core/datamodel/mask.h"

namespace alus {
namespace snapengine {

Mask::Mask(std::string_view name, int width, int height, [[maybe_unused]] std::shared_ptr<ImageType> image_type)
    : Band(name, ProductData::TYPE_UINT8, width, height) {
    //    Assert::NotNull(image_type, "imageType");
    //    image_type_ = image_type;
    //    image_config_listener_ = evt->{
    //        if (IsSourceImageSet()) {
    //            // Added setSourceImage(null), otherwise
    //            // org.esa.snap.core.datamodel.MaskTest.testReassignExpression
    //            // cannot work. (nf 2015-07-27)
    //            //
    //            std::shared_ptr<MultiLevelImage> source_image = GetSourceImage();
    //            SetSourceImage(nullptr);
    //            // The sourceImage.reset() call is left here
    //            // so that old level images are removed from JAI tile cache.
    //            source_image->Reset();
    //        }
    //        //            fireProductNodeChanged(evt.getPropertyName(), evt.getOldValue(), evt.getNewValue());
    //    };
    //    image_config_ = image_type.CreateImageConfig();
    //    //        image_config_->AddPropertyChangeListener(image_config_listener_);
}
}  // namespace snapengine
}  // namespace alus
