///**
// * This file is a filtered duplicate of a SNAP's org.esa.snap.core.datamodel.mask.java
// * ported for native code.
// * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
// * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
// *
// * This program is free software; you can redistribute it and/or modify it
// * under the terms of the GNU General Public License as published by the Free
// * Software Foundation; either version 3 of the License, or (at your option)
// * any later version.
// * This program is distributed in the hope that it will be useful, but WITHOUT
// * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// * more details.
// *
// * You should have received a copy of the GNU General Public License along
// * with this program; if not, see http://www.gnu.org/licenses/
// */
//#pragma once
//
//#include "band.h"
//#include "image_type.h"
//
//namespace alus {
//namespace snapengine {
//
///**
// * A {@code Mask} is used to mask image pixels of other raster data nodes.
// * <p>
// * This is a preliminary API under construction for BEAM 4.7. Not intended for public use.
// *
// * @author Norman Fomferra
// * @since BEAM 4.7
// */
//class Mask : virtual public Band {
//   private:
//    std::shared_ptr<ImageType> image_type_;
//    std::shared_ptr<PropertyChangeListener> image_config_listener_;
//    std::shared_ptr<PropertyContainer> image_config_;
//
//   public:
//    /**
//     * Constructs a new mask.
//     *
//     * @param name      The new mask's name.
//     * @param width     The new mask's raster width.
//     * @param height    The new mask's raster height.
//     * @param imageType The new mask's image type.
//     */
//    Mask(std::string_view name, int width, int height, std::shared_ptr<ImageType> image_type);
//
//    /**
//     * @return The image type of this mask.
//     */
//    std::shared_ptr<ImageType> GetImageType() { return image_type_; }
//
//    /**
//     * @return The image configuration of this mask.
//     */
//    std::shared_ptr<PropertyContainer> GetImageConfig() { return image_config_; }
//
//    Color GetImageColor() { return (Color)image_config_->GetValue(ImageType::PROPERTY_NAME_COLOR); }
//
//    void SetImageColor(Color color) { image_config_->SetValue(ImageType::PROPERTY_NAME_COLOR, color); }
//
//    double GetImageTransparency() { return (Double)image_config_->GetValue(ImageType::PROPERTY_NAME_TRANSPARENCY); }
//
//    void SetImageTransparency(double transparency) {
//        image_config_->SetValue(ImageType::PROPERTY_NAME_TRANSPARENCY, transparency);
//    }
//};
//}  // namespace snapengine
//}  // namespace alus
