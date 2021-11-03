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
#pragma once

class Mask;

namespace alus {
namespace snapengine {
/**
 * Specifies a factory for the {@link RasterDataNode#getSourceImage() source image} used by a {@link Mask}.
 */
class ImageType {
   private:
    std::string name_;

   protected:
    ImageType(std::string_view name) : name_(name) {}
    ImageType(const ImageType&) = delete;
    ImageType& operator=(const ImageType&) = delete;
    virtual ~ImageType() = default;

   public:
    static constexpr std::string_view PROPERTY_NAME_COLOR = "color";
    static constexpr std::string_view PROPERTY_NAME_TRANSPARENCY = "transparency";
//    static Color DEFAULT_COLOR = Color.RED;
    static constexpr double DEFAULT_TRANSPARENCY = 0.5;

    /**
     * Creates the mask's source image.
     *
     * @param mask The mask which requests creation of its source image.
     *
     * @return The image.
     */
    virtual std::shared_ptr<MultiLevelImage> CreateImage(std::shared_ptr<Mask> mask) = 0;

    bool CanTransferMask(std::shared_ptr<Mask> mask, std::shared_ptr<Product> product) { return false; }

    std::shared_ptr<Mask> TransferMask(std::shared_ptr<Mask> mask, std::shared_ptr<Product> product){return nullptr};

    /**
     * Creates a prototype image configuration.
     *
     * @return The image configuration.
     */
    std::shared_ptr<PropertyContainer> CreateImageConfig() {
        std::shared_ptr<PropertyDescriptor> color_type =
            std::make_shared<PropertyDescriptor>(PROPERTY_NAME_COLOR, Color.class);
        color_type->SetNotNull(true);
        color_type->SetDefaultValue(DEFAULT_COLOR);

        std::shared_ptr<PropertyDescriptor> transparency_type =
            std::make_shared<PropertyDescriptor>(PROPERTY_NAME_TRANSPARENCY, Double::TYPE);
        transparency_type->DetDefaultValue(DEFAULT_TRANSPARENCY);

        std::shared_ptr<PropertyContainer> image_config = std::make_shared<PropertyContainer>();
        image_config->AddProperty(std::make_shared<Property>(color_type, std::make_shared<DefaultPropertyAccessor>()));
        image_config->AddProperty(
            std::make_shared<Property>(transparency_type, std::make_shared<DefaultPropertyAccessor>()));

        SetImageStyle(image_config, DEFAULT_COLOR, DEFAULT_TRANSPARENCY);
        return image_config;
    }

    void HandleRename(std::shared_ptr<Mask> mask,
                      std::string_view old_external_name,
                      std::string_view new_external_name) {}

    std::string GetName() { return name_; }
}
};  // namespace snapengine
}  // namespace alus
}  // namespace alus
