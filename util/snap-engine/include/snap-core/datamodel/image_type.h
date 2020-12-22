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
