#include "snap-core/datamodel/quicklooks/quicklook.h"

#include "snap-core/datamodel/product.h"

namespace alus {
namespace snapengine {

Quicklook::Quicklook(const std::shared_ptr<Product>& product, std::string_view name,
                     const boost::filesystem::path& browse_file, const bool product_can_append_files,
                     const boost::filesystem::path& product_quicklook_folder,
                     std::vector<std::shared_ptr<Band>> quicklook_bands)
    : ProductNode(name) {
    browse_file_ = browse_file;
    product_can_append_files_ = product_can_append_files;
    product_quicklook_folder_ = product_quicklook_folder;
    quicklook_bands_ = quicklook_bands;

    SetProduct(product);
    // todo: provide config when needed
    //        final Preferences preferences = Config.instance().preferences();
    //        saveWithProduct = preferences.getBoolean(QuicklookGenerator.PREFERENCE_KEY_QUICKLOOKS_SAVE_WITH_PRODUCT,
    //                                                 QuicklookGenerator.DEFAULT_VALUE_QUICKLOOKS_SAVE_WITH_PRODUCT);
}
void Quicklook::SetProduct(const std::shared_ptr<Product>& product) {
    if (product) {
        product_ = product;
        product_file_ = product->GetFileLocation();
    }
}

Quicklook::Quicklook(const std::shared_ptr<Product>& product, std::string_view name,
                     std::vector<std::shared_ptr<Band>> quicklook_bands)
    : Quicklook(product, name, "", false, "", quicklook_bands) {}

Quicklook::Quicklook(const std::shared_ptr<Product>& product, std::string_view name,
                     const boost::filesystem::path& browse_file)
    : Quicklook(product, name, browse_file, false, "", std::vector<std::shared_ptr<Band>>{}) {}

Quicklook::Quicklook(const std::shared_ptr<Product>& product, std::string_view name)
    : Quicklook(product, name, "", false, "", std::vector<std::shared_ptr<Band>>{}) {}

Quicklook::Quicklook(const boost::filesystem::path& product_file) : Quicklook(nullptr, DEFAULT_QUICKLOOK_NAME) {
    product_file_ = product_file;
}

}  // namespace snapengine
}  // namespace alus
