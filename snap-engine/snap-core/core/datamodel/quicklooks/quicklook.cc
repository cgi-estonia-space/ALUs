/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.quicklooks.Quicklook.java
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
 * with this program; if not, see http://www.gnu.org/licenses/"
 */
#include "snap-core/core/datamodel/quicklooks/quicklook.h"

#include <utility>

#include "snap-core/core/datamodel/product.h"

namespace alus::snapengine {

Quicklook::Quicklook(const std::shared_ptr<Product>& product, std::string_view name,
                     const boost::filesystem::path& browse_file, const bool product_can_append_files,
                     const boost::filesystem::path& product_quicklook_folder,
                     std::vector<std::shared_ptr<Band>> quicklook_bands)
    : ProductNode(name) {
    browse_file_ = browse_file;
    product_can_append_files_ = product_can_append_files;
    product_quicklook_folder_ = product_quicklook_folder;
    quicklook_bands_ = std::move(quicklook_bands);

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
    : Quicklook(product, name, "", false, "", std::move(quicklook_bands)) {}

Quicklook::Quicklook(const std::shared_ptr<Product>& product, std::string_view name,
                     const boost::filesystem::path& browse_file)
    : Quicklook(product, name, browse_file, false, "", std::vector<std::shared_ptr<Band>>{}) {}

Quicklook::Quicklook(const std::shared_ptr<Product>& product, std::string_view name)
    : Quicklook(product, name, "", false, "", std::vector<std::shared_ptr<Band>>{}) {}

Quicklook::Quicklook(const boost::filesystem::path& product_file) : Quicklook(nullptr, DEFAULT_QUICKLOOK_NAME) {
    product_file_ = product_file;
}

}  // namespace alus::snapengine