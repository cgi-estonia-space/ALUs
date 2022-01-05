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
#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include <boost/filesystem/path.hpp>

#include "snap-core/core/datamodel/product_node.h"

namespace alus::snapengine {

class Band;
class Product;
class ProductSubsetDef;
/**
 * java original was Created by luis on 10/01/2016.
 */
class Quicklook : virtual public ProductNode {
private:
    static constexpr std::string_view QUICKLOOK_EXT = ".jpg";

    //    BufferedImage image_;
    std::shared_ptr<Product> product_;
    boost::filesystem::path product_file_;
    boost::filesystem::path browse_file_;
    std::vector<std::shared_ptr<Band>> quicklook_bands_;
    boost::filesystem::path product_quicklook_folder_;
    bool product_can_append_files_;

    std::string quicklook_link_ = "";

public:
    static constexpr std::string_view DEFAULT_QUICKLOOK_NAME = "Quicklook";
    static constexpr std::string_view SNAP_QUICKLOOK_FILE_PREFIX = "snapQL_";

    explicit Quicklook(const boost::filesystem::path& product_file);
    /**
     * Constructor when only a quicklook name is given. Quicklook will be generated using defaults
     *
     * @param product the source product
     * @param name    the name of the quicklook
     */
    Quicklook(const std::shared_ptr<Product>& product, std::string_view name);

    /**
     * Constructor when a browseFile is given. The quicklook is generated from the browse file
     *
     * @param product    the source product
     * @param name       the name of the quicklook
     * @param browseFile the preview or browse image from a product
     */
    Quicklook(const std::shared_ptr<Product>& product, std::string_view name,
              const boost::filesystem::path& browse_file);

    /**
     * Constructor when a browseFile is given. The quicklook is generated from the browse file
     *
     * @param product    the source product
     * @param name       the name of the quicklook
     * @param quicklookBands   the bands to create an RGB quicklook from
     */
    Quicklook(const std::shared_ptr<Product>& product, std::string_view name,
              std::vector<std::shared_ptr<Band>> quicklook_bands);

    /**
     * Constructor when a browseFile is given. The quicklook is generated from the browse file
     *
     * @param product                the source product
     * @param name                   the name of the quicklook
     * @param browseFile             the preview or browse image from a product
     * @param productCanAppendFiles  true when files may be written to the product
     * @param productQuicklookFolder where to write the quicklook files
     */
    Quicklook(const std::shared_ptr<Product>& product, std::string_view name,
              const boost::filesystem::path& browse_file, bool product_can_append_files,
              const boost::filesystem::path& product_quicklook_folder,
              std::vector<std::shared_ptr<Band>> quicklook_bands);

    void SetProduct(const std::shared_ptr<Product>& product);

    /**
     * Gets an estimated, raw storage size in bytes of this product node.
     *
     * @param subsetDef if not <code>null</code> the subset may limit the size returned
     * @return the size in bytes.
     */
    uint64_t GetRawStorageSize([[maybe_unused]] const std::shared_ptr<ProductSubsetDef>& subset_def) override {
        return 0;
    }
};
}  // namespace alus::snapengine
