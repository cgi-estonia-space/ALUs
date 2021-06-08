/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.dataio.ProductIO.java
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
#include "snap-core/dataio/product_i_o.h"

#include "alus_log.h"
#include "snap-core/dataio/i_product_reader.h"
#include "snap-core/dataio/i_product_reader_plug_in.h"
#include "snap-core/dataio/product_i_o_plug_in_manager.h"
#include "snap-core/dataio/product_subset_def.h"
#include "snap-core/datamodel/product.h"
#include "snap-core/util/guardian.h"

namespace alus::snapengine {

std::shared_ptr<IProductReader> ProductIO::GetProductReaderForInput(const std::any& input) {
    LOGV << "Searching reader plugin for '" << input.type().name() << "'";

    std::vector<std::shared_ptr<IProductReaderPlugIn>> plugins =
        ProductIOPlugInManager::GetInstance().GetAllReaderPlugIns();
    std::shared_ptr<IProductReaderPlugIn> selected_plug_in = nullptr;

    for (const auto& plug_in : plugins) {
        try {
            auto decode_qualification = plug_in->GetDecodeQualification(input);
            if (decode_qualification == DecodeQualification::INTENDED) {
                selected_plug_in = plug_in;
                break;
            } else if (decode_qualification == DecodeQualification::SUITABLE) {
                selected_plug_in = plug_in;
            }
        } catch (const std::exception& e) {
            LOGW << "Error attempting to read " << input.type().name() << " with plugin reader "
                      << typeid(plug_in).name() << ": " << e.what();
        }
    }
    if (selected_plug_in) {
        LOGV << "Selected " << typeid(selected_plug_in).name();
        return selected_plug_in->CreateReaderInstance();
    }
    LOGV << "No suitable reader plugin found";
    return nullptr;
}

std::shared_ptr<Product> ProductIO::ReadProduct(const boost::filesystem::path& file) {
    return ReadProductImpl(file, nullptr);
}

std::shared_ptr<Product> ProductIO::ReadProductImpl(const boost::filesystem::path& file,
                                                    const std::shared_ptr<ProductSubsetDef>& subset_def) {
    Guardian::AssertNotNull("file", file);
    if (!boost::filesystem::exists(file)) {
        throw std::runtime_error("File not found: " + file.filename().string());
    }
    const std::shared_ptr<IProductReader> product_reader = GetProductReaderForInput(file);
    if (product_reader) {
        return product_reader->ReadProductNodes(file, subset_def);
    }
    return nullptr;
}
}  // namespace alus::snapengine
