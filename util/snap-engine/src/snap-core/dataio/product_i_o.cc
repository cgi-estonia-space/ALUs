#include "snap-core/dataio/product_i_o.h"

#include <iostream>

#include "snap-core/dataio/i_product_reader.h"
#include "snap-core/dataio/i_product_reader_plug_in.h"
#include "snap-core/dataio/product_i_o_plug_in_manager.h"
#include "snap-core/dataio/product_subset_def.h"
#include "snap-core/datamodel/product.h"
#include "snap-core/util/guardian.h"

namespace alus::snapengine {

std::shared_ptr<IProductReader> ProductIO::GetProductReaderForInput(const std::any& input) {

    std::cerr << "Searching reader plugin for '" << input.type().name() << "'" << std::endl;

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
            std::cerr << "Error attempting to read " << input.type().name() << " with plugin reader "
                      << typeid(plug_in).name() << ": " << e.what() << std::endl;
        }
    }
    if (selected_plug_in) {
        std::cerr << "Selected " << typeid(selected_plug_in).name() << std::endl;
        return selected_plug_in->CreateReaderInstance();
    }
    std::cerr << "No suitable reader plugin found" << std::endl;
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
