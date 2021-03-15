#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "snap-core/datamodel/metadata_attribute.h"
#include "snap-core/datamodel/metadata_element.h"
namespace alus::snapengine {
class Product;
}
namespace alus::s1tbx {

class BandInfo;
/**
 * Supports reading directories for level1, level2, and level0
 */
class ISentinel1Directory {
public:
    static constexpr std::string_view SENTINEL_DATE_FORMAT_PATTERN{"%d-%b-%Y %H:%M:%S"};

    virtual void Close() = 0;

    virtual void ReadProductDirectory() = 0;

    virtual std::shared_ptr<snapengine::Product> CreateProduct() = 0;

    virtual std::shared_ptr<BandInfo> GetBandInfo(const std::shared_ptr<snapengine::Band>& dest_band) = 0;

    virtual bool IsSLC() = 0;

    virtual std::shared_ptr<snapengine::MetadataElement> GetMetadataObject(
        const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root, std::string_view metadata_object_name) {
        const std::shared_ptr<snapengine::MetadataElement>& metadata_section =
            orig_prod_root->GetElement("XFDU")->GetElement("metadataSection");
        const std::vector<std::shared_ptr<snapengine::MetadataElement>> metadata_objects =
            metadata_section->GetElements();

        for (auto elem : metadata_objects) {
            if (elem->GetAttribute("ID")->GetData()->GetElemString() == metadata_object_name) {
                return elem;
            }
        }
        return nullptr;
    }
};
}  // namespace alus::s1tbx
