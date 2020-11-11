#include "metadata_attribute.h"

#include "product_data.h"

namespace alus::snapengine {
class ProductData;

MetadataAttribute::MetadataAttribute(std::string_view name, int type) : MetadataAttribute(name, type, 1) {}
MetadataAttribute::MetadataAttribute(std::string_view name, int type, int num_elems)
    : MetadataAttribute(name, ProductData::CreateInstance(type, num_elems), false) {}
MetadataAttribute::MetadataAttribute(std::string_view name, std::shared_ptr<ProductData> data, bool read_only)
    : DataNode(name, data, read_only) {}

std::shared_ptr<MetadataAttribute> alus::snapengine::MetadataAttribute::CreateDeepClone() {
    auto clone = std::make_shared<MetadataAttribute>(GetName(), GetData()->CreateDeepClone(), IsReadOnly());
    clone->SetDescription(GetDescription());
    clone->SetSynthetic(IsSynthetic());
    clone->SetUnit(GetUnit());
    return clone;
}
bool MetadataAttribute::Equals(MetadataAttribute& object) { return (object.GetData() == GetData()); }

}  // namespace alus::snapengine