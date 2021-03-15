#include "s1tbx-commons/io/x_m_l_product_directory.h"

#include <fstream>
#include <iostream>

#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/product.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata_i_o.h"

namespace alus::s1tbx {

std::shared_ptr<snapengine::MetadataElement> XMLProductDirectory::AddMetaData() {
    std::shared_ptr<snapengine::MetadataElement> root =
        std::make_shared<snapengine::MetadataElement>(snapengine::Product::METADATA_ROOT_NAME);

    auto root_element = xml_doc_.document_element();
    snapengine::AbstractMetadataIO::AddXMLMetadata(root_element,
                                                   snapengine::AbstractMetadata::AddOriginalProductMetadata(root));
    AddAbstractedMetadataHeader(root);

    return root;
}
XMLProductDirectory::XMLProductDirectory(const boost::filesystem::path& input_file)
    : AbstractProductDirectory(input_file) {}

void XMLProductDirectory::ReadProductDirectory() {
    std::fstream is;
    GetInputStream(GetRootFolder() + GetHeaderFileName(), is);
    if (is) {
        xml_doc_.load(is);
    }
    if (is.is_open()) {
        is.close();
    }
}

}  // namespace alus::s1tbx
