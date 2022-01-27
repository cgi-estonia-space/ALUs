/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.IndexCoding.java
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
#include "index_coding.h"

#include "../util/guardian.h"
#include "metadata_attribute.h"

namespace alus::snapengine {

IndexCoding::IndexCoding(std::string_view name) : SampleCoding(name) {}
std::shared_ptr<MetadataAttribute> IndexCoding::GetIndex(std::string_view name) { return GetAttribute(name); }
std::vector<std::string> IndexCoding::GetIndexNames() { return GetAttributeNames(); }
std::shared_ptr<MetadataAttribute> IndexCoding::AddIndex(std::string_view name, int value,
                                                         std::string_view description) {
    return AddSample(name, value, description);
}
int IndexCoding::GetIndexValue(std::string_view name) {
    Guardian::AssertNotNull("name", name);
    std::shared_ptr<MetadataAttribute> attribute = GetAttribute(name);
    if (attribute == nullptr) {
        throw std::invalid_argument("index '" + std::string(name) + "' not found");
    }
    //    Debug::AssertTrue(attribute->GetData()->IsInt());
    //    Debug::AssertTrue(attribute->GetData()->IsScalar());
    return attribute->GetData()->GetElemInt();
}
}  // namespace alus::snapengine