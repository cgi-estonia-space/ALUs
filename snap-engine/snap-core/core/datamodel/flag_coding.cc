/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.FlagCoding.java
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
#include "flag_coding.h"

#include <stdexcept>

#include "../util/guardian.h"

namespace alus::snapengine {

FlagCoding::FlagCoding(std::string_view name) : SampleCoding(name) {}

std::shared_ptr<MetadataAttribute> FlagCoding::GetFlag(std::string_view name) { return GetAttribute(name); }

std::vector<std::string> FlagCoding::GetFlagNames() { return GetAttributeNames(); }

std::shared_ptr<MetadataAttribute> FlagCoding::AddFlag(std::string_view name, int flag_mask,
                                                       std::string_view description) {
    return AddSamples(name, std::vector<int>{flag_mask}, description);
}

std::shared_ptr<MetadataAttribute> FlagCoding::AddFlag(std::string_view name, int flag_mask, int flag_value,
                                                       std::string_view description) {
    return AddSamples(name, std::vector<int>{flag_mask, flag_value}, description);
}

int FlagCoding::GetFlagMask(std::string_view name) {
    Guardian::AssertNotNullOrEmpty("name", name);
    std::shared_ptr<MetadataAttribute> attribute = GetAttribute(name);
    if (attribute == nullptr) {
        throw std::invalid_argument("flag '" + std::string(name) + "' not found");
    }
    //        Debug::AssertTrue(attribute->GetData()->IsInt());
    //        Debug::AssertTrue(attribute->GetData()->GetNumElems() == 1 || attribute->GetData()->GetNumElems() == 2);
    return attribute->GetData()->GetElemInt();
}

}  // namespace alus::snapengine