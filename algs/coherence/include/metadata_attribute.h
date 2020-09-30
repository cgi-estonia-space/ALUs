/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.datamodel.MetadataAttribute.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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
#pragma once

#include "data_node.h"
#include "product_data.h"

namespace alus::snapengine {
// todo: snap extends DataNode which extends ProductNode (change if need arises)
class MetadataAttribute : public DataNode {
   public:
    MetadataAttribute(std::string_view name, int type) : MetadataAttribute(name, type, 1) {}
    MetadataAttribute(std::string_view name, int type, int num_elems)
        : MetadataAttribute(name, ProductData::CreateInstance(type, num_elems), false) {}
    MetadataAttribute(std::string_view name, std::shared_ptr<ProductData> data, bool read_only)
        : DataNode(name, data, read_only) {}

    bool Equals(MetadataAttribute &object) { return (object.GetData() == GetData()); }

    std::shared_ptr<MetadataAttribute> CreateDeepClone() {
        auto clone = std::make_shared<MetadataAttribute>(GetName(), (GetData())->CreateDeepClone(), IsReadOnly());
        clone->SetDescription(GetDescription());
        clone->SetSynthetic(IsSynthetic());
        clone->SetUnit(GetUnit());
        return clone;
    }
};

}  // namespace alus::snapengine