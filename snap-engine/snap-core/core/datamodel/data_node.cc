/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.DataNode.java
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
#include "data_node.h"

#include <cstdint>

#include <boost/format.hpp>

#include "../util/guardian.h"

namespace alus::snapengine {

DataNode::DataNode(std::string_view name, int data_type, int64_t num_elems) : ProductNode(name) {
    if (data_type != ProductData::TYPE_INT8 && data_type != ProductData::TYPE_INT16 &&
        data_type != ProductData::TYPE_INT32 && data_type != ProductData::TYPE_UINT8 &&
        data_type != ProductData::TYPE_UINT16 && data_type != ProductData::TYPE_UINT32 &&
        data_type != ProductData::TYPE_FLOAT32 && data_type != ProductData::TYPE_FLOAT64 &&
        data_type != ProductData::TYPE_ASCII && data_type != ProductData::TYPE_UTC) {
        throw std::invalid_argument("dataType is invalid");
    }
    data_type_ = data_type;
    num_elems_ = num_elems;
    data_ = nullptr;
    read_only_ = false;
}
DataNode::DataNode(std::string_view name, std::shared_ptr<ProductData> data, bool read_only) : ProductNode(name) {
    Guardian::AssertNotNull("data", data);
    data_type_ = data->GetType();
    num_elems_ = data->GetNumElems();
    data_ = std::move(data);
    read_only_ = read_only;
}

void DataNode::SetUnit(std::string_view unit) { unit_ = unit; }
void DataNode::SetUnit(std::optional<std::string> unit) { unit_ = std::move(unit); }

void DataNode::SetSynthetic(bool synthetic) { synthetic_ = synthetic; }

std::shared_ptr<ProductData> DataNode::CreateCompatibleProductData(const int num_elems) const {
    return ProductData::CreateInstance(GetDataType(), num_elems);
}

void DataNode::SetDataElems(const std::any& elems) {
    if (IsReadOnly()) {
        throw std::invalid_argument("attribute is read-only");
    }
    CheckState();
    if (data_ == nullptr) {
        // todo:investigate this limit (ported from java)
        if (num_elems_ > INT32_MAX) {
            throw std::invalid_argument("number of elements must be less than " +
                                        std::to_string(static_cast<int64_t> INT32_MAX + 1));
        }
        data_ = CreateCompatibleProductData(static_cast<int32_t>(num_elems_));
    }

    std::any old_data = data_->GetElems();
    //    todo: snap java version checks for changes, if we need this must be added here
    data_->SetElems(elems);
}
void DataNode::SetReadOnly(bool read_only) {
    bool old_value = read_only_;
    if (old_value != read_only) {
        read_only_ = read_only;
        SetModified(true);
    }
}

void DataNode::CheckDataCompatibility(const std::shared_ptr<ProductData>& data) {
    //    todo: check this debug and provide similar functionality
    //    Debug::AssertNotNull(data);

    if (data->GetType() != GetDataType()) {
        std::string msg_pattern = "Illegal data for data node ''%1%'', type %2% expected";
        throw std::invalid_argument(
            boost::str(boost::format(msg_pattern) % GetName() % ProductData::GetTypeString(GetDataType())));
    }

    if (data->GetNumElems() != GetNumDataElems()) {
        std::string msg_pattern =
            "Illegal number of data elements for data node ''%1%'', %2% elements expected but was %3%";
        throw std::invalid_argument(
            boost::str(boost::format(msg_pattern) % GetName() % GetNumDataElems() % data->GetNumElems()));
    }
}

void DataNode::SetData(const std::shared_ptr<ProductData>& data) {
    if (IsReadOnly()) {
        throw std::invalid_argument("data node '" + std::string(GetName()) + "' is read-only");
    }

    if (data_ == data) {
        return;
    }

    if (data != nullptr) {
        CheckDataCompatibility(data);
    }

    std::shared_ptr<ProductData> old_data = data_;
    data_ = data;

    // if data node already had data before, mark that it has been modified so
    // new data is stored on next incremental save
    if (old_data != nullptr) {
        SetModified(true);
    }
}

void DataNode::Dispose() {
    if (data_ != nullptr) {
        data_->Dispose();
        data_ = nullptr;
    }
    ProductNode::Dispose();
}

uint64_t DataNode::GetRawStorageSize(const std::shared_ptr<ProductSubsetDef>& subset_def) {
    uint64_t size = 0;
    const int estimated_overhead{256};
    if (IsPartOfSubset(subset_def)) {
        size += estimated_overhead;
        size += ProductData::GetElemSize(GetDataType()) * GetNumDataElems();
    }
    return size;
}

}  // namespace alus::snapengine