#include "data_node.h"

#include <cstdint>

#include <boost/format.hpp>

#include "guardian.h"

namespace alus::snapengine {

DataNode::DataNode(std::string_view name, int data_type, long num_elems) : ProductNode(name) {
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
void DataNode::SetUnit(std::optional<std::string> unit) { unit_ = unit; }

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
                                        std::to_string((int64_t)INT32_MAX + 1));
        }
        data_ = CreateCompatibleProductData((int32_t)num_elems_);
    }

    //    todo::better use ProductData* here? check if any has same types and then anycast and equality check?
    //    ProductData* old_data = data_->G;
    // todo: make sure its not just same reference when comparing
    std::any old_data = data_->GetElems();
    // todo::need a good equality check for std::any solution
    //    if (!ObjectUtils::EqualObjects(old_data, elems)) {
    //    old_data==elems
    //    if (!ObjectUtils::EqualObjects(old_data, elems)) {
    data_->SetElems(elems);
    //    }

    //!!!!!!!!!!!THIS IS EXAMPLE HOW TO SOLVE IT!!!!!!!!!!!!
    //    bool UByte::EqualElems(const ProductData *other) const {
    //        if(other == this){
    //            return true;
    //        }else if (other->GetElems().type() == typeid(std::vector<uint8_t>)) {
    //            return (array_ == std::any_cast<std::vector<uint8_t>>(other->GetElems()));
    //        }
    //        return false;
    //    }
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

}  // namespace alus::snapengine