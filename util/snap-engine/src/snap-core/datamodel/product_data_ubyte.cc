#include "product_data_ubyte.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <boost/lexical_cast.hpp>

namespace alus {
namespace snapengine {

UByte::UByte(int num_elems) : UByte(num_elems, true) {}

UByte::UByte(int num_elems, bool is_unsigned) : UByte(std::vector<uint8_t>(num_elems), is_unsigned) {}

UByte::UByte(std::vector<uint8_t> array, bool is_unsigned)
    : ProductData(is_unsigned ? ProductData::TYPE_UINT8 : ProductData::TYPE_INT8) {
    array_ = std::move(array);
}

UByte::UByte(int num_elems, int type) : ProductData(type) { array_ = std::vector<uint8_t>(num_elems); }

UByte::UByte(std::vector<uint8_t> array) : UByte(std::move(array), true) {}

UByte::UByte(std::vector<uint8_t> array, int type) : ProductData(type) { array_ = std::move(array); }

int UByte::GetNumElems() const { return array_.size(); }

void UByte::Dispose() { array_.clear(); }

int UByte::GetElemIntAt(int index) const { return array_.at(index); }
long UByte::GetElemUIntAt(int index) const { return array_.at(index); }
long UByte::GetElemLongAt(int index) const { return array_.at(index); }
float UByte::GetElemFloatAt(int index) const { return array_.at(index); }
double UByte::GetElemDoubleAt(int index) const { return array_.at(index); }
std::string UByte::GetElemStringAt(int index) const { return std::to_string(array_.at(index)); }

void UByte::SetElemIntAt(int index, int value) { array_.at(index) = (uint8_t)value; }
void UByte::SetElemUIntAt(int index, long value) { array_.at(index) = (uint8_t)value; }
void UByte::SetElemLongAt(int index, long value) { array_.at(index) = (uint8_t)value; }
void UByte::SetElemFloatAt(int index, float value) { array_.at(index) = (uint8_t)std::round(value); }
void UByte::SetElemDoubleAt(int index, double value) { array_.at(index) = (uint8_t)std::round(value); }

std::shared_ptr<ProductData> UByte::CreateDeepClone() const {
    //    todo:check if this is correct
    return std::make_shared<UByte>(this->array_);
}

// todo: compare to java implementation (not sure if this is what it should be)
std::any UByte::GetElems() const { return array_; }

void UByte::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<uint8_t>)) {
        array_ = std::any_cast<std::vector<uint8_t>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        std::transform(string_data.begin(), string_data.end(), array_.begin(), [](const std::string &s) {
            if (UINT8_MAX >= std::stoul(s)) {
                return std::stoul(s);
            } else {
                throw std::out_of_range("value is not uint8_t");
            }
        });
    } else {
        throw std::invalid_argument("data is not std::vector<uint8_t> or std::vector<std::string>");
    }
}

bool UByte::EqualElems(const std::shared_ptr<ProductData> other) const {
    if(other.get() == this){
        return true;
    }else if (other->GetElems().type() == typeid(std::vector<uint8_t>)) {
        return (array_ == std::any_cast<std::vector<uint8_t>>(other->GetElems()));
    }
    return false;
}
void UByte::SetElemStringAt(int index, std::string_view value) {
    array_.at(index) = boost::lexical_cast<uint8_t>(value);
}

}  // namespace snapengine
}  // namespace alus