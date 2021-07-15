/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class UShort which is inside org.esa.snap.core.datamodel.ProductData.java
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
#include "product_data_ushort.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <boost/lexical_cast.hpp>

namespace alus {
namespace snapengine {
UShort::UShort(int num_elems) : UShort(num_elems, true) {}

UShort::UShort(int num_elems, bool is_unsigned) : UShort(std::vector<uint16_t>(num_elems), is_unsigned) {}

UShort::UShort(std::vector<uint16_t> array, bool is_unsigned)
    : ProductData(is_unsigned ? ProductData::TYPE_UINT16 : ProductData::TYPE_INT16) {
    array_ = std::move(array);
}

UShort::UShort(std::vector<uint16_t> array) : UShort(std::move(array), true) {}

int UShort::GetNumElems() const { return array_.size(); }

void UShort::Dispose() { array_.clear(); }

int UShort::GetElemIntAt(int index) const { return array_.at(index); }
long UShort::GetElemUIntAt(int index) const { return array_.at(index); }
long UShort::GetElemLongAt(int index) const { return array_.at(index); }
float UShort::GetElemFloatAt(int index) const { return array_.at(index); }
double UShort::GetElemDoubleAt(int index) const { return array_.at(index); }
std::string UShort::GetElemStringAt(int index) const { return std::to_string(array_.at(index)); }

void UShort::SetElemIntAt(int index, int value) { array_.at(index) = (uint16_t)value; }
void UShort::SetElemUIntAt(int index, long value) { array_.at(index) = (uint16_t)value; }
void UShort::SetElemLongAt(int index, long value) { array_.at(index) = (uint16_t)value; }
void UShort::SetElemFloatAt(int index, float value) { array_.at(index) = (uint16_t)std::round(value); }
void UShort::SetElemDoubleAt(int index, double value) { array_.at(index) = (uint16_t)std::round(value); }

std::shared_ptr<alus::snapengine::ProductData> alus::snapengine::UShort::CreateDeepClone() const {
    //    todo:check if this is correct
    return std::make_shared<UShort>(this->array_);
}

// todo: compare to java implementation (not sure if this is what it should be)
std::any UShort::GetElems() const { return array_; }

// todo: check java implementation and add additional safty checks
// todo: support strings?
void UShort::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<uint16_t>)) {
        array_ = std::any_cast<std::vector<uint16_t>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        std::transform(string_data.begin(), string_data.end(), array_.begin(), [](const std::string& s) {
            if (UINT16_MAX >= std::stoul(s)) {
                return std::stoul(s);
            } else {
                throw std::out_of_range("value is not of type uint16_t");
            }
        });
    } else {
        throw std::invalid_argument("data is not std::vector<uint16_t> or std::vector<std::string>");
    }
}

bool UShort::EqualElems(const std::shared_ptr<ProductData> other) const {
    if (other.get() == this) {
        return true;
    } else if (other->GetElems().type() == typeid(std::vector<uint16_t>)) {
        return (array_ == std::any_cast<std::vector<uint16_t>>(other->GetElems()));
    }
    return false;
}
void UShort::SetElemStringAt(int index, std::string_view value) {
    array_.at(index) = boost::lexical_cast<uint16_t>(value);
}

}  // namespace snapengine
}  // namespace alus
