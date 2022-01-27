/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class UInt which is inside org.esa.snap.core.datamodel.ProductData.java
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
#include "product_data_uint.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "boost/lexical_cast.hpp"

namespace alus::snapengine {

UInt::UInt(int num_elems) : UInt(num_elems, true) {}

UInt::UInt(int num_elems, bool is_unsigned) : UInt(std::vector<uint32_t>(num_elems), is_unsigned) {}

UInt::UInt(std::vector<uint32_t> array, bool is_unsigned)
    : ProductData(is_unsigned ? ProductData::TYPE_UINT32 : ProductData::TYPE_INT32) {
    array_ = std::move(array);
}

UInt::UInt(std::vector<uint32_t> array) : UInt(std::move(array), true) {}

int UInt::GetNumElems() const { return array_.size(); }

void UInt::Dispose() { array_.clear(); }

int UInt::GetElemIntAt(int index) const { return array_.at(index); }
int64_t UInt::GetElemUIntAt(int index) const { return array_.at(index); }
int64_t UInt::GetElemLongAt(int index) const { return array_.at(index); }
float UInt::GetElemFloatAt(int index) const { return array_.at(index); }
double UInt::GetElemDoubleAt(int index) const { return array_.at(index); }
std::string UInt::GetElemStringAt(int index) const { return std::to_string(array_.at(index)); }

void UInt::SetElemIntAt(int index, int value) { array_.at(index) = value; }
void UInt::SetElemUIntAt(int index, int64_t value) { array_.at(index) = static_cast<uint32_t>(value); }
void UInt::SetElemLongAt(int index, int64_t value) { array_.at(index) = static_cast<uint32_t>(value); }
void UInt::SetElemFloatAt(int index, float value) { array_.at(index) = static_cast<uint32_t>(std::round(value)); }
void UInt::SetElemDoubleAt(int index, double value) { array_.at(index) = static_cast<uint32_t>(std::round(value)); }

std::shared_ptr<alus::snapengine::ProductData> UInt::CreateDeepClone() const {
    //    todo:check if this is correct
    return std::make_shared<UInt>(this->array_);
}

// todo: compare to java implementation (not sure if this is what it should be)
std::any UInt::GetElems() const { return array_; }

// todo: check java implementation and add additional safty checks
// todo: support strings?
void UInt::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<uint32_t>)) {
        array_ = std::any_cast<std::vector<uint32_t>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        // we need to check stoi vs. stoll
        std::transform(string_data.begin(), string_data.end(), array_.begin(), [](const std::string& s) {
            if (UINT32_MAX >= std::stoull(s)) {
                return std::stoul(s);
            }
            throw std::out_of_range("value is not of type uint32_t");
        });
    } else {
        throw std::invalid_argument("data is not std::vector<uint32_t> or std::vector<std::string>");
    }
}

bool UInt::EqualElems(const std::shared_ptr<ProductData> other) const {
    if (other.get() == this) {
        return true;
    }
    if (other->GetElems().type() == typeid(std::vector<uint32_t>)) {
        return (array_ == std::any_cast<std::vector<uint32_t>>(other->GetElems()));
    }
    return false;
}
void UInt::SetElemStringAt(int index, std::string_view value) {
    array_.at(index) = boost::lexical_cast<uint32_t>(value);
}
/**
 * Returns the internal data array holding this value's data elements.
 *
 * @return the internal data array, never {@code null}
 */
std::vector<uint32_t> UInt::GetArray() const { return array_; }

}  // namespace alus::snapengine