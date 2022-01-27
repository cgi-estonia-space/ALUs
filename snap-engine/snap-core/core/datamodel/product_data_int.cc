/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class Int which is inside org.esa.snap.core.datamodel.ProductData.java
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
#include "product_data_int.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <boost/lexical_cast.hpp>

namespace alus::snapengine {
Int::Int(int num_elems) : Int(num_elems, false) {}

Int::Int(int num_elems, bool is_unsigned) : Int(std::vector<int32_t>(num_elems), is_unsigned) {}

Int::Int(std::vector<int32_t> array, bool is_unsigned)
    : ProductData(is_unsigned ? ProductData::TYPE_UINT32 : ProductData::TYPE_INT32) {
    array_ = std::move(array);
}

Int::Int(std::vector<int32_t> array) : Int(std::move(array), false) {}

int Int::GetNumElems() const { return array_.size(); }

void Int::Dispose() { array_.clear(); }

int Int::GetElemIntAt(int index) const { return array_.at(index); }
int64_t Int::GetElemUIntAt(int index) const { return array_.at(index); }
int64_t Int::GetElemLongAt(int index) const { return array_.at(index); }
float Int::GetElemFloatAt(int index) const { return array_.at(index); }
double Int::GetElemDoubleAt(int index) const { return array_.at(index); }
std::string Int::GetElemStringAt(int index) const { return std::to_string(array_.at(index)); }

void Int::SetElemIntAt(int index, int value) { array_.at(index) = value; }
void Int::SetElemUIntAt(int index, int64_t value) { array_.at(index) = static_cast<int32_t>(value); }
void Int::SetElemLongAt(int index, int64_t value) { array_.at(index) = static_cast<int32_t>(value); }
void Int::SetElemFloatAt(int index, float value) { array_.at(index) = static_cast<int>(std::round(value)); }
void Int::SetElemDoubleAt(int index, double value) { array_.at(index) = static_cast<int>(std::round(value)); }

std::shared_ptr<ProductData> Int::CreateDeepClone() const {
    //    todo:check if this is correct
    return std::make_unique<Int>(this->array_);
}

// todo: compare to java implementation (not sure if this is what it should be)
std::any Int::GetElems() const { return array_; }

void Int::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<int32_t>)) {
        array_ = std::any_cast<std::vector<int32_t>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        std::transform(string_data.begin(), string_data.end(), array_.begin(), [](const std::string& s) {
            // so in a way it could be 64bit and put into 32bit so we check min/max
            if (INT32_MIN <= std::stoll(s) && INT32_MAX >= std::stoll(s)) {
                // std::stol at least 32 and int32_t is exactly 32
                return std::stol(s);
            }
            throw std::out_of_range("value is not int32_t");
        });
    } else {
        throw std::invalid_argument("data is not std::vector<int32_t> or std::vector<std::string>");
    }
}

bool Int::EqualElems(const std::shared_ptr<ProductData> other) const {
    if (other.get() == this) {
        return true;
    }
    if (other->GetElems().type() == typeid(std::vector<int32_t>)) {
        return (array_ == std::any_cast<std::vector<int32_t>>(other->GetElems()));
    }
    return false;
}
void Int::SetElemStringAt(int index, std::string_view value) { array_.at(index) = boost::lexical_cast<int32_t>(value); }

}  // namespace alus::snapengine