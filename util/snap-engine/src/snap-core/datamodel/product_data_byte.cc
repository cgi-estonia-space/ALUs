/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class Byte which is inside org.esa.snap.core.datamodel.ProductData.java
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
#include "product_data_byte.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <boost/lexical_cast.hpp>

namespace alus {
namespace snapengine {

Byte::Byte(int num_elems) : Byte(num_elems, false) {}

Byte::Byte(int num_elems, bool is_unsigned) : Byte(std::vector<int8_t>(num_elems), is_unsigned) {}

Byte::Byte(std::vector<int8_t> array, bool is_unsigned)
    : ProductData(is_unsigned ? ProductData::TYPE_UINT8 : ProductData::TYPE_INT8) {
    array_ = std::move(array);
}

Byte::Byte(int num_elems, int type) : ProductData(type) { array_ = std::move(std::vector<int8_t>(num_elems)); }

Byte::Byte(std::vector<int8_t> array) : Byte(std::move(array), false) {}

Byte::Byte(std::vector<int8_t> array, int type) : ProductData(type) { array_ = std::move(array); }

int Byte::GetNumElems() const { return array_.size(); }

void Byte::Dispose() { array_.clear(); }

int Byte::GetElemIntAt(int index) const { return array_.at(index); }
long Byte::GetElemUIntAt(int index) const { return array_.at(index); }
long Byte::GetElemLongAt(int index) const { return array_.at(index); }
float Byte::GetElemFloatAt(int index) const { return array_.at(index); }
double Byte::GetElemDoubleAt(int index) const { return array_.at(index); }
std::string Byte::GetElemStringAt(int index) const { return std::to_string(array_.at(index)); }

void Byte::SetElemIntAt(int index, int value) { array_.at(index) = (int8_t)value; }
void Byte::SetElemUIntAt(int index, long value) { array_.at(index) = (int8_t)value; }
void Byte::SetElemLongAt(int index, long value) { array_.at(index) = (int8_t)value; }
void Byte::SetElemFloatAt(int index, float value) { array_.at(index) = (int8_t)std::round(value); }
void Byte::SetElemDoubleAt(int index, double value) { array_.at(index) = (int8_t)std::round(value); }

std::shared_ptr<ProductData> Byte::CreateDeepClone() const {
    //    todo:check if this is correct
    return std::make_shared<Byte>(this->array_);
}

std::any Byte::GetElems() const { return array_; }

void Byte::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<int8_t>)) {
        array_ = std::any_cast<std::vector<int8_t>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        std::transform(string_data.begin(), string_data.end(), array_.begin(), [](const std::string& s) {
            auto const v = std::stoi(s);
            if (INT8_MIN <= v && INT8_MAX >= v) {
                return v;
            } else {
                throw std::out_of_range("value is not int8_t");
            }
        });
    } else {
        throw std::invalid_argument("data is not std::vector<int8_t> or std::vector<std::string>");
    }
}

bool Byte::EqualElems(const std::shared_ptr<ProductData> other) const {
    if (other.get() == this) {
        return true;
    } else if (other->GetElems().type() == typeid(std::vector<int8_t>)) {
        return (array_ == std::any_cast<std::vector<int8_t>>(other->GetElems()));
    }
    return false;
}
void Byte::SetElemStringAt(int index, std::string_view value) {
    array_.at(index) = boost::lexical_cast<int16_t>(value);
}

}  // namespace snapengine
}  // namespace alus