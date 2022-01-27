/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class Short which is inside org.esa.snap.core.datamodel.ProductData.java
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
#include "product_data_short.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <boost/lexical_cast.hpp>

namespace alus::snapengine {

Short::Short(int num_elems) : Short(num_elems, false) {}

Short::Short(int num_elems, bool is_unsigned) : Short(std::vector<int16_t>(num_elems), is_unsigned) {}

Short::Short(std::vector<int16_t> array, bool is_unsigned)
    : ProductData(is_unsigned ? ProductData::TYPE_UINT16 : ProductData::TYPE_INT16) {
    array_ = std::move(array);
}

Short::Short(std::vector<int16_t> array) : Short(std::move(array), false) {}

int Short::GetNumElems() const { return array_.size(); }

void Short::Dispose() { array_.clear(); }

int Short::GetElemIntAt(int index) const { return array_.at(index); }
int64_t Short::GetElemUIntAt(int index) const { return array_.at(index); }
int64_t Short::GetElemLongAt(int index) const { return array_.at(index); }
float Short::GetElemFloatAt(int index) const { return array_.at(index); }
double Short::GetElemDoubleAt(int index) const { return array_.at(index); }
std::string Short::GetElemStringAt(int index) const { return std::to_string(array_.at(index)); }

void Short::SetElemIntAt(int index, int value) { array_.at(index) = static_cast<int16_t>(value); }
void Short::SetElemUIntAt(int index, int64_t value) { array_.at(index) = static_cast<int16_t>(value); }
void Short::SetElemLongAt(int index, int64_t value) { array_.at(index) = static_cast<int16_t>(value); }
void Short::SetElemFloatAt(int index, float value) { array_.at(index) = static_cast<int16_t>(std::round(value)); }
void Short::SetElemDoubleAt(int index, double value) { array_.at(index) = static_cast<int16_t>(std::round(value)); }
void Short::SetElemStringAt(int index, std::string_view value) {
    array_.at(index) = boost::lexical_cast<int16_t>(value);
}

std::shared_ptr<alus::snapengine::ProductData> alus::snapengine::Short::CreateDeepClone() const {
    //    todo:check if this is correct
    return std::make_shared<Short>(this->array_);
}

// todo: compare to java implementation (not sure if this is what it should be)
std::any Short::GetElems() const { return array_; }

// todo: check java implementation and add additional safty checks
void Short::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<int16_t>)) {
        array_ = std::any_cast<std::vector<int16_t>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        std::transform(string_data.begin(), string_data.end(), array_.begin(), [](const std::string& s) {
            auto const v = std::stoi(s);
            if (INT16_MIN <= v && INT16_MAX >= v) {
                return v;
            }
            throw std::out_of_range("value is not int16_t");
        });
    } else {
        throw std::invalid_argument("data is not std::vector<int16_t> or std::vector<std::string>");
    }
}

bool Short::EqualElems(const std::shared_ptr<ProductData> other) const {
    if (other.get() == this) {
        return true;
    }
    if (other->GetElems().type() == typeid(std::vector<int16_t>)) {
        return (array_ == std::any_cast<std::vector<int16_t>>(other->GetElems()));
    }
    return false;
}

}  // namespace alus::snapengine