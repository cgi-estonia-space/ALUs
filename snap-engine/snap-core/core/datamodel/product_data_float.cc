/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class Float which is inside org.esa.snap.core.datamodel.ProductData.java
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
#include "product_data_float.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <boost/lexical_cast.hpp>

namespace alus::snapengine {

Float::Float(int num_elems) : ProductData(ProductData::TYPE_FLOAT32) { array_ = std::vector<float>(num_elems); }

Float::Float(std::vector<float> array) : ProductData(ProductData::TYPE_FLOAT32) { array_ = std::move(array); }

int Float::GetNumElems() const { return array_.size(); }

void Float::Dispose() { array_.clear(); }

int Float::GetElemIntAt(int index) const { return static_cast<int>(std::round(array_.at(index))); }
int64_t Float::GetElemUIntAt(int index) const { return static_cast<int64_t>(std::round(array_.at(index))); }
int64_t Float::GetElemLongAt(int index) const { return static_cast<int64_t>(std::round(array_.at(index))); }
float Float::GetElemFloatAt(int index) const { return array_.at(index); }
double Float::GetElemDoubleAt(int index) const { return array_.at(index); }
std::string Float::GetElemStringAt(int index) const { return std::to_string(array_.at(index)); }

void Float::SetElemIntAt(int index, int value) { array_.at(index) = static_cast<float>(value); }
void Float::SetElemUIntAt(int index, int64_t value) { array_.at(index) = static_cast<float>(value); }
void Float::SetElemLongAt(int index, int64_t value) { array_.at(index) = static_cast<float>(value); }
void Float::SetElemFloatAt(int index, float value) { array_.at(index) = value; }
void Float::SetElemDoubleAt(int index, double value) { array_.at(index) = static_cast<float>(value); }

std::shared_ptr<ProductData> Float::CreateDeepClone() const {
    //    todo:check if this is correct
    return std::make_shared<Float>(this->array_);
}

// todo: compare to java implementation (not sure if this is what it should be)
std::any Float::GetElems() const { return array_; }

// todo: check java implementation and add additional safty checks
// todo: support strings?
void Float::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<float>)) {
        array_ = std::any_cast<std::vector<float>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        std::transform(string_data.begin(), string_data.end(), array_.begin(),
                       [](const std::string& s) { return std::stof(s); });
    } else {
        throw std::invalid_argument("data is not std::vector<float> or std::vector<std::string>");
    }
}

bool Float::EqualElems(std::shared_ptr<ProductData> other) const {
    if (other.get() == this) {
        return true;
    }
    //        return array_ == ((std::shared_ptr<Float>)other)->GetArray();
    return (array_ == std::any_cast<std::vector<float>>(other->GetElems()));
}
void Float::SetElemStringAt(int index, std::string_view value) { array_.at(index) = boost::lexical_cast<float>(value); }

}  // namespace alus::snapengine