/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class Double which is inside org.esa.snap.core.datamodel.ProductData.java
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
#include "product_data_double.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <utility>

#include <boost/lexical_cast.hpp>

namespace alus::snapengine {
Double::Double(int num_elems) : ProductData(ProductData::TYPE_FLOAT64) { array_ = std::vector<double>(num_elems); }

Double::Double(std::vector<double> array) : ProductData(ProductData::TYPE_FLOAT64) { array_ = std::move(array); }

int Double::GetNumElems() const { return array_.size(); }

void Double::Dispose() { array_.clear(); }

int Double::GetElemIntAt(int index) const { return static_cast<int>(std::round(array_.at(index))); }
int64_t Double::GetElemUIntAt(int index) const { return static_cast<int64_t>(std::round(array_.at(index))); }
int64_t Double::GetElemLongAt(int index) const { return static_cast<int64_t>(std::round(array_.at(index))); }
float Double::GetElemFloatAt(int index) const { return static_cast<float>(array_.at(index)); }
double Double::GetElemDoubleAt(int index) const { return array_.at(index); }
std::string Double::GetElemStringAt(int index) const {
    std::ostringstream out;
    double integral;
    if (std::modf(array_.at(index), &integral) == 0) {
        out << std::fixed << std::setprecision(1) << integral;
    } else {
        out << std::setprecision(std::numeric_limits<double>::max_digits10) << array_.at(index);
    }
    return out.str();
}

void Double::SetElemIntAt(int index, int value) { array_.at(index) = value; }
void Double::SetElemUIntAt(int index, int64_t value) { array_.at(index) = value; }
void Double::SetElemLongAt(int index, int64_t value) { array_.at(index) = value; }
void Double::SetElemFloatAt(int index, float value) { array_.at(index) = value; }
void Double::SetElemDoubleAt(int index, double value) { array_.at(index) = value; }

std::shared_ptr<alus::snapengine::ProductData> Double::CreateDeepClone() const {
    //    todo:check if this is correct
    return std::make_shared<Double>(this->array_);
}

// todo: compare to java implementation (not sure if this is what it should be)
std::any Double::GetElems() const { return array_; }

// todo: check java implementation and add additional safty checks
void Double::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<double>)) {
        array_ = std::any_cast<std::vector<double>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        std::transform(string_data.begin(), string_data.end(), array_.begin(),
                       [](const std::string& s) { return std::stod(s); });
    } else {
        throw std::invalid_argument("data is not std::vector<double> or std::vector<std::string>");
    }
}

bool Double::EqualElems(const std::shared_ptr<ProductData> other) const {
    if (other.get() == this) {
        return true;
    }
    //        return array_ == ((std::shared_ptr<Double>)other)->GetArray();
    return (array_ == std::any_cast<std::vector<double>>(other->GetElems()));
}
void Double::SetElemStringAt(int index, std::string_view value) {
    array_.at(index) = boost::lexical_cast<double>(value);
}
}  // namespace alus::snapengine