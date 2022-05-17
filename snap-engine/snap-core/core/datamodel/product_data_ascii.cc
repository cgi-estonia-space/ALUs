/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class ASCII which is inside org.esa.snap.core.datamodel.ProductData.java
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
#include "product_data_ascii.h"

#include <algorithm>
#include <stdexcept>

namespace alus::snapengine {

ASCII::ASCII(std::string_view data) : Byte(std::vector<int8_t>(data.begin(), data.end()), ProductData::TYPE_ASCII) {}

ASCII::ASCII(int length) : Byte(length, ProductData::TYPE_ASCII) {}

std::string ASCII::GetElemString() { return std::string(this->array_.begin(), this->array_.end()); }

std::string ASCII::GetElemStringAt(int index) const { return std::string{static_cast<char>(array_.at(index))}; }
void ASCII::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<int8_t>)) {
        array_ = std::any_cast<std::vector<int8_t>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        std::transform(string_data.begin(), string_data.end(), array_.begin(), [](const std::string& s) {
            auto const v = std::stoi(s);
            if (INT8_MIN <= v && INT8_MAX >= v) {
                return v;
            }
            throw std::out_of_range("value is not int8_t");
        });
    } else if (data.type() == typeid(std::string_view)) {
        auto assign_data = std::any_cast<std::string_view>(data);
        array_.assign(assign_data.begin(), assign_data.end());
    } else if (data.type() == typeid(std::string)) {
        auto assign_data = std::any_cast<std::string>(data);
        array_.assign(assign_data.begin(), assign_data.end());
    } else if (data.type() == typeid(const char*)) {
        auto assign_data = std::string(std::any_cast<const char*>(data));
        array_.assign(assign_data.begin(), assign_data.end());
    } else {
        throw std::invalid_argument(
            "data is not std::vector<int8_t> or std::vector<std::string> or std::string or std::string_view or const "
            "char*");
    }
}
std::shared_ptr<ProductData> ASCII::CreateDeepClone() const {
    std::shared_ptr<ASCII> data = std::make_shared<ASCII>(array_.size());

    std::copy(array_.begin(), array_.end(), data->array_.begin());
    //    final ASCII data = new ASCII(_array.length);
    //    System.arraycopy(_array, 0, data._array, 0, _array.length);
    //    return data;
    return data;
}

}  // namespace alus::snapengine