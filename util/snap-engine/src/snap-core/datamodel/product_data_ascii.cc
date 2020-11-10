#include "product_data_ascii.h"

#include <algorithm>

namespace alus {
namespace snapengine {

ASCII::ASCII(std::string_view data) : Byte(std::vector<int8_t>(data.begin(), data.end()), ProductData::TYPE_ASCII) {}

ASCII::ASCII(int length) : Byte(length, ProductData::TYPE_ASCII) {}

std::string ASCII::GetElemString() { return std::string(this->array_.begin(), this->array_.end()); }

std::string ASCII::GetElemStringAt(int index) const { return std::string{(char)array_.at(index)}; }
void ASCII::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<int8_t>)) {
        array_ = std::any_cast<std::vector<int8_t>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        std::transform(string_data.begin(), string_data.end(), array_.begin(), [](const std::string &s) {
            auto const v = std::stoi(s);
            if (INT8_MIN <= v && INT8_MAX >= v) {
                return v;
            } else {
                throw std::out_of_range("value is not int8_t");
            }
        });
    } else if (data.type() == typeid(std::string_view)) {
        auto assign_data = std::any_cast<std::string_view>(data);
        array_.assign(assign_data.begin(), assign_data.end());
    } else if (data.type() == typeid(std::string)) {
        auto assign_data = std::any_cast<std::string>(data);
        array_.assign(assign_data.begin(), assign_data.end());
    } else if (data.type() == typeid(const char *)) {
        auto assign_data = std::string(std::any_cast<const char *>(data));
        array_.assign(assign_data.begin(), assign_data.end());
    } else {
        throw std::invalid_argument(
            "data is not std::vector<int8_t> or std::vector<std::string> or std::string or std::string_view or const "
            "char*");
    }
}

}  // namespace snapengine
}  // namespace alus