#include "product_data_long.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <boost/lexical_cast.hpp>

namespace alus {
namespace snapengine {

Long::Long(int num_elems) : Long(std::vector<int64_t>(num_elems)){};

Long::Long(std::vector<int64_t> array) : ProductData(ProductData::TYPE_INT64) { array_ = std::move(array); }

int Long::GetNumElems() const { return array_.size(); }

void Long::Dispose() { array_.clear(); }

int Long::GetElemIntAt(int index) const { return array_.at(index); }
long Long::GetElemUIntAt(int index) const { return array_.at(index); }
long Long::GetElemLongAt(int index) const { return array_.at(index); }
float Long::GetElemFloatAt(int index) const { return array_.at(index); }
double Long::GetElemDoubleAt(int index) const { return array_.at(index); }
std::string Long::GetElemStringAt(int index) const { return std::to_string(array_.at(index)); }

void Long::SetElemIntAt(int index, int value) { array_.at(index) = value; }
void Long::SetElemUIntAt(int index, long value) { array_.at(index) = value; }
void Long::SetElemLongAt(int index, long value) { array_.at(index) = value; }
void Long::SetElemFloatAt(int index, float value) { array_.at(index) = std::round(value); }
void Long::SetElemDoubleAt(int index, double value) { array_.at(index) = std::round(value); }

std::shared_ptr<ProductData> Long::CreateDeepClone() const {
    //    todo:check if this is correct
    return std::make_shared<Long>(this->array_);
}

// todo: compare to java implementation (not sure if this is what it should be)
std::any Long::GetElems() const { return array_; }

// todo: check java implementation and add additional safty checks
void Long::SetElems(std::any data) {
    if (data.type() == typeid(std::vector<int64_t>)) {
        array_ = std::any_cast<std::vector<int64_t>>(data);
    } else if (data.type() == typeid(std::vector<std::string>)) {
        auto string_data = std::any_cast<std::vector<std::string>>(data);
        std::transform(
            string_data.begin(), string_data.end(), array_.begin(), [](const std::string &s) { return std::stoll(s); });
    } else {
        throw std::invalid_argument("data is not std::vector<int64_t> or std::vector<std::string>");
    }
}

bool Long::EqualElems(const std::shared_ptr<ProductData> other) const {
    if(other.get() == this){
        return true;
    }else if (other->GetElems().type() == typeid(std::vector<int64_t>)) {
        return (array_ == std::any_cast<std::vector<int64_t>>(other->GetElems()));
    }
    return false;
}
void Long::SetElemStringAt(int index, std::string_view value) {
    array_.at(index) = boost::lexical_cast<int64_t>(value);
}

}  // namespace snapengine
}  // namespace alus