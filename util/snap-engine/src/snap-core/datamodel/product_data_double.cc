#include "product_data_double.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <utility>

#include <boost/lexical_cast.hpp>

namespace alus {
namespace snapengine {
Double::Double(int num_elems) : ProductData(ProductData::TYPE_FLOAT64) { array_ = std::vector<double>(num_elems); }

Double::Double(std::vector<double> array) : ProductData(ProductData::TYPE_FLOAT64) { array_ = std::move(array); }

int Double::GetNumElems() const { return array_.size(); }

void Double::Dispose() { array_.clear(); }

int Double::GetElemIntAt(int index) const { return std::round(array_.at(index)); }
long Double::GetElemUIntAt(int index) const { return std::round(array_.at(index)); }
long Double::GetElemLongAt(int index) const { return std::round(array_.at(index)); }
float Double::GetElemFloatAt(int index) const { return (float)array_.at(index); }
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
void Double::SetElemUIntAt(int index, long value) { array_.at(index) = value; }
void Double::SetElemLongAt(int index, long value) { array_.at(index) = value; }
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
    } else {
        //        return array_ == ((std::shared_ptr<Double>)other)->GetArray();
        return (array_ == std::any_cast<std::vector<double>>(other->GetElems()));
    }
}
void Double::SetElemStringAt(int index, std::string_view value) {
    array_.at(index) = boost::lexical_cast<double>(value);
}
}  // namespace snapengine
}  // namespace alus