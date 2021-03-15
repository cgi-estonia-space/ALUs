#include "product_subset_def.h"

#include <algorithm>

#include <boost/algorithm/string.hpp>

#include "guardian.h"
#include "pixel_subset_region.h"

namespace alus::snapengine {

int ProductSubsetDef::GetNodeNameIndex(std::string_view name) {
    if (!node_name_list_.empty()) {
        for (size_t i = 0; i < node_name_list_.size(); i++) {
            auto node_name = node_name_list_.at(i);
            if (boost::iequals(node_name, name)) {
                return i;
            }
        }
    }
    return -1;
}

ProductSubsetDef::ProductSubsetDef(std::string_view subset_name) : subset_name_(subset_name) {}

ProductSubsetDef::ProductSubsetDef() : ProductSubsetDef("") {}

std::vector<std::string> ProductSubsetDef::GetNodeNames() {
    // todo: local variable node_name_list_ needs more thinking
    if (node_name_list_.empty()) {
        return node_name_list_;
    }
    std::vector<std::string> result(node_name_list_.size());
    for (size_t i = 0; i < node_name_list_.size(); i++) {
        result.at(i) = node_name_list_.at(i);
    }
    return result;
}

void ProductSubsetDef::SetNodeNames(const std::vector<std::string>& names) {
    if (!names.empty()) {
        if (!node_name_list_.empty()) {
            node_name_list_.clear();
        }
        AddNodeNames(names);
    } else {
        node_name_list_.clear();
    }
}

void ProductSubsetDef::AddNodeName(std::string_view name) {
    Guardian::AssertNotNullOrEmpty("name", name);
    if (ContainsNodeName(name)) {
        return;
    }
    node_name_list_.emplace_back(name);
}

void ProductSubsetDef::AddNodeNames(const std::vector<std::string>& names) {
    if (names.empty()) {
        return;
    }
    for (const auto& name : names) {
        AddNodeName(name);
    }
}

void ProductSubsetDef::AddNodeNames(const std::unordered_set<std::string>& names) {
    if (names.empty()) {
        return;
    }
    for (const auto& name : names) {
        AddNodeName(name);
    }
}

bool ProductSubsetDef::RemoveNodeName(std::string_view name) {
    int index = GetNodeNameIndex(name);
    if (index < 0) {
        return false;
    }
    node_name_list_.erase(node_name_list_.begin() + index);
    return true;
}

bool ProductSubsetDef::ContainsNodeName(std::string_view name) { return GetNodeNameIndex(name) >= 0; }

bool ProductSubsetDef::IsNodeAccepted(std::string_view name) {
    return node_name_list_.empty() || ContainsNodeName(name);
}

void ProductSubsetDef::SetSubSampling(int sub_sampling_x, int sub_sampling_y) {
    if (sub_sampling_x < 1 || sub_sampling_y < 1) {
        throw std::invalid_argument("invalid sub-sampling");
    }
    sub_sampling_x_ = sub_sampling_x;
    sub_sampling_y_ = sub_sampling_y;
}

bool ProductSubsetDef::IsEntireProductSelected() {
    // return region == null && subSamplingX == 1 && subSamplingY == 1 && nodeNameList == null && !ignoreMetadata;
    return (subset_region_ == nullptr) && sub_sampling_x_ == 1 && sub_sampling_y_ == 1 && node_name_list_.empty() &&
           !ignore_metadata_;
}

void ProductSubsetDef::SetSubsetRegion(const std::shared_ptr<AbstractSubsetRegion>& subset_region) {
    subset_region_ = subset_region;
}

void ProductSubsetDef::SetIgnoreMetadata(bool ignore_metadata) { ignore_metadata_ = ignore_metadata; }

void ProductSubsetDef::SetRegionMap(const std::unordered_map<std::string, custom::Rectangle>& region_map) {
    region_map_ = region_map;
}

std::shared_ptr<custom::Rectangle> ProductSubsetDef::GetRegion() {
    // todo: this might not work correctly (instance of logic), modify if needed
    if (subset_region_ != nullptr && std::dynamic_pointer_cast<PixelSubsetRegion>(subset_region_) != nullptr) {
        auto pixel_subset_region = std::dynamic_pointer_cast<PixelSubsetRegion>(subset_region_);
        return std::make_shared<custom::Rectangle>(pixel_subset_region->GetPixelRegion());
    }
    return nullptr;
}

}  // namespace alus::snapengine
