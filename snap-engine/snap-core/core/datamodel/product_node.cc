/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductNode.java
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
#include "snap-core/core/datamodel/product_node.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "../util/guardian.h"
#include "ceres-core/core/ceres_assert.h"
#include "snap-core/core/dataio/product_subset_def.h"
#include "snap-core/core/datamodel/product.h"

namespace alus::snapengine {

void ProductNode::SetDescription(std::string_view description) { description_ = description; }
void ProductNode::SetDescription(const std::optional<std::string_view>& description) { description_ = description; }

ProductNode::ProductNode(std::string_view name) : ProductNode(name, std::nullopt) {}

ProductNode::ProductNode(std::string_view name, const std::optional<std::string_view>& description) {
    std::string name_str{std::string(name)};
    boost::algorithm::trim(name_str);
    Guardian::AssertNotNullOrEmpty("name", name_str);
    name_ = name;
    description_ = description;
}
void ProductNode::SetOwner(ProductNode* owner) {
    if (owner != owner_) {
        owner_ = owner;
    }
}
Product* ProductNode::GetProduct() {
    //            todo:add support for thread safty (java had synchronized)
    // todo: also check if instanceof logic from java has been replaced like needed
    if (product_ == nullptr) {
        auto* owner = this;
        do {
            auto* check_type = dynamic_cast<Product*>(owner);
            if (check_type != nullptr) {
                product_ = check_type;
                break;
            }
            owner = owner->GetOwner();
        } while (owner != nullptr);
    }
    return product_;
}

void ProductNode::SetModified(bool modified) {
    bool old_state = modified_;
    if (old_state != modified) {
        modified_ = modified;
        auto* owner = GetOwner();
        if (modified_ && owner) {
            owner->SetModified(true);
        }
    }
}

std::shared_ptr<IProductReader> ProductNode::GetProductReader() {
    auto* product = GetProduct();
    if (product) {
        return product->GetProductReader();
    }
    return nullptr;
}

std::shared_ptr<IProductWriter> ProductNode::GetProductWriter() {
    if (auto* product = GetProduct(); product) {
        return product->GetProductWriter();
    }
    return nullptr;
}

std::string ProductNode::GetDisplayName() {
    auto prefix = GetProductRefString();
    if (!prefix.has_value()) {
        return GetName();
    }
    return prefix.value() + " " + GetName();
}

std::optional<std::string> ProductNode::GetProductRefString() {
    auto* product = GetProduct();
    if (product) {
        return std::make_optional(product->GetRefStr());
    }
    return std::nullopt;
}
void ProductNode::SetName(std::string_view name) {
    Guardian::AssertNotNull("name", name);  // NOLINT
    std::string name_str(name);
    boost::algorithm::trim(name_str);
    SetNodeName(name_str, false);
}

void ProductNode::SetNodeName(std::string_view trimmed_name, bool silent) {
    Guardian::AssertNotNullOrEmpty("name contains only spaces", trimmed_name);
    if (name_ != trimmed_name) {
        auto* product = GetProduct();
        if (product) {
            Assert::Argument(!product->ContainsRasterDataNode(trimmed_name),
                             "The Product '" + product->GetName() + "' already contains " +
                                 "a raster data node with the name '" + std::string(trimmed_name) + "'.");
        }
        if (!IsValidNodeName(trimmed_name)) {
            throw std::invalid_argument("The given name '" + std::string(trimmed_name) + "' is not a valid node name.");
        }
        name_ = trimmed_name;
        if (!silent) {
            SetModified(true);
        }
    }
}
bool ProductNode::IsValidNodeName(std::string_view name) {
    if (name == "" || boost::iequals("or", name) || boost::iequals("and", name) || boost::iequals("not", name)) {
        return false;
    }
    std::string name_str(name);
    boost::algorithm::trim(name_str);
    return std::regex_match(name_str, std::regex(R"([^\\/:*?"<>|\.][^\\/:*?"<>|]*)"));
}
bool ProductNode::IsPartOfSubset(const std::shared_ptr<ProductSubsetDef>& subset_def) const {
    return subset_def == nullptr || subset_def->ContainsNodeName(GetName());
}
void ProductNode::Dispose() {
    owner_ = nullptr;
    product_ = nullptr;
    description_ = std::nullopt;
    name_ = "";
}

}  // namespace alus::snapengine