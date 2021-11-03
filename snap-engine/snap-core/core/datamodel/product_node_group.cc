/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductNodeGroup.java
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
#include "product_node_group.h"

#include "ceres-core/core/ceres_assert.h"
#include "snap-core/core/dataio/product_subset_def.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/flag_coding.h"
#include "snap-core/core/datamodel/index_coding.h"
#include "snap-core/core/datamodel/mask.h"
#include "snap-core/core/datamodel/metadata_attribute.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/datamodel/quicklooks/quicklook.h"
#include "snap-core/core/datamodel/tie_point_grid.h"

namespace alus {
namespace snapengine {

template <typename T>
ProductNodeGroup<T>::ProductNodeGroup(ProductNode* owner, std::string_view name,
                                      bool taking_over_node_ownership)
    : ProductNode(name, "") {
    node_list_ = std::make_shared<ProductNodeList<T>>();
    taking_over_node_ownership_ = taking_over_node_ownership;
    SetOwner(owner);
}

template <typename T>
ProductNodeGroup<T>::ProductNodeGroup(std::string_view name) : ProductNodeGroup(nullptr, name, false) {}

template <typename T>
bool ProductNodeGroup<T>::Add(T node) {
    Assert::NotNull(node, "node");
    bool added = node_list_->Add(node);
    if (added) {
        if (taking_over_node_ownership_) {
            node->SetOwner(this);
        }
        SetModified(true);
    }
    return added;
}

template <typename T>
void ProductNodeGroup<T>::Add(int index, T node) {
    Assert::NotNull(node, "node");
    node_list_->Add(index, node);
    // notifyadded...
    if (taking_over_node_ownership_) {
        node->SetOwner(this);
    }
    SetModified(true);
}

template <typename T>
bool ProductNodeGroup<T>::Remove(T node) {
    Assert::NotNull(node, "node");
    bool removed = node_list_->Remove(node);

    if (removed) {
        if (taking_over_node_ownership_ && node->GetOwner() == this) {
            node->SetOwner(nullptr);
        }
        SetModified(true);
    }
    return removed;
}
template <typename T>
std::vector<std::string> ProductNodeGroup<T>::GetNodeNames() {
    return node_list_->GetNames();
}
template <typename T>
bool ProductNodeGroup<T>::Contains(std::string_view name) {
    return node_list_->Contains(name);
}
template <typename T>
int ProductNodeGroup<T>::IndexOf(T element) {
    return node_list_->IndexOf(element);
}

template <typename T>
int ProductNodeGroup<T>::IndexOf(std::string_view name) {
    return node_list_->IndexOf(name);
}

template <typename T>
uint64_t ProductNodeGroup<T>::GetRawStorageSize(const std::shared_ptr<ProductSubsetDef>& subset_def) {
    uint64_t size = 0;
    auto nodes = ToArray();
    for (auto const& node : nodes) {
        if (subset_def->IsNodeAccepted(node->GetName())) {
            size += node->GetRawStorageSize(subset_def);
        }
    }
    return size;
}

template <typename T>
void ProductNodeGroup<T>::SetModified(bool modified) {
    bool old_state = IsModified();
    if (old_state != modified) {
        if (!modified) {
            for (auto const& node : ToArray()) {
                node->SetModified(false);
            }
            ClearRemovedList();
        }
        ProductNode::SetModified(modified);
    }
}

template <typename T>
void ProductNodeGroup<T>::ClearRemovedList() {
    node_list_->ClearRemovedList();
}
template <typename T>
T ProductNodeGroup<T>::Get(int index) {
    return node_list_->GetAt(index);
}
template <typename T>
T ProductNodeGroup<T>::Get(std::string_view name) {
    return node_list_->Get(name);
}

template <typename T>
void ProductNodeGroup<T>::Dispose() {
    node_list_->Dispose();
    ProductNode::Dispose();
}

// explicit init to force compile
template class ProductNodeGroup<std::shared_ptr<ProductNode>>;
template class ProductNodeGroup<std::shared_ptr<TiePointGrid>>;
template class ProductNodeGroup<std::shared_ptr<Band>>;
template class ProductNodeGroup<std::shared_ptr<Mask>>;
// template class ProductNodeGroup<std::shared_ptr<VectorDataNode>>;
template class ProductNodeGroup<std::shared_ptr<MetadataElement>>;
template class ProductNodeGroup<std::shared_ptr<MetadataAttribute>>;
template class ProductNodeGroup<std::shared_ptr<FlagCoding>>;
template class ProductNodeGroup<std::shared_ptr<IndexCoding>>;
template class ProductNodeGroup<std::shared_ptr<Quicklook>>;

}  // namespace snapengine
}  // namespace alus
