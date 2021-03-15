#include "snap-core/datamodel/product_node_list.h"

#include <algorithm>
#include <memory>

#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/flag_coding.h"
#include "snap-core/datamodel/index_coding.h"
#include "snap-core/datamodel/mask.h"
#include "snap-core/datamodel/metadata_attribute.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/product_node.h"
#include "snap-core/datamodel/quicklooks/quicklook.h"
#include "snap-core/datamodel/tie_point_grid.h"

namespace alus {
namespace snapengine {
template <typename T>
std::vector<std::string> ProductNodeList<T>::GetNames() {
    // map to names
    std::vector<std::string> names(nodes_.size());
    std::transform(nodes_.begin(), nodes_.end(), names.begin(), [](const T& s) { return s->GetName(); });
    return names;
}

template <typename T>
std::vector<std::string> ProductNodeList<T>::GetDisplayNames() {
    // map to names
    std::vector<std::string> display_names(nodes_.size());
    std::transform(nodes_.begin(), nodes_.end(), display_names.begin(), [](const T& s) { return s->GetDisplayName(); });
    return display_names;
}

template <typename T>
T ProductNodeList<T>::Get(std::string_view name) {
    int index = IndexOf(name);
    return index >= 0 ? nodes_.at(index) : nullptr;
}

template <typename T>
int ProductNodeList<T>::IndexOf(std::string_view name) {
    Guardian::AssertNotNull("name", name);
    int n = Size();
    for (int i = 0; i < n; i++) {
        if (boost::iequals(GetAt(i)->GetName(), name)) {
            return i;
        }
    }
    return -1;
}

template <typename T>
T ProductNodeList<T>::GetByDisplayName(std::string_view display_name) {
    Guardian::AssertNotNull("display_name", display_name);
    for (T node : nodes_) {
        if (node->GetDisplayName() == display_name) {
            return node;
        }
    }
    return nullptr;
}

template <typename T>
void ProductNodeList<T>::Add(int index, T node) {
    if (node != nullptr) {
        auto it = nodes_.begin();
        nodes_.insert(it + index, node);
    }
}

template <typename T>
bool ProductNodeList<T>::Remove(T node) {
    //       todo:deal with thread safty!
    if (node) {
        auto position = std::find(nodes_.begin(), nodes_.end(), node);
        if (position != nodes_.end()) {
            removed_nodes_.push_back(node);
            nodes_.erase(position);
            return true;
        }
    }
    return false;
}

template <typename T>
void ProductNodeList<T>::RemoveAll() {
    //        todo:thread safty not supported
    removed_nodes_.insert(removed_nodes_.end(), nodes_.begin(), nodes_.end());
    nodes_.clear();
}

template <typename T>
bool ProductNodeList<T>::Add(T node) {
    auto start_size = nodes_.size();
    if (node) {
        nodes_.push_back(node);
        return start_size != nodes_.size();
    }
    return false;
}

template <typename T>
bool ProductNodeList<T>::Contains(T node) {
    return node != nullptr && (std::find(nodes_.begin(), nodes_.end(), node) != nodes_.end());
}

template <typename T>
bool ProductNodeList<T>::Contains(std::string_view name) {
    return IndexOf(name) >= 0;
}

template <typename T>
int ProductNodeList<T>::IndexOf(T node) {
    Guardian::AssertNotNull("node", node);
    auto it = std::find(nodes_.begin(), nodes_.end(), node);
    if (it != nodes_.end()) {
        return distance(nodes_.begin(), it);
    }
    return -1;
}
template <typename T>
void ProductNodeList<T>::Dispose() {
    for (int i = 0; i < Size(); i++) {
        GetAt(i)->Dispose();
    }
    RemoveAll();
    DisposeRemovedList();
}
template <typename T>
void ProductNodeList<T>::DisposeRemovedList() {
    for (T removed_node : removed_nodes_) {
        removed_node->Dispose();
    }
    ClearRemovedList();
}

// explicit instantiations and force the compilation
template class ProductNodeList<std::shared_ptr<ProductNode>>;
template class ProductNodeList<std::shared_ptr<TiePointGrid>>;
template class ProductNodeList<std::shared_ptr<Band>>;
template class ProductNodeList<std::shared_ptr<Mask>>;
// template class ProductNodeList<std::shared_ptr<VectorDataNode>>;
template class ProductNodeList<std::shared_ptr<MetadataElement>>;
template class ProductNodeList<std::shared_ptr<MetadataAttribute>>;
template class ProductNodeList<std::shared_ptr<FlagCoding>>;
template class ProductNodeList<std::shared_ptr<IndexCoding>>;
template class ProductNodeList<std::shared_ptr<Quicklook>>;

}  // namespace snapengine
}  // namespace alus