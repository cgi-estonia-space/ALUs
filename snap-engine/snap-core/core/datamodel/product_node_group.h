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
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "product_node.h"
#include "product_node_list.h"

namespace alus::snapengine {

/**
 * A type-safe container for elements of the type <code>ProductNode</code>.
 *
 * java version @author Norman Fomferra
 */
template <typename T>
class ProductNodeGroup : public ProductNode {
private:
    std::shared_ptr<ProductNodeList<T>> node_list_;
    bool taking_over_node_ownership_;

public:
    /**
     * Constructs a node group with no owner and which will not take ownership of added children.
     *
     * @param name The group name.
     * @since BEAM 4.8
     */
    explicit ProductNodeGroup(std::string_view name);

    /**
     * Constructs a node group for the given owner.
     *
     * @param owner                   The owner of the group.
     * @param name                    The group name.
     * @param takingOverNodeOwnership If {@code true}, child nodes will have this group as owner after adding.
     */
    ProductNodeGroup(ProductNode* owner, std::string_view name, bool taking_over_node_ownership);

    /**
     * Adds the given node to this group.
     *
     * @param node the node to be added, ignored if <code>null</code>
     * @return true, if the node has been added
     */
    bool Add(T node);
    /**
     * Adds the given node to this group.
     *
     * @param index the index.
     * @param node  the node to be added, ignored if <code>null</code>
     */
    void Add(int index, T node);

    /**
     * @param index The node index.
     * @return The product node at the given index.
     */
    T Get(int index);

    /**
     * @param name the name
     * @return the product node with the given name.
     */
    T Get(std::string_view name);

    /**
     * Returns an array of all products currently managed.
     *
     * @return an array containing the products, never <code>null</code>, but the array can have zero length
     */
    std::vector<T> ToArray() { return node_list_->ToArray(); };
    /**
     * @param array the array into which the elements of the list are to be stored, if it is big enough; otherwise, a
     *              new array of the same runtime type is allocated for this purpose.
     * @return an array containing the product nodes, never <code>null</code>, but the array can have zero length
     */
    std::vector<T> ToArray(std::vector<T> array) { return node_list_->ToArray(array); }

    /**
     * @return The number of product nodes in this product group.
     */
    int GetNodeCount() { return node_list_->Size(); }

    /**
     * Removes the given node from this group.
     *
     * @param node the node to be removed
     * @return true, if the node was removed
     */
    bool Remove(T node);

    /**
     * Returns the names of all products currently managed.
     *
     * @return an array containing the names, never <code>null</code>, but the array can have zero length
     */
    std::vector<std::string> GetNodeNames();

    /**
     * Tests whether a node with the given name is contained in this group.
     *
     * @param name the name
     * @return true, if so
     */
    bool Contains(std::string_view name);

    int IndexOf(T element);

    int IndexOf(std::string_view name);

    uint64_t GetRawStorageSize(const std::shared_ptr<ProductSubsetDef>& subset_def) override;

    void ClearRemovedList();

    void SetModified(bool modified) override;

    void Dispose() override;
};

}  // namespace alus::snapengine
