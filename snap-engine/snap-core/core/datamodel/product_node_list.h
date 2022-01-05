/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductNodeList.java
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

#include <string>
#include <string_view>
#include <vector>

#include <boost/algorithm/string.hpp>

#include "guardian.h"
#include "snap-core/core/datamodel/product_node.h"
#include "snap-core/core/datamodel/product_node_list.h"

namespace alus::snapengine {

/**
 * A type-safe list for elements of the type <code>ProductNode</code>.
 *
 * java version @author Norman Fomferra
 */
template <typename T>
class ProductNodeList /*: public ProductNode*/ {
private:
    // todo: make these thread safe
    std::vector<T> nodes_;
    std::vector<T> removed_nodes_;

    void DisposeRemovedList();

public:
    /**
     * @return the size of this list.
     */
    int Size() { return nodes_.size(); }

    /**
     * @param index the index, must be in the range zero to <code>size()</code>
     *
     * @return the element at the spcified index.
     */
    T GetAt(int index) { return nodes_.at(index); }

    /**
     * Gets the names of all nodes contained in this list. If this list is empty a zero-length array is returned.
     *
     * @return a string array containing all node names, never <code>null</code>
     */
    std::vector<std::string> GetNames();

    std::vector<std::string> GetDisplayNames();

    /**
     * Gets the element with the given name. The method performs a case insensitive search.
     *
     * @param name the name of the node, must not be <code>null</code>
     *
     * @return the node with the given name or <code>null</code> if a node with the given name is not contained in this
     *         list
     *
     * @throws IllegalArgumentException if the name is <code>null</code>
     */
    T Get(std::string_view name);

    /**
     * Gets the index of the node with the given name. The method performs a case insensitive search.
     *
     * @param name the name of the node, must not be <code>null</code>
     *
     * @return the index of the node with the given name or <code>-1</code> if a node with the given name is not
     *         contained in this list
     *
     * @throws IllegalArgumentException if the name is <code>null</code>
     */
    int IndexOf(std::string_view name);

    /**
     * Gets the index of the given node.
     *
     * @param node the node to get the index, must not be <code>null</code>
     *
     * @return the index of the given node or <code>-1</code> if the node is not contained in this list
     *
     * @throws IllegalArgumentException if the node is <code>null</code>
     */
    int IndexOf(T node);

    /**
     * Gets the element with the given display name.
     *
     * @param displayName the display name of the node, must not be <code>null</code>
     *
     * @return the node with the given display name or <code>null</code> if a node with the given display name is not
     * contained in this list
     *
     * @throws IllegalArgumentException if the display name is <code>null</code>
     * @see ProductNode#getDisplayName()
     */
    T GetByDisplayName(std::string_view display_name);

    /**
     * Tests if this list contains a node with the given name.
     *
     * @param name the name of the node, must not be <code>null</code>
     *
     * @return true if this list contains a node with the given name.
     *
     * @throws IllegalArgumentException if the name is <code>null</code>
     */
    bool Contains(std::string name) { return IndexOf(name) >= 0; }

    /**
     * Tests if this list contains the given node.
     *
     * @param node the node
     *
     * @return true if this list contains the given node.
     *
     * @throws IllegalArgumentException if the node is <code>null</code>
     */
    bool Contains(T node);

    /**
     * Tests if this list contains a node with the given name.
     *
     * @param name the name of the node, must not be <code>null</code>
     *
     * @return true if this list contains a node with the given name.
     *
     * @throws IllegalArgumentException if the name is <code>null</code>
     */
    bool Contains(std::string_view name);

    /**
     * Adds a new node to this list. Note that <code>null</code> nodes are not added to this list.
     *
     * @param node the node to be added, ignored if <code>null</code>
     *
     * @return true if the node was added, otherwise false.
     */
    bool Add(T node);

    /**
     * Inserts a new node to this list at the given index. Note that <code>null</code> nodes are not added to this
     * list.
     *
     * @param node  the node to be added, ignored if <code>null</code>
     * @param index the insert index
     *
     * @throws ArrayIndexOutOfBoundsException if the index was invalid.
     */
    void Add(int index, T node);

    /**
     * Clears the internal removed product nodes list.
     */
    void ClearRemovedList() { removed_nodes_.clear(); }

    /**
     * Gets all removed product nodes.
     *
     * @return a collection of all removed product nodes.
     */
    std::vector<T> GetRemovedNodes() const { return removed_nodes_; }

    /**
     * Removes the given node from this list. The removed nodes will be added to the internal list of removed product
     * nodes.
     *
     * @param node the node to be removed, ignored if <code>null</code>
     *
     * @return <code>true</code> if the node is a member of this list and could successfully be removed,
     *         <code>false</code> otherwise
     */
    bool Remove(T node);

    /**
     * Removes all nodes from this list.
     */
    void RemoveAll();

    /**
     * Returns the list of named nodes as an array. If this list is empty a zero-length array is returned.
     *
     * @param array the array into which the elements of the list are to be stored, if it is big enough; otherwise, a
     *              new array of the same runtime type is allocated for this purpose.
     *
     * @return an array containing the elements of the list. never <code>null</code>
     */
    std::vector<T> ToArray([[maybe_unused]] std::vector<T> array) {
        //        todo:check if this is ok for our purposes
        //        nodes_.insert(std::end(nodes_), std::begin(array), std::end(array));
        return nodes_;
    }

    /**
     * Returns the list of named nodes as an array. If this list is empty a zero-length array is returned.
     *
     * @return a string array containing all node names, never <code>null</code>
     */
    std::vector<T> ToArray() {
        // todo: should set size?
        return nodes_;  //.ToArray(std::make_shared<ProductNode>());
    }

    /**
     * Releases all of the resources used by this object instance and all of its owned children. Its primary use is to
     * allow the garbage collector to perform a vanilla job.
     * <p>This method should be called only if it is for sure that this object instance will never be used again. The
     * results of referencing an instance of this class after a call to <code>dispose()</code> are undefined.
     */
    void Dispose();
};
}  // namespace alus::snapengine