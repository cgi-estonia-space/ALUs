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
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <string_view>

namespace alus {
namespace snapengine {

/**
 * The <code>ProductNode</code> is the base class for all nodes within a remote sensing data product and even the data
 * product itself.
 *
 * java version @author Norman Fomferra
 */
class Product;
class IProductReader;
class ProductSubsetDef;
class ProductNode : public std::enable_shared_from_this<ProductNode> {
private:
    std::string name_{};
    std::optional<std::string> description_;
    std::shared_ptr<ProductNode> owner_;
    // transient in java version
    bool modified_;
    std::shared_ptr<Product> product_;

protected:
    ProductNode() = default;
    ProductNode(const ProductNode&) = delete;
    ProductNode& operator=(const ProductNode&) = delete;
    virtual ~ProductNode() = default;
    /**
     * Constructs a new product node with the given name.
     *
     * @param name the node name, must not be <code>null</code>
     * @throws IllegalArgumentException if the given name is not a valid node identifier
     */
    explicit ProductNode(std::string_view name);
    /**
     * Constructs a new product node with the given name and an optional description.
     *
     * @param name        the node name, must not be <code>null</code>
     * @param description a descriptive string, can be <code>null</code>
     * @throws IllegalArgumentException if the given name is not a valid node identifier
     */
    ProductNode(std::string_view name, const std::optional<std::string_view>& description);

    template <typename T>
    std::shared_ptr<T> SharedFromBase();

    void SetNodeName(std::string_view trimmed_name, bool silent);

    /**
     * Returns whether or not this node is part of the given subset.
     *
     * @param subsetDef The subset definition.
     * @return <code>true</code> if the subset is not <code>null</code> and it contains a node name equal to this node's
     * name.
     */
    bool IsPartOfSubset(const std::shared_ptr<ProductSubsetDef>& subset_def);

public:
    static constexpr std::string_view PROPERTY_NAME_NAME{"name"};
    static constexpr std::string_view PROPERTY_NAME_DESCRIPTION{"description"};
    // todo:make abstract?
    [[nodiscard]] virtual std::string GetName() const { return name_; };
    [[nodiscard]] virtual std::optional<std::string> GetDescription() const { return description_; }

    /**
     * Sets a short textual description for this products node.
     *
     * @param description a description, can be <code>null</code>
     */
    void SetDescription(std::string_view description);
    /**
     * Sets a short textual description for this products node.
     *
     * @param description a description, can be <code>null</code>
     */
    void SetDescription(const std::optional<std::string_view>& description);
    /**
     * @return The owner node of this node.
     */
    [[nodiscard]] std::shared_ptr<ProductNode> GetOwner() { return owner_; }

    /**
     * Sets the the owner node of this node.
     * <p>Overrides shall finally call <code>super.setOwner(owner)</code>.
     *
     * @param owner the new owner
     */
    void SetOwner(const std::shared_ptr<ProductNode>& owner);

    /**
     * Returns this node's display name. The display name is the product reference string with the node name appended.
     * <p>Example: The string <code>"[2] <i>node-name</i>"</code> means node <code><i>node-name</i></code> of the
     * product with the reference number <code>2</code>.
     *
     * @return this node's name with a product prefix <br>or this node's name only if this node's product prefix is
     * <code>null</code>
     *
     * @see #getProductRefString
     */
    virtual std::string GetDisplayName();

    /**
     * Gets the product reference string. The product reference string is the product reference number enclosed in
     * square brackets. <p>Example: The string <code>"[2]"</code> stands for a product with the reference number
     * <code>2</code>.
     *
     * @return the product reference string. <br>or <code>null</code> if this node has no product <br>or
     * <code>null</code> if its product reference number was inactive
     */
    std::optional<std::string> GetProductRefString();

    /**
     * Sets this node's modified flag.
     * <p>
     * If the modified flag changes to true and this node has an owner, the owner's modified flag is also set to
     * true.
     *
     * @param modified whether or not this node is beeing marked as modified.
     * @see Product#fireNodeChanged
     */
    virtual void SetModified(bool modified);

    /**
     * Sets this product's name.
     *
     * @param name The name.
     */
    void SetName(std::string_view name);

    /**
     * Returns whether or not this node is modified.
     *
     * @return <code>true</code> if so
     */
    [[nodiscard]] bool IsModified() const { return modified_; }

    /**
     * Returns the product to which this node belongs to.
     *
     * @return the product, or <code>null</code> if this node was not owned by a product at the time this method was
     * called
     */
    std::shared_ptr<Product> GetProduct();

    /**
     * Returns the product reader for the product to which this node belongs to.
     *
     * @return the product reader, or <code>null</code> if no such exists
     */
    virtual std::shared_ptr<IProductReader> GetProductReader();

    /**
     * Releases all of the resources used by this object instance and all of its owned children. Its primary use is to
     * allow the garbage collector to perform a vanilla job.
     * <p>This method should be called only if it is for sure that this object instance will never be used again. The
     * results of referencing an instance of this class after a call to <code>dispose()</code> are undefined.
     * <p>Overrides of this method should always call <code>super.dispose();</code> after disposing this instance.
     */
    virtual void Dispose();

    //////////////////////////////////////////////////////////////////////////
    // General utility methods

    /**
     * Tests whether the given name is valid name for a node.
     * A valid node name must not start with a dot. Also a valid node name must not contain
     * any of the character  <code>\/:*?"&lt;&gt;|</code>
     *
     * @param name the name to test
     * @return <code>true</code> if the name is a valid node identifier, <code>false</code> otherwise
     */
    static bool IsValidNodeName(std::string_view name);

    /**
     * Gets an estimated, raw storage size in bytes of this product node.
     *
     * @return the size in bytes.
     */
    virtual uint64_t GetRawStorageSize() { return GetRawStorageSize(nullptr); }

    /**
     * Gets an estimated, raw storage size in bytes of this product node.
     *
     * @param subsetDef if not <code>null</code> the subset may limit the size returned
     * @return the size in bytes.
     */
    virtual uint64_t GetRawStorageSize(const std::shared_ptr<ProductSubsetDef>& subset_def) = 0;
};

////////////////////////////////////////////////////////////////////////
/////TEMPLATED IMPLEMENTATION NEEDS TO BE IN THE SAME FILE
////////////////////////////////////////////////////////////////////////

template <typename T>
std::shared_ptr<T> ProductNode::SharedFromBase() {
    return std::dynamic_pointer_cast<T>(shared_from_this());
}

}  // namespace snapengine
}  // namespace alus
