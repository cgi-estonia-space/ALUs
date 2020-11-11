/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.datamodel.ProductNode.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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

#include <memory>
#include <optional>
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
    [[nodiscard]] std::shared_ptr<ProductNode> GetOwner() const { return owner_; }

    /**
     * Sets the the owner node of this node.
     * <p>Overrides shall finally call <code>super.setOwner(owner)</code>.
     *
     * @param owner the new owner
     */
    void SetOwner(const std::shared_ptr<ProductNode>& owner);

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
     * Releases all of the resources used by this object instance and all of its owned children. Its primary use is to
     * allow the garbage collector to perform a vanilla job.
     * <p>This method should be called only if it is for sure that this object instance will never be used again. The
     * results of referencing an instance of this class after a call to <code>dispose()</code> are undefined.
     * <p>Overrides of this method should always call <code>super.dispose();</code> after disposing this instance.
     */
    virtual void Dispose() {
        owner_ = nullptr;
        product_ = nullptr;
        description_ = nullptr;
        name_ = nullptr;
    }
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
