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
class ProductNode {
   private:
    std::string name_{};
    std::string description_{};
    std::shared_ptr<ProductNode> owner_;
    //    bool modified_;
   protected:
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
    ProductNode(std::string_view name, std::string_view description);
    //        : name_(name), description_(description) {}

   public:
    static constexpr std::string_view PROPERTY_NAME_NAME{"name"};
    static constexpr std::string_view PROPERTY_NAME_DESCRIPTION{"description"};
    // todo:make abstract?
    [[nodiscard]] virtual std::string_view GetName() const { return name_; };
    [[nodiscard]] virtual std::string_view GetDescription() const { return description_; }

    /**
     * Sets a short textual description for this products node.
     *
     * @param description a description, can be <code>null</code>
     */
    void SetDescription(std::string_view description);

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

    //    /**
    //     * Returns whether or not this node is modified.
    //     *
    //     * @return <code>true</code> if so
    //     */
    //   [[nodiscard]] bool IsModified() const{
    //        return modified_;
    //    }
};

}  // namespace snapengine
}  // namespace alus
