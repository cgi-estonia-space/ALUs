/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.datamodel.MetadataElement.java
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

#include <vector>

#include "i_meta_data_reader.h"
#include "product_data_utc.h"
#include "product_node.h"

namespace alus {
namespace snapengine {

class IMetaDataReader;

class MetadataElement : public ProductNode {
   private:
    MetadataElement(std::string_view name, std::string_view description, IMetaDataReader* meta_data_reader);
    IMetaDataReader* meta_data_reader_{};
    //    std::vector<MetadataElement> elements_{};
    std::vector<std::shared_ptr<MetadataElement>> elements_{};
    std::vector<std::shared_ptr<MetadataAttribute>> attributes_{};
    // todo:name and description to superclass ProductNode same for attribtue node?

    [[nodiscard]] static std::string GetAttributeNotFoundMessage(std::string_view name);
    [[nodiscard]] static std::shared_ptr<MetadataElement> GetParentElement(const ProductNode& node);

    std::shared_ptr<MetadataAttribute> GetAndMaybeCreateAttribute(std::string_view name, int type, int num_elems);

   public:
    explicit MetadataElement() : ProductNode(nullptr) {}
    explicit MetadataElement(std::string_view name) : ProductNode(name) {}
    MetadataElement(std::string_view name,
                    std::vector<std::shared_ptr<MetadataElement>> elements,
                    std::vector<std::shared_ptr<MetadataAttribute>> attributes);
    //    void AddElement(const MetadataElement& me);
    void AddElement(const std::shared_ptr<MetadataElement>& me);

    /**
     * Adds an attribute to this node.
     *
     * @param attribute the attribute to be added, <code>null</code> is ignored
     */
    void AddAttribute(const std::shared_ptr<MetadataAttribute>& ma);
    //    MetadataElement GetElementAt(int index);
    /**
     * Returns an std::vector of elements contained in this element.
     *
     * @return an std::vector of elements contained in this product.
     */
    [[nodiscard]] std::vector<std::shared_ptr<MetadataElement>> GetElements() const { return elements_; };
    /**
     * Returns the element with the given name.
     *
     * @param name the element name
     *
     * @return the element with the given name or <code>null</code> if a element with the given name is not contained in
     *         this element.
     */
    [[nodiscard]] std::shared_ptr<MetadataElement> GetElement(std::string_view name);

    [[nodiscard]] std::vector<std::shared_ptr<MetadataAttribute>> GetAttributes() const { return attributes_; };
    /**
     * @return the number of elements contained in this element.
     */
    [[nodiscard]] int GetNumElements() const;

    /**
     * Returns the number of attributes attached to this node.
     *
     * @return the number of attributes
     */
    [[nodiscard]] int GetNumAttributes() const;

    /**
     * Returns the attribute at the given index.
     *
     * @param index the attribute index
     *
     * @return the attribute
     *
     * @throws std::out_of_range
     */
    [[nodiscard]] auto GetAttributeAt(int index) const { return attributes_.at(index); }

    /**
     * Tests if a element with the given name is contained in this element.
     *
     * @param name the name, must not be <code>null</code>
     *
     * @return <code>true</code> if a element with the given name is contained in this element, <code>false</code>
     *         otherwise
     */
    [[nodiscard]] auto ContainsElement(std::string_view name) const;

    /**
     * Checks whether this node has an element with the given name.
     *
     * @param name the attribute name
     *
     * @return <code>true</code> if so
     */
    [[nodiscard]] bool ContainsAttribute(std::string_view name) const;

    /**
     * Returns the names of all attributes of this node.
     *
     * @return the attribute name array, never <code>null</code>
     */
    [[nodiscard]] std::vector<std::string> GetAttributeNames() const;

    /**
     * Removes the given attribute from this annotation. If an attribute with the same name already exists, the method
     * does nothing.
     *
     * @param attribute the attribute to be removed, <code>null</code> is ignored
     *
     * @return <code>true</code> if it was removed
     */
    bool RemoveAttribute(std::shared_ptr<MetadataAttribute>& ma);
    /**
     * Gets the index of the given element.
     *
     * @param element The element .
     *
     * @return The element's index, or -1.*/
    [[nodiscard]] int GetElementIndex(const std::shared_ptr<MetadataElement>& element) const;

    /**
     * Gets the index of the given attribute.
     *
     * @param attribute The attribute.
     *
     * @return The attribute's index, or -1.
     */
    [[nodiscard]] int GetAttributeIndex(const std::shared_ptr<MetadataAttribute>& attribute) const;

    /**
     * Returns the integer value of the attribute with the given name. <p>The given default value is returned if an
     * attribute with the given name could not be found in this node.
     *
     * @param name         the attribute name
     * @param defaultValue the default value
     *
     * @return the attribute value as integer.
     *
     * @throws NumberFormatException if the attribute type is ASCII but cannot be converted to a number
     */
    [[nodiscard]] int GetAttributeInt(std::string_view name, int default_value) const;
    /**
     * Returns the integer value of the attribute with the given name. <p>An Exception is thrown if an
     * attribute with the given name could not be found in this node.
     *
     * @param name the attribute name
     *
     * @return the attribute value as integer.
     *
     * @throws NumberFormatException    if the attribute type is ASCII but cannot be converted to a number
     * @throws IllegalArgumentException if an attribute with the given name could not be found
     */
    [[nodiscard]] int GetAttributeInt(std::string_view name) const;

    /**
     * Returns the double value of the attribute with the given name. <p>The given default value is returned if an
     * attribute with the given name could not be found in this node.
     *
     * @param name         the attribute name
     * @param defaultValue the default value
     *
     * @return the attribute value as double.
     *
     * @throws NumberFormatException if the attribute type is ASCII but cannot be converted to a number
     */
    [[nodiscard]] double GetAttributeDouble(std::string_view name, double default_value) const;

    /**
     * Returns the double value of the attribute with the given name. <p>An Exception is thrown if an
     * attribute with the given name could not be found in this node.
     *
     * @param name the attribute name
     *
     * @return the attribute value as double.
     *
     * @throws NumberFormatException    if the attribute type is ASCII but cannot be converted to a number
     * @throws IllegalArgumentException if an attribute with the given name could not be found
     */
    [[nodiscard]] double GetAttributeDouble(std::string_view name) const;
    /**
     * Returns the attribute with the given name.
     *
     * @param name the attribute name
     *
     * @return the attribute with the given name or <code>null</code> if it could not be found
     */
    [[nodiscard]] std::shared_ptr<MetadataAttribute> GetAttribute(std::string_view name) const;

    /**
     * Returns the UTC value of the attribute with the given name. <p>The given default value is returned if an
     * attribute with the given name could not be found in this node.
     *
     * @param name         the attribute name
     * @param defaultValue the default value
     *
     * @return the attribute value as UTC.
     */
    [[nodiscard]] const std::shared_ptr<Utc> GetAttributeUtc(std::string_view name,
                                                             std::shared_ptr<Utc> default_value) const;

    /**
     * Returns the UTC value of the attribute with the given name.
     *
     * @param name the attribute name
     *
     * @return the attribute value as UTC.
     *
     * @throws IllegalArgumentException if an attribute with the given name could not be found
     */
    [[nodiscard]] const std::shared_ptr<Utc> GetAttributeUtc(std::string_view name) const;

    /**
     * Returns the string value of the attribute with the given name. <p>The given default value is returned if an
     * attribute with the given name could not be found in this node.
     *
     * @param name         the attribute name
     * @param defaultValue the default value
     *
     * @return the attribute value as integer.
     */
    [[nodiscard]] std::string GetAttributeString(std::string_view name, std::string_view default_value) const;

    /**
     * Returns the string value of the attribute with the given name. <p>An Exception is thrown if an
     * attribute with the given name could not be found in this node.
     *
     * @param name the attribute name
     *
     * @return the attribute value as integer.
     *
     * @throws IllegalArgumentException if an attribute with the given name could not be found
     */
    [[nodiscard]] std::string GetAttributeString(std::string_view name) const;

    /**
     * Sets the attribute with the given name to the given integer value. <p>A new attribute with
     * <code>ProductData.TYPE_INT32</code> is added to this node if an attribute with the given name could not be found
     * in this node.
     *
     * @param name  the attribute name
     * @param value the new value
     */
    void SetAttributeInt(std::string_view name, int value);

    /**
     * Sets the attribute with the given name to the given double value. <p>A new attribute with
     * <code>ProductData.TYPE_FLOAT64</code> is added to this node if an attribute with the given name could not be
     * found in this node.
     *
     * @param name  the attribute name
     * @param value the new value
     */
    void SetAttributeDouble(std::string_view name, double value);

    /**
     * Sets the attribute with the given name to the given utc value. <p>A new attribute with
     * <code>ProductData.UTC</code> is added to this node if an attribute with the given name could not be found
     * in this node.
     *
     * @param name  the attribute name
     * @param value the new value
     */
    void SetAttributeUTC(std::string_view name, const Utc& value);

    /**
     * Sets the attribute with the given name to the given string value. <p>A new attribute with
     * <code>ProductData.TYPE_ASCII</code> is added to this node if an attribute with the given name could not be found
     * in this node.
     *
     * @param name  the attribute name
     * @param value the new value
     */
    void SetAttributeString(std::string_view name, std::string_view value);

    std::shared_ptr<MetadataElement> CreateDeepClone();
};
}  // namespace snapengine
}  // namespace alus
