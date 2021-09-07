
/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.MetadataElement.java
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

#include "i_meta_data_reader.h"
#include "product_data_utc.h"
#include "product_node.h"

namespace alus {
namespace snapengine {

template <typename T>
class ProductNodeGroup;
class ProductSubsetDef;
class IMetaDataReader;
class MetadataAttribute;

class MetadataElement : public ProductNode {
private:
    MetadataElement(std::string_view name, std::string_view description, IMetaDataReader* meta_data_reader);
    IMetaDataReader* meta_data_reader_{};
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<MetadataElement>>> elements_{};
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<MetadataAttribute>>> attributes_{};

    [[nodiscard]] static std::string GetAttributeNotFoundMessage(std::string_view name);
    [[nodiscard]] static std::shared_ptr<MetadataElement> GetParentElement(const ProductNode& node);

    std::shared_ptr<MetadataAttribute> GetAndMaybeCreateAttribute(std::string_view name, int type, int num_elems);

public:
    explicit MetadataElement() : ProductNode(nullptr) {}
    explicit MetadataElement(std::string_view name) : ProductNode(name) {}
    virtual void AddElement(std::shared_ptr<MetadataElement> me);

    /**
     * Adds an attribute to this node.
     *
     * @param attribute the attribute to be added, <code>null</code> is ignored
     */
    virtual void AddAttribute(std::shared_ptr<MetadataAttribute> ma);

    /**
     * Returns an std::vector of elements contained in this element.
     *
     * @return an std::vector of elements contained in this product.
     */
    [[nodiscard]] std::vector<std::shared_ptr<MetadataElement>> GetElements();

    /**
     * Returns the element with the given name.
     *
     * @param name the element name
     *
     * @return the element with the given name or <code>null</code> if a element with the given name is not contained in
     *         this element.
     */
    [[nodiscard]] std::shared_ptr<MetadataElement> GetElement(std::string_view name);

    /**
     * Returns the element at the given index.
     *
     * @param index the element index
     *
     * @return the element at the given index
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    std::shared_ptr<MetadataElement> GetElementAt(int index);

    /**
     * Returns a vector of strings containing the names of the groups contained in this element.
     *
     * @return a string vector containing the names of the groups contained in this element. If this element has no
*         groups an empty vector is returned.
     */
    std::vector<std::string> GetElementNames() const;

    [[nodiscard]] std::vector<std::shared_ptr<MetadataAttribute>> GetAttributes();

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
    [[nodiscard]] std::shared_ptr<MetadataAttribute> GetAttributeAt(int index) const;

    /**
     * Tests if a element with the given name is contained in this element.
     *
     * @param name the name, must not be <code>null</code>
     *
     * @return <code>true</code> if a element with the given name is contained in this element, <code>false</code>
     *         otherwise
     */
    [[nodiscard]] bool ContainsElement(std::string_view name) const;

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
    [[nodiscard]] std::shared_ptr<Utc> GetAttributeUtc(std::string_view name, std::shared_ptr<Utc> default_value) const;

    /**
     * Returns the UTC value of the attribute with the given name.
     *
     * @param name the attribute name
     *
     * @return the attribute value as UTC.
     *
     * @throws IllegalArgumentException if an attribute with the given name could not be found
     */
    [[nodiscard]] std::shared_ptr<Utc> GetAttributeUtc(std::string_view name) const;

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
    void SetAttributeUtc(std::string_view name, const std::shared_ptr<Utc>& value);

    /**
     * Sets the attribute with the given name to the given string value. <p>A new attribute with
     * <code>ProductData.TYPE_ASCII</code> is added to this node if an attribute with the given name could not be found
     * in this node.
     *
     * @param name  the attribute name
     * @param value the new value
     */
    void SetAttributeString(std::string_view name, std::string_view value);

    /**
     * Gets an estimated, raw storage size in bytes of this product node.
     *
     * @param subsetDef if not <code>null</code> the subset may limit the size returned
     *
     * @return the size in bytes.
     */
    uint64_t GetRawStorageSize(const std::shared_ptr<ProductSubsetDef>& subset_def) override;

    std::shared_ptr<MetadataElement> CreateDeepClone();

    /**
     * Adds the given element to this element at index.
     *
     * @param element the element to added, ignored if <code>null</code>
     * @param index   where to put it
     */
    void AddElementAt(const std::shared_ptr<MetadataElement>& element, int index);

    /**
     * Removes the given element from this element.
     *
     * @param element the element to be removed, ignored if <code>null</code>
     *
     * @return true, if so
     */
    bool RemoveElement(const std::shared_ptr<MetadataElement>& element);
    /**
     * Removes the given attribute from this annotation. If an attribute with the same name already exists, the method
     * does nothing.
     *
     * @param attribute the attribute to be removed, <code>null</code> is ignored
     *
     * @return <code>true</code> if it was removed
     */
    bool RemoveAttribute(std::shared_ptr<MetadataAttribute> attribute);

    /**
     * Releases all of the resources used by this object instance and all of its owned children. Its primary use is to
     * allow the garbage collector to perform a vanilla job.
     * <p>This method should be called only if it is for sure that this object instance will never be used again. The
     * results of referencing an instance of this class after a call to <code>dispose()</code> are undefined.
     * <p>Overrides of this method should always call <code>super.dispose();</code> after disposing this instance.
     */
    void Dispose() override;
};
}  // namespace snapengine
}  // namespace alus
