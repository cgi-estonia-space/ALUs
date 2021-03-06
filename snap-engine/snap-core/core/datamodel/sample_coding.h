/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.SampleCoding.java
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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "snap-core/core/datamodel/metadata_element.h"

namespace alus::snapengine {
/**
 * Provides the information required to decode integer sample values that
 * represent index values (e.g. types, classes, categories).
 * @since BEAM 4.2
 */
class SampleCoding : public MetadataElement {
public:
    explicit SampleCoding(std::string_view name);

    /**
     * Overrides the base class <code>addElement</code> in order to <b>not</b> add an element to this flag coding
     * because flag codings do not support inner elements.
     *
     * @param element the element to be added, always ignored
     */
    void AddElement(const std::shared_ptr<MetadataElement>& element) override;

    /**
     * Adds an attribute to this node. If an attribute with the same name already exists, the method does nothing.
     *
     * @param attribute the attribute to be added
     * @throws IllegalArgumentException if the attribute added is not an integer or does not have a scalar value
     */
    void AddAttribute(const std::shared_ptr<MetadataAttribute>& attribute) override;

    /**
     * Adds a new coding value to this sample coding.
     *
     * @param name        the coding name
     * @param value       the value
     * @param description the description text
     * @return A new attribute representing the coded sample.
     * @throws IllegalArgumentException if <code>name</code> is null
     */
    std::shared_ptr<MetadataAttribute> AddSample(std::string_view name, int value, std::string_view description);

    /**
     * Adds a new coding value to this sample coding.
     *
     * @param name        the coding name
     * @param values      the values
     * @param description the description text
     * @return A new attribute representing the coded sample.
     * @throws IllegalArgumentException if <code>name</code> is null
     */
    std::shared_ptr<MetadataAttribute> AddSamples(std::string_view name, const std::vector<int>& values,
                                                  std::string_view description);

    /**
     * Gets the number of coded sample values.
     *
     * @return the number of coded sample values
     */
    int GetSampleCount();

    /**
     * Gets the sample name at the specified attribute index.
     *
     * @param index the attribute index.
     * @return the sample name.
     */
    std::string GetSampleName(int index);

    /**
     * Gets the sample value at the specified attribute index.
     *
     * @param index the attribute index.
     * @return the sample value.
     */
    int GetSampleValue(int index);
};
}  // namespace alus::snapengine
