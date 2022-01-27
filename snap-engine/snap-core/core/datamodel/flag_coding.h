/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.FlagCoding.java
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

#include "metadata_attribute.h"
#include "sample_coding.h"

namespace alus::snapengine {

/**
 * Provides the information required to decode integer sample values that
 * are combined of single flags (bit indexes).
 */
class FlagCoding : public virtual SampleCoding {
public:
    /**
     * Constructs a new flag coding object with the given name.
     *
     * @param name the name
     */
    explicit FlagCoding(std::string_view name);

    /**
     * Returns a metadata attribute wich is the representation of the flag with the given name. This method delegates to
     * getPropertyValue(String).
     *
     * @param name the flag name
     * @return a metadata attribute wich is the representation of the flag with the given name
     */
    std::shared_ptr<MetadataAttribute> GetFlag(std::string_view name);

    /**
     * Returns a string array which contains the names of all flags contained in this <code>FlagCoding</code> object.
     *
     * @return a string array which contains all names of this <code>FlagCoding</code>.<br> If this
     *         <code>FlagCoding</code> does not contain any flag, <code>null</code> is returned
     */
    std::vector<std::string> GetFlagNames();

    /**
     * Adds a new flag definition to this flags coding.
     *
     * @param name        the flag name
     * @param flagMask    the flag's bit mask
     * @param description the description text
     * @throws IllegalArgumentException if <code>name</code> is null
     * @return A new attribute representing the flag.
     */
    std::shared_ptr<MetadataAttribute> AddFlag(std::string_view name, int flag_mask, std::string_view description);

    /**
     * Adds a new flag definition to this flags coding.
     *
     * @param name        the flag name
     * @param flagMask    the flag's bit mask
     * @param description the description text
     * @throws IllegalArgumentException if <code>name</code> is null
     * @return A new attribute representing the flag.
     * @since SNAP 0.5
     */
    std::shared_ptr<MetadataAttribute> AddFlag(std::string_view name, int flag_mask, int flag_value,
                                               std::string_view description);

    /**
     * Returns the flag mask value for the specified flag name.
     *
     * @param name the flag name
     * @return flagMask the flag's bit mask as a 32 bit integer
     * @throws IllegalArgumentException if <code>name</code> is null, or a flag with the name does not exist
     */
    int GetFlagMask(std::string_view name);
};
}  // namespace alus::snapengine
