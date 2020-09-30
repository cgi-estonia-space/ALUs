/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class ASCII which is inside org.esa.snap.core.datamodel.ProductData.java
 * ported for native code. Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally
 * stated to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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

#include "product_data_byte.h"

namespace alus {
namespace snapengine {

/**
 * The {@code ProductData.ASCII} class is a {@code ProductData.Byte} specialisation representing textual
 * values.
 * <p> Internally, data is stored in an array of the type {@code byte[]}.
 */
class ASCII : public Byte {
   public:
    /**
     * Constructs a new {@code ASCII} value.
     *
     * @param length the ASCII string length
     */
    explicit ASCII(int length);

    /**
     * Constructs a new {@code ASCII} value.
     *
     * @param data the ASCII string data
     */
    explicit ASCII(std::string_view data);

    /**
     * Returns a textual representation of this value's value. The text returned is a string created from the bytes
     * array elements in this value interpreted as ASCII values.
     *
     * @return a text representing this product data, never {@code null}
     */
    [[nodiscard]] std::string GetElemString() override;

    /**
     * Returns a textual representation of this product data. The text returned is a string cretaed from the bytes
     * array elements in this value interpreted as ASCII values.
     *
     * @return a text representing this product data, never {@code null}
     */
    [[nodiscard]] std::string GetElemStringAt(int index) const override;

    /**
     * Sets the data of this value. The data must be a string, an byte or an char array.
     * Each has to have at least a length of one.
     *
     * @param data the data array
     *
     * @throws IllegalArgumentException if data is {@code null} or it is not an array of the required type or
     *                                  does the array length is less than one.
     */
    void SetElems(std::any data) override;
};

}  // namespace snapengine
}  // namespace alus