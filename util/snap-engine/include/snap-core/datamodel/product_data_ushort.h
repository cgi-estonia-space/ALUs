/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class UShort which is inside org.esa.snap.core.datamodel.ProductData.java
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

#include "product_data.h"

#include <cstdint>
#include <vector>

namespace alus {
namespace snapengine {
class ProductData;

/**
 * The {@code Short} class is a {@code ProductData} specialisation for signed 16-bit integer fields.
 * <p> Internally, data is stored in an array of the type {@code short[]}.
 */
class UShort : public ProductData {
protected:
    /**
     * The internal data array holding this value's data elements.
     */
    std::vector<uint16_t> array_;

    /**
     * Constructs a new signed {@code short} value.
     *
     * @param numElems the number of elements, must not be less than one
     * @param unsigned if {@code true} an unsigned value type is constructed
     */
    UShort(int num_elems, bool is_unsigned);

    /**
     * Constructs a new signed {@code short} value.
     *
     * @param array    the elements
     * @param unsigned if {@code true} an unsigned value type is constructed
     */
    UShort(std::vector<uint16_t> array, bool is_unsigned);

    /**
     * Retuns a "deep" copy of this product data.
     *
     * @return a copy of this product data
     */
    [[nodiscard]] std::shared_ptr<ProductData> CreateDeepClone() const override;

public:
    /**
     * Constructs a new signed {@code short} value.
     *
     * @param num_elems the number of elements, must not be less than one
     */
    explicit UShort(int num_elems);

    /**
     * Constructs a new signed {@code short} value.
     *
     * @param array the elements
     */
    explicit UShort(std::vector<uint16_t> array);

    /**
     * Returns the number of data elements this value has.
     */
    [[nodiscard]] int GetNumElems() const override;
    /**
     * Please refer to {@link ProductData#getElemIntAt(int)}.
     */
    [[nodiscard]] int GetElemIntAt(int index) const override;
    /**
     * Please refer to {@link ProductData#getElemUIntAt(int)}.
     */
    [[nodiscard]] long GetElemUIntAt(int index) const override;
    /**
     * Please refer to {@link ProductData#getElemLongAt(int)}.
     */
    [[nodiscard]] long GetElemLongAt(int index) const override;
    /**
     * Please refer to {@link ProductData#getElemFloatAt(int)}.
     */
    [[nodiscard]] float GetElemFloatAt(int index) const override;
    /**
     * Please refer to {@link ProductData#getElemDoubleAt(int)}.
     */
    [[nodiscard]] double GetElemDoubleAt(int index) const override;
    /**
     * Please refer to {@link ProductData#getElemStringAt(int)}.
     */
    [[nodiscard]] std::string GetElemStringAt(int index) const override;
    /**
     * Please refer to {@link ProductData#setElemIntAt(int, int)}.
     */
    void SetElemIntAt(int index, int value) override;
    /**
     * Please refer to {@link ProductData#setElemUIntAt(int, long)}.
     */
    void SetElemUIntAt(int index, long value) override;
    /**
     * Please refer to {@link ProductData#setElemLongAt(int, long)}.
     */
    void SetElemLongAt(int index, long value) override;
    /**
     * Please refer to {@link ProductData#setElemFloatAt(int, float)}.
     */
    void SetElemFloatAt(int index, float value) override;
    /**
     * Please refer to {@link ProductData#setElemDoubleAt(int, double)}.
     */
    void SetElemDoubleAt(int index, double value) override;
    /**
     * Please refer to {@link ProductData#setElemDoubleAt(int, double)}.
     */
    void SetElemStringAt(int index, std::string_view value) override;

    /**
     * Returns the internal data array holding this value's data elements.
     *
     * @return the internal data array, never {@code null}
     */
    [[nodiscard]] std::vector<uint16_t> GetArray() const { return array_; }

    /**
     * Gets the actual value value(s). The value returned can safely been casted to an array object of the type
     * {@code short[]}.
     *
     * @return this value's value, always a {@code short[]}, never {@code null}
     */
    [[nodiscard]] std::any GetElems() const override;
    /**
     * Sets the data of this value. The data must be an array of the type {@code short[]} or
     * {@code String[]} and have a length that is equal to the value returned by the
     * {@code getNumDataElems} method.
     *
     * @param data the data array
     *
     * @throws IllegalArgumentException if data is {@code null} or it is not an array of the required type or
     *                                  does not have the required array length.
     */
    void SetElems(std::any data) override;
    /**
     * Releases all of the resources used by this object instance and all of its owned children. Its primary use is
     * to allow the garbage collector to perform a vanilla job.
     * <p>This method should be called only if it is for sure that this object instance will never be used again.
     * The results of referencing an instance of this class after a call to {@code dispose()} are undefined.
     * <p>Overrides of this method should always call {@code super.dispose();} after disposing this instance.
     */
    void Dispose() override;

    [[nodiscard]] bool EqualElems(std::shared_ptr<ProductData> other) const override;
};

}  // namespace snapengine
}  // namespace alus