/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class Float which is inside org.esa.snap.core.datamodel.ProductData.java
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

#include <vector>

#include "product_data.h"

/**
 * The {@code ProductData.Float} class is a {@code ProductData} specialisation for 32-bit floating point
 * fields.
 * <p> Internally, data is stored in an array of the type {@code float[]}.
 */
namespace alus {
namespace snapengine {
class ProductData;
class Float : public ProductData {
   protected:
    /**
     * The internal data array holding this value's data elements.
     */
    std::vector<float> array_;

    /**
     * Retuns a "deep" copy of this product data.
     *
     * @return a copy of this product data
     */
    [[nodiscard]] std::shared_ptr<ProductData> CreateDeepClone() const override;

   public:
    /**
     * Constructs a new {@code Float} instance with the given number of elements.
     *
     * @param numElems the number of elements, must not be less than one
     */
    explicit Float(int num_elems);

    /**
     * Constructs a new {@code Float} instance for the given array reference.
     *
     * @param array the array reference
     */
    explicit Float(std::vector<float> array);

    /**
     * Returns the number of data elements this value has.
     */
    [[nodiscard]] int GetNumElems() const override;
    [[nodiscard]] int GetElemIntAt(int index) const override;
    [[nodiscard]] long GetElemUIntAt(int index) const override;
    [[nodiscard]] long GetElemLongAt(int index) const override;
    [[nodiscard]] float GetElemFloatAt(int index) const override;
    [[nodiscard]] double GetElemDoubleAt(int index) const override;
    [[nodiscard]] std::string GetElemStringAt(int index) const override;
    void SetElemIntAt(int index, int value) override;
    void SetElemUIntAt(int index, long value) override;
    void SetElemLongAt(int index, long value) override;
    void SetElemFloatAt(int index, float value) override;
    void SetElemDoubleAt(int index, double value) override;
    /**
     * Please refer to {@link ProductData#setElemDoubleAt(int, double)}.
     */
    void SetElemStringAt(int index, std::string_view value) override;

    /**
     * Gets the actual value value(s). The value returned can safely been casted to an array object of the type
     * {@code float[]}.
     *
     * @return this value's value, always a {@code float[]}, never {@code null}
     */
    [[nodiscard]] std::any GetElems() const override;
    /**
     * Sets the data of this value. The data must be an array of the type {@code float[]} or
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
    /**
    * Tests whether this ProductData is equal to another one.
    * Performs an element-wise comparision if the other object is a {@link ProductData} instance of the same
    data type.
    * Otherwise the method behaves like {@link Object#equals(Object)}.
    *
    * @param other the other one
    */
    [[nodiscard]] bool EqualElems(std::shared_ptr<ProductData> other) const override;

    /**
     * Returns the internal data array holding this value's data elements.
     *
     * @return the internal data array, never {@code null}
     */
    [[nodiscard]] std::vector<float> GetArray() const { return array_; }
};

}  // namespace snapengine
}  // namespace alus