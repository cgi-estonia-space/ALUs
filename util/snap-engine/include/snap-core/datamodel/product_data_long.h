/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class LONG which is inside org.esa.snap.core.datamodel.ProductData.java
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

#include <cstdint>
#include <vector>

#include "product_data.h"

/**
 * The @code{.cpp} Long @endcode class is a @code{.cpp} ProductData @endcode specialisation for signed 64-bit integer
 * fields. <p> Internally, data is stored in an array of the type @code{.cpp} long[] @endcode.
 */
namespace alus {
namespace snapengine {
class ProductData;
class Long : public ProductData {
   protected:
    /**
     * The internal data array holding this value's data elements.
     */
    std::vector<int64_t> array_;

    /**
     * Returns a "deep" copy of this product data.
     *
     * @return a copy of this product data
     */
    [[nodiscard]] std::shared_ptr<ProductData> CreateDeepClone() const override;

   public:
    /**
     * Constructs a new @code{.cpp} Long @endcode instance.
     *
     * @param num_elems the number of elements, must not be less than one
     */
    explicit Long(int num_elems);

    /**
     * Constructs a new @code{.cpp} Long @endcode instance.
     *
     * @param array of the elements
     */
    explicit Long(std::vector<int64_t> array);

    /**
     * Returns the number of data elements this value has.
     */
    [[nodiscard]] int GetNumElems() const override;
    /**
     * Please refer to @link ProductData::GetElemIntAt(int) @endlink.
     */
    [[nodiscard]] int GetElemIntAt(int index) const override;
    /**
     * Please refer to {@link ProductData#GetElemUIntAt(int)}.
     */
    [[nodiscard]] long GetElemUIntAt(int index) const override;
    /**
     * Please refer to {@link ProductData#GetElemLongAt(int)}.
     */
    [[nodiscard]] long GetElemLongAt(int index) const override;
    /**
     * Please refer to {@link ProductData#GetElemFloatAt(int)}.
     */
    [[nodiscard]] float GetElemFloatAt(int index) const override;
    /**
     * Please refer to {@link ProductData#GetElemDoubleAt(int)}.
     */
    [[nodiscard]] double GetElemDoubleAt(int index) const override;
    /**
     * Please refer to {@link ProductData#GetElemStringAt(int)}.
     */
    [[nodiscard]] std::string GetElemStringAt(int index) const override;
    /**
     * Please refer to {@link ProductData#SetElemIntAt(int, int)}.
     */
    void SetElemIntAt(int index, int value) override;
    /**
     * Please refer to {@link ProductData#SetElemUIntAt(int, long)}.
     */
    void SetElemUIntAt(int index, long value) override;
    /**
     * Please refer to {@link ProductData#SetElemLongAt(int, long)}.
     */
    void SetElemLongAt(int index, long value) override;
    /**
     * Please refer to {@link ProductData#SetElemFloatAt(int, float)}.
     */
    void SetElemFloatAt(int index, float value) override;
    /**
     * Please refer to {@link ProductData::SetElemDoubleAt(int, double)}.
     */
    void SetElemDoubleAt(int index, double value) override;
    /**
     * Please refer to {@link ProductData#setElemDoubleAt(int, double)}.
     */
    void SetElemStringAt(int index, std::string_view value) override;
    /**
     * Returns the internal data array holding this value's data elements.
     *
     * @return the internal data array, never @code nullptr @endcode
     */
    [[nodiscard]] std::vector<int64_t> GetArray() const { return array_; }

    /**
     * Gets the actual value value(s). The value returned can safely been casted to an array object of the type
     * @code int[] @endcode.
     *
     * @return this value's value, always a @code std::vector<int> @endcode, never @code nullptr @endcode
     */
    [[nodiscard]] std::any GetElems() const override;

    /**
     * Sets the data of this value. The data must be an array of the type @code std::vector<int> @endcode or
     * @code std::vector<std::String> @endcode and have a length that is equal to the value returned by the
     * @code GetNumDataElems @endcode method.
     *
     * @param data the data array
     *
     * @throws IllegalArgumentException if data is @code nullptr @endcode or it is not an array of the required type or
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