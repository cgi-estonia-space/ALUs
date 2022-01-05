/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class Int which is inside org.esa.snap.core.datamodel.ProductData.java
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
#include <vector>

#include <boost/lexical_cast.hpp>

#include "product_data.h"

namespace alus::snapengine {
class ProductData;

/**
 * The {@code Int} class is a {@code ProductData} specialisation for signed 32-bit integer fields.
 * <p> Internally, data is stored in an array of the type {@code int[]}.
 */
class Int : public ProductData {
protected:
    /**
     * The internal data array holding this value's data elements.
     */
    std::vector<int32_t> array_;

    /**
     * Constructs a new {@code int} instance.
     *
     * @param numElems the number of elements, must not be less than one
     * @param unsigned if {@code true} an unsigned value type is constructed
     */
    Int(int num_elems, bool is_unsigned);

    /**
     * Constructs a new signed {@code int} value.
     *
     * @param array    the elements
     * @param unsigned if {@code true} an unsigned value type is constructed
     */
    Int(std::vector<int32_t> array, bool is_unsigned);

    [[nodiscard]] std::shared_ptr<ProductData> CreateDeepClone() const override;

public:
    /**
     * Constructs a new signed {@code int} value.
     *
     * @param numElems the number of elements, must not be less than one
     */
    explicit Int(int num_elems);

    /**
     * Constructs a new signed {@code int} value.
     *
     * @param array the elements
     */
    explicit Int(std::vector<int32_t> array);

    /**
     * Returns the number of data elements this value has.
     */
    [[nodiscard]] int GetNumElems() const override;
    void Dispose() override;
    [[nodiscard]] int GetElemIntAt(int index) const override;
    [[nodiscard]] int64_t GetElemUIntAt(int index) const override;
    [[nodiscard]] int64_t GetElemLongAt(int index) const override;
    [[nodiscard]] float GetElemFloatAt(int index) const override;
    [[nodiscard]] double GetElemDoubleAt(int index) const override;
    [[nodiscard]] std::string GetElemStringAt(int index) const override;
    void SetElemIntAt(int index, int value) override;
    void SetElemUIntAt(int index, int64_t value) override;
    void SetElemLongAt(int index, int64_t value) override;
    void SetElemFloatAt(int index, float value) override;
    void SetElemDoubleAt(int index, double value) override;
    /**
     * Please refer to {@link ProductData#setElemDoubleAt(int, double)}.
     */
    void SetElemStringAt(int index, std::string_view value) override;
    //    [[nodiscard]] void *GetElems() const override;
    [[nodiscard]] std::any GetElems() const override;
    //    void SetElems(void *data) override;
    /**
     * Sets the data of this value. The data must be a {@code std::vector<int32_t>} or
     * {@code std::vector<std::string>} and have a length that is equal to the value returned by the
     * {@code GetNumDataElems} method.
     *
     * @param data the data vector
     *
     * @throws std::bad_any_cast if data is it is not a vector of the required type.
     */
    void SetElems(std::any data) override;
    [[nodiscard]] bool EqualElems(std::shared_ptr<ProductData> other) const override;

    /**
     * Returns the internal data array holding this value's data elements.
     *
     * @return the internal data array, never {@code null}
     */
    [[nodiscard]] std::vector<int32_t> GetArray() const { return array_; }
};

}  // namespace alus::snapengine