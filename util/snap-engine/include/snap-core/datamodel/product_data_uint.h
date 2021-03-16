/**
 * This file is a filtered duplicate of a SNAP's
 * static nested class UInt which is inside org.esa.snap.core.datamodel.ProductData.java
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

#include "product_data.h"

namespace alus {
namespace snapengine {
class ProductData;
class UInt : public ProductData {
protected:
    /**
     * The internal data array holding this value's data elements.
     */
    std::vector<uint32_t> array_;

    /**
     * Constructs a new {@code int} instance.
     *
     * @param numElems the number of elements, must not be less than one
     * @param unsigned if {@code true} an unsigned value type is constructed
     */
    UInt(int num_elems, bool is_unsigned);

protected:
    /**
     * Constructs a new signed {@code int} value.
     *
     * @param array    the elements
     * @param unsigned if {@code true} an unsigned value type is constructed
     */
    UInt(std::vector<uint32_t> array, bool is_unsigned);

    [[nodiscard]] std::shared_ptr<ProductData> CreateDeepClone() const override;

public:
    /**
     * Constructs a new signed {@code int} value.
     *
     * @param numElems the number of elements, must not be less than one
     */
    explicit UInt(int num_elems);

    /**
     * Constructs a new signed {@code int} value.
     *
     * @param array the elements
     */
    explicit UInt(std::vector<uint32_t> array);

    /**
     * Returns the number of data elements this value has.
     */
    [[nodiscard]] int GetNumElems() const override;
    void Dispose() override;
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
    [[nodiscard]] std::any GetElems() const override;
    void SetElems(std::any data) override;
    [[nodiscard]] bool EqualElems(std::shared_ptr<ProductData> other) const override;

    [[nodiscard]] std::vector<uint32_t> GetArray() const;
};

}  // namespace snapengine
}  // namespace alus