/**
 * This file is a filtered duplicate of a SNAP's  org.esa.snap.core.datamodel.ProductData.java ported
 * for native code. Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
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

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace alus {
namespace snapengine {

class ProductData {
    //    todo:check over type mappings e.g in java Int is long int here?
private:
    /**
     * The type ID of this value.
     */
    // todo: remove this comment if done _type
    int type_;
    // todo: remove this comment if done _elemSize
    int elem_size_;

protected:
    /**
     * Constructs a new value of the given type.
     *
     * @param type the value's type
     */
    explicit ProductData(int type);

    /**
     * Retuns a "deep" copy of this product data.
     *
     * @return a copy of this product data
     */
    [[nodiscard]] virtual std::shared_ptr<ProductData> CreateDeepClone() const = 0;

public:
    /**
     * The ID for an undefined data type.
     */
    static constexpr int TYPE_UNDEFINED = 0;

    /**
     * The ID for a signed 8-bit integer data type.
     */
    static constexpr int TYPE_INT8 = 10;

    /**
     * The ID for a signed 16-bit integer data type.
     */
    static constexpr int TYPE_INT16 = 11;

    /**
     * The ID for a signed 32-bit integer data type.
     */
    static constexpr int TYPE_INT32 = 12;

    /**
     * The ID for a signed 64-bit integer data type.
     */
    static constexpr int TYPE_INT64 = 13;

    /**
     * The ID for an unsigned 8-bit integer data type.
     */
    static constexpr int TYPE_UINT8 = 20;

    /**
     * The ID for an unsigned 16-bit integer data type.
     */
    static constexpr int TYPE_UINT16 = 21;

    /**
     * The ID for an unsigned 32-bit integer data type.
     */
    static constexpr int TYPE_UINT32 = 22;

    /**
     * The ID for a signed 32-bit floating point data type.
     */
    static constexpr int TYPE_FLOAT32 = 30;

    /**
     * The ID for a signed 64-bit floating point data type.
     */
    static constexpr int TYPE_FLOAT64 = 31;

    /**
     * The ID for a ASCII string represented by an array of bytes ({@code byte[]}).
     */
    static constexpr int TYPE_ASCII = 41;

    /**
     * The ID for a UTC date/time value represented as Modified Julian Day (MJD) (an {@code int[3]}: int[0] = days,
     * int[1] = seconds, int[2] = micro-seconds).
     */
    static constexpr int TYPE_UTC = 51;

    /**
     * The string representation of {@code TYPE_INT8}
     */
    static constexpr std::string_view TYPESTRING_INT8 = "int8";
    /**
     * The string representation of {@code TYPE_INT16}
     */
    static constexpr std::string_view TYPESTRING_INT16 = "int16";
    /**
     * The string representation of {@code TYPE_INT32}
     */
    static constexpr std::string_view TYPESTRING_INT32 = "int32";
    /**
     * The string representation of {@code TYPE_INT64}
     */
    static constexpr std::string_view TYPESTRING_INT64 = "int64";
    /**
     * The string representation of {@code TYPE_UINT8}
     */
    static constexpr std::string_view TYPESTRING_UINT8 = "uint8";
    /**
     * The string representation of {@code TYPE_UINT16}
     */
    static constexpr std::string_view TYPESTRING_UINT16 = "uint16";
    /**
     * The string representation of {@code TYPE_UINT32}
     */
    static constexpr std::string_view TYPESTRING_UINT32 = "uint32";
    /**
     * The string representation of {@code TYPE_FLOAT32}
     */
    static constexpr std::string_view TYPESTRING_FLOAT32 = "float32";
    /**
     * The string representation of {@code TYPE_FLOAT64}
     */
    static constexpr std::string_view TYPESTRING_FLOAT64 = "float64";
    /**
     * The string representation of {@code TYPE_ASCII}
     */
    static constexpr std::string_view TYPESTRING_ASCII = "ascii";

    /**
     * The string representation of {@code TYPE_UTC}
     */
    static constexpr std::string_view TYPESTRING_UTC = "utc";

    /**
     * Gets the element size of an element of the given type in bytes.
     *
     * @param type the element type
     *
     * @return the size of a single element in bytes.
     *
     * @throws IllegalArgumentException if the type is not supported.
     */
    static uint64_t GetElemSize(int type);

    /**
     * Factory method which creates a value instance of the given type and with exactly one element.
     *
     * @param type the value's type
     *
     * @return a new value instance, {@code null} if the given type is not known
     */
    static std::shared_ptr<ProductData> CreateInstance(int type);

    /**
     * Factory method which creates a value instance of the given type and with the specified number of elements.
     *
     * @param type     the value's type
     * @param numElems the number of elements, must be greater than zero if type is not {@link ProductData#TYPE_UTC}
     *
     * @return a new value instance
     *
     * @throws IllegalArgumentException if one of the arguments is invalid
     */
    static std::shared_ptr<ProductData> CreateInstance(int type, int num_elems);

    static std::shared_ptr<ProductData> CreateInstance(std::string_view data);
    //    static ProductData* CreateInstance(std::vector<int8_t> data);

    static std::shared_ptr<ProductData> CreateInstance(std::vector<float> elems);

    static std::shared_ptr<ProductData> CreateInstance(std::vector<int> elems);
    /**
     * Returns a textual representation of the given data type.
     *
     * @return a data type string, {@code null} if the type is unknown
     */
    static std::string GetTypeString(int type) {
        switch (type) {
            case TYPE_INT8:
                return std::string(TYPESTRING_INT8);
            case TYPE_INT16:
                return std::string(TYPESTRING_INT16);
            case TYPE_INT32:
                return std::string(TYPESTRING_INT32);
            case TYPE_INT64:
                return std::string(TYPESTRING_INT64);
            case TYPE_UINT8:
                return std::string(TYPESTRING_UINT8);
            case TYPE_UINT16:
                return std::string(TYPESTRING_UINT16);
            case TYPE_UINT32:
                return std::string(TYPESTRING_UINT32);
            case TYPE_FLOAT32:
                return std::string(TYPESTRING_FLOAT32);
            case TYPE_FLOAT64:
                return std::string(TYPESTRING_FLOAT64);
            case TYPE_ASCII:
                return std::string(TYPESTRING_ASCII);
            case TYPE_UTC:
                return std::string(TYPESTRING_UTC);
            default:
                return nullptr;
        }
    }

    /**
     * Returns a integer representation of the given data type string.
     *
     * @return a data type integer, {@code null} if the type is unknown
     */
    static int GetType(std::string_view type) {
        if (type == TYPESTRING_INT8) {
            return TYPE_INT8;
        } else if (type == TYPESTRING_INT16) {
            return TYPE_INT16;
        } else if (type == TYPESTRING_INT32) {
            return TYPE_INT32;
        } else if (type == TYPESTRING_INT64) {
            return TYPE_INT64;
        } else if (type == TYPESTRING_UINT8) {
            return TYPE_UINT8;
        } else if (type == TYPESTRING_UINT16) {
            return TYPE_UINT16;
        } else if (type == TYPESTRING_UINT32) {
            return TYPE_UINT32;
        } else if (type == TYPESTRING_FLOAT32) {
            return TYPE_FLOAT32;
        } else if (type == TYPESTRING_FLOAT64) {
            return TYPE_FLOAT64;
        } else if (type == TYPESTRING_ASCII) {
            return TYPE_ASCII;
        } else if (type == TYPESTRING_UTC) {
            return TYPE_UTC;
        } else {
            return TYPE_UNDEFINED;
        }
    }

    /**
     * Tests whether the given value type is an unsigned integer type.
     *
     * @return true, if so
     */
    static bool IsUIntType(int type) { return type >= 20 && type < 30; }

    /**
     * Tests whether the given value type is a signed or unsigned integer type.
     *
     * @return true, if so
     */
    static bool IsIntType(int type) { return type >= 10 && type < 30; }

    /**
     * Tests whether the given value type is a floating point type.
     *
     * @return true, if so
     */
    static bool IsFloatingPointType(int type) { return type >= 30 && type < 40; }

    /**
     * Gets the element size of an element of this product data in bytes.
     *
     * @return the size of a single element in bytes
     */
    [[nodiscard]] int GetElemSize() const { return elem_size_; }

    /**
     * Returns the number of data elements this value has.
     */
    [[nodiscard]] virtual int GetNumElems() const = 0;

    /**
     * Returns the internal value. The actual type of the returned object should only be one of <ol>
     * <li>{@code byte[]} - for signed/unsigned 8-bit integer fields</li> <li>{@code short[]} - for
     * signed/unsigned 16-bit integer fields</li> <li>{@code int[]} - for signed/unsigned 32-bit integer
     * fields</li> <li>{@code long[]} - for signed/unsigned 64-bit integer
     * fields</li> <li>{@code float[]} - for signed 32-bit floating point fields</li> <li>{@code double[]} -
     * for signed 64-bit floating point fields</li> </ol>
     *
     * @return an array of one of the described types
     */
    //    [[nodiscard]] virtual void* GetElems() const = 0;
    [[nodiscard]] virtual std::any GetElems() const = 0;

    /**
     * Sets the internal value. The actual type of the given data object should only be one of <ol>
     * <li>{@code byte[]} - for signed/unsigned 8-bit integer fields</li> <li>{@code short[]} - for
     * signed/unsigned 16-bit integer fields</li> <li>{@code int[]} - for signed/unsigned 32-bit integer
     * fields</li> <li>{@code long[]} - for signed/unsigned 64-bit integer
     * fields</li> <li>{@code float[]} - for signed 32-bit floating point fields</li> <li>{@code double[]} -
     * for signed 64-bit floating point fields</li> <li>{@code String[]} - for all field types</li> </ol>
     *
     * @param data an array of one of the described types
     */
    //    virtual void SetElems(void* data) = 0;
    virtual void SetElems(std::any data) = 0;

    /**
     * Releases all of the resources used by this object instance and all of its owned children. Its primary use is
     * to allow the garbage collector to perform a vanilla job. <p>This method should be called only if it is for
     * sure that this object instance will never be used again. The results of referencing an instance of this class
     * after a call to {@code dispose()} are undefined.
     */
    virtual void Dispose() = 0;

    /**
     * Gets the value element with the given index as an {@code int}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    [[nodiscard]] virtual int GetElemIntAt(int index) const = 0;

    /**
     * Gets the value element with the given index as a {@code long}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    [[nodiscard]] virtual long GetElemUIntAt(int index) const = 0;

    /**
     * Gets the value element with the given index as an {@code long}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    [[nodiscard]] virtual long GetElemLongAt(int index) const = 0;

    /**
     * Gets the value element with the given index as a {@code float}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    [[nodiscard]] virtual float GetElemFloatAt(int index) const = 0;

    /**
     * Gets the value element with the given index as a {@code double}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    [[nodiscard]] virtual double GetElemDoubleAt(int index) const = 0;

    /**
     * Gets the value element with the given index as a {@code String}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    [[nodiscard]] virtual std::string GetElemStringAt(int index) const = 0;

    /**
     * Sets the value at the specified index as an {@code int}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     * @param value the value to be set
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    virtual void SetElemIntAt(int index, int value) = 0;

    /**
     * Sets the value at the specified index as an unsigned {@code int} given as a {@code long}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     * @param value the value to be set
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    virtual void SetElemUIntAt(int index, long value) = 0;

    /**
     * Sets the value at the specified index as a {@code long}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     * @param value the value to be set
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    virtual void SetElemLongAt(int index, long value) = 0;

    /**
     * Sets the value at the specified index as a {@code float}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     * @param value the value to be set
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    virtual void SetElemFloatAt(int index, float value) = 0;

    /**
     * Sets the value at the specified index as a {@code double}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     * @param value the value to be set
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    virtual void SetElemDoubleAt(int index, double value) = 0;

    /**
     * Sets the value at the specified index as a {@code String}.
     * <p><i>THE METHOD IS CURRENTLY NOT IMPLEMENTED.</i>
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     * @param value the value to be set
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    virtual void SetElemStringAt(int index, std::string_view value) = 0;

    /**
     * Tests whether this ProductData is equal to another one.
     * Performs an element-wise comparision if the other object is a {@link ProductData} instance of the same data
     * type. Otherwise the method behaves like {@link Object#equals(Object)}.
     *
     * @param other the other one
     */
    [[nodiscard]] virtual bool EqualElems(std::shared_ptr<ProductData> other) const = 0;
    /**
     * Returns the value as an {@code int}.
     * <p>The method assumes that this value is a scalar and therefore simply returns {@code getElemIntAt(0)}.
     *
     * @see #getElemIntAt(int index)
     */
    [[nodiscard]] int GetElemInt() const { return GetElemIntAt(0); }

    /**
     * Sets the value as an {@code int}. <p>The method assumes that this value is a scalar and therefore simply
     * calls {@code setElemIntAt(0, value)}.
     *
     * @param value the value to be set
     *
     * @see #setElemIntAt(int index, int value)
     */
    void SetElemInt(int value);

    /**
     * Returns the value as an unsigned {@code int} given as a {@code long}.
     * <p>The method assumes that this value is a scalar and therefore simply returns {@code getElemUIntAt(0)}.
     *
     * @see #getElemUIntAt(int index)
     */
    [[nodiscard]] long GetElemUInt() const { return GetElemUIntAt(0); }

    /**
     * Returns the value as a {@code long}.
     * <p>The method assumes that this value is a scalar and therefore simply returns {@code getElemLongAt(0)}.
     *
     * @see #getElemLongAt(int index)
     */
    [[nodiscard]] long GetElemLong() const { return GetElemLongAt(0); }

    /**
     * Returns the value as an {@code float}.
     * <p>The method assumes that this value is a scalar and therefore simply returns {@code getElemFloatAt(0)}.
     *
     * @see #getElemFloatAt(int index)
     */
    [[nodiscard]] float GetElemFloat() const { return GetElemFloatAt(0); }

    /**
     * Returns the value as an {@code double}.
     * <p>The method assumes that this value is a scalar and therefore simply returns {@code getElemDoubleAt(0)}.
     *
     * @see #getElemDoubleAt(int index)
     */
    [[nodiscard]] double GetElemDouble() const { return GetElemDoubleAt(0); }

    /**
     * Sets the value as a {@code double}. <p>The method assumes that this value is a scalar and therefore simply
     * calls {@code setElemDoubleAt(0)}.
     *
     * @param value the value to be set
     *
     * @see #setElemDoubleAt(int index, double value)
     */
    void SetElemDouble(double value);

    /**
     * Sets the value as a {@code float}. <p>The method assumes that this value is a scalar and therefore simply
     * calls {@code setElemFloatAt(0, value)}.
     *
     * @param value the value to be set
     *
     * @see #setElemFloatAt(int index, float value)
     */
    void SetElemFloat(float value);

    /**
     * Sets the value as a {@code String}. <p>The method assumes that this value is a scalar and therefore simply
     * calls {@code setElemStringAt(0)}.
     *
     * @param value the value to be set
     *
     * @see #setElemStringAt
     */
    void SetElemString(std::string_view value);

    /**
     * Sets the value as an unsigned {@code int} given as a {@code long}. <p>The method assumes that this
     * value is a scalar and therefore simply calls {@code setElemUIntAt(0, value)}.
     *
     * @param value the value to be set
     *
     * @see #setElemUIntAt(int index, long value)
     */
    void SetElemUInt(int value);
    /**
     * Sets the value as a {@code boolean}. <p>The method assumes that this value is a scalar and therefore simply
     * calls {@code setElemDoubleAt(0)}.
     *
     * @param value the value to be set
     *
     * @see #setElemBooleanAt(int index, boolean value)
     */
    void SetElemBoolean(bool value);

    /**
     * Sets the value as a {@code long}. <p>The method assumes that this value is a scalar and therefore simply
     * calls {@code setElemLongInt(0, value)}.
     *
     * @param value the value to be set
     *
     * @see #setElemLongAt(int index, long value)
     */
    void SetElemLong(long value);

    /**
     * Sets the value at the specified index as a {@code boolean}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     * @param value the value to be set
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    void SetElemBooleanAt(int index, bool value);

    /**
     * Returns the value as an {@code boolean}. <p>The method assumes that this value is a scalar and therefore
     * simply returns {@code getElemBooleanAt(0)}.
     *
     * @see #getElemBooleanAt(int index)
     */
    [[nodiscard]] bool GetElemBoolean() const { return GetElemBooleanAt(0); }

    /**
     * Gets the value element with the given index as a {@code boolean}.
     *
     * @param index the value index, must be {@code &gt;=0} and {@code &lt;getNumDataElems()}
     *
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    [[nodiscard]] bool GetElemBooleanAt(int index) const { return (GetElemIntAt(index) != 0); }

    /**
     * Returns this value's type ID.
     */
    [[nodiscard]] int GetType() const { return type_; }

    /**
     * Returns the value as a {@code String}. The text returned is the comma-separated list of elements contained
     * in this value.
     *
     * @return a text representing this fields value, never {@code null}
     */
    [[nodiscard]] virtual std::string GetElemString();

    /**
     * Tests if this value is a scalar.
     *
     * @return true, if so
     */
    [[nodiscard]] bool IsScalar() const { return GetNumElems() == 1; }

    [[nodiscard]] bool IsInt() const { return IsIntType(type_); }

    /**
     * Returns a string representation of this value which can be used for debugging purposes.
     */
    [[nodiscard]] std::string ToString() { return GetElemString(); }

    /**
     * Returns this value's data type String.
     */
    virtual std::string GetTypeString() { return GetTypeString(GetType()); }

    friend class MetadataAttribute;  // friend class
    friend class DataNode;           // friend class
};

}  // namespace snapengine
}  // namespace alus