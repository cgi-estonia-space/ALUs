/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductData.java
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
#include "product_data.h"

#include <sstream>
#include <stdexcept>

#include "guardian.h"
#include "product_data_ascii.h"
#include "product_data_byte.h"
#include "product_data_double.h"
#include "product_data_float.h"
#include "product_data_int.h"
#include "product_data_long.h"
#include "product_data_short.h"
#include "product_data_ubyte.h"
#include "product_data_uint.h"
#include "product_data_ushort.h"
#include "product_data_utc.h"

namespace alus {
namespace snapengine {

ProductData::ProductData(int type) {
    type_ = type;
    elem_size_ = GetElemSize(type);
}

// changed to sizeof (java implementation has fixed size)
uint64_t ProductData::GetElemSize(int type) {
    switch (type) {
        case TYPE_INT8:
            return sizeof(int8_t);
        case TYPE_ASCII:
            return sizeof(char);
        case TYPE_UINT8:
            return sizeof(uint8_t);
        case TYPE_INT16:
            return sizeof(int16_t);
        case TYPE_UINT16:
            return sizeof(uint16_t);
        case TYPE_INT32:
            return sizeof(int32_t);
        case TYPE_UINT32:
            return sizeof(uint32_t);
        case TYPE_FLOAT32:
            return sizeof(float);
        case TYPE_UTC:
            // java uses int[3]
            return 3 * sizeof(uint32_t);
        case TYPE_INT64:
            return sizeof(int64_t);
        case TYPE_FLOAT64:
            return sizeof(double);
        default:
            throw std::invalid_argument("type is not supported");
    }
}

std::shared_ptr<ProductData> ProductData::CreateInstance(int type) { return CreateInstance(type, 1); }

std::shared_ptr<ProductData> ProductData::CreateInstance(int type, int num_elems) {
    if (num_elems < 1 && type != ProductData::TYPE_UTC) {
        throw std::invalid_argument("num_elems is less than one");
    }
    switch (type) {
        case ProductData::TYPE_INT8:
            return std::make_shared<Byte>(num_elems);
        case ProductData::TYPE_INT16:
            return std::make_shared<Short>(num_elems);
        case ProductData::TYPE_INT32:
            return std::make_shared<Int>(num_elems);
        case ProductData::TYPE_INT64:
            return std::make_shared<Long>(num_elems);
        case ProductData::TYPE_UINT8:
            return std::make_shared<UByte>(num_elems);
        case ProductData::TYPE_UINT16:
            return std::make_shared<UShort>(num_elems);
        case ProductData::TYPE_UINT32:
            return std::make_shared<UInt>(num_elems);
        case ProductData::TYPE_FLOAT32:
            //            todo:in the end check over if underlying types are mapped correctly
            return std::make_shared<Float>(num_elems);
        case ProductData::TYPE_FLOAT64:
            return std::make_shared<Double>(num_elems);
        case ProductData::TYPE_ASCII:
            return std::make_shared<ASCII>(num_elems);
        case ProductData::TYPE_UTC:
            return std::make_shared<Utc>();
        default:
            throw std::invalid_argument("Unknown type. Cannot create product data instance.");
    }
}

std::string ProductData::GetElemString() {
    if (IsScalar()) {
        return GetElemStringAt(0);
    }
    std::stringstream ss;
    for (int i = 0; i < GetNumElems(); i++) {
        if (i > 0) {
            ss << (",");
        }
        ss << GetElemStringAt(i);
    }
    return ss.str();
}

std::shared_ptr<ProductData> ProductData::CreateInstance(std::string_view data) {
    return (std::shared_ptr<ProductData>)std::make_shared<ASCII>(std::string(data));
}

void ProductData::SetElemInt(int value) { SetElemIntAt(0, value); }

void ProductData::SetElemUInt(int value) { SetElemUIntAt(0, value); }

void ProductData::SetElemDouble(double value) { SetElemDoubleAt(0, value); }

void ProductData::SetElemFloat(float value) { SetElemFloatAt(0, value); }

void ProductData::SetElemString(std::string_view value) { SetElemStringAt(0, value); }

void ProductData::SetElemBoolean(bool value) { SetElemBooleanAt(0, value); }

void ProductData::SetElemLong(long value) { SetElemLongAt(0, value); }

void ProductData::SetElemBooleanAt(int index, bool value) { SetElemIntAt(index, value ? 1 : 0); }

std::shared_ptr<ProductData> ProductData::CreateInstance(std::vector<float> elems) {
    return std::make_shared<Float>(elems);
}
std::shared_ptr<ProductData> ProductData::CreateInstance(std::vector<int> elems) {
    // vector is never nullptr
    //    snapengine::Guardian::AssertNotNull("elems", elems);
    return std::make_shared<Int>(elems);
}

}  // namespace snapengine
}  // namespace alus
