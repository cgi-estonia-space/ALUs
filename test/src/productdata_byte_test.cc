/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductDataByteTest.java
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
#include <cstdint>
#include <fstream>

#include "gmock/gmock.h"
#include "product_data_byte.h"

namespace {

using alus::snapengine::ProductData;

class ProductDataByteTest {};

TEST(ProductDataByte, testSingleValueConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_INT8);
    instance->SetElems(std::vector<int8_t>{127});  // NOLINT

    ASSERT_EQ(ProductData::TYPE_INT8, instance->GetType());
    ASSERT_EQ(127, instance->GetElemInt());
    ASSERT_EQ(127L, instance->GetElemUInt());
    ASSERT_FLOAT_EQ(127.0F, instance->GetElemFloat());
    ASSERT_DOUBLE_EQ(127.0, instance->GetElemDouble());
    ASSERT_EQ("127", instance->GetElemString());
    ASSERT_EQ(1, instance->GetNumElems());

    auto data = instance->GetElems();
    ASSERT_EQ(true, data.type() == typeid(std::vector<int8_t>));
    ASSERT_EQ(1, std::any_cast<std::vector<int8_t>>(data).size());

    ASSERT_EQ(true, instance->IsScalar());
    ASSERT_EQ(true, instance->IsInt());
    ASSERT_EQ("127", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_INT8);
    expected_equal->SetElems(std::vector<int8_t>{127});  // NOLINT
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_INT8);
    expected_unequal->SetElems(std::vector<int8_t>{126});  // NOLINT
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));

    ////        StreamTest
    //    ProductData dataFromStream = null;
    //    try {
    //        instance.writeTo(_outputStream);
    //        dataFromStream = ProductData.createInstance(ProductData.TYPE_INT8);
    //        dataFromStream.readFrom(_inputStream);
    //    } catch (IOException e) {
    //        fail("IOException not expected");
    //    }
    //    ASSERT_EQ(true, instance.equalElems(dataFromStream));
}

TEST(ProductDataByte, testConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_INT8, 4);  // NOLINT
    instance->SetElems(std::vector<int8_t>{-1, 127, -128, 0});                                       // NOLINT

    ASSERT_EQ(ProductData::TYPE_INT8, instance->GetType());
    ASSERT_EQ(-1, instance->GetElemIntAt(0));
    ASSERT_EQ(127, instance->GetElemIntAt(1));
    ASSERT_EQ(-128, instance->GetElemIntAt(2));
    ASSERT_EQ(-1L, instance->GetElemUIntAt(0));
    ASSERT_EQ(127L, instance->GetElemUIntAt(1));
    ASSERT_EQ(-128L, instance->GetElemUIntAt(2));
    ASSERT_FLOAT_EQ(-1.0F, instance->GetElemFloatAt(0));
    ASSERT_FLOAT_EQ(127.0F, instance->GetElemFloatAt(1));
    ASSERT_FLOAT_EQ(-128.0F, instance->GetElemFloatAt(2));
    ASSERT_DOUBLE_EQ(-1.0, instance->GetElemDoubleAt(0));
    ASSERT_DOUBLE_EQ(127.0, instance->GetElemDoubleAt(1));
    ASSERT_DOUBLE_EQ(-128.0, instance->GetElemDoubleAt(2));
    ASSERT_EQ("-1", instance->GetElemStringAt(0));
    ASSERT_EQ("127", instance->GetElemStringAt(1));
    ASSERT_EQ("-128", instance->GetElemStringAt(2));
    ASSERT_EQ(true, instance->GetElemBooleanAt(0));
    ASSERT_EQ(true, instance->GetElemBooleanAt(1));
    ASSERT_EQ(false, instance->GetElemBooleanAt(3));
    ASSERT_EQ(4, instance->GetNumElems());

    auto data2 = instance->GetElems();
    ASSERT_EQ(true, data2.type() == typeid(std::vector<int8_t>));
    ASSERT_EQ(4, std::any_cast<std::vector<int8_t>>(data2).size());

    ASSERT_EQ(false, instance->IsScalar());
    ASSERT_EQ(true, instance->IsInt());
    ASSERT_EQ("-1,127,-128,0", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_INT8, 4);  // NOLINT
    expected_equal->SetElems(std::vector<int8_t>{-1, 127, -128, 0});                                       // NOLINT
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_INT8, 4);  // NOLINT
    expected_unequal->SetElems(std::vector<int8_t>{-1, 127, -127, 0});                                       // NOLINT
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));

    //        StreamTest
    //    ProductData dataFromStream = null;
    //    try {
    //        instance.writeTo(_outputStream);
    //        dataFromStream = ProductData.createInstance(ProductData.TYPE_INT8, 4);
    //        dataFromStream.readFrom(_inputStream);
    //    } catch (IOException e) {
    //        fail("IOException not expected");
    //    }
    //    ASSERT_EQ(true, instance.equalElems(dataFromStream));
}

TEST(ProductDataByte, testSetElemsAsString) {
    std::shared_ptr<ProductData> pd = ProductData::CreateInstance(ProductData::TYPE_INT8, 3);  // NOLINT
    pd->SetElems(std::vector<std::string>{std::to_string(static_cast<int8_t> INT8_MAX),
                                          std::to_string(static_cast<int8_t> INT8_MIN), std::to_string(0)});

    ASSERT_EQ(INT8_MAX, pd->GetElemIntAt(0));
    ASSERT_EQ(INT8_MIN, pd->GetElemIntAt(1));
    ASSERT_EQ(0, pd->GetElemIntAt(2));
}

TEST(ProductDataByte, testSetElemsAsStringOutOfRange) {
    std::string expected{"value is not int8_t"};
    std::shared_ptr<ProductData> pd1 = ProductData::CreateInstance(ProductData::TYPE_INT8, 1);
    try {
        auto str = std::to_string(static_cast<int16_t> INT8_MAX + 1);
        pd1->SetElems(std::vector<std::string>{str});
    } catch (const std::exception& actual) {
        ASSERT_EQ(expected, actual.what());
    }

    std::shared_ptr<ProductData> pd2 = ProductData::CreateInstance(ProductData::TYPE_INT8, 1);
    try {
        auto str = std::to_string(static_cast<int16_t> INT8_MIN - 1);
        pd2->SetElems(std::vector<std::string>{str});
    } catch (const std::exception& actual) {
        ASSERT_EQ(expected, actual.what());
    }
}

}  // namespace