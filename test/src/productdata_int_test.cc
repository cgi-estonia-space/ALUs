/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductDataIntTest.java
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
#include "product_data_int.h"

namespace {

using alus::snapengine::ProductData;

class ProductDataIntTest {};

TEST(ProductDataInt, testSingleValueConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_INT32);
    instance->SetElems(std::vector<int32_t>{2147483647});  // NOLINT

    ASSERT_EQ(ProductData::TYPE_INT32, instance->GetType());
    ASSERT_EQ(2147483647, instance->GetElemInt());
    ASSERT_EQ(2147483647L, instance->GetElemUInt());
    ASSERT_FLOAT_EQ(2147483647.0F, instance->GetElemFloat());
    ASSERT_DOUBLE_EQ(2147483647.0, instance->GetElemDouble());
    ASSERT_EQ("2147483647", instance->GetElemString());
    ASSERT_EQ(true, instance->GetElemBoolean());
    ASSERT_EQ(1, instance->GetNumElems());
    auto data = instance->GetElems();
    ASSERT_EQ(true, data.type() == typeid(std::vector<int32_t>));
    ASSERT_EQ(1, std::any_cast<std::vector<int32_t>>(data).size());
    ASSERT_EQ(true, instance->IsScalar());
    ASSERT_EQ(true, instance->IsInt());
    ASSERT_EQ("2147483647", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_INT32);
    expected_equal->SetElems(std::vector<int32_t>{2147483647});  // NOLINT
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_INT32);
    expected_unequal->SetElems(std::vector<int32_t>{2147483646});  // NOLINT
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));

    //        StreamTest
    //    ProductData dataFromStream = null;
    //    try {
    //        instance.writeTo(_outputStream);
    //        dataFromStream = ProductData.createInstance(ProductData.TYPE_INT32);
    //        dataFromStream.readFrom(_inputStream);
    //    } catch (IOException e) {
    //        fail("IOException not expected");
    //    }
    //    ASSERT_EQ(true, instance->EqualElems(dataFromStream));
}

TEST(ProductDataInt, testConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_INT32, 3);  // NOLINT
    instance->SetElems(std::vector<int32_t>{-1, 2147483647, -2147483648});                            // NOLINT

    ASSERT_EQ(ProductData::TYPE_INT32, instance->GetType());
    ASSERT_EQ(-1, instance->GetElemIntAt(0));
    ASSERT_EQ(2147483647, instance->GetElemIntAt(1));
    ASSERT_EQ(-2147483648, instance->GetElemIntAt(2));
    ASSERT_EQ(-1L, instance->GetElemUIntAt(0));
    ASSERT_EQ(2147483647L, instance->GetElemUIntAt(1));
    ASSERT_EQ(-2147483648L, instance->GetElemUIntAt(2));
    ASSERT_FLOAT_EQ(-1.0F, instance->GetElemFloatAt(0));
    ASSERT_FLOAT_EQ(2147483647.0F, instance->GetElemFloatAt(1));
    ASSERT_FLOAT_EQ(-2147483648.0F, instance->GetElemFloatAt(2));
    ASSERT_DOUBLE_EQ(-1.0, instance->GetElemDoubleAt(0));
    ASSERT_DOUBLE_EQ(2147483647.0, instance->GetElemDoubleAt(1));
    ASSERT_DOUBLE_EQ(-2147483648.0, instance->GetElemDoubleAt(2));
    ASSERT_EQ("-1", instance->GetElemStringAt(0));
    ASSERT_EQ("2147483647", instance->GetElemStringAt(1));
    ASSERT_EQ("-2147483648", instance->GetElemStringAt(2));
    ASSERT_EQ(true, instance->GetElemBooleanAt(0));
    ASSERT_EQ(true, instance->GetElemBooleanAt(1));
    ASSERT_EQ(true, instance->GetElemBooleanAt(2));
    ASSERT_EQ(3, instance->GetNumElems());
    auto data2 = instance->GetElems();
    ASSERT_EQ(true, (data2.type() == typeid(std::vector<int32_t>)));
    ASSERT_EQ(3, std::any_cast<std::vector<int32_t>>(data2).size());
    ASSERT_EQ(false, instance->IsScalar());
    ASSERT_EQ(true, instance->IsInt());
    ASSERT_EQ("-1,2147483647,-2147483648", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_INT32, 3);  // NOLINT
    expected_equal->SetElems(std::vector<int32_t>{-1, 2147483647, -2147483648});                            // NOLINT
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_INT32, 3);  // NOLINT
    expected_unequal->SetElems(std::vector<int32_t>{-1, 2147483647, -2147483647});                            // NOLINT
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));

    //        StreamTest
    //    ProductData dataFromStream = null;
    //    try {
    //        instance.writeTo(_outputStream);
    //        dataFromStream = ProductData.createInstance(ProductData.TYPE_INT32, 3);
    //        dataFromStream.readFrom(_inputStream);
    //    } catch (IOException e) {
    //        fail("IOException not expected");
    //    }
    //    ASSERT_EQ(true, instance->EqualElems(dataFromStream));
}

TEST(ProductDataInt, testSetElemsAsString) {
    std::shared_ptr<ProductData> pd = ProductData::CreateInstance(ProductData::TYPE_INT32, 3);  // NOLINT
    pd->SetElems(std::vector<std::string>{std::to_string(static_cast<int32_t> INT32_MAX),
                                          std::to_string(static_cast<int32_t>(INT32_MIN)), std::to_string(0)});
    ASSERT_EQ(INT32_MAX, pd->GetElemIntAt(0));
    ASSERT_EQ(INT32_MIN, pd->GetElemIntAt(1));
    ASSERT_EQ(0, pd->GetElemIntAt(2));
}

TEST(ProductDataInt, TestSetElemsAsStringOutOfRange) {
    std::string expected{"value is not int32_t"};

    std::shared_ptr<ProductData> pd1 = ProductData::CreateInstance(ProductData::TYPE_INT32, 1);
    try {
        auto str = std::to_string(static_cast<int64_t> INT32_MAX + 1);
        pd1->SetElems(std::vector<std::string>{str});
    } catch (const std::exception& actual) {
        ASSERT_EQ(expected, actual.what());
    }

    std::shared_ptr<ProductData> pd2 = ProductData::CreateInstance(ProductData::TYPE_INT32, 1);
    try {
        auto str = std::to_string(static_cast<int64_t> INT32_MIN - 1);
        pd2->SetElems(std::vector<std::string>{str});
    } catch (const std::exception& actual) {
        ASSERT_EQ(expected, actual.what());
    }
}

}  // namespace