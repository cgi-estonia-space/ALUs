/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductDataShortTest.java
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

#include "product_data_short.h"

namespace {

using alus::snapengine::ProductData;

class ProductDataShortTest {};

TEST(ProductDataShort, testSingleValueConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_INT16);
    instance->SetElems(std::vector<int16_t>{32767});  // NOLINT

    ASSERT_EQ(ProductData::TYPE_INT16, instance->GetType());
    ASSERT_EQ(32767, instance->GetElemInt());
    ASSERT_EQ(32767L, instance->GetElemUInt());
    ASSERT_FLOAT_EQ(32767.0F, instance->GetElemFloat());
    ASSERT_DOUBLE_EQ(32767.0, instance->GetElemDouble());
    ASSERT_EQ("32767", instance->GetElemString());
    ASSERT_EQ(1, instance->GetNumElems());

    auto data = instance->GetElems();
    ASSERT_EQ(true, (data.type() == typeid(std::vector<int16_t>)));
    ASSERT_EQ(1, std::any_cast<std::vector<int16_t>>(data).size());

    ASSERT_EQ(true, instance->IsScalar());
    ASSERT_EQ(true, instance->IsInt());
    ASSERT_EQ("32767", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_INT16);
    expected_equal->SetElems(std::vector<int16_t>{32767});  // NOLINT
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_INT16);
    expected_unequal->SetElems(std::vector<int16_t>{32766});  // NOLINT
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));

    //        StreamTest
    //    ProductData *dataFromStream = null;
    //    try {
    //        instance.writeTo(_outputStream);
    //        dataFromStream = ProductData::CreateInstance(ProductData::TYPE_INT16);
    //        dataFromStream.readFrom(_inputStream);
    //    } catch (IOException e) {
    //        fail("IOException not expected");
    //    }
    //    ASSERT_EQ(true, instance->EqualElems(dataFromStream));
}

TEST(ProductDataShort, testConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_INT16, 3);  // NOLINT
    instance->SetElems(std::vector<int16_t>{-1, 32767, -32768});                                      // NOLINT

    ASSERT_EQ(ProductData::TYPE_INT16, instance->GetType());
    ASSERT_EQ(-1, instance->GetElemIntAt(0));
    ASSERT_EQ(32767, instance->GetElemIntAt(1));
    ASSERT_EQ(-32768, instance->GetElemIntAt(2));
    ASSERT_EQ(-1L, instance->GetElemUIntAt(0));
    ASSERT_EQ(32767L, instance->GetElemUIntAt(1));
    ASSERT_EQ(-32768L, instance->GetElemUIntAt(2));
    ASSERT_FLOAT_EQ(-1.0F, instance->GetElemFloatAt(0));
    ASSERT_FLOAT_EQ(32767.0F, instance->GetElemFloatAt(1));
    ASSERT_FLOAT_EQ(-32768.0F, instance->GetElemFloatAt(2));
    ASSERT_DOUBLE_EQ(-1.0, instance->GetElemDoubleAt(0));
    ASSERT_DOUBLE_EQ(32767.0, instance->GetElemDoubleAt(1));
    ASSERT_DOUBLE_EQ(-32768.0, instance->GetElemDoubleAt(2));
    ASSERT_EQ("-1", instance->GetElemStringAt(0));
    ASSERT_EQ("32767", instance->GetElemStringAt(1));
    ASSERT_EQ("-32768", instance->GetElemStringAt(2));
    ASSERT_EQ(3, instance->GetNumElems());

    auto data2 = instance->GetElems();
    ASSERT_EQ(true, (data2.type() == typeid(std::vector<int16_t>)));
    ASSERT_EQ(3, std::any_cast<std::vector<int16_t>>(data2).size());

    ASSERT_EQ(false, instance->IsScalar());
    ASSERT_EQ(true, instance->IsInt());
    ASSERT_EQ("-1,32767,-32768", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_INT16, 3);  // NOLINT
    expected_equal->SetElems(std::vector<int16_t>{-1, 32767, -32768});                                      // NOLINT
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_INT16, 3);  // NOLINT
    expected_unequal->SetElems(std::vector<int16_t>{-1, 32767, -32767});                                      // NOLINT
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));

    //        StreamTest
    //    ProductData *dataFromStream = null;
    //    try {
    //        instance.writeTo(_outputStream);
    //        dataFromStream = ProductData::CreateInstance(ProductData::TYPE_INT16, 3);
    //        dataFromStream.readFrom(_inputStream);
    //    } catch (IOException e) {
    //        fail("IOException not expected");
    //    }
    //    ASSERT_EQ(true, instance->EqualElems(dataFromStream));
}

TEST(ProductDataShort, testSetElemsAsString) {
    std::shared_ptr<ProductData> pd = ProductData::CreateInstance(ProductData::TYPE_INT16, 3);  // NOLINT
    pd->SetElems(std::vector<std::string>{std::to_string(INT16_MAX), std::to_string(0), std::to_string(INT16_MIN)});

    ASSERT_EQ(INT16_MAX, pd->GetElemIntAt(0));
    ASSERT_EQ(0, pd->GetElemIntAt(1));
    ASSERT_EQ(INT16_MIN, pd->GetElemIntAt(2));
}

TEST(ProductDataShort, testSetElemsAsStringOutOfRange) {
    std::shared_ptr<ProductData> pd1 = ProductData::CreateInstance(ProductData::TYPE_INT16, 1);
    EXPECT_THROW(pd1->SetElems(std::vector<std::string>{std::to_string(INT16_MAX + 1)}), std::out_of_range);
    std::shared_ptr<ProductData> pd2 = ProductData::CreateInstance(ProductData::TYPE_INT16, 1);
    EXPECT_THROW(pd2->SetElems(std::vector<std::string>{std::to_string(INT16_MIN - 1)}), std::out_of_range);
}

}  // namespace