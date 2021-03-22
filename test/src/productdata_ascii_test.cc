/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductDataAsciiTest.java
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
#include "product_data_ascii.h"

namespace {
using namespace alus::snapengine;

class ProductDataASCIITest {};

TEST(ProductDataASCII, testDataTypeInconsistency) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_ASCII);
    ASSERT_EQ(ProductData::TYPE_ASCII, instance->GetType());
}

TEST(ProductDataASCII, testSingleValueConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_ASCII);
    instance->SetElems(std::vector<int8_t>{'#'});

    ASSERT_EQ(35, instance->GetElemInt());
    ASSERT_EQ(35L, instance->GetElemUInt());
    ASSERT_FLOAT_EQ(35.0F, instance->GetElemFloat());
    ASSERT_DOUBLE_EQ(35.0, instance->GetElemDouble());
    ASSERT_EQ("#", instance->GetElemString());
    ASSERT_EQ(true, instance->GetElemBoolean());
    ASSERT_EQ(1, instance->GetNumElems());

    auto data = instance->GetElems();
    ASSERT_EQ(true, data.type() == typeid(std::vector<int8_t>));
    ASSERT_EQ(1, std::any_cast<std::vector<int8_t>>(data).size());

    ASSERT_EQ(true, instance->IsScalar());
    ASSERT_EQ(false, instance->IsInt());
    ASSERT_EQ("#", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_ASCII);
    expected_equal->SetElems(std::vector<int8_t>{35});
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_ASCII);
    expected_unequal->SetElems(std::vector<int8_t>{126});
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));
}

TEST(ProductDataASCII, testConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_ASCII, 10);
    instance->SetElems(std::vector<int8_t>{'a', 'A', 'e', 'i', 'n', ' ', 'T', 'e', 's', 't'});

    ASSERT_EQ(97, instance->GetElemIntAt(0));
    ASSERT_EQ(65, instance->GetElemIntAt(1));
    ASSERT_EQ(101, instance->GetElemIntAt(2));
    ASSERT_EQ(97L, instance->GetElemUIntAt(0));
    ASSERT_EQ(65L, instance->GetElemUIntAt(1));
    ASSERT_EQ(101L, instance->GetElemUIntAt(2));
    ASSERT_FLOAT_EQ(97.0F, instance->GetElemFloatAt(0));
    ASSERT_FLOAT_EQ(65.0F, instance->GetElemFloatAt(1));
    ASSERT_FLOAT_EQ(101.0F, instance->GetElemFloatAt(2));
    ASSERT_DOUBLE_EQ(97.0, instance->GetElemDoubleAt(0));
    ASSERT_DOUBLE_EQ(65.0, instance->GetElemDoubleAt(1));
    ASSERT_DOUBLE_EQ(101.0, instance->GetElemDoubleAt(2));
    ASSERT_EQ("a", instance->GetElemStringAt(0));
    ASSERT_EQ("A", instance->GetElemStringAt(1));
    ASSERT_EQ("e", instance->GetElemStringAt(2));
    ASSERT_EQ(10, instance->GetNumElems());

    auto data2 = instance->GetElems();
    ASSERT_EQ(true, (data2.type() == typeid(std::vector<int8_t>)));
    ASSERT_EQ(10, std::any_cast<std::vector<int8_t>>(data2).size());
    ASSERT_EQ(false, instance->IsScalar());
    ASSERT_EQ(false, instance->IsInt());
    ASSERT_EQ("aAein Test", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_ASCII, 10);
    expected_equal->SetElems(std::vector<int8_t>{'a', 'A', 'e', 'i', 'n', ' ', 'T', 'e', 's', 't'});
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_ASCII, 10);
    expected_unequal->SetElems(std::vector<int8_t>{'A', 'a', 'e', 'i', 'n', ' ', 'T', 'e', 's', 't'});
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));
}

}  // namespace