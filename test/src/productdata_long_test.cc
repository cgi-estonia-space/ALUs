#include <cstdint>
#include <fstream>

#include "gmock/gmock.h"
#include "product_data_long.h"

namespace {
using namespace alus::snapengine;

class ProductDataLongTest {};

TEST(ProductDataLong, testSingleValueConstructor) {
    int64_t test_value = 2147483652L;
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_INT64);
    instance->SetElems(std::vector<int64_t>{test_value});

    ASSERT_EQ(ProductData::TYPE_INT64, instance->GetType());
    ASSERT_EQ(2147483652L, instance->GetElemLongAt(0));
    ASSERT_EQ(2147483652L, instance->GetElemUInt());
    ASSERT_FLOAT_EQ(2147483652.0F, instance->GetElemFloat());
    ASSERT_DOUBLE_EQ(2147483652.0, instance->GetElemDouble());
    ASSERT_EQ("2147483652", instance->GetElemString());
    ASSERT_EQ(true, instance->GetElemBoolean());
    ASSERT_EQ(1, instance->GetNumElems());

    auto data = instance->GetElems();
    ASSERT_EQ(true, data.type() == typeid(std::vector<int64_t>));
    ASSERT_EQ(1, std::any_cast<std::vector<int64_t>>(data).size());

    ASSERT_EQ(true, instance->IsScalar());
    ASSERT_EQ(true, instance->IsInt());
    ASSERT_EQ("2147483652", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_INT64);
    expected_equal->SetElems(std::vector<int64_t>{test_value});
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_INT64);
    expected_unequal->SetElems(std::vector<int64_t>{test_value - 1});
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));

    //        StreamTest
    //    ProductData dataFromStream = null;
    //    try {
    //        instance.writeTo(_outputStream);
    //        dataFromStream = ProductData.createInstance(ProductData.TYPE_INT64);
    //        dataFromStream.readFrom(_inputStream);
    //    } catch (IOException e) {
    //        fail("IOException not expected");
    //    }
    //    ASSERT_EQ(true, instance->EqualElems(dataFromStream));
}

TEST(ProductDataLong, testConstructor) {
    int64_t test_value = 2147483652L;
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_INT64, 3);
    instance->SetElems(std::vector<int64_t>{-1, test_value, -1 * test_value});

    ASSERT_EQ(ProductData::TYPE_INT64, instance->GetType());
    ASSERT_EQ(-1L, instance->GetElemLongAt(0));
    ASSERT_EQ(2147483652L, instance->GetElemLongAt(1));
    ASSERT_EQ(-2147483652L, instance->GetElemLongAt(2));
    ASSERT_EQ(-1, instance->GetElemIntAt(0));
    ASSERT_EQ(-1L, instance->GetElemUIntAt(0));
    ASSERT_EQ(2147483652L, instance->GetElemUIntAt(1));
    ASSERT_EQ(-2147483652L, instance->GetElemUIntAt(2));
    ASSERT_FLOAT_EQ(-1.0F, instance->GetElemFloatAt(0));
    ASSERT_FLOAT_EQ(2147483652.0F, instance->GetElemFloatAt(1));
    ASSERT_FLOAT_EQ(-2147483652.0F, instance->GetElemFloatAt(2));
    ASSERT_DOUBLE_EQ(-1.0, instance->GetElemDoubleAt(0));
    ASSERT_DOUBLE_EQ(2147483652.0, instance->GetElemDoubleAt(1));
    ASSERT_DOUBLE_EQ(-2147483652.0, instance->GetElemDoubleAt(2));
    ASSERT_EQ("-1", instance->GetElemStringAt(0));
    ASSERT_EQ("2147483652", instance->GetElemStringAt(1));
    ASSERT_EQ("-2147483652", instance->GetElemStringAt(2));
    ASSERT_EQ(true, instance->GetElemBooleanAt(0));
    ASSERT_EQ(true, instance->GetElemBooleanAt(1));
    ASSERT_EQ(true, instance->GetElemBooleanAt(2));
    ASSERT_EQ(3, instance->GetNumElems());

    auto data2 = instance->GetElems();
    ASSERT_EQ(true, (data2.type() == typeid(std::vector<int64_t>)));
    ASSERT_EQ(3, std::any_cast<std::vector<int64_t>>(data2).size());

    ASSERT_EQ(false, instance->IsScalar());
    ASSERT_EQ(true, instance->IsInt());
    ASSERT_EQ("-1,2147483652,-2147483652", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_INT64, 3);
    expected_equal->SetElems(std::vector<int64_t>{-1, 2147483652L, -1 * 2147483652L});
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_INT64, 3);
    expected_unequal->SetElems(std::vector<int64_t>{-1, 2147483652L, -1 * 2147483647 + 5});
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));

    //        StreamTest
    //    ProductData dataFromStream = null;
    //    try {
    //        instance.writeTo(_outputStream);
    //        dataFromStream = ProductData.createInstance(ProductData.TYPE_INT64, 3);
    //        dataFromStream.readFrom(_inputStream);
    //    } catch (IOException e) {
    //        fail("IOException not expected");
    //    }
    //    ASSERT_EQ(true, instance->EqualElems(dataFromStream));
}

TEST(ProductDataLong, testSetElemsAsString) {
    std::shared_ptr<ProductData> pd = ProductData::CreateInstance(ProductData::TYPE_INT64, 3);
    pd->SetElems(std::vector<std::string>{
        std::to_string(INT64_MAX),
        std::to_string(INT64_MIN),
        std::to_string(0),
    });

    ASSERT_EQ((int64_t)INT64_MAX, pd->GetElemLongAt(0));
    ASSERT_EQ((int64_t)INT64_MIN, pd->GetElemLongAt(1));
    ASSERT_EQ(0, pd->GetElemLongAt(2));
}

TEST(ProductDataLong, testSetElemsAsString_OutOfRange) {
    std::string expected{"value is not int64_t"};
    std::shared_ptr<ProductData> pd1 = ProductData::CreateInstance(ProductData::TYPE_INT64, 1);
    EXPECT_THROW(pd1->SetElems(std::vector<std::string>{"9223372036854775808"}), std::out_of_range);

    std::shared_ptr<ProductData> pd2 = ProductData::CreateInstance(ProductData::TYPE_INT64, 1);
    EXPECT_THROW(pd2->SetElems(std::vector<std::string>{"-9223372036854775809"}), std::out_of_range);
}

}  // namespace