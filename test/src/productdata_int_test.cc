#include <cstdint>
#include <fstream>

#include "gmock/gmock.h"
#include "product_data_int.h"

namespace {
using namespace alus::snapengine;

class ProductDataIntTest {};

TEST(ProductDataInt, testSingleValueConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_INT32);
    instance->SetElems(std::vector<int32_t>{2147483647});

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
    expected_equal->SetElems(std::vector<int32_t>{2147483647});
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_INT32);
    expected_unequal->SetElems(std::vector<int32_t>{2147483646});
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
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_INT32, 3);
    instance->SetElems(std::vector<int32_t>{-1, 2147483647, -2147483648});

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

    std::shared_ptr<ProductData> expectedEqual = ProductData::CreateInstance(ProductData::TYPE_INT32, 3);
    expectedEqual->SetElems(std::vector<int32_t>{-1, 2147483647, -2147483648});
    ASSERT_EQ(true, instance->EqualElems(expectedEqual));

    std::shared_ptr<ProductData> expectedUnequal = ProductData::CreateInstance(ProductData::TYPE_INT32, 3);
    expectedUnequal->SetElems(std::vector<int32_t>{-1, 2147483647, -2147483647});
    ASSERT_EQ(false, instance->EqualElems(expectedUnequal));

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
    std::shared_ptr<ProductData> pd = ProductData::CreateInstance(ProductData::TYPE_INT32, 3);
    pd->SetElems(std::vector<std::string>{
        std::to_string((int32_t)INT32_MAX), std::to_string((int32_t)INT32_MIN), std::to_string(0)});
    ASSERT_EQ(INT32_MAX, pd->GetElemIntAt(0));
    ASSERT_EQ(INT32_MIN, pd->GetElemIntAt(1));
    ASSERT_EQ(0, pd->GetElemIntAt(2));
}

TEST(ProductDataInt, testSetElemsAsString_OutOfRange) {
    std::string expected{"value is not int32_t"};

    std::shared_ptr<ProductData> pd1 = ProductData::CreateInstance(ProductData::TYPE_INT32, 1);
    try {
        auto str = std::to_string((int64_t)INT32_MAX + 1);
        pd1->SetElems(std::vector<std::string>{str});
    } catch (const std::exception& actual) {
        ASSERT_EQ(expected, actual.what());
    }

    std::shared_ptr<ProductData> pd2 = ProductData::CreateInstance(ProductData::TYPE_INT32, 1);
    try {
        auto str = std::to_string((int64_t)INT32_MIN - 1);
        pd2->SetElems(std::vector<std::string>{str});
    } catch (const std::exception& actual) {
        ASSERT_EQ(expected, actual.what());
    }
}

}  // namespace