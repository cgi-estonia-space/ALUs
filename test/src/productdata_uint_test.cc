#include <cstdint>
#include <fstream>

#include "gmock/gmock.h"
#include "product_data_uint.h"

namespace {
using namespace alus::snapengine;

class ProductDataUIntTest {};

TEST(ProductDataUInt, testSingleValueConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_UINT32);
    instance->SetElems(std::vector<uint32_t>{static_cast<uint32_t>(-1)});

    ASSERT_EQ(ProductData::TYPE_UINT32, instance->GetType());
    ASSERT_EQ(-1, instance->GetElemInt());
    ASSERT_EQ(4294967295L, instance->GetElemUInt());
    ASSERT_FLOAT_EQ(4294967295.0F, instance->GetElemFloat());
    ASSERT_DOUBLE_EQ(4294967295.0, instance->GetElemDouble());
    ASSERT_EQ("4294967295", instance->GetElemString());
    ASSERT_EQ(1, instance->GetNumElems());

    auto data = instance->GetElems();
    ASSERT_EQ(true, (data.type() == typeid(std::vector<uint32_t>)));
    ASSERT_EQ(1, std::any_cast<std::vector<uint32_t>>(data).size());

    ASSERT_EQ(true, instance->IsScalar());
    ASSERT_EQ(true, instance->IsInt());
    ASSERT_EQ("4294967295", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_UINT32);
    expected_equal->SetElems(std::vector<uint32_t>{static_cast<uint32_t>(-1)});
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_UINT32);
    expected_unequal->SetElems(std::vector<uint32_t>{static_cast<uint32_t>(-2)});
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));

    ////        StreamTest
    //    ProductData dataFromStream = null;
    //    try {
    //        instance.writeTo(_outputStream);
    //        dataFromStream = ProductData::CreateInstance(ProductData::TYPE_UINT32);
    //        dataFromStream.readFrom(_inputStream);
    //    } catch (IOException e) {
    //        fail("IOException not expected");
    //    }
    //    ASSERT_EQ(true, instance->EqualElems(dataFromStream));
}

TEST(ProductDataUInt, testConstructor) {
    std::shared_ptr<ProductData> instance = ProductData::CreateInstance(ProductData::TYPE_UINT32, 3);
    instance->SetElems(
        std::vector<uint32_t>{static_cast<uint32_t>(-1), 2147483647, static_cast<uint32_t>(-2147483648)});

    ASSERT_EQ(ProductData::TYPE_UINT32, instance->GetType());
    ASSERT_EQ(-1, instance->GetElemIntAt(0));
    ASSERT_EQ(2147483647, instance->GetElemIntAt(1));
    ASSERT_EQ(-2147483648, instance->GetElemIntAt(2));
    ASSERT_EQ(4294967295L, instance->GetElemUIntAt(0));
    ASSERT_EQ(2147483647L, instance->GetElemUIntAt(1));
    ASSERT_EQ(2147483648L, instance->GetElemUIntAt(2));
    ASSERT_FLOAT_EQ(4294967295.0F, instance->GetElemFloatAt(0));
    ASSERT_FLOAT_EQ(2147483647.0F, instance->GetElemFloatAt(1));
    ASSERT_FLOAT_EQ(2147483648.0F, instance->GetElemFloatAt(2));
    ASSERT_DOUBLE_EQ(4294967295.0, instance->GetElemDoubleAt(0));
    ASSERT_DOUBLE_EQ(2147483647.0, instance->GetElemDoubleAt(1));
    ASSERT_DOUBLE_EQ(2147483648.0, instance->GetElemDoubleAt(2));
    ASSERT_EQ("4294967295", instance->GetElemStringAt(0));
    ASSERT_EQ("2147483647", instance->GetElemStringAt(1));
    ASSERT_EQ("2147483648", instance->GetElemStringAt(2));
    ASSERT_EQ(3, instance->GetNumElems());

    auto data2 = instance->GetElems();
    ASSERT_EQ(true, (data2.type() == typeid(std::vector<uint32_t>)));
    ASSERT_EQ(3, std::any_cast<std::vector<uint32_t>>(data2).size());

    ASSERT_EQ(false, instance->IsScalar());
    ASSERT_EQ(true, instance->IsInt());
    ASSERT_EQ("4294967295,2147483647,2147483648", instance->ToString());

    std::shared_ptr<ProductData> expected_equal = ProductData::CreateInstance(ProductData::TYPE_UINT32, 3);
    expected_equal->SetElems(
        std::vector<uint32_t>{static_cast<uint32_t>(-1), 2147483647, static_cast<uint32_t>(-2147483648)});
    ASSERT_EQ(true, instance->EqualElems(expected_equal));

    std::shared_ptr<ProductData> expected_unequal = ProductData::CreateInstance(ProductData::TYPE_UINT32, 3);
    expected_unequal->SetElems(
        std::vector<uint32_t>{static_cast<uint32_t>(-1), 2147483647, static_cast<uint32_t>(-2147483647)});
    ASSERT_EQ(false, instance->EqualElems(expected_unequal));

    ////        StreamTest
    //    ProductData dataFromStream = null;
    //    try {
    //        instance.writeTo(_outputStream);
    //        dataFromStream = ProductData::CreateInstance(ProductData::TYPE_UINT32, 3);
    //        dataFromStream.readFrom(_inputStream);
    //    } catch (IOException e) {
    //        fail("IOException not expected");
    //    }
    //    ASSERT_EQ(true, instance->EqualElems(dataFromStream));
}

TEST(ProductDataUInt, testSetElemsAsString) {
    auto pd = ProductData::CreateInstance(ProductData::TYPE_UINT32, 3);
    pd->SetElems(std::vector<std::string>{std::to_string(UINT32_MAX), std::to_string(0), std::to_string(0)});

    ASSERT_EQ(UINT32_MAX, pd->GetElemUIntAt(0));
    ASSERT_EQ(0, pd->GetElemUIntAt(1));
    ASSERT_EQ(0, pd->GetElemIntAt(2));
}

TEST(ProductDataUInt, testSetElemsAsString_OutOfRange) {
    auto pd1 = ProductData::CreateInstance(ProductData::TYPE_UINT32, 1);
    EXPECT_THROW(pd1->SetElems(std::vector<std::string>{std::to_string((uint64_t)UINT32_MAX + 1)}), std::out_of_range);

    auto pd2 = ProductData::CreateInstance(ProductData::TYPE_UINT32, 1);
    EXPECT_THROW(pd2->SetElems(std::vector<std::string>{std::to_string((uint64_t)0 - 1)}), std::out_of_range);
}

}  // namespace