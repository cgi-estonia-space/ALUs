#include <fstream>

#include "gmock/gmock.h"
#include "metadata_attribute.h"
#include "product_data_ascii.h"

namespace {
using namespace alus::snapengine;

class ProductDataMetadataAttributeTest : public ::testing::Test {
   public:
    ~ProductDataMetadataAttributeTest() override = default;

    //   protected:
    std::shared_ptr<MetadataAttribute> attribute_int_;
    std::shared_ptr<MetadataAttribute> attribute_float_;
    std::shared_ptr<MetadataAttribute> attribute_string_;

    void SetUp() override {
        attribute_int_ = std::make_shared<MetadataAttribute>(
            "attributeInt", ProductData::CreateInstance(ProductData::TYPE_INT32, 3), false);
        attribute_float_ = std::make_shared<MetadataAttribute>(
            "attributeFloat", ProductData::CreateInstance(ProductData::TYPE_FLOAT32), false);
        attribute_string_ = std::make_shared<MetadataAttribute>(
            "attributeString", ProductData::CreateInstance(ProductData::TYPE_ASCII, 32), false);
    }
    //    void TearDown() override { Test::TearDown(); }
};

/**
 * Tests construction failures
 */
TEST(MetadataAttribute, testRsAttribute) {
    EXPECT_THROW(MetadataAttribute(nullptr, ProductData::CreateInstance(ProductData::TYPE_FLOAT32), false),
                 std::invalid_argument);
    EXPECT_THROW(MetadataAttribute("a2", nullptr, false), std::invalid_argument);
}
TEST_F(ProductDataMetadataAttributeTest, testSetValueWithWrongType) {
    // rejects illegal type?
    EXPECT_THROW(attribute_int_->SetDataElems("5"), std::invalid_argument);
}

TEST_F(ProductDataMetadataAttributeTest, testSetData) {
    EXPECT_THROW(attribute_int_->SetDataElems(nullptr), std::invalid_argument);

    // null --> new value: is modified ?
    attribute_int_->SetDataElems(std::vector<int32_t>{1, 2, 3});

    ASSERT_EQ(true,
              (std::vector<int32_t>{1, 2, 3} == std::any_cast<std::vector<int32_t>>(attribute_int_->GetDataElems())));
    //    ASSERT_EQ(true, attribute_int_->IsModified());

    // old value == new value?
    attribute_int_->SetDataElems(std::vector<int32_t>{1, 2, 3});
    ASSERT_EQ(true,
              (std::vector<int32_t>{1, 2, 3} == std::any_cast<std::vector<int32_t>>(attribute_int_->GetDataElems())));
    //    ASSERT_EQ(true, attribute_int_->IsModified());
}
// SET DESCRIPTION
TEST_F(ProductDataMetadataAttributeTest, testSetDescription) {
    // INT
    attribute_int_->SetDescription(std::nullopt);
    ASSERT_EQ(std::nullopt, attribute_int_->GetDescription());

    // null --> new value: is modified ?
    attribute_int_->SetDescription(std::make_optional("The sensor type"));
    ASSERT_EQ("The sensor type", attribute_int_->GetDescription());

    // old value == new value?
    attribute_int_->SetDescription(std::make_optional("The sensor type"));
    ASSERT_EQ("The sensor type", attribute_int_->GetDescription());

    // old value != new value?
    attribute_int_->SetDescription(std::make_optional("Upper left point"));
    ASSERT_EQ("Upper left point", attribute_int_->GetDescription());

    // FLOAT
    attribute_float_->SetDescription(std::nullopt);
    ASSERT_EQ(std::nullopt, attribute_float_->GetDescription());

    // null --> new value: is modified ?
    attribute_float_->SetDescription(std::make_optional("The sensor type"));
    ASSERT_EQ("The sensor type", attribute_float_->GetDescription());

    // old value == new value?
    attribute_float_->SetDescription(std::make_optional("The sensor type"));
    ASSERT_EQ("The sensor type", attribute_float_->GetDescription());

    // old value != new value?
    attribute_float_->SetDescription(std::make_optional("Upper left point"));
    ASSERT_EQ("Upper left point", attribute_float_->GetDescription());

    // STRING
    attribute_string_->SetDescription(std::nullopt);
    ASSERT_EQ(std::nullopt, attribute_string_->GetDescription());

    // null --> new value: is modified ?
    attribute_string_->SetDescription(std::make_optional("The sensor type"));
    ASSERT_EQ("The sensor type", attribute_string_->GetDescription());

    // old value == new value?
    attribute_string_->SetDescription(std::make_optional("The sensor type"));
    ASSERT_EQ("The sensor type", attribute_string_->GetDescription());

    // old value != new value?
    attribute_string_->SetDescription(std::make_optional("Upper left point"));
    ASSERT_EQ("Upper left point", attribute_string_->GetDescription());
}

TEST_F(ProductDataMetadataAttributeTest, testSetUnit) {
    // INT
    // old value --> null ?
    attribute_int_->SetUnit(std::nullopt);

    ASSERT_EQ(std::nullopt, attribute_int_->GetUnit());

    // nullptr --> new value: is modified ?
    attribute_int_->SetUnit(std::make_optional("mg/m^3"));
    ASSERT_EQ("mg/m^3", attribute_int_->GetUnit());

    // old value == new value?
    attribute_int_->SetUnit(std::make_optional("mg/m^3"));
    ASSERT_EQ("mg/m^3", attribute_int_->GetUnit());

    // old value != new value?
    attribute_int_->SetUnit(std::make_optional("g/cm^3"));
    ASSERT_EQ("g/cm^3", attribute_int_->GetUnit());
    // FLOAT
    // old value --> null ?
    attribute_float_->SetUnit(std::nullopt);
    ASSERT_EQ(std::nullopt, attribute_float_->GetUnit());

    // nullptr --> new value: is modified ?
    attribute_float_->SetUnit(std::make_optional("mg/m^3"));
    ASSERT_EQ("mg/m^3", attribute_float_->GetUnit());

    // old value == new value?
    attribute_float_->SetUnit(std::make_optional("mg/m^3"));
    ASSERT_EQ("mg/m^3", attribute_float_->GetUnit());

    // old value != new value?
    attribute_float_->SetUnit(std::make_optional("g/cm^3"));
    ASSERT_EQ("g/cm^3", attribute_float_->GetUnit());
    // STRING
    // old value --> null ?
    attribute_string_->SetUnit(std::nullopt);
    ASSERT_EQ(std::nullopt, attribute_string_->GetUnit());

    // nullptr --> new value: is modified ?
    attribute_string_->SetUnit(std::make_optional("mg/m^3"));
    ASSERT_EQ("mg/m^3", attribute_string_->GetUnit());

    // old value == new value?
    attribute_string_->SetUnit(std::make_optional("mg/m^3"));
    ASSERT_EQ("mg/m^3", attribute_string_->GetUnit());

    // old value != new value?
    attribute_string_->SetUnit(std::make_optional("g/cm^3"));
    ASSERT_EQ("g/cm^3", attribute_string_->GetUnit());
}
TEST(MetadataAttribute, testASCIIAttribute) {
    auto attribute1 = std::make_shared<MetadataAttribute>("name", ProductData::TYPE_ASCII, 1);
    attribute1->GetData()->SetElems(std::string_view("new data"));
    // attribute should be of type ASCII and not INT8
    ASSERT_EQ(attribute1->GetDataType(), ProductData::TYPE_ASCII);

    auto attribute1_2 = std::make_shared<MetadataAttribute>("name", ProductData::TYPE_ASCII, 1);
    attribute1->GetData()->SetElems("new data 1 2");
    // attribute should be of type ASCII and not INT8
    ASSERT_EQ(attribute1_2->GetDataType(), ProductData::TYPE_ASCII);

    auto attribute2 = std::make_shared<MetadataAttribute>("name", std::make_shared<ASCII>("my string"), false);
    // attribute should be of type ASCII and not INT8
    ASSERT_EQ(attribute2->GetDataType(), ProductData::TYPE_ASCII);
}

}  // namespace