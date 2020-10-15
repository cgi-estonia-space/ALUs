#include <cstdint>
#include <fstream>

#include "gmock/gmock.h"
#include "metadata_element.h"

namespace {
using namespace alus::snapengine;

class ProductDataMetadataElementTest : public ::testing::Test {
   public:
    ~ProductDataMetadataElementTest() override = default;

   protected:
    MetadataElement test_group_;

    void SetUp() override { test_group_ = MetadataElement{"test"}; }
    //    void TearDown() override { Test::TearDown(); }
};

/**
 * Tests construction failures
 */
TEST(MetadataElement, testRsAnnotation) { EXPECT_THROW(MetadataElement(nullptr), std::invalid_argument); }

/**
 * Tests the functionality for addAttribute
 */
TEST(MetadataElement, testAddAttribute) {
    MetadataElement annot{"test_me"};

    // allow null argument, but ignore it... (just like in snap)
    EXPECT_NO_THROW(annot.AddAttribute(nullptr));

    // add an attribute
    auto att =
        std::make_shared<MetadataAttribute>("Test1", ProductData::CreateInstance(ProductData::TYPE_INT32), false);
    annot.AddAttribute(att);
    ASSERT_EQ(1, annot.GetNumAttributes());
    ASSERT_EQ(att, annot.GetAttributeAt(0));
}

/**
 * Tests the functionality for containsAttribute()
 */
TEST(MetadataElement, testContainsAttribute) {
    MetadataElement annot{"test_me"};
    std::shared_ptr<MetadataAttribute> att =
        std::make_shared<MetadataAttribute>("Test1", ProductData::CreateInstance(ProductData::TYPE_INT32), false);

    // should not contain anything now
    ASSERT_EQ(false, annot.ContainsAttribute("Test1"));

    // add attribute an check again
    annot.AddAttribute(att);
    ASSERT_EQ(true, annot.ContainsAttribute("Test1"));

    // tis one should not be there
    ASSERT_EQ(false, annot.ContainsAttribute("NotMe"));
}

/**
 * Tests the functionality for getPropertyValue()
 */
TEST(MetadataElement, testGetAttribute) {
    MetadataElement annot{"yepp"};

    // a new object should not return anything on this request
    EXPECT_THROW(annot.GetAttributeAt(0), std::out_of_range);

    auto att = std::make_shared<MetadataAttribute>(
        "GuiTest_DialogAndModalDialog", ProductData::CreateInstance(ProductData::TYPE_INT32), false);
    annot.AddAttribute(att);
    ASSERT_EQ(att, annot.GetAttributeAt(0));
}

/**
 * Tests the functionality for getAttributeNames
 */
TEST(MetadataElement, testGetAttributeNames) {
    MetadataElement annot{"yepp"};

    auto att = std::make_shared<MetadataAttribute>(
        "GuiTest_DialogAndModalDialog", ProductData::CreateInstance(ProductData::TYPE_INT32), false);

    // initially no strings should be returned
    ASSERT_EQ(0, annot.GetAttributeNames().size());

    // now add one attribute and check again
    annot.AddAttribute(att);
    ASSERT_EQ(1, annot.GetAttributeNames().size());
    ASSERT_EQ("GuiTest_DialogAndModalDialog", annot.GetAttributeNames().at(0));
}

/**
 * GuiTest_DialogAndModalDialog the functionality for getNumAttributes()
 */
TEST(MetadataElement, testGetNumAttributes) {
    MetadataElement annot{"yepp"};
    auto att = std::make_shared<MetadataAttribute>(
        "GuiTest_DialogAndModalDialog", ProductData::CreateInstance(ProductData::TYPE_INT32), false);

    // a new object should not have any attributes
    ASSERT_EQ(0, annot.GetNumAttributes());

    // add one and test again
    annot.AddAttribute(att);
    ASSERT_EQ(1, annot.GetNumAttributes());
}

/**
 * Tests the functionality for removeAttribute()
 */
TEST(MetadataElement, testRemoveAttribute) {
    MetadataElement annot{"yepp"};
    auto att = std::make_shared<MetadataAttribute>(
        "GuiTest_DialogAndModalDialog", ProductData::CreateInstance(ProductData::TYPE_INT32), false);
    auto att2 = std::make_shared<MetadataAttribute>(
        "GuiTest_DialogAndModalDialog", ProductData::CreateInstance(ProductData::TYPE_INT32), false);

    // add one, check, remove again, check again
    annot.AddAttribute(att);
    ASSERT_EQ(1, annot.GetNumAttributes());
    annot.RemoveAttribute(att);
    ASSERT_EQ(0, annot.GetNumAttributes());

    // try to add existent attribute name
    annot.AddAttribute(att);
    ASSERT_EQ(1, annot.GetNumAttributes());
    annot.AddAttribute(att2);
    ASSERT_EQ(2, annot.GetNumAttributes());

    // try to remove non existent attribute
    auto att3 = std::make_shared<MetadataAttribute>(
        "DifferentName", ProductData::CreateInstance(ProductData::TYPE_INT32), false);
    annot.RemoveAttribute(att3);
    ASSERT_EQ(2, annot.GetNumAttributes());
}

// original test was covering some functionality we do not use atm
TEST(MetadataElement, testSetAtributeInt) {
    MetadataElement elem{"test"};
    ASSERT_EQ(0, elem.GetNumAttributes());

    elem.SetAttributeInt("counter", 3);

    ASSERT_EQ(1, elem.GetNumAttributes());
    auto attrib = elem.GetAttributeAt(0);
    ASSERT_NE(attrib, nullptr);
    ASSERT_EQ("counter", attrib->GetName());
    ASSERT_EQ(3, attrib->GetData()->GetElemInt());

    elem.SetAttributeInt("counter", -3);

    ASSERT_EQ(1, elem.GetNumAttributes());
    ASSERT_EQ("counter", attrib->GetName());
    ASSERT_EQ(-3, attrib->GetData()->GetElemInt());
}

}  // namespace