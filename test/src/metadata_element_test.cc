/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.MetadataElementTest.java
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
#include <memory>
#include <stdexcept>

#include "gmock/gmock.h"

#include "metadata_attribute.h"
#include "metadata_element.h"

namespace {
using namespace alus::snapengine;

/*
class ProductDataMetadataElementTest : public ::testing::Test {
public:
    ~ProductDataMetadataElementTest() override = default;

protected:
    std::unique_ptr<MetadataElement> test_group_;

    void SetUp() override { test_group_ = std::make_unique<MetadataElement>("test"); }
    //    void TearDown() override { Test::TearDown(); }
};
*/
/**
 * Tests construction failures
 */
TEST(MetadataElement, testRsAnnotation) { EXPECT_THROW(MetadataElement(nullptr), std::invalid_argument); }

/**
 * Tests the functionality for addAttribute
 */
TEST(MetadataElement, testAddAttribute) {
    auto annot = std::make_shared<MetadataElement>("test_me");

    // allow null argument, but ignore it... (just like in snap)
    EXPECT_NO_THROW(annot->AddAttribute(nullptr));

    // add an attribute
    auto att =
        std::make_shared<MetadataAttribute>("Test1", ProductData::CreateInstance(ProductData::TYPE_INT32), false);
    annot->AddAttribute(att);
    ASSERT_EQ(1, annot->GetNumAttributes());
    ASSERT_EQ(att, annot->GetAttributeAt(0));
}

/**
 * Tests the functionality for containsAttribute()
 */
TEST(MetadataElement, testContainsAttribute) {
    auto annot = std::make_shared<MetadataElement>("test_me");
    std::shared_ptr<MetadataAttribute> att =
        std::make_shared<MetadataAttribute>("Test1", ProductData::CreateInstance(ProductData::TYPE_INT32), false);

    // should not contain anything now
    ASSERT_EQ(false, annot->ContainsAttribute("Test1"));

    // add attribute an check again
    annot->AddAttribute(att);
    ASSERT_EQ(true, annot->ContainsAttribute("Test1"));

    // tis one should not be there
    ASSERT_EQ(false, annot->ContainsAttribute("NotMe"));
}

/**
 * Tests the functionality for getPropertyValue()
 */
TEST(MetadataElement, testGetAttribute) {
    auto annot = std::make_shared<MetadataElement>("yepp");

    // a new object should not return anything on this request
    EXPECT_THROW(annot->GetAttributeAt(0), std::runtime_error);

    auto att = std::make_shared<MetadataAttribute>("GuiTest_DialogAndModalDialog",
                                                   ProductData::CreateInstance(ProductData::TYPE_INT32), false);
    annot->AddAttribute(att);
    ASSERT_EQ(att, annot->GetAttributeAt(0));
}

/**
 * Tests the functionality for getAttributeNames
 */
TEST(MetadataElement, testGetAttributeNames) {
    auto annot = std::make_shared<MetadataElement>("yepp");

    auto att = std::make_shared<MetadataAttribute>("GuiTest_DialogAndModalDialog",
                                                   ProductData::CreateInstance(ProductData::TYPE_INT32), false);

    // initially no strings should be returned
    ASSERT_EQ(0, annot->GetAttributeNames().size());

    // now add one attribute and check again
    annot->AddAttribute(att);
    ASSERT_EQ(1, annot->GetAttributeNames().size());
    ASSERT_EQ("GuiTest_DialogAndModalDialog", annot->GetAttributeNames().at(0));
}

/**
 * GuiTest_DialogAndModalDialog the functionality for getNumAttributes()
 */
TEST(MetadataElement, testGetNumAttributes) {
    auto annot = std::make_shared<MetadataElement>("yepp");
    auto att = std::make_shared<MetadataAttribute>("GuiTest_DialogAndModalDialog",
                                                   ProductData::CreateInstance(ProductData::TYPE_INT32), false);

    // a new object should not have any attributes
    ASSERT_EQ(0, annot->GetNumAttributes());

    // add one and test again
    annot->AddAttribute(att);
    ASSERT_EQ(1, annot->GetNumAttributes());
}

/**
 * Tests the functionality for removeAttribute()
 */
TEST(MetadataElement, testRemoveAttribute) {
    auto annot = std::make_shared<MetadataElement>("yepp");
    auto att = std::make_shared<MetadataAttribute>("GuiTest_DialogAndModalDialog",
                                                   ProductData::CreateInstance(ProductData::TYPE_INT32), false);
    auto att2 = std::make_shared<MetadataAttribute>("GuiTest_DialogAndModalDialog",
                                                    ProductData::CreateInstance(ProductData::TYPE_INT32), false);

    // add one, check, remove again, check again
    annot->AddAttribute(att);
    ASSERT_EQ(1, annot->GetNumAttributes());
    annot->RemoveAttribute(att);
    ASSERT_EQ(0, annot->GetNumAttributes());

    // try to add existent attribute name
    annot->AddAttribute(att);
    ASSERT_EQ(1, annot->GetNumAttributes());
    annot->AddAttribute(att2);
    ASSERT_EQ(2, annot->GetNumAttributes());

    // try to remove non existent attribute
    auto att3 = std::make_shared<MetadataAttribute>("DifferentName",
                                                    ProductData::CreateInstance(ProductData::TYPE_INT32), false);
    annot->RemoveAttribute(att3);
    ASSERT_EQ(2, annot->GetNumAttributes());
}

// original test was covering some functionality we do not use atm
TEST(MetadataElement, testSetAtributeInt) {
    auto elem = std::make_shared<MetadataElement>("test");
    ASSERT_EQ(0, elem->GetNumAttributes());

    elem->SetAttributeInt("counter", 3);

    ASSERT_EQ(1, elem->GetNumAttributes());
    auto attrib = elem->GetAttributeAt(0);
    ASSERT_NE(attrib, nullptr);
    ASSERT_EQ("counter", attrib->GetName());
    ASSERT_EQ(3, attrib->GetData()->GetElemInt());

    elem->SetAttributeInt("counter", -3);

    ASSERT_EQ(1, elem->GetNumAttributes());
    ASSERT_EQ("counter", attrib->GetName());
    ASSERT_EQ(-3, attrib->GetData()->GetElemInt());
}

}  // namespace