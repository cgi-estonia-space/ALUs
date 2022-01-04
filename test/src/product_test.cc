/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductTest.java
 * ported for native code.
 * Copied from(https://github.com/senbox-org/snap-engine). It was originally stated:
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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "gmock/gmock.h"

#include "d_geo_coding.h"
#include "dummy_product_reader.h"
#include "dummy_product_reader_plug_in.h"
#include "s_geo_coding.h"
#include "snap-core/core/dataio/abstract_product_reader.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/flag_coding.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/datamodel/product_data.h"
#include "snap-core/core/datamodel/product_node_group.h"
#include "snap-core/core/datamodel/tie_point_grid.h"
#include "snap-core/core/datamodel/virtual_band.h"

namespace {

using alus::snapengine::Band;
using alus::snapengine::DGeoCoding;
using alus::snapengine::DummyProductReader;
using alus::snapengine::DummyProductReaderPlugIn;
using alus::snapengine::MetadataElement;
using alus::snapengine::Product;
using alus::snapengine::ProductData;
using alus::snapengine::ProductNode;
using alus::snapengine::ProductNodeGroup;
using alus::snapengine::SGeoCoding;
using alus::snapengine::TiePointGrid;
using alus::snapengine::VirtualBand;
using alus::snapengine::custom::Dimension;

class ProductTest : public ::testing::Test {
protected:
    static constexpr std::string_view PROD_TYPE{"TestProduct"};
    static constexpr int SCENE_WIDTH = 20;
    static constexpr int SCENE_HEIGHT = 30;
    std::shared_ptr<Product> product_;

    void SetUp() override {
        product_ = Product::CreateProduct("product", PROD_TYPE, SCENE_WIDTH, SCENE_HEIGHT);
        product_->SetModified(false);
    }

    void TearDown() override { Test::TearDown(); }
};

TEST_F(ProductTest, testSetAndGetReader) {
    auto product = Product::CreateProduct("name", "MER_RR__1P", 312, 213);  // NOLINT

    ASSERT_FALSE(product->GetProductReader());

    auto reader1 = std::make_shared<DummyProductReader>(std::make_shared<DummyProductReaderPlugIn>());
    product->SetProductReader(reader1);
    ASSERT_EQ(reader1, product->GetProductReader());

    auto reader2 = std::make_shared<DummyProductReader>(std::make_shared<DummyProductReaderPlugIn>());
    product->SetProductReader(reader2);
    ASSERT_EQ(reader2, product->GetProductReader());

    ASSERT_THROW(product->SetProductReader(nullptr), std::invalid_argument);
}

TEST_F(ProductTest, testAddBandWithBandParameters) {
    ASSERT_EQ(0, product_->GetNumBands());
    ASSERT_EQ(0, product_->GetBandNames().size());

    ASSERT_FALSE(product_->GetBand("band1"));
    ASSERT_EQ(false, product_->ContainsBand("band1"));

    product_->AddBand(std::make_shared<Band>("band1", ProductData::TYPE_FLOAT32, ProductTest::SCENE_WIDTH,
                                             ProductTest::SCENE_HEIGHT));

    ASSERT_EQ(1, product_->GetNumBands());
    ASSERT_EQ("band1", product_->GetBandNames().at(0));
    ASSERT_EQ(true, product_->ContainsBand("band1"));
    ASSERT_TRUE(product_->GetBandAt(0));
    ASSERT_EQ("band1", product_->GetBandAt(0)->GetName());
    ASSERT_TRUE(product_->GetBand("band1"));
    ASSERT_EQ("band1", product_->GetBand("band1")->GetName());
}

TEST_F(ProductTest, testGetType) {
    std::shared_ptr<Product> prod;

    prod = Product::CreateProduct("TestName", "TEST", SCENE_WIDTH, SCENE_HEIGHT);
    ASSERT_EQ("TEST", prod->GetProductType());

    prod = Product::CreateProduct("TestName", "TEST", SCENE_WIDTH, SCENE_HEIGHT, nullptr);
    ASSERT_EQ("TEST", prod->GetProductType());
}

TEST_F(ProductTest, testGetSceneRasterWidth) {
    std::shared_ptr<Product> prod;

    const int raster_width_1{243};
    prod = Product::CreateProduct("TestName", PROD_TYPE, raster_width_1, SCENE_HEIGHT);
    ASSERT_EQ(raster_width_1, prod->GetSceneRasterWidth());

    const int raster_width_2{789};
    prod = Product::CreateProduct("TestName", PROD_TYPE, raster_width_2, SCENE_HEIGHT, nullptr);
    ASSERT_EQ(raster_width_2, prod->GetSceneRasterWidth());
}

TEST_F(ProductTest, testGetSceneRasterHeight) {
    std::shared_ptr<Product> prod;

    const int raster_height_1{373};
    prod = Product::CreateProduct("TestName", PROD_TYPE, SCENE_WIDTH, raster_height_1);
    ASSERT_EQ(373, prod->GetSceneRasterHeight());

    const int raster_height_2{427};
    prod = Product::CreateProduct("TestName", PROD_TYPE, SCENE_WIDTH, raster_height_2, nullptr);
    ASSERT_EQ(raster_height_2, prod->GetSceneRasterHeight());
}

// todo: when some sort of image logic gets integrated (e.g java snap version has class MultiLevelImage)
// TEST_F(ProductTest, testBitmaskHandlingByte) {
//    std::shared_ptr<Product> product = Product::CreateProduct("Y", "X", 4, 4);
//
//    std::shared_ptr<Band> band = product->AddBand("flags", ProductData::TYPE_INT8);
//    int8_t f1 = 0x01;
//    int8_t f2 = 0x02;
//    int8_t f3 = 0x04;
//
//    auto flag_coding = std::make_shared<FlagCoding>(std::string("flags"));
//    flag_coding->AddFlag("F1", f1, nullptr);
//    flag_coding->AddFlag("F2", f2, nullptr);
//    flag_coding->AddFlag("F3", f3, nullptr);
//
//    product->GetFlagCodingGroup()->Add(flag_coding);
//    band->SetSampleCoding(flag_coding);
//
//    band->EnsureRasterData();
//    std::vector<int8_t> elems = {0, f1, f2, f3, f1, 0, f1 + f2, f1 + f3, f2, f1 + f2, 0, f2 + f3,f3, f1 + f3, f2 + f3,
//    0}; band->GetRasterData()->SetElems(elems); product->SetModified(false);
//
//    std::vector<int32_t> f1_mask ={
//        0, 255, 0, 0,
//        255, 0, 255, 255,
//        0, 255, 0, 0,
//        0, 255, 0, 0
//    };
//    TestBitmaskHandling(product, "flags.F1", f1_mask);
//
//    std::vector<int32_t> f1_and_f2_mask = {0, 0, 0, 0,0, 0, 255, 0,0, 255, 0, 0,0, 0, 0, 0 };
//    TestBitmaskHandling(product, "flags.F1 AND flags.F2", f1_and_f2_mask);
//
//    std::vector<int32_t> f1_and_f2_or_f3_mask = { 0, 0, 0, 255, 0, 0, 255, 255, 0, 255, 0, 255, 255, 255, 255, 0 };
//    TestBitmaskHandling(product, "(flags.F1 AND flags.F2) OR flags.F3", f1_and_f2_or_f3_mask);
//}

TEST_F(ProductTest, testGetAndSetRefNo) {
    ASSERT_EQ(0, product_->GetRefNo());

    ASSERT_THROW(product_->SetRefNo(0), std::invalid_argument);

    const int ref_no_1{14};
    product_->SetRefNo(ref_no_1);
    ASSERT_EQ(ref_no_1, product_->GetRefNo());

    ASSERT_THROW(product_->SetRefNo(23), std::runtime_error);

    // no exception expected when the reference number to be set is the same as the one already set
    product_->SetRefNo(ref_no_1);
    ASSERT_EQ(ref_no_1, product_->GetRefNo());
}

// currently we have not ported listeners which track property changes
TEST_F(ProductTest, testGetAndSetFileLocationProperty) {
    std::shared_ptr<Product> product = Product::CreateProduct("A", "B", 10, 10);  // NOLINT

    ASSERT_EQ("", product->GetFileLocation());

    product->SetFileLocation(boost::filesystem::path("test1.nc"));
    ASSERT_EQ(boost::filesystem::path("test1.nc"), product->GetFileLocation());

    product->SetFileLocation(boost::filesystem::path("test2.nc"));
    ASSERT_EQ(boost::filesystem::path("test2.nc"), product->GetFileLocation());

    product->SetFileLocation("");
    ASSERT_EQ("", product->GetFileLocation());
}

TEST_F(ProductTest, testModifiedProperty) {
    //"product should be initially un-modified"
    ASSERT_EQ(false, product_->IsModified());
    product_->SetModified(true);
    ASSERT_EQ(true, product_->IsModified());
    product_->SetModified(false);
    ASSERT_EQ(false, product_->IsModified());
}

TEST_F(ProductTest, testModifiedFlagAfterBandHasBeenAddedAndRemoved) {
    auto band = std::make_shared<Band>("band1", ProductData::TYPE_FLOAT32, SCENE_WIDTH, SCENE_HEIGHT);
    ASSERT_EQ(nullptr, product_->GetBand("band1"));
    product_->AddBand(band);
    ASSERT_EQ(band, product_->GetBand("band1"));
    //"added band, modified flag should be set"
    ASSERT_EQ(true, product_->IsModified());
    product_->SetModified(false);
    product_->RemoveBand(band);
    ASSERT_EQ(nullptr, product_->GetBand("band1"));
    //"removed band, modified flag should be set",
    ASSERT_EQ(true, product_->IsModified());
}

TEST_F(ProductTest, testModifiedFlagAfterBandHasBeenModified) {
    auto band = std::make_shared<Band>("band1", ProductData::TYPE_FLOAT32, SCENE_WIDTH, SCENE_HEIGHT);
    product_->AddBand(band);
    product_->SetModified(false);

    band->SetData(ProductData::CreateInstance(std::vector<float>(SCENE_WIDTH * SCENE_HEIGHT)));
    // data initialized, modified flag should not be set
    ASSERT_EQ(false, product_->IsModified());

    band->SetData(ProductData::CreateInstance(std::vector<float>(SCENE_WIDTH * SCENE_HEIGHT)));
    // data modified, modified flag should be set
    ASSERT_EQ(true, product_->IsModified());

    band->SetModified(false);
    product_->SetModified(false);

    band->SetData(nullptr);
    // data set to null, modified flag should be set
    ASSERT_EQ(true, product_->IsModified());
}

TEST_F(ProductTest, testModifiedFlagDelegation) {
    auto band1 = std::make_shared<Band>("band1", ProductData::TYPE_FLOAT32, SCENE_WIDTH, SCENE_HEIGHT);
    auto band2 = std::make_shared<Band>("band2", ProductData::TYPE_FLOAT32, SCENE_WIDTH, SCENE_HEIGHT);

    product_->AddBand(band1);
    product_->AddBand(band2);
    product_->SetModified(false);

    band1->SetModified(true);
    ASSERT_EQ(true, band1->IsModified());
    ASSERT_EQ(false, band2->IsModified());
    ASSERT_EQ(true, product_->IsModified());

    band2->SetModified(true);
    ASSERT_EQ(true, band1->IsModified());
    ASSERT_EQ(true, band2->IsModified());
    ASSERT_EQ(true, product_->IsModified());

    product_->SetModified(false);
    ASSERT_EQ(false, band1->IsModified());
    ASSERT_EQ(false, band2->IsModified());
    ASSERT_EQ(false, product_->IsModified());
}

TEST_F(ProductTest, testDefaultGroups) {
    auto p = Product::CreateProduct("n", "t", 10, 10);  // NOLINT
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<ProductNode>>> groups = p->GetGroups();
    ASSERT_TRUE(groups);

    ASSERT_TRUE(p->GetGroup("bands"));
    ASSERT_EQ(p->GetGroup("bands"), groups->Get(std::string("bands")));

    ASSERT_TRUE(p->GetGroup("tie_point_grids"));
    ASSERT_EQ(p->GetGroup("tie_point_grids"), groups->Get(std::string("tie_point_grids")));

    ASSERT_TRUE(p->GetGroup("index_codings"));
    ASSERT_EQ(p->GetGroup("index_codings"), groups->Get(std::string("index_codings")));

    ASSERT_TRUE(p->GetGroup("flag_codings"));
    ASSERT_EQ(p->GetGroup("flag_codings"), groups->Get(std::string("flag_codings")));

    ASSERT_TRUE(p->GetGroup("masks"));
    ASSERT_EQ(p->GetGroup("masks"), groups->Get(std::string("masks")));

    //    todo: if we support vector data in the future
    //    ASSERT_TRUE(p->GetGroup("vector_data"));
    //    ASSERT_EQ(p->GetGroup("vector_data"), groups->Get(std::string("vector_data")));
}

// events are not ported and removed from ported tests
TEST_F(ProductTest, testAddAndRemoveGroup) {
    auto p = Product::CreateProduct("n", "t", 10, 10);  // NOLINT
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<ProductNode>>> groups = p->GetGroups();
    ASSERT_TRUE(groups);

    int node_count_0 = groups->GetNodeCount();
    auto spectra = std::make_shared<ProductNodeGroup<std::shared_ptr<MetadataElement>>>(p.get(), "spectra", true);
    groups->Add(spectra);
    ASSERT_EQ(groups->GetNodeCount(), node_count_0 + 1);

    ASSERT_EQ(spectra, p->GetGroup("spectra"));
    spectra->Add(std::make_shared<MetadataElement>("radiance"));
    spectra->Add(std::make_shared<MetadataElement>("reflectance"));
    // FYI: our test actually is different from snap java version from here
    ASSERT_EQ(spectra->GetNodeCount(), 2);
    std::shared_ptr<MetadataElement> remove_node = spectra->Get("radiance");
    spectra->Remove(remove_node);
    ASSERT_EQ("radiance", remove_node->GetName());
    ASSERT_EQ(spectra->GetNodeCount(), 1);
}

TEST_F(ProductTest, testUniqueGeoCodings) {
    std::shared_ptr<Product> p = Product::CreateProduct("N", "T", 4, 4);  // NOLINT

    ASSERT_FALSE(p->IsUsingSingleGeoCoding());

    //    todo: make some workaround
    auto gc1 = std::make_shared<SGeoCoding>();
    auto gc2 = std::make_shared<DGeoCoding>();
    p->SetSceneGeoCoding(gc1);

    ASSERT_TRUE(p->IsUsingSingleGeoCoding());

    p->AddBand("A", ProductData::TYPE_INT8);
    p->AddBand("B", ProductData::TYPE_INT8);

    ASSERT_TRUE(p->IsUsingSingleGeoCoding());

    p->GetBand("A")->SetGeoCoding(gc1);
    p->GetBand("B")->SetGeoCoding(gc2);

    ASSERT_FALSE(p->IsUsingSingleGeoCoding());

    p->GetBand("B")->SetGeoCoding(gc1);

    ASSERT_TRUE(p->IsUsingSingleGeoCoding());
}

TEST_F(ProductTest, testContainsPixel) {
    std::shared_ptr<Product> p = Product::CreateProduct("x", "y", 1121, 2241);  // NOLINT

    ASSERT_TRUE(p->ContainsPixel(0.0F, 0.0F));
    ASSERT_TRUE(p->ContainsPixel(0.0F, 2241.0F));
    ASSERT_TRUE(p->ContainsPixel(1121.0F, 0.0F));
    ASSERT_TRUE(p->ContainsPixel(1121.0F, 2241.0F));
    ASSERT_TRUE(p->ContainsPixel(500.0F, 1000.0F));

    ASSERT_FALSE(p->ContainsPixel(-0.1F, 0.0F));
    ASSERT_FALSE(p->ContainsPixel(0.0F, 2241.1F));
    ASSERT_FALSE(p->ContainsPixel(1121.0F, -0.1F));
    ASSERT_FALSE(p->ContainsPixel(1121.1F, 2241.0F));
    ASSERT_FALSE(p->ContainsPixel(-1, -1));

    p->Dispose();
}

// DISABLED_
// todo: need events support for this test to pass
TEST_F(ProductTest, DISABLED_testExpressionIsChangedIfANodeNameIsChanged) {
    std::shared_ptr<Product> product = Product::CreateProduct("p", "t", 10, 10);  // NOLINT
    auto virtual_band =
        std::make_shared<VirtualBand>("vb", ProductData::TYPE_FLOAT32, 10, 10, "band1 + band2 - band3");  // NOLINT
    boost::filesystem::path file_location("dummy.dim");
    product->SetFileLocation(file_location);
    product->AddBand(virtual_band);
    product->AddBand("band1", ProductData::TYPE_FLOAT32);
    product->AddBand("band2", ProductData::TYPE_FLOAT32);
    product->AddBand("band3", ProductData::TYPE_FLOAT32);

    product->GetBand("band1")->SetName("b1");
    //"Name 'band1' is not changed",
    ASSERT_EQ("b1 + band2 - band3", virtual_band->GetExpression());

    ASSERT_EQ(file_location, product->GetFileLocation());
}

TEST_F(ProductTest, testThatAddBandThrowExceptionIfNameIsNotUnique) {
    std::shared_ptr<Product> product = Product::CreateProduct("p", "t", 1, 1);
    product->AddBand("band1", ProductData::TYPE_FLOAT32);
    product->AddTiePointGrid(
        std::make_shared<TiePointGrid>("grid", 2, 2, 0, 0, 1, 1, std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F}));

    try {
        product->AddBand("band1", ProductData::TYPE_FLOAT32);
    } catch (const std::invalid_argument& e) {
        ASSERT_TRUE(boost::contains(e.what(), "name"));
    }

    try {
        product->AddBand("grid", ProductData::TYPE_FLOAT32);
    } catch (const std::invalid_argument& e) {
        ASSERT_TRUE(boost::contains(e.what(), "name"));
    }
}

TEST_F(ProductTest, testThatAddTiePointGridThrowExceptionIfNameIsNotUnique) {
    std::shared_ptr<Product> product = Product::CreateProduct("p", "t", 1, 1);
    product->AddBand("band1", ProductData::TYPE_FLOAT32);
    product->AddTiePointGrid(
        std::make_shared<TiePointGrid>("grid", 2, 2, 0, 0, 1, 1, std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F}));

    try {
        product->AddTiePointGrid(
            std::make_shared<TiePointGrid>("grid", 2, 2, 0, 0, 1, 1, std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F}));
    } catch (const std::invalid_argument& e) {
        ASSERT_TRUE(boost::contains(e.what(), "name"));
    }

    try {
        product->AddTiePointGrid(
            std::make_shared<TiePointGrid>("band1", 2, 2, 0, 0, 1, 1, std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F}));
    } catch (const std::invalid_argument& e) {
        ASSERT_TRUE(boost::contains(e.what(), "name"));
    }
}

TEST_F(ProductTest, testPreferredTileSizeProperty) {
    std::shared_ptr<Product> product;

    product = Product::CreateProduct("A", "B", 1000, 2000);  // NOLINT
    ASSERT_EQ(nullptr, product->GetPreferredTileSize());

    auto dim = std::make_shared<Dimension>(128, 256);  // NOLINT
    product->SetPreferredTileSize(dim);
    ASSERT_EQ(dim, product->GetPreferredTileSize());
    auto dim2 = std::make_shared<Dimension>(300, 400);  // NOLINT
    product->SetPreferredTileSize(dim2);
    ASSERT_EQ(dim2, product->GetPreferredTileSize());

    product->SetPreferredTileSize(nullptr);
    ASSERT_EQ(nullptr, product->GetPreferredTileSize());
}

}  // namespace