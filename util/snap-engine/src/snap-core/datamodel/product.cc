/**
 * This file is a filtered and modified duplicate of a SNAP's
 * org.esa.snap.core.datamodel.Product.java
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
#include "product.h"

#include <stdexcept>

// TEMPORARY// todo: try to move behind IProductReader
#include "custom/i_image_reader.h"
#include "custom/i_image_writer.h"
#include "i_meta_data_reader.h"
#include "i_meta_data_writer.h"

#include "ceres-core/ceres_assert.h"
#include "snap-core/dataio/product_subset_def.h"
#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/flag_coding.h"
#include "snap-core/datamodel/geo_pos.h"
#include "snap-core/datamodel/i_geo_coding.h"
#include "snap-core/datamodel/index_coding.h"
#include "snap-core/datamodel/mask.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/pixel_pos.h"
#include "snap-core/datamodel/product_data_utc.h"
#include "snap-core/datamodel/product_node_group.h"
#include "snap-core/datamodel/quicklooks/quicklook.h"
#include "snap-core/datamodel/raster_data_node.h"
#include "snap-core/datamodel/tie_point_grid.h"
#include "snap-core/datamodel/virtual_band.h"
#include "snap-core/util/guardian.h"
#include "snap-core/util/math/math_utils.h"

namespace alus {
namespace snapengine {

Product::Product(std::string_view name, std::string_view type, int scene_raster_width, int scene_raster_height)
    : Product(name, type, scene_raster_width, scene_raster_height, nullptr) {}
Product::Product(std::string_view name, std::string_view type, int scene_raster_width, int scene_raster_height,
                 const std::shared_ptr<IProductReader>& reader)
    : Product(name, type, std::make_shared<custom::Dimension>(scene_raster_width, scene_raster_height), reader) {}
Product::Product(std::string_view name, std::string_view type) : Product(name, type, nullptr) {}

Product::Product(std::string_view name, std::string_view type, const std::shared_ptr<IProductReader>& reader)
    : Product(name, type, nullptr, reader) {}

Product::Product(std::string_view name, std::string_view type,
                 const std::shared_ptr<custom::Dimension>& scene_raster_size,
                 const std::shared_ptr<IProductReader>& reader)
    : ProductNode(name) {
    Guardian::AssertNotNullOrEmpty("type", type);
    product_type_ = type;
    scene_raster_size_ = scene_raster_size;
    reader_ = reader;
    metadata_root_ = std::make_shared<MetadataElement>(METADATA_ROOT_NAME);
}

void Product::SetModified(bool modified) {
    bool old_state = IsModified();
    if (old_state != modified) {
        ProductNode::SetModified(modified);
        if (!modified) {
            band_group_->SetModified(false);
            tie_point_grid_group_->SetModified(false);
            mask_group_->SetModified(false);
            quicklook_group_->SetModified(false);
            //                vectorDataGroup.setModified(false);
            flag_coding_group_->SetModified(false);
            index_coding_group_->SetModified(false);
            GetMetadataRoot()->SetModified(false);
        }
    }
}

std::vector<std::shared_ptr<Band>> Product::GetBands() {
    return band_group_->ToArray(std::vector<std::shared_ptr<Band>>(GetNumBands()));
}
std::shared_ptr<Band> Product::GetBand(std::string_view name) {
    Guardian::AssertNotNullOrEmpty("name", name);
    return band_group_->Get(name);
}
int Product::GetBandIndex(std::string_view name) {
    Guardian::AssertNotNullOrEmpty("name", name);
    return band_group_->IndexOf(name);
}
bool Product::ContainsBand(std::string_view name) {
    Guardian::AssertNotNullOrEmpty("name", name);
    return band_group_->Contains(name);
}

std::shared_ptr<Band> Product::AddBand(std::string_view band_name, std::string_view expression, int data_type) {
    std::shared_ptr<Band> band =
        std::make_shared<VirtualBand>(band_name, data_type, GetSceneRasterWidth(), GetSceneRasterHeight(), expression);
    AddBand(band);
    return band;
}

std::shared_ptr<Band> Product::AddBand(std::string_view band_name, std::string_view expression) {
    return AddBand(band_name, expression, ProductData::TYPE_FLOAT32);
}

std::shared_ptr<Band> Product::AddBand(std::string_view band_name, int data_type) {
    std::shared_ptr<Band> band =
        std::make_shared<Band>(band_name, data_type, GetSceneRasterWidth(), GetSceneRasterHeight());
    AddBand(band);
    return band;
}
void Product::AddBand(std::shared_ptr<Band> band) {
    Assert::NotNull(band, "band");
    Assert::Argument(!ContainsRasterDataNode(band->GetName()), "The Product '" + GetName() + "' already contains " +
                                                                   "a band with the name '" + band->GetName() + "'.");
    band_group_->Add(band);
}
bool Product::ContainsRasterDataNode(std::string_view name) {
    return ContainsBand(name) || ContainsTiePointGrid(name) || GetMaskGroup()->Contains(name);
}

std::shared_ptr<RasterDataNode> Product::GetRasterDataNode(std::string_view name) {
    std::shared_ptr<RasterDataNode> raster_data_node = GetBand(name);
    if (raster_data_node != nullptr) {
        return raster_data_node;
    }
    raster_data_node = GetTiePointGrid(name);
    if (raster_data_node != nullptr) {
        return raster_data_node;
    }
    return std::dynamic_pointer_cast<RasterDataNode>(GetMaskGroup()->Get(name));
}

std::vector<std::shared_ptr<RasterDataNode>> Product::GetRasterDataNodes() {
    // 32 came from original java version :)
    std::vector<std::shared_ptr<RasterDataNode>> raster_data_nodes(32);
    //    todo:why idea does not ask for cast for band_group and other groups need it!?
    auto band_group = GetBandGroup();
    for (int i = 0; i < band_group->GetNodeCount(); i++) {
        raster_data_nodes.push_back(band_group->Get(i));
    }
    auto mask_group = GetMaskGroup();
    for (int i = 0; i < mask_group->GetNodeCount(); i++) {
        raster_data_nodes.push_back(mask_group->Get(i));
    }
    auto tpg_group = GetTiePointGridGroup();
    for (int i = 0; i < tpg_group->GetNodeCount(); i++) {
        raster_data_nodes.push_back(tpg_group->Get(i));
    }
    return raster_data_nodes;
}

std::shared_ptr<custom::Dimension> Product::GetSceneRasterSize() {
    if (scene_raster_size_) {
        return scene_raster_size_;
    }
    //        todo: add if we need this
    //        if (!InitSceneProperties()) {
    //            throw std::runtime_error("scene raster size not set and no reference band found to derive it
    //            from");
    //        }
    throw std::runtime_error("scene raster size not set and no reference band found to derive it from");
}

int Product::GetSceneRasterWidth() { return GetSceneRasterSize()->width; }
int Product::GetSceneRasterHeight() { return GetSceneRasterSize()->height; }

const std::shared_ptr<IDataTileReader>& Product::GetReader() const { return reader_old_; }
const std::shared_ptr<custom::IImageReader>& Product::GetImageReader() const { return image_reader_; }
const std::shared_ptr<custom::IImageWriter>& Product::GetImageWriter() const { return image_writer_; }
void Product::SetReader(const std::shared_ptr<IDataTileReader>& reader) { reader_old_ = reader; }
void Product::SetImageReader(const std::shared_ptr<custom::IImageReader>& reader) { image_reader_ = reader; }

const std::shared_ptr<IDataTileWriter>& Product::GetWriter() const { return writer_old_; }
void Product::SetWriter(const std::shared_ptr<IDataTileWriter>& writer) { writer_old_ = writer; }

void Product::SetImageWriter(const std::shared_ptr<custom::IImageWriter>& writer) { image_writer_ = writer; }

const std::shared_ptr<IMetaDataReader>& Product::GetMetadataReader() const {
    if (metadata_reader_) {
        return metadata_reader_;
    }
    throw std::runtime_error("no metadata reader set");
}

void Product::SetMetadataReader(const std::shared_ptr<IMetaDataReader>& metadata_reader) {
    metadata_reader_ = metadata_reader;
    metadata_reader_->SetProduct(SharedFromBase<Product>());
}

bool Product::HasMetaDataReader() const { return metadata_reader_ != nullptr; }
const std::shared_ptr<IMetaDataWriter>& Product::GetMetadataWriter() const {
    if (metadata_writer_) {
        return metadata_writer_;
    }
    throw std::runtime_error("no metadata reader set");
}

void Product::SetMetadataWriter(const std::shared_ptr<IMetaDataWriter>& metadata_writer) {
    metadata_writer_ = metadata_writer;
    metadata_writer_->SetProduct(SharedFromBase<Product>());
}

bool Product::RemoveTiePointGrid(std::shared_ptr<TiePointGrid> tie_point_grid) {
    return tie_point_grid_group_->Remove(tie_point_grid);
}
int Product::GetNumTiePointGrids() { return tie_point_grid_group_->GetNodeCount(); }
std::shared_ptr<TiePointGrid> Product::GetTiePointGridAt(int index) { return tie_point_grid_group_->Get(index); }
std::vector<std::string> Product::GetTiePointGridNames() { return tie_point_grid_group_->GetNodeNames(); }
bool Product::RemoveBand(std::shared_ptr<Band> band) { return band_group_->Remove(band); }
int Product::GetNumBands() { return band_group_->GetNodeCount(); }
std::shared_ptr<Band> Product::GetBandAt(int index) { return band_group_->Get(index); }
std::vector<std::string> Product::GetBandNames() { return band_group_->GetNodeNames(); }
bool Product::ContainsTiePointGrid(std::string_view name) {
    Guardian::AssertNotNullOrEmpty("name", name);
    return tie_point_grid_group_->Contains(name);
}

std::shared_ptr<TiePointGrid> Product::GetTiePointGrid(std::string_view name) {
    Guardian::AssertNotNullOrEmpty("name", name);
    return tie_point_grid_group_->Get(name);
}

void Product::SetProductReader(std::shared_ptr<IProductReader> reader) {
    Guardian::AssertNotNull("ProductReader", reader);
    reader_ = reader;
}

void Product::SetPreferredTileSize(std::shared_ptr<custom::Dimension> preferred_tile_size) {
    preferred_tile_size_ = preferred_tile_size;
}

void Product::SetStartTime(std::shared_ptr<Utc> start_time) {
    std::shared_ptr<Utc> old = start_time_;
    if (start_time != old) {
        SetModified(true);
    }
    start_time_ = start_time;
}

void Product::SetEndTime(const std::shared_ptr<Utc>& end_time) {
    std::shared_ptr<Utc> old = end_time_;
    if (end_time != old) {
        SetModified(true);
    }
    end_time_ = end_time;
}

std::shared_ptr<ProductNodeGroup<std::shared_ptr<TiePointGrid>>> Product::GetTiePointGridGroup() {
    return tie_point_grid_group_;
}

void Product::SetSceneGeoCoding(const std::shared_ptr<IGeoCoding>& scene_geo_coding) {
    //    todo::support check if needed
    //    CheckGeoCoding(scene_geo_coding);
    if (scene_geo_coding_ != scene_geo_coding) {
        scene_geo_coding_ = scene_geo_coding;
        SetModified(true);
    }
}

std::shared_ptr<Quicklook> Product::GetQuicklook(std::string_view name) {
    Guardian::AssertNotNullOrEmpty("name", name);
    return quicklook_group_->Get(name);
}

std::shared_ptr<Quicklook> Product::GetDefaultQuicklook() {
    if (quicklook_group_->GetNodeCount() == 0) {
        bool was_modified = IsModified();
        quicklook_group_->Add(
            std::make_shared<Quicklook>(SharedFromBase<Product>(), Quicklook::DEFAULT_QUICKLOOK_NAME));
        if (!was_modified) {
            SetModified(false);
        }
    }
    return quicklook_group_->Get(0);
}
void Product::SetProductType(std::string_view product_type) {
    Guardian::AssertNotNullOrEmpty("productType", product_type);
    if (product_type_ != product_type) {
        product_type_ = product_type;
        SetModified(true);
    }
}

uint64_t Product::GetRawStorageSize(const std::shared_ptr<ProductSubsetDef>& subset_def) {
    uint64_t size = 0;
    for (int i = 0; i < GetNumBands(); i++) {
        size += GetBandAt(i)->GetRawStorageSize(subset_def);
    }
    for (int i = 0; i < GetNumTiePointGrids(); i++) {
        size += GetTiePointGridAt(i)->GetRawStorageSize(subset_def);
    }
    for (int i = 0; i < GetFlagCodingGroup()->GetNodeCount(); i++) {
        size += GetFlagCodingGroup()->Get(i)->GetRawStorageSize(subset_def);
    }
    for (int i = 0; i < GetMaskGroup()->GetNodeCount(); i++) {
        size += GetMaskGroup()->Get(i)->GetRawStorageSize(subset_def);
    }
    for (int i = 0; i < GetQuicklookGroup()->GetNodeCount(); i++) {
        size += GetQuicklookGroup()->Get(i)->GetRawStorageSize(subset_def);
    }
    size += GetMetadataRoot()->GetRawStorageSize(subset_def);
    return size;
}

void Product::AddTiePointGrid(std::shared_ptr<TiePointGrid> tie_point_grid) {
    if (ContainsRasterDataNode(tie_point_grid->GetName())) {
        throw std::invalid_argument("The Product '" + GetName() + "' already contains " +
                                    "a tie-point grid with the name '" + tie_point_grid->GetName() + "'.");
    }
    tie_point_grid_group_->Add(tie_point_grid);
}

bool Product::IsCompatibleProduct(const std::shared_ptr<Product>& product, float eps) {
    Guardian::AssertNotNull("product", product);
    if (SharedFromBase<Product>() == product) {
        return true;
    }

    if (GetSceneRasterWidth() != product->GetSceneRasterWidth()) {
        std::cerr << "raster width " << product->GetSceneRasterWidth() << " not equal to " << GetSceneRasterWidth()
                  << std::endl;
        return false;
    }
    if (GetSceneRasterHeight() != product->GetSceneRasterHeight()) {
        std::cerr << "raster height " << product->GetSceneRasterHeight() << " not equal to " << GetSceneRasterHeight()
                  << std::endl;
        return false;
    }
    if (GetSceneGeoCoding() == nullptr && product->GetSceneGeoCoding() != nullptr) {
        std::cerr << "no geocoding in master but in source" << std::endl;
        return false;
    }
    if (GetSceneGeoCoding() != nullptr) {
        if (product->GetSceneGeoCoding() == nullptr) {
            std::cerr << "no geocoding in source but in master" << std::endl;
            return false;
        }

        auto pixel_pos = std::make_shared<PixelPos>();
        auto geo_pos1 = std::make_shared<GeoPos>();
        auto geo_pos2 = std::make_shared<GeoPos>();

        pixel_pos->x_ = 0.5f;
        pixel_pos->y_ = 0.5f;
        GetSceneGeoCoding()->GetGeoPos(pixel_pos, geo_pos1);
        product->GetSceneGeoCoding()->GetGeoPos(pixel_pos, geo_pos2);
        if (!EqualsLatLon(geo_pos1, geo_pos2, eps)) {
            //            todo: probably needs override operator or toString method when run
            std::cerr << "first scan line left corner " << geo_pos2 << " not equal to " << geo_pos1 << std::endl;
            return false;
        }

        pixel_pos->x_ = GetSceneRasterWidth() - 1 + 0.5f;
        pixel_pos->y_ = 0.5f;
        GetSceneGeoCoding()->GetGeoPos(pixel_pos, geo_pos1);
        product->GetSceneGeoCoding()->GetGeoPos(pixel_pos, geo_pos2);
        if (!EqualsLatLon(geo_pos1, geo_pos2, eps)) {
            std::cerr << "first scan line right corner " << geo_pos2 << " not equal to " << geo_pos1 << std::endl;
            return false;
        }

        pixel_pos->x_ = 0.5f;
        pixel_pos->y_ = GetSceneRasterHeight() - 1 + 0.5f;
        GetSceneGeoCoding()->GetGeoPos(pixel_pos, geo_pos1);
        product->GetSceneGeoCoding()->GetGeoPos(pixel_pos, geo_pos2);
        if (!EqualsLatLon(geo_pos1, geo_pos2, eps)) {
            std::cerr << "last scan line left corner " << geo_pos2 << " not equal to " << geo_pos1 << std::endl;
            return false;
        }

        pixel_pos->x_ = GetSceneRasterWidth() - 1 + 0.5f;
        pixel_pos->y_ = GetSceneRasterHeight() - 1 + 0.5f;
        GetSceneGeoCoding()->GetGeoPos(pixel_pos, geo_pos1);
        product->GetSceneGeoCoding()->GetGeoPos(pixel_pos, geo_pos2);
        if (!EqualsLatLon(geo_pos1, geo_pos2, eps)) {
            std::cerr << "last scan line right corner " << geo_pos2 << " not equal to " << geo_pos1 << std::endl;
            return false;
        }
    }
    return true;
}
bool Product::EqualsLatLon(const std::shared_ptr<GeoPos>& pos1, const std::shared_ptr<GeoPos>& pos2, float eps) {
    return EqualsOrNaN(pos1->lat_, pos2->lat_, eps) && EqualsOrNaN(pos1->lon_, pos2->lon_, eps);
}

bool Product::EqualsOrNaN(double v1, double v2, float eps) {
    return MathUtils::EqualValues(v1, v2, eps) || (isnan(v1) && isnan(v2));
}
std::vector<std::shared_ptr<TiePointGrid>> Product::GetTiePointGrids() {
    auto tie_point_grids = std::vector<std::shared_ptr<TiePointGrid>>(GetNumTiePointGrids());
    for (std::size_t i = 0; i < tie_point_grids.size(); i++) {
        tie_point_grids.at(i) = GetTiePointGridAt(i);
    }
    return tie_point_grids;
}
std::shared_ptr<Product> Product::CreateProduct(std::string_view name, std::string_view type, int scene_raster_width,
                                                int scene_raster_height) {
    auto product = std::shared_ptr<Product>(new Product(name, type, scene_raster_width, scene_raster_height));
    return InitProductMembers(product);
}

std::shared_ptr<Product> Product::CreateProduct(std::string_view name, std::string_view type, int scene_raster_width,
                                                int scene_raster_height,
                                                const std::shared_ptr<IProductReader>& reader) {
    auto product = std::shared_ptr<Product>(new Product(name, type, scene_raster_width, scene_raster_height, reader));
    return InitProductMembers(product);
}

std::shared_ptr<Product> Product::CreateProduct(std::string_view name, std::string_view type) {
    auto product = std::shared_ptr<Product>(new Product(name, type));
    return InitProductMembers(product);
}
std::shared_ptr<Product> Product::CreateProduct(std::string_view name, std::string_view type,
                                                const std::shared_ptr<IProductReader>& reader) {
    auto product = std::shared_ptr<Product>(new Product(name, type, reader));
    return InitProductMembers(product);
}

std::shared_ptr<Product> Product::InitProductMembers(const std::shared_ptr<Product>& product) {
    //    TODO: THIS IS NOT CORRECT CONSTURCTOR!!! SWAP IT LATER
    product->metadata_root_->SetOwner(product->SharedFromBase<Product>());

    product->band_group_ =
        std::make_shared<ProductNodeGroup<std::shared_ptr<Band>>>(product->SharedFromBase<Product>(), "bands", true);
    product->tie_point_grid_group_ = std::make_shared<ProductNodeGroup<std::shared_ptr<TiePointGrid>>>(
        product->SharedFromBase<Product>(), "tie_point_grids", true);
    //    vector_data_group_ = new VectorDataNodeProductNodeGroup();
    product->index_coding_group_ = std::make_shared<ProductNodeGroup<std::shared_ptr<IndexCoding>>>(
        product->SharedFromBase<Product>(), "index_codings", true);
    product->flag_coding_group_ = std::make_shared<ProductNodeGroup<std::shared_ptr<FlagCoding>>>(
        product->SharedFromBase<Product>(), "flag_codings", true);
    product->mask_group_ =
        std::make_shared<ProductNodeGroup<std::shared_ptr<Mask>>>(product->SharedFromBase<Product>(), "masks", true);
    product->quicklook_group_ = std::make_shared<ProductNodeGroup<std::shared_ptr<Quicklook>>>(
        product->SharedFromBase<Product>(), "quicklooks", true);

    //    todo: implement if we use it
    //    pin_group_ = CreatePinGroup();
    //    gcp_group_ = CreateGcpGroup();

    product->groups_ = std::make_shared<ProductNodeGroup<std::shared_ptr<ProductNode>>>(
        product->SharedFromBase<Product>(), "groups", false);

    product->groups_->Add(product->band_group_);
    product->groups_->Add(product->quicklook_group_);
    product->groups_->Add(product->tie_point_grid_group_);
    //    groups_->Add(vector_data_group_);
    product->groups_->Add(product->index_coding_group_);
    product->groups_->Add(product->flag_coding_group_);
    product->groups_->Add(product->mask_group_);
    //    groups_->Add(pin_group_);
    //    groups_->Add(gcp_group_);

    product->SetModified(false);
    return product;
}

void Product::SetRefNo(int ref_no) {
    Guardian::AssertWithinRange("refNo", ref_no, 1, std::numeric_limits<int>::max());
    if (ref_no_ != 0 && ref_no_ != ref_no) {
        throw std::runtime_error("this.refNo != 0 && this.refNo != refNo");
    }
    ref_no_ = ref_no;
    ref_str_ = "[" + std::to_string(ref_no_) + "]";
}

std::shared_ptr<ProductNodeGroup<std::shared_ptr<ProductNode>>> Product::GetGroups() { return groups_; }

std::shared_ptr<ProductNode> Product::GetGroup(std::string_view name) { return groups_->Get(name); }

bool Product::IsUsingSingleGeoCoding() {
    std::shared_ptr<IGeoCoding> geo_coding = GetSceneGeoCoding();
    if (geo_coding == nullptr) {
        return false;
    }
    std::vector<std::shared_ptr<RasterDataNode>> raster_data_nodes = GetRasterDataNodes();
    for (const auto& raster_data_node : raster_data_nodes) {
        if (raster_data_node && geo_coding != raster_data_node->GetGeoCoding()) {
            return false;
        }
    }
    return true;
}
bool Product::ContainsPixel(double x, double y) {
    return x >= 0.0f && x <= GetSceneRasterWidth() && y >= 0.0f && y <= GetSceneRasterHeight();
}
bool Product::ContainsPixel(const std::shared_ptr<PixelPos>& pixel_pos) {
    return ContainsPixel(pixel_pos->x_, pixel_pos->y_);
}

void Product::CloseProductReader() {
    if (reader_) {
        reader_->Close();
        reader_ = nullptr;
    }
}

void Product::CloseProductWriter() {
    if (writer_) {
        //    todo: if we port writers we need modifications
        //        writer_->Flush();
        //        writer_->Close();
        writer_ = nullptr;
    }
}
void Product::CloseIO() {
    std::exception e_1;
    bool e_1_bool = false;
    try {
        CloseProductReader();
    } catch (const std::exception& e) {
        e_1 = e;
        e_1_bool = true;
    }
    std::exception e_O;
    bool e_0_bool = false;
    try {
        CloseProductWriter();
    } catch (const std::exception& e) {
        e_O = e;
        e_0_bool = true;
    }
    if (e_1_bool) {
        throw e_1;
    }
    if (e_0_bool) {
        throw e_O;
    }
}
void Product::Dispose() {
    try {
        CloseIO();
    } catch ([[maybe_unused]] const std::exception& ignore) {
        // ignore
    }

    reader_ = nullptr;
    writer_ = nullptr;

    metadata_root_->Dispose();
    band_group_->Dispose();
    tie_point_grid_group_->Dispose();
    flag_coding_group_->Dispose();
    index_coding_group_->Dispose();
    mask_group_->Dispose();
    quicklook_group_->Dispose();

    if (scene_geo_coding_) {
        scene_geo_coding_->Dispose();
        scene_geo_coding_ = nullptr;
    }
}

// void Product::CheckGeoCoding(const std::shared_ptr<GeoCoding>& geo_coding) {
//    if (std::dynamic_pointer_cast<TiePointGeoCoding>(geo_coding)) {
//        std::shared_ptr<TiePointGeoCoding> gc = std::dynamic_pointer_cast<TiePointGridCoding>(geo_coding);
////        todo: add check if needed
////        Guardian::AssertSame("gc.getLatGrid()", gc->GetLatGrid(), GetTiePointGrid(gc->GetLatGrid()->GetName()));
////        Guardian::AssertSame("gc.getLonGrid()", gc->GetLonGrid(), GetTiePointGrid(gc->GetLonGrid()->GetName()));
//    } else if (std::dynamic_pointer_cast<MapGeoCoding>(geo_coding)) {
//        std::shared_ptr<MapGeoCoding> gc = std::dynamic_pointer_cast<MapGeoCoding> geo_coding;
//        std::shared_ptr<MapInfo> map_info = gc->GetMapInfo();
//        Guardian::AssertNotNull("mapInfo", map_info);
//        Guardian::AssertEquals("mapInfo.getSceneWidth()", map_info->GetSceneWidth(), GetSceneRasterWidth());
//        Guardian::AssertEquals("mapInfo.getSceneHeight()", map_info->GetSceneHeight(), GetSceneRasterHeight());
//    }
//}

}  // namespace snapengine
}  // namespace alus