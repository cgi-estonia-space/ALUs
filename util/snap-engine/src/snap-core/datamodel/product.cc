#include "product.h"

#include "../../../../../algs/coherence/include/i_data_tile_reader.h"
#include "flag_coding.h"
#include "guardian.h"
#include "i_meta_data_reader.h"
#include "i_meta_data_writer.h"
#include "index_coding.h"
#include "metadata_element.h"
#include "product_node_group.h"
#include "virtual_band.h"

namespace alus {
namespace snapengine {

Product::Product(std::string_view name, std::string_view type, const std::shared_ptr<IDataTileReader>& reader,
                 const std::shared_ptr<IMetaDataReader>& metadata_reader)
    : Product(name, type, nullptr, reader, metadata_reader) {}
Product::Product(std::string_view name, std::string_view type) : Product(name, type, nullptr, nullptr) {}
Product::Product(std::string_view name, std::string_view type, int scene_raster_width, int scene_raster_height,
                 const std::shared_ptr<IDataTileReader>& reader,
                 const std::shared_ptr<IMetaDataReader>& metadata_reader)
    : Product(name, type, std::make_shared<Dimension>(scene_raster_width, scene_raster_height), reader,
              metadata_reader) {}
Product::Product(std::string_view name, std::string_view type, int scene_raster_width, int scene_raster_height)
    : Product(name, type, scene_raster_width, scene_raster_height, nullptr, nullptr) {}
Product::Product(std::string_view name, std::string_view type, const std::shared_ptr<Dimension>& scene_raster_size,
                 const std::shared_ptr<IDataTileReader>& reader,
                 const std::shared_ptr<IMetaDataReader>& metadata_reader)
    : ProductNode(name) {
    Guardian::AssertNotNullOrEmpty("type", type);
    product_type_ = type;
    scene_raster_size_ = scene_raster_size;
    reader_ = reader;
    // todo:check if this constructor works like expected
    if (metadata_reader) {
        metadata_reader_ = metadata_reader;
        metadata_reader_->SetProduct(SharedFromBase<Product>());
    }
    // can't set it in constructor because reader also needs to know product itself at construction time ? probably
    // setter would do it and use set here
    //    metadata_reader_ = metadata_reader;

    metadata_root_ = std::make_shared<MetadataElement>(METADATA_ROOT_NAME);
    // todo:check if this works like expected
    //  metadata_root_->SetOwner(SharedFromBase<Product>());
    // todo:this need a lot of more work just wanted to provide a proof of concept example
    //    band_group_ = std::make_shared<ProductNodeGroup<std::shared_ptr<Band>>>(SharedFromBase<Product>(), "bands",
    //    true);
    // groups_ = std::make_shared<ProductNodeGroup<std::shared_ptr<ProductNode>>>(SharedFromBase<Product>(), "groups",
    // false);
    SetModified(false);
}
void Product::SetModified(bool modified) {
    bool old_state = IsModified();
    if (old_state != modified) {
        // this should come from product node SetModified(modified);
        if (!modified) {
            ProductNode::SetModified(false);
            tie_point_grid_group_->SetModified(false);
            //                maskGroup.setModified(false);
            //                quicklookGroup.setModified(false);
            //                vectorDataGroup.setModified(false);
            flag_coding_group_->SetModified(false);
            index_coding_group_->SetModified(false);
            //                getMetadataRoot().setModified(false);
        }
    }
}

// std::vector<std::shared_ptr<Band>> Product::GetBands() {
//    return band_group_->ToArray(std::vector<std::shared_ptr<Band>>(GetNumBands()));
//}
// std::shared_ptr<Band> Product::GetBand(std::string_view name) {
//    Guardian::AssertNotNullOrEmpty("name", name);
//    return band_group_->Get(name);
//}
// int Product::GetBandIndex(std::string_view name) {
//    Guardian::AssertNotNullOrEmpty("name", name);
//    return band_group_->IndexOf(name);
//}
// bool Product::ContainsBand(std::string_view name) {
//    Guardian::AssertNotNullOrEmpty("name", name);
//    return band_group_->Contains(name);
//}

// std::shared_ptr<Band> Product::AddBand(std::string_view band_name, std::string_view expression, int data_type) {
//    std::shared_ptr<Band> band =
//        std::make_shared<VirtualBand>(band_name, data_type, GetSceneRasterWidth(), GetSceneRasterHeight(),
//        expression);
//    AddBand(band);
//    return band;
//}

// std::shared_ptr<Band> Product::AddBand(std::string_view band_name, std::string_view expression) {
//    return AddBand(band_name, expression, ProductData::TYPE_FLOAT32);
//}

// std::shared_ptr<Band> Product::AddBand(std::string_view band_name, int data_type) {
//    std::shared_ptr<Band> band =
//        std::make_shared<Band>(band_name, data_type, GetSceneRasterWidth(), GetSceneRasterHeight());
//    AddBand(band);
//    return band;
//}
// void Product::AddBand(std::shared_ptr<Band> band) {
//    //    Assert::NotNull(band, "band");
//    //    Assert::Argument(!ContainsRasterDataNode(band->GetName()), "The Product '" + GetName() + "' already contains
//    "
//    //    + "a band with the name '" + band->GetName() + "'.");
//    band_group_->Add(band);
//}
// bool Product::ContainsRasterDataNode(std::string_view name) {
//    return ContainsBand(name) /*|| ContainsTiePointGrid(name)|| GetMaskGroup()->Contains(name)*/;
//}
// std::shared_ptr<RasterDataNode> Product::GetRasterDataNode(std::string_view name) {
//    std::shared_ptr<RasterDataNode> raster_data_node = GetBand(name);
//    if (raster_data_node != nullptr) {
//        return raster_data_node;
//    }
//    raster_data_node = GetTiePointGrid(name);
//    if (raster_data_node != nullptr) {
//        return raster_data_node;
//    }
////    return GetMaskGroup()->Get(name);
//}
std::vector<std::shared_ptr<RasterDataNode>> Product::GetRasterDataNodes() {
    // 32 came from original java version :)
    std::vector<std::shared_ptr<RasterDataNode>> raster_data_nodes(32);
    auto band_group = GetBandGroup();
    for (int i = 0; i < band_group->GetNodeCount(); i++) {
        raster_data_nodes.push_back(band_group->Get(i));
    }
    //    auto mask_group = GetMaskGroup();
    //    for (int i = 0; i < mask_group->GetNodeCount(); i++) {
    //        raster_data_nodes.push_back(mask_group->Get(i));
    //    }
    auto tpg_group = GetTiePointGridGroup();
    for (int i = 0; i < tpg_group->GetNodeCount(); i++) {
        raster_data_nodes.push_back(tpg_group->Get(i));
    }
    return raster_data_nodes;
}
std::shared_ptr<Dimension> Product::GetSceneRasterSize() {
    if (scene_raster_size_ != nullptr) {
        return scene_raster_size_;
    }
    //        todo: add if we need this
    //        if (!InitSceneProperties()) {
    //            throw std::runtime_error("scene raster size not set and no reference band found to derive it
    //            from");
    //        }
    return scene_raster_size_;
}

int Product::GetSceneRasterWidth() { return GetSceneRasterSize()->width; }
int Product::GetSceneRasterHeight() { return GetSceneRasterSize()->height; }

const std::shared_ptr<IDataTileReader>& Product::GetReader() const { return reader_; }
void Product::SetReader(const std::shared_ptr<IDataTileReader>& reader) { reader_ = reader; }

const std::shared_ptr<IDataTileWriter>& Product::GetWriter() const { return writer_; }
void Product::SetWriter(const std::shared_ptr<IDataTileWriter>& writer) { writer_ = writer; }

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

}  // namespace snapengine
}  // namespace alus