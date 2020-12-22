#include "product_utils.h"

#include <type_traits>

#include "band.h"
#include "ceres_assert.h"
#include "guardian.h"
#include "index_coding.h"
#include "metadata_element.h"
#include "product.h"
#include "product_node_group.h"

namespace alus {
namespace snapengine {

void ProductUtils::CopyProductNodes(const std::shared_ptr<Product>& source_product,
                                    const std::shared_ptr<Product>& target_product) {
    ProductUtils::CopyMetadata(source_product, target_product);
    //    ProductUtils::CopyTiePointGrids(source_product, target_product);
    //    ProductUtils::CopyFlagCodings(source_product, target_product);
    //    ProductUtils::CopyFlagBands(source_product, target_product, true);
    //    ProductUtils::CopyGeoCoding(source_product, target_product);
    //    ProductUtils::CopyMasks(source_product, target_product);
    //    ProductUtils::CopyVectorData(source_product, target_product);
    //    ProductUtils::CopyIndexCodings(source_product, target_product);
    //    ProductUtils::CopyQuicklookBandName(source_product, target_product);
    target_product->SetStartTime(source_product->GetStartTime());
    target_product->SetEndTime(source_product->GetEndTime());
    target_product->SetDescription(source_product->GetDescription());
    // todo: add autogrouping if need arises
    //    target_product->SetAutoGrouping(source_product->GetAutoGrouping());
}
void ProductUtils::CopyMetadata(const std::shared_ptr<Product>& source, const std::shared_ptr<Product>& target) {
    //    todo:add assert
    Assert::NotNull(source, "source");
    Assert::NotNull(target, "target");
    CopyMetadata(source->GetMetadataRoot(), target->GetMetadataRoot());
}

void ProductUtils::CopyMetadata(const std::shared_ptr<MetadataElement>& source,
                                const std::shared_ptr<MetadataElement>& target) {
    Assert::NotNull(source, "source");
    Assert::NotNull(target, "target");
    for (const std::shared_ptr<MetadataElement>& element : source->GetElements()) {
        target->AddElement(element->CreateDeepClone());
    }
    for (const std::shared_ptr<MetadataAttribute>& attribute : source->GetAttributes()) {
        target->AddAttribute(attribute->CreateDeepClone());
    }
}

// void ProductUtils::CopyTiePointGrids(std::shared_ptr<Product> source_product, std::shared_ptr<Product>
// target_product) {
//    for (int i = 0; i < source_product->GetNumTiePointGrids(); i++) {
//        std::shared_ptr<TiePointGrid> src_t_p_g = source_product->GetTiePointGridAt(i);
//        if (!target_product->ContainsRasterDataNode(src_t_p_g->GetName())) {
//            target_product->AddTiePointGrid(src_t_p_g->CloneTiePointGrid());
//        }
//    }
//}
void ProductUtils::CopyFlagCodings(const std::shared_ptr<Product>& source, const std::shared_ptr<Product>& target) {
    Guardian::AssertNotNull("source", source);
    Guardian::AssertNotNull("target", target);

    int num_codings = source->GetFlagCodingGroup()->GetNodeCount();
    for (int n = 0; n < num_codings; n++) {
        std::shared_ptr<FlagCoding> source_flag_coding = source->GetFlagCodingGroup()->Get(n);
        CopyFlagCoding(source_flag_coding, target);
    }
}
std::shared_ptr<FlagCoding> ProductUtils::CopyFlagCoding(const std::shared_ptr<FlagCoding>& source_flag_coding,
                                                         const std::shared_ptr<Product>& target) {
    std::shared_ptr<FlagCoding> flag_coding = target->GetFlagCodingGroup()->Get(source_flag_coding->GetName());
    if (flag_coding == nullptr) {
        flag_coding = std::make_shared<FlagCoding>(source_flag_coding->GetName());
        flag_coding->SetDescription(source_flag_coding->GetDescription());
        target->GetFlagCodingGroup()->Add(flag_coding);
        CopyMetadata(source_flag_coding, flag_coding);
    }
    return flag_coding;
}
// void ProductUtils::CopyFlagBands(std::shared_ptr<Product> source_product,
//                                 std::shared_ptr<Product> target_product,
//                                 bool copy_source_image) {
//    Guardian::AssertNotNull("source", source_product);
//    Guardian::AssertNotNull("target", target_product);
//    if (source_product->GetFlagCodingGroup()->GetNodeCount() > 0) {
//        // loop over bands and check if they have a flags coding attached
//        for (int i = 0; i < source_product->GetNumBands(); i++) {
//            std::shared_ptr<Band> source_band = source_product->GetBandAt(i);
//            std::string band_name = source_band->GetName();
//            if (source_band->IsFlagBand() && target_product->GetBand(band_name) == nullptr) {
//                CopyBand(band_name, source_product, target_product, copy_source_image);
//            }
//        }
//
//        // first the bands have to be copied and then the masks
//        // other wise the referenced bands, e.g. flag band, is not contained in the target product
//        // and the mask is not copied
//        CopyMasks(source_product, target_product);
//        CopyOverlayMasks(source_product, target_product);
//    }
//}

// void ProductUtils::CopyMasks(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product) {
//    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> source_mask_group = source_product->GetMaskGroup();
//    for (int i = 0; i < source_mask_group->GetNodeCount(); i++) {
//        std::shared_ptr<Mask> mask = source_mask_group->Get(i);
//        if (!target_product->GetMaskGroup()->Contains(mask->GetName()) &&
//            mask->GetImageType()->CanTransferMask(mask, target_product)) {
//            mask->GetImageType()->TransferMask(mask, target_product);
//        }
//    }
//}

// std::shared_ptr<Band> ProductUtils::CopyBand(std::string_view source_band_name,
//                                             std::shared_ptr<Product> source_product,
//                                             std::string_view target_band_name,
//                                             std::shared_ptr<Product> target_product,
//                                             bool copy_source_image) {
//    Guardian::AssertNotNull(std::string_view("sourceProduct"), source_product);
//    Guardian::AssertNotNull("targetProduct", target_product);
//
//    if (source_band_name == nullptr || source_band_name.length() == 0) {
//        return nullptr;
//    }
//    std::shared_ptr<Band> source_band = source_product->GetBand(source_band_name);
//    if (source_band == nullptr) {
//        return nullptr;
//    }
//    std::shared_ptr<Band> target_band = std::make_shared<Band>(
//        target_band_name, source_band->GetDataType(), source_band->GetRasterWidth(), source_band->GetRasterHeight());
//    target_product->AddBand(target_band);
//    CopyRasterDataNodeProperties(source_band, target_band);
//    if (copy_source_image) {
//        target_band->SetSourceImage(source_band->GetSourceImage());
//    }
//    return target_band;
//}

// void ProductUtils::CopyRasterDataNodeProperties(std::shared_ptr<RasterDataNode> source_raster,
//                                                std::shared_ptr<RasterDataNode> target_raster) {
//    target_raster->SetDescription(source_raster->GetDescription());
//    target_raster->SetUnit(source_raster->GetUnit());
//    target_raster->SetScalingFactor(source_raster->GetScalingFactor());
//    target_raster->SetScalingOffset(source_raster->GetScalingOffset());
//    target_raster->SetLog10Scaled(source_raster->IsLog10Scaled());
//    target_raster->SetNoDataValueUsed(source_raster->IsNoDataValueUsed());
//    target_raster->SetNoDataValue(source_raster->GetNoDataValue());
//    target_raster->SetValidPixelExpression(source_raster->GetValidPixelExpression());
//
//    std::shared_ptr<Band> source_band = std::dynamic_pointer_cast<Band>(source_raster);
//    std::shared_ptr<Band> target_band = std::dynamic_pointer_cast<Band>(target_raster);
//    // todo: not sure if these are nullptr or we need to check if these are empty (needs check!)
//    // java code:    if (sourceRaster instanceof Band && targetRaster instanceof Band) {
//    if (source_band && target_band) {
//        CopySpectralBandProperties(source_band, target_band);
//        std::shared_ptr<Product> target_product = target_band->GetProduct();
//        if (target_product == nullptr) {
//            return;
//        }
//        if (source_band->GetFlagCoding() != nullptr) {
//            std::shared_ptr<FlagCoding> src_flag_coding = source_band->GetFlagCoding();
//            CopyFlagCoding(src_flag_coding, target_product);
//            target_band->SetSampleCoding(target_product->GetFlagCodingGroup()->Get(src_flag_coding->GetName()));
//        }
//        if (source_band->GetIndexCoding() != nullptr) {
//            std::shared_ptr<IndexCoding> src_index_coding = source_band->GetIndexCoding();
//            CopyIndexCoding(src_index_coding, target_product);
//            target_band->SetSampleCoding(target_product->GetIndexCodingGroup()->Get(src_index_coding->GetName()));
//
////            std::shared_ptr<ImageInfo> image_info = source_band->GetImageInfo();
////            if (image_info != nullptr) {
////                target_band->SetImageInfo(image_info->Clone());
////            }
//        }
//    }
//}

///**
// * Copies the spectral properties from source band to target band. These properties are:
// * <ul>
// * <li>{@link Band#getSpectralBandIndex() spectral band index},</li>
// * <li>{@link Band#getSpectralWavelength() the central wavelength},</li>
// * <li>{@link Band#getSpectralBandwidth() the spectral bandwidth} and</li>
// * <li>{@link Band#getSolarFlux() the solar spectral flux}.</li>
// * </ul>
// *
// * @param sourceBand the source band
// * @param targetBand the target band
// * @see #copyRasterDataNodeProperties(RasterDataNode, RasterDataNode)
// */
// void ProductUtils::CopySpectralBandProperties(std::shared_ptr<Band> source_band, std::shared_ptr<Band> target_band) {
//    //    todo::support asserts
//        Guardian::AssertNotNull("source", source_band);
//        Guardian::AssertNotNull("target", target_band);
//
//    target_band->SetSpectralBandIndex(source_band->GetSpectralBandIndex());
//    target_band->SetSpectralWavelength(source_band->GetSpectralWavelength());
//    target_band->SetSpectralBandwidth(source_band->GetSpectralBandwidth());
//    target_band->SetSolarFlux(source_band->GetSolarFlux());
//}

// std::shared_ptr<Band> ProductUtils::CopyBand(std::string_view source_band_name,
//                                             std::shared_ptr<Product> source_product,
//                                             std::shared_ptr<Product> target_product,
//                                             bool copy_source_image) {
//    return CopyBand(source_band_name, source_product, source_band_name, target_product, copy_source_image);
//}

// void ProductUtils::CopyOverlayMasks(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product)
// {
//    for (std::shared_ptr<RasterDataNode> source_node : source_product->GetTiePointGrids()) {
//        CopyOverlayMasks(source_node, target_product);
//    }
//    for (std::shared_ptr<RasterDataNode> source_node : source_product->GetBands()) {
//        CopyOverlayMasks(source_node, target_product);
//    }
//}
// void ProductUtils::CopyOverlayMasks(std::shared_ptr<RasterDataNode> source_node,
//                                    std::shared_ptr<Product> target_product) {
//    std::vector<std::string> mask_names = source_node->GetOverlayMaskGroup()->GetNodeNames();
//    std::shared_ptr<RasterDataNode> target_node = target_product->GetRasterDataNode(source_node->GetName());
//    if (target_node != nullptr) {
//        std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> overlay_mask_group =
//            target_node->GetOverlayMaskGroup();
//        std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> mask_group = target_product->GetMaskGroup();
//        AddMasksToGroup(mask_names, mask_group, overlay_mask_group);
//    }
//}
// void ProductUtils::AddMasksToGroup(std::vector<std::string> mask_names,
//                                   std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> mask_group,
//                                   std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> special_mask_group) {
//    for (std::string mask_name : mask_names) {
//        std::shared_ptr<Mask> mask = mask_group->Get(mask_name);
//        if (mask != nullptr) {
//            special_mask_group->Add(mask);
//        }
//    }
//}

// void ProductUtils::CopyGeoCoding(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product) {
////    Guardian::AssertNotNull("sourceProduct", source_product);
////    Guardian::AssertNotNull("targetProduct", target_product);
////    source_product->TransferGeoCodingTo(target_product, nullptr);
//}

// void ProductUtils::CopyVectorData(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product) {
//    std::shared_ptr<ProductNodeGroup<std::shared_ptr<VectorDataNode>>> vector_data_group =
//        source_product->GetVectorDataGroup();
//
//    // Note that since BEAM 4.10, we always have 2 permanent VDNs: pins and ground_control_points
//    bool all_permanent_and_empty = true;
//    for (int i = 0; i < vector_data_group->GetNodeCount(); i++) {
//        std::shared_ptr<VectorDataNode> source_v_d_n = vector_data_group->Get(i);
//        if (!source_v_d_n->IsPermanent() || !source_v_d_n->GetFeatureCollection()->IsEmpty()) {
//            all_permanent_and_empty = false;
//            break;
//        }
//    }
//    if (all_permanent_and_empty) {
//        return;
//    }
//
//    if (source_product->IsCompatibleProduct(target_product, 1.0e-3f)) {
//        for (int i = 0; i < vector_data_group->GetNodeCount(); i++) {
//            std::shared_ptr<VectorDataNode> source_v_d_n = vector_data_group->Get(i);
//            std::string name = source_v_d_n->GetName();
//
//            std::shared_ptr<FeatureCollection<std::shared_ptr<SimpleFeatureType>, std::shared_ptr<SimpleFeature>>>
//                feature_collection = source_v_d_n->GetFeatureCollection();
//            feature_collection = std::make_shared<DefaultFeatureCollection>(feature_collection);
//            if (!target_product->GetVectorDataGroup()->Contains(name)) {
//                target_product->GetVectorDataGroup()->Add(
//                    std::make_shared<VectorDataNode>(name, feature_collection->GetSchema()));
//            }
//            std::shared_ptr<VectorDataNode> target_v_d_n = target_product->GetVectorDataGroup()->Get(name);
//            target_v_d_n->GetFeatureCollection()->AddAll(feature_collection);
//            target_v_d_n->SetDefaultStyleCss(source_v_d_n->GetDefaultStyleCss());
//            target_v_d_n->SetDescription(source_v_d_n->GetDescription());
//        }
//    } else {
//        if (source_product->GetSceneGeoCoding() == nullptr || target_product->GetSceneGeoCoding() == nullptr) {
//            return;
//        }
//        std::shared_ptr<Geometry> clip_geometry;
//        try {
//            std::shared_ptr<Geometry> source_geometry_wgs84 = FeatureUtils::CreateGeoBoundaryPolygon(source_product);
//            std::shared_ptr<Geometry> target_geometry_wgs84 = FeatureUtils::CreateGeoBoundaryPolygon(target_product);
//            if (!source_geometry_wgs84->Intersects(target_geometry_wgs84)) {
//                return;
//            }
//            clip_geometry = source_geometry_wgs84->Intersection(target_geometry_wgs84);
//        } catch (std::exception& e) {
//            return;
//        }
//
//        std::shared_ptr<CoordinateReferenceSystem> src_model_crs = source_product->GetSceneCRS();
//        std::shared_ptr<CoordinateReferenceSystem> target_model_crs = target_product->GetSceneCRS();
//
//        for (int i = 0; i < vector_data_group->GetNodeCount(); i++) {
//            std::shared_ptr<VectorDataNode> source_v_d_n = vector_data_group->Get(i);
//            std::string name = source_v_d_n->GetName();
//            std::shared_ptr<FeatureCollection<std::shared_ptr<SimpleFeatureType>, std::shared_ptr<SimpleFeature>>>
//                feature_collection = source_v_d_n->GetFeatureCollection();
//            feature_collection = FeatureUtils::ClipCollection(feature_collection,
//                                                            src_model_crs,
//                                                            clip_geometry,
//                                                            DefaultGeographicCRS::WGS84,
//                                                            nullptr,
//                                                            target_model_crs
//                /*  ,
//                ProgressMonitor.NULL*/);
//            if (!target_product->GetVectorDataGroup()->Contains(name)) {
//                target_product->GetVectorDataGroup()->Add(
//                    std::make_shared<VectorDataNode>(name, feature_collection->GetSchema()));
//            }
//            std::shared_ptr<VectorDataNode> target_v_d_n = target_product->GetVectorDataGroup()->Get(name);
//            target_v_d_n->GetPlacemarkGroup();
//            target_v_d_n->GetFeatureCollection()->AddAll(feature_collection);
//            target_v_d_n->SetDefaultStyleCss(source_v_d_n->GetDefaultStyleCss());
//            target_v_d_n->SetDescription(source_v_d_n->GetDescription());
//        }
//    }
//}
void ProductUtils::CopyIndexCodings(const std::shared_ptr<Product>& source, const std::shared_ptr<Product>& target) {
    Guardian::AssertNotNull("source", source);
    Guardian::AssertNotNull("target", target);

    int num_codings = source->GetIndexCodingGroup()->GetNodeCount();
    for (int n = 0; n < num_codings; n++) {
        std::shared_ptr<IndexCoding> source_flag_coding = source->GetIndexCodingGroup()->Get(n);
        CopyIndexCoding(source_flag_coding, target);
    }
}

std::shared_ptr<IndexCoding> ProductUtils::CopyIndexCoding(const std::shared_ptr<IndexCoding>& source_index_coding,
                                                           const std::shared_ptr<Product>& target) {
    std::shared_ptr<IndexCoding> index_coding = target->GetIndexCodingGroup()->Get(source_index_coding->GetName());
    if (index_coding == nullptr) {
        index_coding = std::make_shared<IndexCoding>(source_index_coding->GetName());
        index_coding->SetDescription(source_index_coding->GetDescription());
        target->GetIndexCodingGroup()->Add(index_coding);
        CopyMetadata(source_index_coding, index_coding);
    }
    return index_coding;
}
// void ProductUtils::CopyQuicklookBandName(std::shared_ptr<Product> source, std::shared_ptr<Product> target) {
//    Guardian::AssertNotNull("source", source);
//    Guardian::AssertNotNull("target", target);
//    if (target->GetQuicklookBandName().empty() && !source->GetQuicklookBandName().empty()) {
//        if (target->GetBand(source->GetQuicklookBandName()) != nullptr) {
//            target->SetQuicklookBandName(source->GetQuicklookBandName());
//        }
//    }
//}

}  // namespace snapengine
}  // namespace alus