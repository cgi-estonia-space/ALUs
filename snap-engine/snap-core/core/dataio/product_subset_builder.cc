/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.dataio.ProductSubsetBuilder.java
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
#include "snap-core/core/dataio/product_subset_builder.h"

#include "alus_log.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/flag_coding.h"
#include "snap-core/core/datamodel/geo_pos.h"
#include "snap-core/core/datamodel/index_coding.h"
#include "snap-core/core/datamodel/metadata_attribute.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/datamodel/pixel_pos.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/datamodel/product_data.h"
#include "snap-core/core/datamodel/product_data_utc.h"
#include "snap-core/core/datamodel/product_node_group.h"
#include "snap-core/core/datamodel/raster_data_node.h"
#include "snap-core/core/datamodel/tie_point_grid.h"
#include "snap-core/core/datamodel/virtual_band.h"
#include "snap-core/core/util/product_utils.h"

namespace alus::snapengine {

ProductSubsetBuilder::ProductSubsetBuilder() : AbstractProductBuilder(false) {}

std::shared_ptr<Product> ProductSubsetBuilder::ReadProductNodesImpl() {
    std::any input = GetInput();
    try {
        source_product_ = std::any_cast<std::shared_ptr<Product>>(input);
        // Debug.assertNotNull(source_product_);
        scene_raster_width_ = source_product_->GetSceneRasterWidth();
        scene_raster_height_ = source_product_->GetSceneRasterHeight();
        if (GetSubsetDef() != nullptr) {
            std::shared_ptr<custom::Dimension> s =
                GetSubsetDef()->GetSceneRasterSize(scene_raster_width_, scene_raster_height_);
            scene_raster_width_ = s->width;
            scene_raster_height_ = s->height;
        }

        std::shared_ptr<Product> target_product = CreateProduct();
        UpdateMetadata(source_product_, target_product, GetSubsetDef());
        return target_product;
    } catch (const std::bad_any_cast& e) {
        LOGE << e.what();
        throw std::runtime_error(
            "ProductSubsetBuilder at ReadProductNodesImpl had an issue with input not being a Product class");
    }
}

void ProductSubsetBuilder::UpdateMetadata(std::shared_ptr<Product> source_product,
                                          std::shared_ptr<Product> target_product,
                                          std::shared_ptr<ProductSubsetDef> subset_def) {
    std::shared_ptr<MetadataElement> src_root = source_product->GetMetadataRoot();
    std::shared_ptr<MetadataElement> src_abs_root = src_root->GetElement("Abstracted_Metadata");
    if (src_abs_root == nullptr) {
        return;
    }

    std::shared_ptr<MetadataElement> trg_root = target_product->GetMetadataRoot();
    std::shared_ptr<MetadataElement> trg_abs_root = trg_root->GetElement("Abstracted_Metadata");
    if (trg_abs_root == nullptr) {
        trg_abs_root = std::make_shared<MetadataElement>("Abstracted_Metadata");
        trg_root->AddElement(trg_abs_root);
        ProductUtils::CopyMetadata(src_abs_root, trg_abs_root);
    }

    std::shared_ptr<custom::Rectangle> region = subset_def->GetRegion();
    std::shared_ptr<MetadataAttribute> height = trg_abs_root->GetAttribute("num_output_lines");
    if (height != nullptr) height->GetData()->SetElemUInt(target_product->GetSceneRasterHeight());

    std::shared_ptr<MetadataAttribute> width = trg_abs_root->GetAttribute("num_samples_per_line");
    if (width != nullptr) width->GetData()->SetElemUInt(target_product->GetSceneRasterWidth());

    std::shared_ptr<MetadataAttribute> offset_x = trg_abs_root->GetAttribute("subset_offset_x");
    if (offset_x != nullptr && region != nullptr) offset_x->GetData()->SetElemUInt(region->x);

    std::shared_ptr<MetadataAttribute> offset_y = trg_abs_root->GetAttribute("subset_offset_y");
    if (offset_y != nullptr && region != nullptr) offset_y->GetData()->SetElemUInt(region->y);

    bool is_sar_product = trg_abs_root->GetAttributeDouble("radar_frequency", 99999) != 99999;  // NOLINT
    if (!is_sar_product) return;

    // update subset metadata for SAR products

    bool near_range_on_left = IsNearRangeOnLeft(target_product);

    int source_image_height = source_product->GetSceneRasterHeight();
    double src_first_line_time = Utc::Parse(src_abs_root->GetAttributeString("first_line_time"))->GetMjd();  // in days
    double src_last_line_time = Utc::Parse(src_abs_root->GetAttributeString("last_line_time"))->GetMjd();    // in days
    double line_time_interval = (src_last_line_time - src_first_line_time) / (source_image_height - 1);      // in days
    if (region != nullptr) {
        int region_y = region->y;
        double region_height = region->height;
        double new_first_line_time = src_first_line_time + line_time_interval * region_y;
        double new_last_line_time = new_first_line_time + line_time_interval * (region_height - 1);
        std::shared_ptr<MetadataAttribute> first_line_time = trg_abs_root->GetAttribute("first_line_time");
        if (first_line_time != nullptr) {
            first_line_time->GetData()->SetElems((Utc(new_first_line_time)).GetArray());
        }
        std::shared_ptr<MetadataAttribute> last_line_time = trg_abs_root->GetAttribute("last_line_time");
        if (last_line_time != nullptr) {
            last_line_time->GetData()->SetElems((Utc(new_last_line_time)).GetArray());
        }
    }

    std::shared_ptr<MetadataAttribute> total_size = trg_abs_root->GetAttribute("total_size");
    if (total_size != nullptr)
        total_size->GetData()->SetElemUInt(static_cast<int>(target_product->GetRawStorageSize()));

    if (near_range_on_left) {
        SetLatLongMetadata(target_product, trg_abs_root, "first_near_lat", "first_near_long", 0.5F, 0.5F);
        SetLatLongMetadata(target_product, trg_abs_root, "first_far_lat", "first_far_long",
                           static_cast<float>(target_product->GetSceneRasterWidth()) - 1 + 0.5F, 0.5F);

        SetLatLongMetadata(target_product, trg_abs_root, "last_near_lat", "last_near_long", 0.5F,
                           static_cast<float>(target_product->GetSceneRasterHeight()) - 1 + 0.5F);
        SetLatLongMetadata(target_product, trg_abs_root, "last_far_lat", "last_far_long",
                           static_cast<float>(target_product->GetSceneRasterWidth()) - 1 + 0.5F,
                           static_cast<float>(target_product->GetSceneRasterHeight()) - 1 + 0.5F);
    } else {
        SetLatLongMetadata(target_product, trg_abs_root, "first_near_lat", "first_near_long",
                           static_cast<float>(target_product->GetSceneRasterWidth()) - 1 + 0.5F, 0.5F);
        SetLatLongMetadata(target_product, trg_abs_root, "first_far_lat", "first_far_long", 0.5F, 0.5F);

        SetLatLongMetadata(target_product, trg_abs_root, "last_near_lat", "last_near_long",
                           static_cast<float>(target_product->GetSceneRasterWidth()) - 1 + 0.5F,
                           static_cast<float>(target_product->GetSceneRasterHeight()) - 1 + 0.5F);
        SetLatLongMetadata(target_product, trg_abs_root, "last_far_lat", "last_far_long", 0.5F,
                           static_cast<float>(target_product->GetSceneRasterHeight()) - 1 + 0.5F);
    }

    std::shared_ptr<MetadataAttribute> slant_range = trg_abs_root->GetAttribute("slant_range_to_first_pixel");
    if (slant_range != nullptr) {
        std::shared_ptr<TiePointGrid> sr_tpg = target_product->GetTiePointGrid("slant_range_time");
        if (sr_tpg != nullptr && region != nullptr) {
            bool srgr_flag = src_abs_root->GetAttributeInt("srgr_flag") != 0;
            double slant_range_dist;
            if (srgr_flag) {
                double slant_range_time;
                if (near_range_on_left) {
                    slant_range_time =
                        sr_tpg->GetPixelDouble(region->x, region->y) / 1000000000.0;  // ns to s  // NOLINT
                } else {
                    slant_range_time = sr_tpg->GetPixelDouble(target_product->GetSceneRasterWidth() - region->x - 1,
                                                              region->y) /
                                       1000000000.0;  // ns to s  // NOLINT
                }
                double half_light_speed = 299792458.0 / 2.0;  // NOLINT
                slant_range_dist = slant_range_time * half_light_speed;
                slant_range->GetData()->SetElemDouble(slant_range_dist);
            } else {
                double slant_range_to_first_pixel = src_abs_root->GetAttributeDouble("slant_range_to_first_pixel");
                double range_spacing = src_abs_root->GetAttributeDouble("RANGE_SPACING", 0);
                if (near_range_on_left) {
                    slant_range_dist = slant_range_to_first_pixel + region->x * range_spacing;
                } else {
                    slant_range_dist = slant_range_to_first_pixel +
                                       (target_product->GetSceneRasterWidth() - region->x - 1) * range_spacing;
                }
                slant_range->GetData()->SetElemDouble(slant_range_dist);
            }
        }
    }

    SetSubsetSRGRCoefficients(source_product, target_product, subset_def, trg_abs_root, near_range_on_left);
}

bool ProductSubsetBuilder::IsNearRangeOnLeft(std::shared_ptr<Product>& product) {
    std::shared_ptr<TiePointGrid> incidence_angle = product->GetTiePointGrid("incident_angle");
    if (incidence_angle != nullptr) {
        double incidence_angle_to_first_pixel = incidence_angle->GetPixelDouble(0, 0);
        double incidence_angle_to_last_pixel = incidence_angle->GetPixelDouble(product->GetSceneRasterWidth() - 1, 0);
        return (incidence_angle_to_first_pixel < incidence_angle_to_last_pixel);
    }
    return true;
}

void ProductSubsetBuilder::SetSubsetSRGRCoefficients(std::shared_ptr<Product>& source_product,
                                                     std::shared_ptr<Product>& target_product,
                                                     std::shared_ptr<ProductSubsetDef>& subset_def,
                                                     std::shared_ptr<MetadataElement>& abs_root,
                                                     bool near_range_on_left) {
    std::shared_ptr<MetadataElement> srgr_coefficients_elem = abs_root->GetElement("SRGR_Coefficients");
    if (srgr_coefficients_elem != nullptr) {
        double range_spacing = abs_root->GetAttributeDouble("RANGE_SPACING", 0);
        double col_index = subset_def->GetRegion() == nullptr ? 0 : subset_def->GetRegion()->x;

        for (const auto& srgr_list : srgr_coefficients_elem->GetElements()) {
            double ground_range_origin = srgr_list->GetAttributeDouble("ground_range_origin", 0);
            double ground_range_origin_subset;
            if (near_range_on_left) {
                ground_range_origin_subset = ground_range_origin + col_index * range_spacing;
            } else {
                double col_index_from_right =
                    source_product->GetSceneRasterWidth() - col_index - target_product->GetSceneRasterWidth();
                ground_range_origin_subset = ground_range_origin + col_index_from_right * range_spacing;
            }
            srgr_list->SetAttributeDouble("ground_range_origin", ground_range_origin_subset);
        }
    }
}

void ProductSubsetBuilder::SetLatLongMetadata(std::shared_ptr<Product>& product,
                                              std::shared_ptr<MetadataElement>& abs_root, std::string_view tag_lat,
                                              std::string_view tag_lon, float x, float y) {
    std::shared_ptr<PixelPos> pixel_pos = std::make_shared<PixelPos>(x, y);
    std::shared_ptr<GeoPos> geo_pos = std::make_shared<GeoPos>();
    if (product->GetSceneGeoCoding() == nullptr) return;
    product->GetSceneGeoCoding()->GetGeoPos(pixel_pos, geo_pos);

    std::shared_ptr<MetadataAttribute> lat = abs_root->GetAttribute(tag_lat);
    if (lat != nullptr) lat->GetData()->SetElemDouble(geo_pos->GetLat());
    std::shared_ptr<MetadataAttribute> lon = abs_root->GetAttribute(tag_lon);
    if (lon != nullptr) lon->GetData()->SetElemDouble(geo_pos->GetLon());
}

std::shared_ptr<Product> ProductSubsetBuilder::CreateProduct() {
    std::shared_ptr<Product> source_product = GetSourceProduct();
    // Debug.assertNotNull(sourceProduct);
    // Debug.assertTrue(getSceneRasterWidth() > 0);
    // Debug.assertTrue(getSceneRasterHeight() > 0);
    std::string new_product_name;
    if (new_product_name_.empty()) {
        new_product_name = source_product->GetName();
    } else {
        new_product_name = new_product_name_;
    }

    std::shared_ptr<Product> product =
        Product::CreateProduct(new_product_name, source_product->GetProductType(), GetSceneRasterWidth(),
                               GetSceneRasterHeight(), shared_from_this());
    // product->SetPointingFactory(sourceProduct.getPointingFactory());
    if (new_product_desc_.empty()) {
        product->SetDescription(source_product->GetDescription());
    } else {
        product->SetDescription(std::string_view(new_product_desc_));
    }
    if (!IsMetadataIgnored()) {
        ProductUtils::CopyMetadata(source_product, product);
    }
    // List of everything missing that perhaps we should have.

    // AddTiePointGridsToProduct(product);
    AddBandsToProduct(product);
    // ProductUtils::CopyMasks(sourceProduct, product);
    // AddFlagCodingsToProduct(product);
    // AddGeoCodingToProduct(product);

    // only copy index codings associated with accepted nodes
    // copyAcceptedIndexCodings(product);

    // ProductUtils::CopyVectorData(sourceProduct, product);
    // ProductUtils::CopyOverlayMasks(sourceProduct, product);
    // ProductUtils::CopyPreferredTileSize(sourceProduct, product);
    // setSceneRasterStartAndStopTime(product);
    // addSubsetInfoMetadata(product);
    if (!source_product->GetQuicklookBandName().empty() && product->GetQuicklookBandName().empty() &&
        product->ContainsBand(source_product->GetQuicklookBandName())) {
        product->SetQuicklookBandName(source_product->GetQuicklookBandName());
    }
    // product->SetAutoGrouping(sourceProduct->GetAutoGrouping());

    return product;
}

void ProductSubsetBuilder::AddBandsToProduct(const std::shared_ptr<Product>& product) {
    // Debug.assertNotNull(this.getSourceProduct());
    // Debug.assertNotNull(product);

    for (int i = 0; i < GetSourceProduct()->GetNumBands(); ++i) {
        std::shared_ptr<Band> source_band = GetSourceProduct()->GetBandAt(i);
        std::string band_name = source_band->GetName();
        if (IsNodeAccepted(band_name)) {
            bool treat_virtual_bands_as_real_bands = false;
            if (GetSubsetDef() != nullptr && GetSubsetDef()->GetTreatVirtualBandsAsRealBands()) {
                treat_virtual_bands_as_real_bands = true;
            }

            std::shared_ptr<Band> dest_band;
            if (!treat_virtual_bands_as_real_bands && dynamic_cast<VirtualBand*>(source_band.get()) != nullptr) {
                std::shared_ptr<VirtualBand> virtual_source = std::dynamic_pointer_cast<VirtualBand>(source_band);
                if (GetSubsetDef() == nullptr) {
                    dest_band =
                        std::make_shared<VirtualBand>(band_name, source_band->GetDataType(), GetSceneRasterWidth(),
                                                      GetSceneRasterHeight(), virtual_source->GetExpression());
                } else {
                    std::shared_ptr<custom::Dimension> dim = GetSubsetDef()->GetSceneRasterSize(
                        source_band->GetRasterWidth(), source_band->GetRasterHeight(), source_band->GetName());
                    dest_band = std::make_shared<VirtualBand>(band_name, source_band->GetDataType(), dim->width,
                                                              dim->height, virtual_source->GetExpression());
                }
            } else if (GetSubsetDef() == nullptr) {
                dest_band = std::make_shared<Band>(band_name, source_band->GetDataType(), GetSceneRasterWidth(),
                                                   GetSceneRasterHeight());
            } else {
                std::shared_ptr<custom::Dimension> dim = GetSubsetDef()->GetSceneRasterSize(
                    source_band->GetRasterWidth(), source_band->GetRasterHeight(), source_band->GetName());
                dest_band = std::make_shared<Band>(band_name, source_band->GetDataType(), dim->width, dim->height);
            }

            if (source_band->GetUnit().has_value()) {
                dest_band->SetUnit(source_band->GetUnit());
            }

            if (source_band->GetDescription().has_value()) {
                dest_band->SetDescription(source_band->GetDescription());
            }

            dest_band->SetScalingFactor(source_band->GetScalingFactor());
            dest_band->SetScalingOffset(source_band->GetScalingOffset());
            dest_band->SetLog10Scaled(source_band->IsLog10Scaled());
            dest_band->SetSpectralBandIndex(source_band->GetSpectralBandIndex());
            dest_band->SetSpectralWavelength(source_band->GetSpectralWavelength());
            dest_band->SetSpectralBandwidth(source_band->GetSpectralBandwidth());
            dest_band->SetSolarFlux(source_band->GetSolarFlux());
            if (source_band->IsNoDataValueSet()) {
                dest_band->SetNoDataValue(source_band->GetNoDataValue());
            }

            dest_band->SetNoDataValueUsed(source_band->IsNoDataValueUsed());
            dest_band->SetValidPixelExpression(source_band->GetValidPixelExpression());
            std::shared_ptr<FlagCoding> source_flag_coding = source_band->GetFlagCoding();
            std::shared_ptr<IndexCoding> source_index_coding = source_band->GetIndexCoding();
            std::string index_coding_name;
            if (source_flag_coding != nullptr) {
                index_coding_name = source_flag_coding->GetName();
                std::shared_ptr<FlagCoding> dest_flag_coding =
                    std::dynamic_pointer_cast<FlagCoding>(product->GetFlagCodingGroup()->Get(index_coding_name));
                if (dest_flag_coding == nullptr) {
                    dest_flag_coding = ProductUtils::CopyFlagCoding(source_flag_coding, product);
                }

                dest_band->SetSampleCoding(dest_flag_coding);
            } else if (source_index_coding != nullptr) {
                index_coding_name = source_index_coding->GetName();
                std::shared_ptr<IndexCoding> dest_index_coding =
                    std::dynamic_pointer_cast<IndexCoding>(product->GetIndexCodingGroup()->Get(index_coding_name));
                if (dest_index_coding == nullptr) {
                    dest_index_coding = ProductUtils::CopyIndexCoding(source_index_coding, product);
                }

                dest_band->SetSampleCoding(dest_index_coding);
            } else {
                dest_band->SetSampleCoding(nullptr);
            }

            // TODO: what is even stx?  // NOLINT
            /*if (IsFullScene(GetSubsetDef(), sourceBand) && sourceBand->IsStxSet()) {
                CopyStx(sourceBand, std::dynamic_pointer_cast<RasterDataNode>(destBand));
            }*/

            product->AddBand(dest_band);
            band_map_.insert(std::pair<std::shared_ptr<Band>, std::shared_ptr<RasterDataNode>>(dest_band, source_band));
        }
    }

    // TODO: Image info? You'll know when you need it.  // NOLINT
    /*auto var11 = band_map_.begin();

    for(var11; var11 != band_map_.end(); ++var11) {
        CopyImageInfo((RasterDataNode)var11->second, (RasterDataNode)var11->first);
    }*/
}

/*void CopyImageInfo(RasterDataNode* sourceRaster, RasterDataNode* targetRaster) {
    ImageInfo imageInfo;
    if (sourceRaster->GetImageInfo() != nullptr) {
        imageInfo = sourceRaster.getImageInfo().createDeepCopy();
        targetRaster.setImageInfo(imageInfo);
    }
}*/

// TODO: Once you start implementing the raster data copy, you need all of the following and perhaps more.  // NOLINT
/*void ProductSubsetBuilder::ReadBandRasterDataImpl(int sourceOffsetX, int sourceOffsetY, int sourceWidth, int
sourceHeight, int sourceStepX, int sourceStepY, std::shared_ptr<Band> destBand, int destOffsetX, int destOffsetY, int
destWidth, int destHeight, const std::shared_ptr<ProductData>& dest_buffer, std::shared_ptr<ceres::IProgressMonitor>
pm){ auto sourceBand = band_map_.at(destBand); if (sourceBand->GetRasterData() != nullptr) { if
(sourceBand->GetRasterWidth() == destWidth && sourceBand->GetRasterHeight() == destHeight) {
            CopyBandRasterDataFully(sourceBand, dest_buffer, destWidth, destHeight);
        } else {
            CopyBandRasterDataSubSampling(sourceBand, sourceOffsetX, sourceOffsetY, sourceWidth, sourceHeight,
sourceStepX, sourceStepY, dest_buffer, destWidth);
        }
    } else if (sourceWidth == destWidth && sourceHeight == destHeight) {
        ReadBandRasterDataRegion(sourceBand, sourceOffsetX, sourceOffsetY, sourceWidth, sourceHeight, dest_buffer, pm);
    } else {
        Rectangle destRect = new Rectangle(destOffsetX, destOffsetY, destWidth, destHeight);
        ReadBandRasterDataSubsampled(sourceBand, dest_buffer, destRect, sourceOffsetX, sourceOffsetY, sourceWidth,
sourceHeight, sourceStepX, sourceStepY);
    }

}
//TODO: So from 1 buffer to another?  // NOLINT
void copyBandRasterDataFully(Band sourceBand, ProductData destBuffer, int destWidth, int destHeight) {
    copyData(sourceBand.getRasterData(),
             0,
             destBuffer,
             0,
             destWidth * destHeight);
}

void copyBandRasterDataSubSampling(Band sourceBand,
                                           int sourceOffsetX, int sourceOffsetY,
                                           int sourceWidth, int sourceHeight,
                                           int sourceStepX, int sourceStepY,
                                           ProductData destBuffer,
                                           int destWidth) {
    final int sourceMinY = sourceOffsetY;
    final int sourceMaxY = sourceOffsetY + sourceHeight - 1;
    int destPos = 0;
    for (int sourceY = sourceMinY; sourceY <= sourceMaxY; sourceY += sourceStepY) {
        // no subsampling in x-direction
        if (sourceStepX == 1) {
            copyData(sourceBand.getRasterData(),
                     sourceY * sourceBand.getRasterWidth() + sourceOffsetX,
                     destBuffer,
                     destPos,
                     destWidth);
        } else {
            copyLine(sourceBand.getRasterData(),
                     sourceY * sourceBand.getRasterWidth() + sourceOffsetX,
                     sourceWidth,
                     sourceStepX,
                     destBuffer,
                     destPos);
        }
        destPos += destWidth;
    }
}

void readBandRasterDataRegion(Band sourceBand,
                                      int sourceOffsetX, int sourceOffsetY,
                                      int sourceWidth, int sourceHeight,
                                      ProductData destBuffer,
                                      ProgressMonitor pm) throws IOException {
    sourceBand.readRasterData(sourceOffsetX,
                              sourceOffsetY,
                              sourceWidth,
                              sourceHeight,
                              destBuffer, pm);
}

private static void readBandRasterDataSubsampled(Band band, ProductData destData, Rectangle destRect, int sourceOffsetX,
int sourceOffsetY, int sourceWidth, int sourceHeight, int sourceStepX, int sourceStepY) throws IOException {

    Point[] tileIndices = band.getSourceImage().getTileIndices(new Rectangle(sourceOffsetX, sourceOffsetY, sourceWidth,
sourceHeight)); HashMap<Rectangle, ProductData> tileMap = new HashMap<>(); for (Point tileIndex : tileIndices) {
        Rectangle tileRect = band.getSourceImage().getTileRect(tileIndex.x, tileIndex.y);
        if (tileRect.isEmpty()) {
            continue;
        }
        final ProductData tileData = ProductData.createInstance(band.getDataType(), tileRect.width * tileRect.height);
        band.readRasterData(tileRect.x, tileRect.y, tileRect.width, tileRect.height, tileData, ProgressMonitor.NULL);
        tileMap.put(tileRect, tileData);
    }

    for (int y = 0; y < destRect.height; y++) {
        final int currentSrcYOffset = sourceOffsetY + y * sourceStepY;
        int currentDestYOffset = y * destRect.width;
        for (int x = 0; x < destRect.width; x++) {
            double value = getSourceValue(band, tileMap, sourceOffsetX + x * sourceStepX, currentSrcYOffset);
            destData.setElemDoubleAt(currentDestYOffset + x, value);
        }

    }
}*/

}  // namespace alus::snapengine