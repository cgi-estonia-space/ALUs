/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.util.ProductUtils.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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
#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "geo_pos.h"

namespace alus {
namespace snapengine {

class MetadataElement;
class Product;
class Band;
class RasterDataNode;
class IndexCoding;
class FlagCoding;
/**
 * This class provides many static factory methods to be used in conjunction with data products.
 *
 * @see Product
 */
class ProductUtils {
private:
    //    static void CopyOverlayMasks(std::shared_ptr<RasterDataNode> source_node, std::shared_ptr<Product>
    //    target_product);

    //    static void AddMasksToGroup(std::vector<std::string> mask_names,
    //                                std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> mask_group,
    //                                std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> special_mask_group);
    /**
     * Copies all properties, except bands, from source product to the target product. only those bands are copied which
     * are used by copied properties. For example latitude and longitude bands of a pixel-based geo-coding.
     *
     * @param sourceProduct the source product
     * @param targetProduct the target product
     */
public:
    static void CopyProductNodes(const std::shared_ptr<Product>& source_product,
                                 const std::shared_ptr<Product>& target_product);

    /**
     * Copies all metadata elements and attributes of the source product to the target product.
     * The copied elements and attributes are deeply cloned.
     *
     * @param source the source product.
     * @param target the target product.
     * @throws NullPointerException if the source or the target product is {@code null}.
     */
    static void CopyMetadata(const std::shared_ptr<Product>& source, const std::shared_ptr<Product>& target);

    /**
     * Copies all metadata elements and attributes of the source element to the target element.
     * The copied elements and attributes are deeply cloned.
     *
     * @param source the source element.
     * @param target the target element.
     * @throws NullPointerException if the source or the target element is {@code null}.
     */
    static void CopyMetadata(const std::shared_ptr<MetadataElement>& source,
                             const std::shared_ptr<MetadataElement>& target);

    /**
     * Copies all tie point grids from one product to another.
     *
     * @param sourceProduct the source product
     * @param targetProduct the target product
     */
    static void CopyTiePointGrids(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product);

    /**
     * Copies the flag codings from the source product to the target.
     *
     * @param source the source product
     * @param target the target product
     */
    static void CopyFlagCodings(const std::shared_ptr<Product>& source, const std::shared_ptr<Product>& target);

    /**
     * Copies the given source flag coding to the target product.
     * If it exists already, the method simply returns the existing instance.
     *
     * @param sourceFlagCoding the source flag coding
     * @param target           the target product
     * @return The flag coding.
     */
    static std::shared_ptr<FlagCoding> CopyFlagCoding(const std::shared_ptr<FlagCoding>& source_flag_coding,
                                                      const std::shared_ptr<Product>& target);

    /**
     * Copies all bands which contain a flag-coding from the source product to the target product.
     *
     * @param sourceProduct   the source product
     * @param targetProduct   the target product
     * @param copySourceImage whether the source image of the source band should be copied.
     * @since BEAM 4.10
     */
    static void CopyFlagBands(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product,
                              bool copy_source_image);

    /**
     * Copies the {@link Mask}s from the source product to the target product.
     * <p>
     * The method does not copy any image geo-coding/geometry information.
     * Use the {@link #copyImageGeometry(RasterDataNode, RasterDataNode, boolean)} to do so.
     * <p>
     * IMPORTANT NOTE: This method should only be used, if it is known that all masks
     * in the source product will also be valid in the target product. This method does
     * <em>not</em> copy overlay masks from the source bands to the target bands. Also
     * note that a source mask is not copied to the target product, when there already
     * is a mask in the target product with the same name as the source mask.
     *
     * @param sourceProduct the source product
     * @param targetProduct the target product
     */
    static void CopyMasks(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product);

    /**
     * Copies all properties from source band to the target band.
     *
     * @param sourceRaster the source band
     * @param targetRaster the target band
     * @see #copySpectralBandProperties(Band, Band)
     */
    static void CopyRasterDataNodeProperties(std::shared_ptr<RasterDataNode> source_raster,
                                             std::shared_ptr<RasterDataNode> target_raster);

    /**
     * Copies the named band from the source product to the target product.
     * <p>
     * The method does not copy any image geo-coding/geometry information.
     * Use the {@link #copyImageGeometry(RasterDataNode, RasterDataNode, boolean)} to do so.
     *
     * @param sourceBandName  the name of the band to be copied.
     * @param sourceProduct   the source product.
     * @param targetProduct   the target product.
     * @param copySourceImage whether the source image of the source band should be copied.
     * @return the copy of the band, or {@code null} if the sourceProduct does not contain a band with the given name.
     * @since BEAM 4.10
     */
    static std::shared_ptr<Band> CopyBand(std::string_view source_band_name, std::shared_ptr<Product> source_product,
                                          std::shared_ptr<Product> target_product, bool copy_source_image);

    /**
     * Copies the named band from the source product to the target product.
     * <p>
     * The method does not copy any image geo-coding/geometry information.
     * Use the {@link #copyImageGeometry(RasterDataNode, RasterDataNode, boolean)} to do so.
     *
     * @param sourceBandName  the name of the band to be copied.
     * @param sourceProduct   the source product.
     * @param targetBandName  the name of the band copied.
     * @param targetProduct   the target product.
     * @param copySourceImage whether the source image of the source band should be copied.
     * @return the copy of the band, or {@code null} if the sourceProduct does not contain a band with the given name.
     * @since BEAM 4.10
     */
    static std::shared_ptr<Band> CopyBand(std::string_view source_band_name, std::shared_ptr<Product> source_product,
                                          std::string_view target_band_name, std::shared_ptr<Product> target_product,
                                          bool copy_source_image);

    /**
     * Copies the overlay {@link Mask}s from the source product's raster data nodes to
     * the target product's raster data nodes.
     * <p>
     * IMPORTANT NOTE: This method should only be used, if it is known that all masks
     * in the source product will also be valid in the target product. This method does
     * <em>not</em> copy overlay masks, which are not contained in the target product's
     * mask group.
     *
     * @param sourceProduct the source product
     * @param targetProduct the target product
     */
    static void CopyOverlayMasks(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product);

    /**
     * Copies the geo-coding from the source product to target product.
     *
     * @param sourceProduct the source product
     * @param targetProduct the target product
     * @throws IllegalArgumentException if one of the params is {@code null}.
     */
    static void CopyGeoCoding(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product);

    static void CopyVectorData(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product);

    /**
     * Copies the index codings from the source product to the target.
     *
     * @param source the source product
     * @param target the target product
     */
    static void CopyIndexCodings(const std::shared_ptr<Product>& source, const std::shared_ptr<Product>& target);

    /**
     * Copies the given source index coding to the target product
     * If it exists already, the method simply returns the existing instance.
     *
     * @param sourceIndexCoding the source index coding
     * @param target            the target product
     * @return The index coding.
     */
    static std::shared_ptr<IndexCoding> CopyIndexCoding(const std::shared_ptr<IndexCoding>& source_index_coding,
                                                        const std::shared_ptr<Product>& target);

    /**
     * Copies the quicklook band name if not currently set and band also exists in target
     *
     * @param source the source product
     * @param target the target product
     */
    static void CopyQuicklookBandName(std::shared_ptr<Product> source, std::shared_ptr<Product> target);

    /**
     * Copies the spectral properties from source band to target band. These properties are:
     * <ul>
     * <li>{@link Band#getSpectralBandIndex() spectral band index},</li>
     * <li>{@link Band#getSpectralWavelength() the central wavelength},</li>
     * <li>{@link Band#getSpectralBandwidth() the spectral bandwidth} and</li>
     * <li>{@link Band#getSolarFlux() the solar spectral flux}.</li>
     * </ul>
     *
     * @param sourceBand the source band
     * @param targetBand the target band
     * @see #copyRasterDataNodeProperties(RasterDataNode, RasterDataNode)
     */
    static void CopySpectralBandProperties(std::shared_ptr<Band> source_band, std::shared_ptr<Band> target_band);

    /**
     * Normalizes the given geographical polygon so that maximum longitude differences between two points are 180
     * degrees. The method operates only on the longitude values of the given polygon.
     *
     * @param polygon a geographical, closed polygon
     * @return 0 if normalizing has not been applied , -1 if negative normalizing has been applied, 1 if positive
     * normalizing has been applied, 2 if positive and negative normalising has been applied
     * @see #denormalizeGeoPolygon(GeoPos[])
     */
    static int NormalizeGeoPolygon(std::vector<GeoPos> polygon);
};
}  // namespace snapengine
}  // namespace alus
