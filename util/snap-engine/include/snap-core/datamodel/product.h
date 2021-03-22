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
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <boost/filesystem.hpp>

// TEMPORARY// todo: move readers writers behind IProductReader/writer interface
#include "../../../../../algs/coherence/include/i_data_tile_reader.h"
#include "../../../../../algs/coherence/include/i_data_tile_writer.h"

#include "custom/dimension.h"
#include "snap-core/dataio/i_product_reader.h"
#include "snap-core/datamodel/product_node.h"

namespace alus {
namespace snapengine {

/**
 * {@code Product} instances are an in-memory representation of a remote sensing data product. The product is more
 * an abstract hull containing references to the data of the product or readers to retrieve the data on demand. The
 * product itself does not hold the remote sensing data. Data products can contain multiple geophysical parameters
 * stored as bands and can also have multiple metadata attributes. Also, a {@code Product} can contain any number
 * of {@code TiePointGrids} holding the tie point data.
 * <p>
 * <p>Every product can also have a product reader and writer assigned to it. The reader represents the data source from
 * which a product was created, whereas the writer represents the data sink. Both, the source and the sink must not
 * necessarily store data in the same format. Furthermore, it is not mandatory for a product to have both of them.
 *
 * java version author was Norman Fomferra
 */
// TEMPORARY// todo: try to move behind IProductReader/IProductWriter interfaces, these are custom relics
class IMetaDataReader;
class IMetaDataWriter;

template <typename T>
class ProductNodeGroup;
class MetadataElement;
class IProductWriter;
class RasterDataNode;
class FlagCoding;
class IndexCoding;
class TiePointGrid;
class Utc;
class Band;
class Mask;
class IGeoCoding;
class Quicklook;
class GeoPos;
class PixelPos;
class Product : public ProductNode {
private:
    /**
     * The internal reference number of this product
     */
    int ref_no_ = 0;

    /**
     * This product's type ID.
     */
    std::string product_type_;

    std::string quicklook_band_name_;

    /**
     * The location file of this product.
     */
    boost::filesystem::path file_location_;

    // TEMPORARY//todo: this is currently placeholder for any reader interface, might want to add different abstractions
    std::shared_ptr<IDataTileReader> reader_old_;
    std::shared_ptr<IDataTileWriter> writer_old_;
    std::shared_ptr<IMetaDataReader> metadata_reader_;
    std::shared_ptr<IMetaDataWriter> metadata_writer_;

    /**
     * The reader for this product. Once the reader is set, and can never be changed again.
     */
    std::shared_ptr<IProductReader> reader_;
    /**
     * The writer for this product. The writer is an exchangeable property of a product.
     */
    std::shared_ptr<IProductWriter> writer_;

    std::shared_ptr<MetadataElement> metadata_root_;

    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Band>>> band_group_;
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<TiePointGrid>>> tie_point_grid_group_;
    //    std::shared_ptr < ProductNodeGroup<std::shared_ptr<VectorDataNode>>> vector_data_group_;
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<FlagCoding>>> flag_coding_group_;
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<IndexCoding>>> index_coding_group_;
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> mask_group_;
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Quicklook>>> quicklook_group_;

    std::shared_ptr<ProductNodeGroup<std::shared_ptr<ProductNode>>> groups_;

    /**
     * The start time of the first raster line.
     */
    std::shared_ptr<Utc> start_time_;

    /**
     * The start time of the first raster line.
     */
    std::shared_ptr<Utc> end_time_;

    /**
     * The product's scene raster size in pixels.
     */
    std::shared_ptr<custom::Dimension> scene_raster_size_;

    std::shared_ptr<custom::Dimension> preferred_tile_size_;

    /**
     * The geo-coding of this product, if any.
     */
    std::shared_ptr<IGeoCoding> scene_geo_coding_;

    /**
     * The internal reference string of this product
     */
    std::string ref_str_;

    /**
     * workaround from snap implementation to provide shared_from_this after Product construction (currently can't use
     * shared_from_this inside constructor)
     * @param product
     * @return
     */
    static std::shared_ptr<Product> InitProductMembers(const std::shared_ptr<Product>& product);
    /**
     * Creates a new product without any reader (in-memory product)
     *
     * @param name              the product name
     * @param type              the product type
     * @param sceneRasterWidth  the scene width in pixels for this data product
     * @param sceneRasterHeight the scene height in pixels for this data product
     */
    Product(std::string_view name, std::string_view type, int scene_raster_width, int scene_raster_height);

    /**
     * Constructs a new product with the given name and the given reader.
     *
     * @param name              the product identifier
     * @param type              the product type
     * @param sceneRasterWidth  the scene width in pixels for this data product
     * @param sceneRasterHeight the scene height in pixels for this data product
     * @param reader            the reader used to create this product and read data from it.
     * @see ProductReader
     */
    Product(std::string_view name, std::string_view type, int scene_raster_width, int scene_raster_height,
            const std::shared_ptr<IProductReader>& reader);

    /**
     * Constructs a new product with the given name and type.
     *
     * @param name the product identifier
     * @param type the product type
     */
    Product(std::string_view name, std::string_view type);

    /**
     * Constructs a new product with the given name, type and the reader.
     *
     * @param name   the product identifier
     * @param type   the product type
     * @param reader the reader used to create this product and read data from it.
     * @see ProductReader
     */
    Product(std::string_view name, std::string_view type, const std::shared_ptr<IProductReader>& reader);

    /**
     * Internally used constructor. Is kept private to keep product name and file location consistent.
     *
     */
    Product(std::string_view name, std::string_view type, const std::shared_ptr<custom::Dimension>& scene_raster_size,
            const std::shared_ptr<IProductReader>& reader);

    void CheckGeoCoding(const std::shared_ptr<IGeoCoding>& geo_coding);
    static bool EqualsLatLon(const std::shared_ptr<GeoPos>& pos1, const std::shared_ptr<GeoPos>& pos2, float eps);
    static bool EqualsOrNaN(double v1, double v2, float eps);

public:
    static constexpr std::string_view TIE_POINT_GRID_DIR_NAME = "tie_point_grids";

    static constexpr std::string_view METADATA_ROOT_NAME = "metadata";
    static constexpr std::string_view HISTORY_ROOT_NAME = "history";

    static constexpr std::string_view PROPERTY_NAME_SCENE_CRS = "sceneCRS";
    static constexpr std::string_view PROPERTY_NAME_SCENE_GEO_CODING = "sceneGeoCoding";
    static constexpr std::string_view PROPERTY_NAME_SCENE_TIME_CODING = "sceneTimeCoding";
    static constexpr std::string_view PROPERTY_NAME_PRODUCT_TYPE = "productType";
    static constexpr std::string_view PROPERTY_NAME_FILE_LOCATION = "fileLocation";

    // modified
    static constexpr std::string_view GEOMETRY_FEATURE_TYPE_NAME = "org.esa.snap.Geometry";
    static constexpr std::string_view PIN_GROUP_NAME = "pins";
    static constexpr std::string_view GCP_GROUP_NAME = "ground_control_points";

    //    // todo:probably good idea to provide single parent reader/writer which is composition of different
    //    implementations
    //    // e.g pugixml to write metadata and gdal to write geotiff etc.
    const std::shared_ptr<IDataTileReader>& GetReader() const;
    void SetReader(const std::shared_ptr<IDataTileReader>& reader);
    const std::shared_ptr<IDataTileWriter>& GetWriter() const;
    void SetWriter(const std::shared_ptr<IDataTileWriter>& writer);

    const std::shared_ptr<IMetaDataReader>& GetMetadataReader() const;
    void SetMetadataReader(const std::shared_ptr<IMetaDataReader>& metadata_reader);
    const std::shared_ptr<IMetaDataWriter>& GetMetadataWriter() const;
    void SetMetadataWriter(const std::shared_ptr<IMetaDataWriter>& metadata_writer);

    /**
     * Workaround static function which calls constructor with same parameters and also inits members which need
     * construction time shared_from_this Creates a new product without any reader (in-memory product)
     *
     * @param name              the product name
     * @param type              the product type
     * @param sceneRasterWidth  the scene width in pixels for this data product
     * @param sceneRasterHeight the scene height in pixels for this data product
     */
    static std::shared_ptr<Product> CreateProduct(std::string_view name, std::string_view type, int scene_raster_width,
                                                  int scene_raster_height);

    /**
     * Workaround static function which calls constructor with same parameters and also inits members which need
     * construction time shared_from_this Constructs a new product with the given name and the given reader.
     *
     * @param name              the product identifier
     * @param type              the product type
     * @param sceneRasterWidth  the scene width in pixels for this data product
     * @param sceneRasterHeight the scene height in pixels for this data product
     * @param reader            the reader used to create this product and read data from it.
     * @see ProductReader
     */
    static std::shared_ptr<Product> CreateProduct(std::string_view name, std::string_view type, int scene_raster_width,
                                                  int scene_raster_height,
                                                  const std::shared_ptr<IProductReader>& reader);
    /**
     * Workaround static function which calls constructor with same parameters and also inits members which need
     * construction time shared_from_this Constructs a new product with the given name and type.
     *
     * @param name the product identifier
     * @param type the product type
     */
    static std::shared_ptr<Product> CreateProduct(std::string_view name, std::string_view type);

    /**
     * Workaround static function which calls constructor with same parameters and also inits members which need
     * construction time shared_from_this Constructs a new product with the given name, type and the reader.
     *
     * @param name   the product identifier
     * @param type   the product type
     * @param reader the reader used to create this product and read data from it.
     * @see ProductReader
     */
    static std::shared_ptr<Product> CreateProduct(std::string_view name, std::string_view type,
                                                  const std::shared_ptr<IProductReader>& reader);

    void SetModified(bool modified) override;

    /**
     * Gets the root element of the associated metadata.
     *
     * @return the metadata root element
     */
    std::shared_ptr<MetadataElement> GetMetadataRoot() { return metadata_root_; }

    /**
     * Gets the product type string.
     *
     * @return the product type string
     */
    std::string GetProductType() { return product_type_; }

    /**
     * Gets the (sensing) start time associated with the first raster data line.
     * <p>For Level-1/2 products this is
     * the data-take time associated with the first raster data line.
     * For Level-3 products, this could be the start time of first input product
     * contributing data.
     *
     * @return the sensing start time, can be null e.g. for non-swath products
     */
    std::shared_ptr<Utc> GetStartTime() { return start_time_; }

    /**
     * Sets the (sensing) start time of this product.
     * <p>For Level-1/2 products this is
     * the data-take time associated with the first raster data line.
     * For Level-3 products, this could be the start time of first input product
     * contributing data.
     *
     * @param startTime the sensing start time, can be null
     */
    void SetStartTime(std::shared_ptr<Utc> start_time);

    /**
     * Sets the product type of this product.
     *
     * @param productType the product type.
     */
    void SetProductType(std::string_view product_type);

    /**
     * Sets the (sensing) stop time associated with the first raster data line.
     * <p>For Level-1/2 products this is
     * the data-take time associated with the last raster data line.
     * For Level-3 products, this could be the end time of last input product
     * contributing data.
     *
     * @param endTime the sensing stop time, can be null
     */
    void SetEndTime(const std::shared_ptr<Utc>& end_time);

    /**
     * Gets the (sensing) stop time associated with the last raster data line.
     * <p>For Level-1/2 products this is
     * the data-take time associated with the last raster data line.
     * For Level-3 products, this could be the end time of last input product
     * contributing data.
     *
     * @return the stop time , can be null e.g. for non-swath products
     */
    std::shared_ptr<Utc> GetEndTime() { return end_time_; }

    /**
     * Sets the geo-coding to be associated with the scene raster.
     *
     * @param sceneGeoCoding the geo-coding, or {@code null}
     * @throws IllegalArgumentException <br>- if the given {@code GeoCoding} is a {@code TiePointGeoCoding}
     *                                  and {@code latGrid} or {@code lonGrid} are not instances of tie point
     *                                  grids in this product. <br>- if the given {@code GeoCoding} is a
     *                                  {@code MapGeoCoding} and its {@code MapInfo} is {@code null}
     *                                  <br>- if the given {@code GeoCoding} is a {@code MapGeoCoding} and the
     *                                  {@code sceneWith} or {@code sceneHeight} of its {@code MapInfo}
     *                                  is not equal to this products {@code sceneRasterWidth} or
     *                                  {@code sceneRasterHeight}
     */
    void SetSceneGeoCoding(const std::shared_ptr<IGeoCoding>& scene_geo_coding);

    /**
     * Tests if all bands of this product are using a single, uniform geo-coding. Uniformity is tested by comparing
     * the band's geo-coding against the geo-coding of this product using the {@link Object#equals(Object)} method.
     * If this product does not have a geo-coding, the method returns false.
     *
     * @return true, if so
     */
    bool IsUsingSingleGeoCoding();

    /**
     * @return The scene raster width in pixels.
     * @throws IllegalStateException if the scene size wasn't specified yet and cannot be derived
     */
    int GetSceneRasterWidth();

    /**
     * @return The scene raster height in pixels
     * @throws IllegalStateException if the scene size wasn't specified yet and cannot be derived
     */
    int GetSceneRasterHeight();

    /**
     * @return The scene size in pixels.
     * @throws IllegalStateException if the scene size wasn't specified yet and cannot be derived
     */
    std::shared_ptr<custom::Dimension> GetSceneRasterSize();

    /**
     * Sets the product reader which will be used to create this product in-memory represention from an external source
     * and which will be used to (re-)load band rasters.
     *
     * @param reader the product reader.
     * @throws IllegalArgumentException if the given reader is null.
     */
    void SetProductReader(std::shared_ptr<IProductReader> reader);

    /**
     * Sets the preferred tile size which may be used for a the {@link java.awt.image.RenderedImage rendered image}
     * created for a {@link RasterDataNode} of this product.
     *
     * @param preferred_tile_size the preferred tile size, may be {@code null} if not specified
     * @see RasterDataNode#getSourceImage()
     * @see RasterDataNode#setSourceImage(java.awt.image.RenderedImage)
     */
    void SetPreferredTileSize(std::shared_ptr<custom::Dimension> preferred_tile_size);

    /**
     * Gets the geo-coding associated with the scene raster.
     *
     * @return the geo-coding, can be {@code null} if this product is not geo-coded.
     */
    std::shared_ptr<IGeoCoding> GetSceneGeoCoding() { return scene_geo_coding_; }

    /**
     * Gets the preferred tile size which may be used for a the {@link java.awt.image.RenderedImage rendered image}
     * created for a {@link RasterDataNode} of this product.
     *
     * @return the preferred tile size, may be {@code null} if not specified
     * @see RasterDataNode#getSourceImage()
     * @see RasterDataNode#setSourceImage(java.awt.image.RenderedImage)
     */
    std::shared_ptr<custom::Dimension> GetPreferredTileSize() { return preferred_tile_size_; }

    /**
     * Returns the reference string of this product.
     *
     * @return the reference string.
     */
    std::string GetRefStr() { return ref_str_; }

    /**
     * @return The reference number of this product.
     */
    int GetRefNo() const { return ref_no_; }

    /**
     * Sets the reference number.
     *
     * @param refNo the reference number to set must be in the range 1 .. Integer.MAX_VALUE
     * @throws IllegalArgumentException if the refNo is out of range
     */
    void SetRefNo(int ref_no);

    void ResetRefNo() {
        ref_no_ = 0;
        ref_str_ = nullptr;
    }

    /**
     * Sets the file location for this product.
     *
     * @param fileLocation the file location, may be {@code null}
     */
    void SetFileLocation(boost::filesystem::path file_location) { file_location_ = file_location; }

    /**
     * Retrieves the disk location of this product. The return value can be {@code null} when the product has no
     * disk location (pure virtual memory product)
     *
     * @return the file location, may be {@code null}
     */
    boost::filesystem::path GetFileLocation() { return file_location_; }

    //    InitSceneProperties() {
    //        Comparator<Band> maxAreaComparator = (o1, o2)->{
    //            final long a1 = o1.getRasterWidth() * (long)o1.getRasterHeight();
    //            final long a2 = o2.getRasterWidth() * (long)o2.getRasterHeight();
    //            return Long.compare(a2, a1);
    //        };
    //        Band refBand = Stream.of(GetBands())
    //                           .filter(b->b.getGeoCoding() != nullptr)
    //                           .sorted(maxAreaComparator)
    //                           .findFirst()
    //                           .orElse(nullptr);
    //        if (refBand == nullptr) {
    //            refBand = Stream.of(GetBands()).sorted(maxAreaComparator).findFirst().orElse(nullptr);
    //        }
    //        if (refBand != nullptr) {
    //            if (scene_raster_size_ == nullptr) {
    //                scene_raster_size_ = std::make_shared<custom::Dimension>(refBand.GetRasterWidth(),
    //                refBand.GetRasterHeight()); if (scene_geo_coding_ == nullptr) {
    //                    scene_geo_coding_ = refBand.GetGeoCoding();
    //                }
    //            }
    //            return true;
    //        }
    //        return false;
    //    }

    /**
     * Returns the reader which was used to create this product in-memory represention from an external source and which
     * will be used to (re-)load band rasters.
     *
     * @return the product reader, can be {@code null}
     */
    std::shared_ptr<IProductReader> GetProductReader() override { return reader_; }

    //////////////////////////////////////////////////////////////////////////
    // Group support

    /**
     * @return The group which contains all other product node groups.
     * @since BEAM 5.0
     */
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<ProductNode>>> GetGroups();

    /**
     * @param name The group name.
     * @return The group with the given name, or {@code null} if no such group exists.
     * @since BEAM 5.0
     */
    std::shared_ptr<ProductNode> GetGroup(std::string_view name);

    //////////////////////////////////////////////////////////////////////////
    // Tie-point grid support

    /**
     * Gets the tie-point grid group of this product.
     *
     * @return The group of all tie-point grids.
     * @since BEAM 4.7
     */
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<TiePointGrid>>> GetTiePointGridGroup();

    /**
     * Adds the given tie-point grid to this product.
     *
     * @param tiePointGrid the tie-point grid to added, ignored if {@code null}
     */
    void AddTiePointGrid(std::shared_ptr<TiePointGrid> tie_point_grid);

    /**
     * Removes the tie-point grid from this product.
     *
     * @param tiePointGrid the tie-point grid to be removed, ignored if {@code null}
     * @return {@code true} if node could be removed
     */

    bool RemoveTiePointGrid(std::shared_ptr<TiePointGrid> tie_point_grid);

    /**
     * Returns the number of tie-point grids contained in this product
     *
     * @return the number of tie-point grids
     */
    int GetNumTiePointGrids();

    /**
     * Returns the tie-point grid at the given index.
     *
     * @param index the tie-point grid index
     * @return the tie-point grid at the given index
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    std::shared_ptr<TiePointGrid> GetTiePointGridAt(int index);

    /**
     * Returns a string array containing the names of the tie-point grids contained in this product
     *
     * @return a string array containing the names of the tie-point grids contained in this product. If this product has
     * no tie-point grids a zero-length-array is returned.
     */
    std::vector<std::string> GetTiePointGridNames();

    /**
     * Returns an array of tie-point grids contained in this product
     *
     * @return an array of tie-point grids contained in this product. If this product has no  tie-point grids a
     * zero-length-array is returned.
     */
    std::vector<std::shared_ptr<TiePointGrid>> GetTiePointGrids();

    /**
     * Returns the tie-point grid with the given name.
     *
     * @param name the tie-point grid name
     * @return the tie-point grid with the given name or {@code null} if a tie-point grid with the given name is
     * not contained in this product.
     */
    std::shared_ptr<TiePointGrid> GetTiePointGrid(std::string_view name);

    /**
     * Tests if a tie-point grid with the given name is contained in this product.
     *
     * @param name the name, must not be {@code null}
     * @return {@code true} if a tie-point grid with the given name is contained in this product,
     * {@code false} otherwise
     */
    bool ContainsTiePointGrid(std::string_view name);

    //////////////////////////////////////////////////////////////////////////
    // Band support

    /**
     * Gets the band group of this product.
     *
     * @return The group of all bands.
     * @since BEAM 4.7
     */
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Band>>> GetBandGroup() { return band_group_; }

    /**
     * Adds the given band to this product.
     *
     * @param band the band to added, must not be {@code null}
     */
    void AddBand(std::shared_ptr<Band> band);

    /**
     * Creates a new band with the given name and data type and adds it to this product and returns it.
     *
     * @param bandName the new band's name
     * @param dataType the raster data type, must be one of the multiple <code>ProductData.TYPE_<i>X</i></code>
     *                 constants
     * @return the new band which has just been added
     */
    std::shared_ptr<Band> AddBand(std::string_view band_name, int data_type);

    /**
     * Creates a new band with the given name and adds it to this product and returns it.
     * The new band's data type is {@code float} and it's samples are computed from the given band maths expression.
     *
     * @param bandName   the new band's name
     * @param expression the band maths expression
     * @return the new band which has just been added
     * @since BEAM 4.9
     */
    std::shared_ptr<Band> AddBand(std::string_view band_name, std::string_view expression);

    /**
     * Creates a new band with the given name and data type and adds it to this product and returns it.
     * The new band's samples are computed from the given band maths expression.
     *
     * @param bandName   the new band's name
     * @param expression the band maths expression
     * @param dataType   the raster data type, must be one of the multiple <code>ProductData.TYPE_<i>X</i></code>
     *                   constants
     * @return the new band which has just been added
     * @since BEAM 4.9
     */
    std::shared_ptr<Band> AddBand(std::string_view band_name, std::string_view expression, int data_type);

    /**
     * Removes the given band from this product.
     *
     * @param band the band to be removed, ignored if {@code null}
     * @return {@code true} if removed succesfully, otherwise {@code false}
     */
    bool RemoveBand(std::shared_ptr<Band> band);

    /**
     * @return the number of bands contained in this product.
     */
    int GetNumBands();

    /**
     * Returns the band at the given index.
     *
     * @param index the band index
     * @return the band at the given index
     * @throws IndexOutOfBoundsException if the index is out of bounds
     */
    std::shared_ptr<Band> GetBandAt(int index);

    /**
     * Returns a string array containing the names of the bands contained in this product
     *
     * @return a string array containing the names of the bands contained in this product. If this product has no bands
     * a zero-length-array is returned.
     */
    std::vector<std::string> GetBandNames();

    /**
     * Returns an array of bands contained in this product
     *
     * @return an array of bands contained in this product. If this product has no bands a zero-length-array is
     * returned.
     */
    std::vector<std::shared_ptr<Band>> GetBands();

    /**
     * Returns the band with the given name.
     *
     * @param name the band name
     * @return the band with the given name or {@code null} if a band with the given name is not contained in this
     * product.
     * @throws IllegalArgumentException if the given name is {@code null} or empty.
     */
    std::shared_ptr<Band> GetBand(std::string_view name);

    /**
     * Returns the index for the band with the given name.
     *
     * @param name the band name
     * @return the band index or {@code -1} if a band with the given name is not contained in this product.
     * @throws IllegalArgumentException if the given name is {@code null} or empty.
     */
    int GetBandIndex(std::string_view name);

    /**
     * Tests if a band with the given name is contained in this product.
     *
     * @param name the name, must not be {@code null}
     * @return {@code true} if a band with the given name is contained in this product, {@code false}
     * otherwise
     * @throws IllegalArgumentException if the given name is {@code null} or empty.
     */
    bool ContainsBand(std::string_view name);

    //////////////////////////////////////////////////////////////////////////
    // Raster data node  support

    /**
     * Tests if a raster data node with the given name is contained in this product. Raster data nodes can be bands or
     * tie-point grids.
     *
     * @param name the name, must not be {@code null}
     * @return {@code true} if a raster data node with the given name is contained in this product,
     * {@code false} otherwise
     */
    bool ContainsRasterDataNode(std::string_view name);

    /**
     * Gets the raster data node with the given name. The method first searches for bands with the given name, then for
     * tie-point grids. If neither bands nor tie-point grids exist with the given name, {@code null} is returned.
     *
     * @param name the name, must not be {@code null}
     * @return the raster data node with the given name or {@code null} if a raster data node with the given name
     * is not contained in this product.
     */
    std::shared_ptr<RasterDataNode> GetRasterDataNode(std::string_view name);

    /**
     * Gets all raster data nodes contained in this product including bands, masks and tie-point grids.
     *
     * @return List of all raster data nodes which may be empty.
     * @since SNAP 2.0
     */
    //     todo:java had synchronized, check if we get issues
    std::vector<std::shared_ptr<RasterDataNode>> GetRasterDataNodes();

    //////////////////////////////////////////////////////////////////////////
    // Mask support

    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> GetMaskGroup() { return mask_group_; }

    //////////////////////////////////////////////////////////////////////////
    // Quicklook support

    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Quicklook>>> GetQuicklookGroup() { return quicklook_group_; }

    std::shared_ptr<Quicklook> GetDefaultQuicklook();

    /**
     * Returns the Quicklook with the given name.
     *
     * @param name the quicklook name
     * @return the quicklook with the given name or {@code null} if a quicklook with the given name is not contained in
     * this product.
     * @throws IllegalArgumentException if the given name is {@code null} or empty.
     */
    std::shared_ptr<Quicklook> GetQuicklook(std::string_view name);

    /**
     * Gets the name of the band suitable for quicklook generation.
     *
     * @return the name of the quicklook band, or null if none has been defined
     */
    std::string GetQuicklookBandName() { return quicklook_band_name_; }

    /**
     * Sets the name of the band suitable for quicklook generation.
     *
     * @param quicklookBandName the name of the quicklook band, or null
     */
    void SetQuicklookBandName(std::string_view quicklook_band_name) { quicklook_band_name_ = quicklook_band_name; }

    //////////////////////////////////////////////////////////////////////////
    // Sample-coding support

    std::shared_ptr<ProductNodeGroup<std::shared_ptr<FlagCoding>>> GetFlagCodingGroup() { return flag_coding_group_; }

    std::shared_ptr<ProductNodeGroup<std::shared_ptr<IndexCoding>>> GetIndexCodingGroup() {
        return index_coding_group_;
    }
    //////////////////////////////////////////////////////////////////////////
    // Pixel Coordinate Tests

    /**
     * Tests if the given pixel position is within the product pixel bounds.
     *
     * @param x the x coordinate of the pixel position
     * @param y the y coordinate of the pixel position
     * @return true, if so
     * @see #containsPixel(PixelPos)
     */
    bool ContainsPixel(double x, double y);

    /**
     * Tests if the given pixel position is within the product pixel bounds.
     *
     * @param pixelPos the pixel position, must not be null
     * @return true, if so
     * @see #containsPixel(double, double)
     */
    bool ContainsPixel(const std::shared_ptr<PixelPos>& pixel_pos);

    /**
     * Gets an estimated, raw storage size in bytes of this product node.
     *
     * @param subsetDef if not {@code null} the subset may limit the size returned
     * @return the size in bytes.
     */

    uint64_t GetRawStorageSize(const std::shared_ptr<ProductSubsetDef>& subsetDef) override;

    /**
     * Checks whether or not the given product is compatible with this product.
     *
     * @param product the product to compare with
     * @param eps     the maximum lat/lon error in degree
     * @return {@code false} if the scene dimensions or geocoding are different, {@code true} otherwise.
     */
    bool IsCompatibleProduct(const std::shared_ptr<Product>& product, float eps);

    /**
     * Closes and clears this product's reader (if any).
     *
     * @throws IOException if an I/O error occurs
     * @see #closeIO
     */
    void CloseProductReader();

    /**
     * Closes and clears this product's writer (if any).
     *
     * @throws IOException if an I/O error occurs
     * @see #closeIO
     */
    void CloseProductWriter();

    /**
     * Closes the file I/O for this product. Calls in sequence <code>{@link #closeProductReader}</code>  and
     * <code>{@link #closeProductWriter}</code>. The <code>{@link #dispose}</code> method is <b>not</b> called, but
     * should be called if the product instance is no longer in use.
     *
     * @throws IOException if an I/O error occurs
     * @see #closeProductReader
     * @see #closeProductWriter
     * @see #dispose
     */
    void CloseIO();

    /**
     * Releases all of the resources used by this object instance and all of its owned children. Its primary use is to
     * allow the garbage collector to perform a vanilla job.
     * <p>This method should be called only if it is for sure that this object instance will never be used again. The
     * results of referencing an instance of this class after a call to {@code dispose()} are undefined.
     * <p>
     * <p>Overrides of this method should always call {@code super.dispose();} after disposing this instance.
     * <p>
     * <p>This implementation also calls the {@code closeIO} in order to release all open I/O resources.
     */
    void Dispose() override;
};

}  // namespace snapengine
}  // namespace alus
