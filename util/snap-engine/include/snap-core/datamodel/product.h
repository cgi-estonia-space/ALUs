#pragma once

#include <boost/filesystem.hpp>
#include <string_view>

#include "../../../../../algs/apply-orbit-file-op/include/dimension.h"
#include "../../../../../algs/coherence/include/i_data_tile_reader.h"
#include "../../../../../algs/coherence/include/i_data_tile_writer.h"
#include "band.h"
//#include "mask.h"
#include "product_node.h"
#include "tie_point_grid.h"

/**
 * This file is a filtered and modified duplicate of a SNAP's  org.esa.snap.core.datamodel.Product.java ported
 * for native code. Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
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
template <typename T>
class ProductNodeGroup;
class MetadataElement;
class IMetaDataReader;
class IMetaDataWriter;
class Product : virtual public ProductNode {
private:
    /**
     * This product's type ID.
     */
    std::string product_type_;

    std::string quick_look_band_name_;

    /**
     * The location file of this product.
     */
    boost::filesystem::path file_location_;

    /**
     * The reader for this product. Once the reader is set, and can never be changed again.
     */
    //     todo: this is currently placeholder for any reader interface, might want to add different abstractions
    std::shared_ptr<IDataTileReader> reader_;
    std::shared_ptr<IDataTileWriter> writer_;
    std::shared_ptr<IMetaDataReader> metadata_reader_;
    std::shared_ptr<IMetaDataWriter> metadata_writer_;

    // todo: maybe make it better in the future
    //    /**
    //    * The reader for this product. Once the reader is set, and can never be changed again.
    //    */
    //    std::shared_ptr<ProductReader> reader_;

    std::shared_ptr<MetadataElement> metadata_root_;

    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Band>>> band_group_;

    std::shared_ptr<ProductNodeGroup<std::shared_ptr<TiePointGrid>>> tie_point_grid_group_;

    //    std::shared_ptr < ProductNodeGroup<std::shared_ptr<VectorDataNode>>> vector_data_group_;
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<FlagCoding>>> flag_coding_group_;
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<IndexCoding>>> index_coding_group_;
    //    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> mask_group_;

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
    std::shared_ptr<Dimension> scene_raster_size_;

    /**
     * The geo-coding of this product, if any.
     */
    //    std::shared_ptr<GeoCoding> scene_geo_coding_;
    /*
     * Internally used constructor. Is kept private to keep product name and file location consistent.
     */
    Product(std::string_view name, std::string_view type, const std::shared_ptr<Dimension>& scene_raster_size,
            const std::shared_ptr<IDataTileReader>& reader, const std::shared_ptr<IMetaDataReader>& metadata_reader);

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

    // todo:probably good idea to provide single parent reader/writer which is composition of different implementations
    // e.g pugixml to write metadata and gdal to write geotiff etc.
    const std::shared_ptr<IDataTileReader>& GetReader() const;
    void SetReader(const std::shared_ptr<IDataTileReader>& reader);
    const std::shared_ptr<IDataTileWriter>& GetWriter() const;
    void SetWriter(const std::shared_ptr<IDataTileWriter>& writer);

    const std::shared_ptr<IMetaDataReader>& GetMetadataReader() const;
    void SetMetadataReader(const std::shared_ptr<IMetaDataReader>& metadata_reader);
    const std::shared_ptr<IMetaDataWriter>& GetMetadataWriter() const;
    void SetMetadataWriter(const std::shared_ptr<IMetaDataWriter>& metadata_writer);

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
            const std::shared_ptr<IDataTileReader>& reader, const std::shared_ptr<IMetaDataReader>& metadata_reader);

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
    Product(std::string_view name, std::string_view type, const std::shared_ptr<IDataTileReader>& reader,
            const std::shared_ptr<IMetaDataReader>& metadata_reader);

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
    void SetStartTime(std::shared_ptr<Utc> start_time) {
        std::shared_ptr<Utc> old = start_time_;
        if (start_time != old) {
            SetModified(true);
        }
        start_time_ = start_time;
    }

    /**
     * Sets the (sensing) stop time associated with the first raster data line.
     * <p>For Level-1/2 products this is
     * the data-take time associated with the last raster data line.
     * For Level-3 products, this could be the end time of last input product
     * contributing data.
     *
     * @param endTime the sensing stop time, can be null
     */
    void SetEndTime(const std::shared_ptr<Utc>& end_time) {
        std::shared_ptr<Utc> old = end_time_;
        if (end_time != old) {
            SetModified(true);
        }
        end_time_ = end_time;
    }

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
    std::shared_ptr<Dimension> GetSceneRasterSize();

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
    //                scene_raster_size_ = std::make_shared<Dimension>(refBand.GetRasterWidth(),
    //                refBand.GetRasterHeight()); if (scene_geo_coding_ == nullptr) {
    //                    scene_geo_coding_ = refBand.GetGeoCoding();
    //                }
    //            }
    //            return true;
    //        }
    //        return false;
    //    }

    /**
     * Returns an array of bands contained in this product
     *
     * @return an array of bands contained in this product. If this product has no bands a zero-length-array is
     * returned.
     */
    //    std::vector<Band> GetBands() { return band_group_->ToArray(std::vector<std::shared_ptr<Band>>(GetNumBands()));
    //    }

    /**
     * Returns the reader which was used to create this product in-memory represention from an external source and which
     * will be used to (re-)load band rasters.
     *
     * @return the product reader, can be {@code null}
     */
    //    std::shared_ptr<ProductReader> GetProductReader() override {
    auto GetProductReader() { return reader_; }

    //////////////////////////////////////////////////////////////////////////
    // Tie-point grid support

    /**
     * Gets the tie-point grid group of this product.
     *
     * @return The group of all tie-point grids.
     * @since BEAM 4.7
     */
    std::shared_ptr<ProductNodeGroup<std::shared_ptr<TiePointGrid>>> GetTiePointGridGroup() {
        return tie_point_grid_group_;
    }

    /**
     * Adds the given tie-point grid to this product.
     *
     * @param tiePointGrid the tie-point grid to added, ignored if {@code null}
     */

    void AddTiePointGrid(std::shared_ptr<TiePointGrid> tie_point_grid);
    //    {
    //        if (containsRasterDataNode(tiePointGrid.getName())) {
    //            throw new IllegalArgumentException("The Product '" + getName() + "' already contains " +
    //                                               "a tie-point grid with the name '" + tiePointGrid.getName() +
    //                                               "'.");
    //        }
    //        tiePointGridGroup.add(tiePointGrid);
    //    }

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
    //    {
    //        final TiePointGrid[] tiePointGrids = new TiePointGrid[getNumTiePointGrids()];
    //        for (int i = 0; i < tiePointGrids.length; i++) {
    //            tiePointGrids[i] = getTiePointGridAt(i);
    //        }
    //        return tiePointGrids;
    //    }

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
     * Gets the name of the band suitable for quicklook generation.
     *
     * @return the name of the quicklook band, or null if none has been defined
     */
    std::string GetQuicklookBandName() { return quick_look_band_name_; }
    /**
     * Sets the name of the band suitable for quicklook generation.
     *
     * @param quicklookBandName the name of the quicklook band, or null
     */
    void SetQuicklookBandName(std::string_view quick_look_band_name) { quick_look_band_name_ = quick_look_band_name; }

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

    //    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> GetMaskGroup() { return mask_group_; }

    //////////////////////////////////////////////////////////////////////////
    // Sample-coding support

    std::shared_ptr<ProductNodeGroup<std::shared_ptr<FlagCoding>>> GetFlagCodingGroup() { return flag_coding_group_; }

    std::shared_ptr<ProductNodeGroup<std::shared_ptr<IndexCoding>>> GetIndexCodingGroup() {
        return index_coding_group_;
    }
};

}  // namespace snapengine
}  // namespace alus
