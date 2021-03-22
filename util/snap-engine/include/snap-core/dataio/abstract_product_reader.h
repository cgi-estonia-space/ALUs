/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.dataio.AbstractProductReader.java
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

#include <any>
#include <memory>
#include <utility>

#include "i_product_reader.h"
#include "i_product_reader_plug_in.h"
#include "product.h"
#include "product_subset_def.h"
#include "tie_point_grid.h"

namespace alus::snapengine {

/**
 * The {@code AbstractProductReader}  class can be used as a base class for new product reader implementations. The
 * only two methods which clients must implement are {@code readProductNodes()} and {@code readBandData}
 * methods.
 *
 * @author Norman Fomferra
 * @version $Revision$ $Date$
 * @see #readProductNodes
 * @see #readBandRasterData
 */
class AbstractProductReader : public std::enable_shared_from_this<AbstractProductReader>, public IProductReader {
private:
    /**
     * @since BEAM 4.9
     */
    static constexpr std::string_view SYSPROP_READER_TILE_WIDTH = "snap.dataio.reader.tileWidth";
    /**
     * @since BEAM 4.9
     */
    static constexpr std::string_view SYSPROP_READER_TILE_HEIGHT = "snap.dataio.reader.tileHeight";

    /**
     * The input source
     */
    //    todo:this needs more thinking later
    std::any input_;

    /**
     * The spectral and spatial subset definition used to read from the original data source.
     */
    std::shared_ptr<ProductSubsetDef> subset_def_;

    /**
     * The reader plug-in responsible for creating this reader.
     */
    std::shared_ptr<IProductReaderPlugIn> reader_plug_in_;

protected:
    //    AbstractProductReader() = default;
    /**
     * Constructs a new abstract product reader.
     *
     * @param readerPlugIn the reader plug-in which created this reader, can be {@code null} for internal reader
     *                     implementations
     */
    explicit AbstractProductReader(std::shared_ptr<IProductReaderPlugIn> reader_plug_in)
        : reader_plug_in_(std::move(reader_plug_in)) {}

    static std::shared_ptr<custom::Dimension> GetConfiguredTileSize(std::shared_ptr<Product> product,
                                                            std::string_view tile_width_str,
                                                            std::string_view tile_height_str);
    static int ParseTileSize(std::string_view size_str, int max_size);
    /**
     * Sets the subset information. This implemetation is protected to overwrite in the inherided class to ensure that
     * the subset information cannot be set from the {@code readProductNodes} method.
     *
     * @param subsetDef the subset definition
     */
    void SetSubsetDef(std::shared_ptr<ProductSubsetDef> subset_def) { subset_def_ = std::move(subset_def); }

    /**
     * Provides an implementation of the {@code readProductNodes} interface method. Clients implementing this
     * method can be sure that the input object and eventually the subset information has already been set.
     * <p>This method is called as a last step in the {@code readProductNodes(input, subsetInfo)} method.
     *
     * @return a new product instance
     * @throws IOException if an I/O error occurs
     */
    virtual std::shared_ptr<Product> ReadProductNodesImpl() = 0;

    /**
     * Checks if the given object is an instance of one of the valid input types for this product reader.
     *
     * @param input the input object passed to {@link #readProductNodes(Object, ProductSubsetDef)}
     * @return {@code true} if so
     * @see ProductReaderPlugIn#getInputTypes()
     */
    bool IsInstanceOfValidInputType(std::any input);

    void SetInput(std::any input);

    /**
     * The template method which is called by the method after an optional spatial subset has been applied to the input
     * parameters.
     * <p>The destination band, buffer and region parameters are exactly the ones passed to the original  call. Since
     * the {@code destOffsetX} and {@code destOffsetY} parameters are already taken into account in the
     * {@code sourceOffsetX} and {@code sourceOffsetY} parameters, an implementor of this method is free to
     * ignore them.
     *
     * @param sourceOffsetX the absolute X-offset in source raster co-ordinates
     * @param sourceOffsetY the absolute Y-offset in source raster co-ordinates
     * @param sourceWidth   the width of region providing samples to be read given in source raster co-ordinates
     * @param sourceHeight  the height of region providing samples to be read given in source raster co-ordinates
     * @param sourceStepX   the sub-sampling in X direction within the region providing samples to be read
     * @param sourceStepY   the sub-sampling in Y direction within the region providing samples to be read
     * @param destBand      the destination band which identifies the data source from which to read the sample values
     * @param destOffsetX   the X-offset in the band's raster co-ordinates
     * @param destOffsetY   the Y-offset in the band's raster co-ordinates
     * @param destWidth     the width of region to be read given in the band's raster co-ordinates
     * @param destHeight    the height of region to be read given in the band's raster co-ordinates
     * @param destBuffer    the destination buffer which receives the sample values to be read
     * @param pm            a monitor to inform the user about progress
     * @throws IOException if an I/O error occurs
     * @see #readBandRasterData
     * @see #getSubsetDef
     */
    virtual void ReadBandRasterDataImpl(int source_offset_x, int source_offset_y, int source_width, int source_height,
                                        int source_step_x, int source_step_y, std::shared_ptr<Band> dest_band,
                                        int dest_offset_x, int dest_offset_y, int dest_width, int dest_height,
                                        const std::shared_ptr<ProductData>& dest_buffer,
                                        std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    template <typename T>
    std::shared_ptr<T> SharedFromBase();

public:
    static void ConfigurePreferredTileSize(const std::shared_ptr<Product>& product);

    std::any GetInput() override;

    /**
     * Returns the subset information with which this data product is read from its physical source.
     *
     * @return the subset information, can be {@code null}
     */
    std::shared_ptr<ProductSubsetDef> GetSubsetDef() override { return subset_def_; };
    void ReadBandRasterData(std::shared_ptr<Band> dest_band, int dest_off_set_x, int dest_offset_y, int dest_width,
                            int dest_height, std::shared_ptr<ProductData> dest_buffer,
                            std::shared_ptr<ceres::IProgressMonitor> pm) override;
    void Close() override;

    /**
     * Checks if this reader ignores metadata or not.
     *
     * @return {@code true} if so
     */
    bool IsMetadataIgnored();

    /**
     * Tests whether or not a product node (a band, a tie-point grid or metadata element) with the given name is
     * accepted with respect to the optional spectral band subset. All accepted nodes will be part of the product read.
     *
     * @param name the node name
     * @return {@code true} if so
     */
    bool IsNodeAccepted(std::string_view name);

    /**
     * Reads a data product and returns a in-memory representation of it.
     * <p> The given subset info can be used to specify spatial and spectral portions of the original product. If the
     * subset is omitted, the complete product is read in.
     * <p> Whether the band data - the actual pixel values - is read in immediately or later when pixels are requested,
     * is up to the implementation.
     *
     * @param input     an object representing a valid output for this product reader, might be a
     *                  {@code ImageInputStream} or other {@code Object} to use for future decoding.
     * @param subsetDef a spectral or spatial subset (or both) of the product. If {@code null}, the entire product
     *                  is read in
     * @throws IllegalArgumentException   if {@code input} is {@code null} or it's type is not one of the
     *                                    supported input sources.
     * @throws IOException                if an I/O error occurs
     * @throws IllegalFileFormatException if the file format is illegal
     */
    std::shared_ptr<Product> ReadProductNodes(std::any input, std::shared_ptr<ProductSubsetDef> subset_def) override;

    //    void ReadTiePointGridRasterData(std::shared_ptr<TiePointGrid> tpg, int dest_offset_x, int dest_offset_y,
    //                                    int dest_width, int dest_height,
    //                                    std::shared_ptr<ProductData> dest_buffer,
    //                                    std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Returns a string representation of the reader.
     *
     * @return a string representation of the object.
     */
    std::string ToString();
};

////////////////////////////////////////////////////////////////////////
/////TEMPLATED IMPLEMENTATION NEEDS TO BE IN THE SAME FILE
////////////////////////////////////////////////////////////////////////
template <typename T>
std::shared_ptr<T> AbstractProductReader::SharedFromBase() {
    return std::dynamic_pointer_cast<T>(shared_from_this());
}

}  // namespace alus::snapengine
