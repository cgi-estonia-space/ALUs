/**
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
#include <vector>

#include "snap-core/datamodel/product_data.h"

namespace alus::snapengine {

class ITile {
public:
    /**
     * <p>Obtains access to the underlying raw sample buffer. The data buffer holds the
     * raw (unscaled, uncalibrated) sample data (e.g. detector counts).
     * Elements in this array must be addressed
     * by an index computed via the {@link #getScanlineStride() scanlineStride} and
     * {@link #getScanlineOffset() scanlineOffset} properties.
     * The index can also be directly computed using the  {@link #getDataBufferIndex(int, int)} method.
     * <p>The most efficient way to access and/or modify the samples in the raw data buffer is using
     * the following nested loops:
     * <pre>
     *   int lineStride = tile.{@link #getScanlineStride()};
     *   int lineOffset = tile.{@link #getScanlineOffset()};
     *   for (int y = tile.{@link #getMinY()}; y &lt;= tile.{@link #getMaxY()}; y++) {
     *      int index = lineOffset;
     *      for (int x = tile.{@link #getMinX()}; x &lt;= tile.{@link #getMaxX()}; x++) {
     *           // use index here to access raw data buffer...
     *           index++;
     *       }
     *       lineOffset += lineStride;
     *   }
     * </pre>
     * <p>If the absolute x,y pixel coordinates are not required, the following construct maybe more
     * readable:
     * <pre>
     *   int lineStride = tile.{@link #getScanlineStride()};
     *   int lineOffset = tile.{@link #getScanlineOffset()};
     *   for (int y = 0; y &lt; tile.{@link #getHeight()}; y++) {
     *      int index = lineOffset;
     *      for (int x = 0; x &lt; tile.{@link #getWidth()}; x++) {
     *           // use index here to access raw data buffer...
     *           index++;
     *       }
     *       lineOffset += lineStride;
     *   }
     * </pre>
     *
     * @return the sample data
     */
    //    virtual std::shared_ptr<ProductData>& GetDataBuffer() = 0;

    /**
     * Gets the minimum pixel x-coordinate within the scene covered by the tile's {@link #getRasterDataNode
     * RasterDataNode}.
     *
     * @return The minimum pixel x-coordinate.
     */
    virtual int GetMinX() const = 0;

    /**
     * Gets the maximum pixel x-coordinate within the scene covered by the tile's {@link #getRasterDataNode
     * RasterDataNode}.
     *
     * @return The maximum pixel x-coordinate.
     */
    virtual int GetMaxX() const = 0;

    /**
     * Gets the minimum pixel y-coordinate within the scene covered by the tile's {@link #getRasterDataNode
     * RasterDataNode}.
     *
     * @return The minimum pixel y-coordinate.
     */
    virtual int GetMinY() const = 0;

    /**
     * Gets the maximum pixel y-coordinate within the scene covered by the tile's {@link #getRasterDataNode
     * RasterDataNode}.
     *
     * @return The maximum pixel y-coordinate.
     */
    virtual int GetMaxY() const = 0;

    /**
     * Gets the scanline offset.
     * The scanline offset is the index to the first valid sample element in the data buffer.
     *
     * @return The raster scanline offset.
     *
     * @see #getScanlineStride()
     */
    virtual int GetScanlineOffset() const = 0;

    /**
     * Gets the raster scanline stride for addressing the internal data buffer.
     * The scanline stride is added to the scanline offset in order to compute offsets of subsequent scanlines.
     *
     * @return The raster scanline stride.
     *
     * @see #getScanlineOffset()
     */
    virtual int GetScanlineStride() const = 0;

    virtual std::vector<float>& GetSimpleDataBuffer() = 0;

    virtual ~ITile() = default;
};

}  // namespace alus::snapengine