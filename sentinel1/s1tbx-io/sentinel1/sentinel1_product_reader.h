/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.sentinel1.Sentinel1ProductReader.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
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
#include <vector>

#include "s1tbx-commons/io/sar_reader.h"
#include "snap-core/core/dataio/i_product_reader_plug_in.h"

namespace alus::ceres {
class IProgressMonitor;
}

namespace alus::snapengine {
class Band;
class ProductData;
}  // namespace alus::snapengine

namespace alus::s1tbx {

class ISentinel1Directory;
class DataCache;
class Sentinel1Directory;
class BandInfo;

/**
 * The product reader for Sentinel1 products.
 */
class Sentinel1ProductReader : public SARReader {
private:
    std::shared_ptr<DataCache> cache_;
    bool use_cache_ = false;

    boost::filesystem::path GetQuicklookFile();

    void ReadSLCRasterBand(int source_offset_x, int source_offset_y, int source_step_x, int source_step_y,
                           const std::shared_ptr<snapengine::ProductData>& dest_buffer, int dest_offset_x,
                           int dest_offset_y, int dest_width, int dest_height,
                           const std::shared_ptr<BandInfo>& band_info);

    /**
     * Uses provided reader (from band_info->imageiofile->reader) to carry out actual io operation for reading band data
     * provided parameters are mostly due to fact that esa snap supports downsampling
     *
     * @param band_info
     * @param source_offset_x
     * @param source_offset_y
     * @param source_step_x
     * @param source_step_y
     * @param dest_rect
     * @return std::vector which holds data
     */
    std::vector<int32_t> ReadRect(const std::shared_ptr<BandInfo>& band_info, int source_offset_x, int source_offset_y,
                                  int source_step_x, int source_step_y,
                                  const std::shared_ptr<snapengine::custom::Rectangle>& dest_rect);

protected:
    std::shared_ptr<ISentinel1Directory> data_dir_ = nullptr;
    // todo: provide plugin functionality
    Sentinel1ProductReader();

    /**
     * Provides an implementation of the <code>readProductNodes</code> interface method. Clients implementing this
     * method can be sure that the input object and eventually the subset information has already been set.
     * <p>
     * <p>This method is called as a last step in the <code>readProductNodes(input, subsetInfo)</code> method.
     *
     * @throws java.io.IOException if an I/O error occurs
     */
    std::shared_ptr<snapengine::Product> ReadProductNodesImpl() override;

    /**
     * {@inheritDoc}
     */
    //     todo:!! must be thread safe, can be called in any order!!
    void ReadBandRasterDataImpl(int source_offset_x, int source_offset_y, [[maybe_unused]] int source_width,
                                [[maybe_unused]] int source_height, int source_step_x, int source_step_y,
                                std::shared_ptr<snapengine::Band> dest_band, int dest_offset_x, int dest_offset_y,
                                int dest_width, int dest_height,
                                const std::shared_ptr<snapengine::ProductData>& dest_buffer,
                                [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override;

public:
    /**
     * Constructs a new abstract product reader.
     *
     * @param readerPlugIn the reader plug-in which created this reader, can be <code>null</code> for internal reader
     *                     implementations
     */
    explicit Sentinel1ProductReader(const std::shared_ptr<snapengine::IProductReaderPlugIn>& reader_plug_in);

    /**
     * Closes the access to all currently opened resources such as file input streams and all resources of this children
     * directly owned by this reader. Its primary use is to allow the garbage collector to perform a vanilla job.
     * <p>
     * <p>This method should be called only if it is for sure that this object instance will never be used again. The
     * results of referencing an instance of this class after a call to <code>close()</code> are undefined.
     * <p>
     * <p>Overrides of this method should always call <code>super.close();</code> after disposing this instance.
     *
     * @throws java.io.IOException if an I/O error occurs
     */

    void Close() override;
};
}  // namespace alus::s1tbx
