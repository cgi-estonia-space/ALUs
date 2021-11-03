/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.io.SARReader.java
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

#include <any>
#include <array>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <boost/filesystem/path.hpp>

#include "snap-core/core/dataio/abstract_product_reader.h"
#include "snap-core/core/dataio/i_product_reader_plug_in.h"

namespace alus::ceres {
class IProgressMonitor;
}

namespace alus::snapengine {
class Band;
class Product;
class ProductData;
}  // namespace alus::snapengine

namespace alus::s1tbx {

/**
 * Common functions for readers
 */
class SARReader : public snapengine::AbstractProductReader {
public:
    static void CreateVirtualIntensityBand(const std::shared_ptr<snapengine::Product>& product,
                                           const std::shared_ptr<snapengine::Band>& band, std::string_view count_str);
    static std::string FindPolarizationInBandName(std::string_view band_name);
    static void DiscardUnusedMetadata(std::shared_ptr<snapengine::Product> product);
    void HandleReaderException(const std::exception& e);
    static bool CheckIfCrossMeridian(std::vector<float> longitude_list);
    /**
     * Returns a <code>Path</code> if the given input is a <code>String</code> or <code>File</code>,
     * otherwise it returns null;
     *
     * @param input an input object of unknown type
     * @return a <code>Path</code> or <code>null</code> it the input can not be resolved to a <code>Path</code>.
     */
    static boost::filesystem::path GetPathFromInput(const std::any& input);

protected:
    explicit SARReader(const std::shared_ptr<snapengine::IProductReaderPlugIn>& reader_plug_in);
    //    todo: intention is to override abstract method with abstract method like in original
    virtual std::shared_ptr<alus::snapengine::Product> ReadProductNodesImpl() override = 0;
    virtual void ReadBandRasterDataImpl(int source_offset_x, int source_offset_y, int source_width, int source_height,
                                        int source_step_x, int source_step_y,
                                        std::shared_ptr<snapengine::Band> dest_band, int dest_offset_x,
                                        int dest_offset_y, int dest_width, int dest_height,
                                        const std::shared_ptr<snapengine::ProductData>& dest_buffer,
                                        std::shared_ptr<ceres::IProgressMonitor> pm) override = 0;

    static void SetQuicklookBandName(const std::shared_ptr<snapengine::Product>& product);
    void AddQuicklook(const std::shared_ptr<snapengine::Product>& product, std::string_view name,
                      const boost::filesystem::path& ql_file);
    void AddCommonSARMetadata(const std::shared_ptr<snapengine::Product>& product);

private:
    static constexpr std::array<std::string_view, 5> ELEMS_TO_KEEP = {
        "Abstracted_Metadata", "MAIN_PROCESSING_PARAMS_ADS", "DSD", "SPH", "lutSigma"};
    static void RemoveUnusedMetadata(const std::shared_ptr<snapengine::MetadataElement>& root);
};

}  // namespace alus::s1tbx
