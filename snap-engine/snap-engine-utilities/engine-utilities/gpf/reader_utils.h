/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.ReaderUtils.java
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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <boost/filesystem.hpp>

namespace alus::snapengine {

class Product;
class Band;
class MetadataElement;
class Utc;

/**
 * Common functions for readers
 */
class ReaderUtils {
private:
    static std::string CreateName(std::string_view orig_name, std::string_view new_prefix);
    static std::string CreateValidUTCString(std::string_view name, std::vector<char> valid_chars, char replace_char);

public:
    static void AddMetadataIncidenceAngles(const std::shared_ptr<Product>& product);
    static void AddMetadataProductSize(const std::shared_ptr<Product>& product);
    static int GetTotalSize(const std::shared_ptr<Product>& product);
    static std::shared_ptr<Band> CreateVirtualIntensityBand(const std::shared_ptr<Product>& product,
                                                            const std::shared_ptr<Band>& band_i,
                                                            const std::shared_ptr<Band>& band_q,
                                                            std::string_view suffix);
    static std::shared_ptr<Band> CreateVirtualIntensityBand(const std::shared_ptr<Product>& product,
                                                            const std::shared_ptr<Band>& band_i,
                                                            const std::shared_ptr<Band>& band_q,
                                                            std::string_view band_name, std::string_view suffix);
    static std::shared_ptr<Band> CreateVirtualPhaseBand(const std::shared_ptr<Product>& product,
                                                        const std::shared_ptr<Band>& band_i,
                                                        const std::shared_ptr<Band>& band_q,
                                                        std::string_view count_str);

    static void AddGeoCoding(const std::shared_ptr<Product>& product, const std::vector<float>& lat_corners,
                             const std::vector<float>& lon_corners);

    static void CreateFineTiePointGrid(int coarse_grid_width, int coarse_grid_height, int fine_grid_width,
                                       int fine_grid_height, std::vector<float> coarse_tie_points,
                                       std::vector<float> fine_tie_points);

    static std::shared_ptr<Utc> GetTime(const std::shared_ptr<snapengine::MetadataElement>& elem, std::string_view tag,
                                        std::string_view time_format);

    /**
     * Returns a <code>Path</code> if the given input is a <code>String</code> or <code>File</code>,
     * otherwise it returns null;
     *
     * @param input an input object of unknown type
     * @return a <code>Path</code> or <code>null</code> it the input can not be resolved to a <code>Path</code>.
     */
    static std::optional<boost::filesystem::path> GetPathFromInput(const std::any& input);
};

}  // namespace alus::snapengine
