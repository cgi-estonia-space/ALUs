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

#include <cstdint>
#include <memory>
#include <vector>

#include <gdal_priv.h>

#include "custom/i_image_reader.h"
#include "custom/rectangle.h"

namespace alus {
namespace snapengine {
namespace custom {

// Placeholder interface for image readers, currently only viable solution is GDAL anyway.
class GdalImageReader : virtual public IImageReader {
private:
    GDALDataset* dataset_{};

public:
    /**
     * avoid tight coupling to data... (swappable sources, targets, types etc..)
     */
    GdalImageReader();

    /**
     * set input path for source to read from
     * @param path_to_band_file
     */
    void SetInputPath(std::string_view path_to_band_file) override;
    /**
     * Make sure std::vector data has correct size before using gdal to fill it
     * if it has wrong size it gets resized
     * @param rectangle
     * @param data
     */
    void ReadSubSampledData(const std::shared_ptr<custom::Rectangle>& rectangle, std::vector<int32_t>& data) override;

    ~GdalImageReader();
};
}  // namespace custom
}  // namespace snapengine
}  // namespace alus