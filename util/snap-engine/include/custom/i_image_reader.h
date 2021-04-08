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

#include "custom/rectangle.h"

namespace alus {
namespace snapengine {
namespace custom {

// Placeholder interface for image readers, currently only viable solution is GDAL anyway.
class IImageReader {
public:
    IImageReader() = default;
    IImageReader(const IImageReader&) = delete;
    IImageReader& operator=(const IImageReader&) = delete;
    virtual ~IImageReader() = default;
    /***
     * Just to provide interface between different implementations of band data readers e.g gdal RasterIO
     *
     * @param 2D rectangle area to be read from data
     * @param data container into which data will be placed
     */
    virtual void ReadSubSampledData(const std::shared_ptr<custom::Rectangle>& rectangle,
                                    std::vector<int32_t>& data) = 0;

    virtual void SetInputPath(std::string_view path_to_band_file) = 0;
};
}  // namespace custom
}  // namespace snapengine
}  // namespace alus