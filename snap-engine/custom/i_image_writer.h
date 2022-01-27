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

#include "custom/rectangle.h"

namespace alus::snapengine::custom {

// Placeholder interface for image readers, currently only viable solution is GDAL anyway.
class IImageWriter {
public:
    IImageWriter() = default;
    IImageWriter(const IImageWriter&) = delete;
    IImageWriter& operator=(const IImageWriter&) = delete;
    virtual ~IImageWriter() = default;

    /***
     * Just to provide interface between different implementations of band data readers e.g gdal RasterIO
     *
     * @param 2D rectangle area to be read from data
     * @param data container into which data will be placed
     */
    virtual void WriteSubSampledData(const custom::Rectangle& rectangle, std::vector<float>& data, int band_indx) = 0;
    virtual void WriteSubSampledData(const alus::Rectangle& rectangle, std::vector<float>& data, int band_indx) = 0;

    //    todo: virtual and generics do not match, might want to go through ProductData, currently not important to
    //    support more than float virtual void WriteSubSampledData(const std::shared_ptr<custom::Rectangle>& rectangle,
    //                                    std::vector<int32_t>& data) = 0;

    virtual void Open(std::string_view path_to_band_file, int raster_size_x, int raster_size_y,
                      std::vector<double> affine_geo_transform_out, std::string_view data_projection_out,
                      bool in_memory_file) = 0;

    /**
     * close dataset
     */
    virtual void Close() = 0;
};
}  // namespace alus::snapengine::custom