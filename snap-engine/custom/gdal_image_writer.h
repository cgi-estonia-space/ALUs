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

#include "custom/i_image_writer.h"
#include "custom/rectangle.h"

namespace alus::snapengine::custom {

// Placeholder interface for image readers, currently only viable solution is GDAL anyway.
class GdalImageWriter : public IImageWriter {
private:
    GDALDataset* dataset_{};

public:
    /**
     * avoid tight coupling to data... (swappable sources, targets, types etc..)
     */
    GdalImageWriter() = default;
    GdalImageWriter(const GdalImageWriter&) = delete;
    GdalImageWriter& operator=(const GdalImageWriter&) = delete;
    ~GdalImageWriter() override;

    /**
     * set input path for source to read from
     * @param path_to_band_file
     */
    void Open(std::string_view path_to_band_file, int raster_size_x, int raster_size_y,
              std::vector<double> affine_geo_transform_out, std::string_view data_projection_out,
              bool in_memory_file) override;

    // TODO internal dataset exposed to work together with different GDAL wrappers in the project  // NOLINT
    GDALDataset* GetDataset() { return dataset_; }
    void ReleaseDataset() { dataset_ = nullptr; }

    // todo: add support for subsampling
    void WriteSubSampledData(const custom::Rectangle& rectangle, std::vector<float>& data, int band_index) override;
    void WriteSubSampledData(const alus::Rectangle& rectangle, std::vector<float>& data, int band_index) override;

    void Close() override;
};
}  // namespace alus::snapengine::custom