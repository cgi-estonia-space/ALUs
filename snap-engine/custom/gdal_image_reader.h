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

namespace alus::snapengine::custom {

// Placeholder interface for image readers, currently only viable solution is GDAL anyway.
class GdalImageReader : virtual public IImageReader {
private:
    GDALDataset* dataset_{};
    std::string data_projection_;
    std::vector<double> affine_geo_transform_;
    std::string file_;

    // todo: might want to avoid internal buffer and attach this to data (more flexibility with types)
    std::vector<float> data_{};

    void InitializeDatasetProperties(GDALDataset* dataset, bool has_transform, bool has_correct_proj);

public:
    /**
     * avoid tight coupling to data... (swappable sources, targets, types etc..)
     */
    GdalImageReader() = default;
    GdalImageReader(const GdalImageReader&) = delete;
    GdalImageReader& operator=(const GdalImageReader&) = delete;
    ~GdalImageReader() override;

    /**
     * set input path for source to read from
     * @param path_to_band_file
     */
    void Open(std::string_view path_to_band_file, bool has_transform, bool has_correct_proj) override;

    void TakeExternalDataset(GDALDataset* dataset);

    /**
     * Make sure std::vector data has correct size before using gdal to fill it
     * if it has wrong size it gets resized
     * @param rectangle
     * @param data
     */
    void ReadSubSampledData(const std::shared_ptr<custom::Rectangle>& rectangle, std::vector<int32_t>& data) override;

    /**
     * different use case, might want to generalize at some point
     * @param rectangle
     * @param band_indx
     */
    void ReadSubSampledData(const custom::Rectangle& rectangle, int band_indx) override;
    void ReadSubSampledData(const alus::Rectangle& rectangle, int band_indx) override;
    [[nodiscard]] std::string GetDataProjection() const;
    [[nodiscard]] std::vector<double> GetGeoTransform() const;
    [[nodiscard]] const std::vector<float>& GetData() const override { return data_; }
    void Close() override;
    void ReleaseDataset();
};
}  // namespace alus::snapengine::custom