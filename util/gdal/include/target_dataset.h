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

#include <deque>
#include <mutex>
#include <vector>

#include <gdal_priv.h>

#include "alus_file_writer.h"
#include "dataset.h"
#include "raster_properties.h"
#include "shapes.h"

namespace alus {

struct TargetDatasetParams {
    GDALDriver* driver;
    std::string_view filename;
    int band_count;
    bool dataset_per_band;
    RasterDimension dimension{};
    const double* transform;
    const char* projectionRef;
};

template <typename OutputType>
class TargetDataset : public AlusFileWriter<OutputType> {
public:
    explicit TargetDataset(TargetDatasetParams params);  // TODO: any better way to pass confs?  // NOLINT
    TargetDataset() = default;

    /**
     * Writes to dataset using GDAL RasterIO().
     *
     * @param from
     * @param to x and y offset of a raster where data will be written to.
     * @param howMuch When default value (x/width:0, y/height:0) is used a
     *                this->dimensions are applied.
     */
    void WriteRectangle(OutputType* from, Rectangle area, int band_nr) override;

    [[nodiscard]] size_t GetSize() const { return this->dimensions_.columnsX * this->dimensions_.rowsY; }
    [[nodiscard]] RasterDimension GetDimensions() const { return this->dimensions_; }

    [[nodiscard]] std::vector<GDALDataset*> GetDataset() { return datasets_; }
    void ReleaseDataset() { datasets_.clear(); }

    ~TargetDataset() override;

private:
    std::vector<GDALDataset*> datasets_;
    std::deque<std::mutex> mutexes_;
    bool dataset_per_band_;
    GDALDataType gdal_data_type_;
    RasterDimension const dimensions_{};
};
}  // namespace alus