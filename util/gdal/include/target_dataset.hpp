#pragma once

#include <vector>

#include <gdal_priv.h>

#include "dataset.h"
#include "raster_properties.hpp"

namespace alus {

class TargetDataset final {
   public:
    TargetDataset(Dataset& dsWithProperties, std::string_view filename);
    TargetDataset();

    /**
     * Writes to dataset using GDAL RasterIO().
     *
     * @param from
     * @param to x and y offset of a raster where data will be written to.
     * @param howMuch When default value (x/width:0, y/height:0) is used a
     *                this->dimensions are applied.
     */
    void write(std::vector<double>& from, RasterPoint to = {0, 0},
               RasterDimension howMuch = {0, 0});

    [[nodiscard]] size_t getSize() const { return this->dimensions.columnsX *
                                                  this->dimensions.rowsY; }
    [[nodiscard]] RasterDimension getDimensions() const { return
                                                          this->dimensions; }

    ~TargetDataset();

   private:
    GDALDataset* gdalDs;
    RasterDimension const dimensions{};
};
}  // namespace alus