#pragma once

#include <string_view>
#include <vector>

#include <gdal_priv.h>

#include "gdal_util.hpp"
#include "i_data_tile_writer.h"
#include "tile.h"

namespace alus {
class GdalTileWriter : virtual public IDataTileWriter {
   private:
    GDALDataset *output_dataset_{};

   public:
    GdalTileWriter(std::string_view file_name,
                   std::vector<int> band_map,
                   int &band_count,
                   const int &band_x_size,
                   const int &band_y_size,
                   int band_x_min,
                   int band_y_min,
                   std::vector<double> affine_geo_transform_out,
                   std::string_view data_projection_out);
    void WriteTile(const Tile &tile, void *tile_data) override;
    void CloseDataSet();
    ~GdalTileWriter() override;
};
}  // namespace alus
