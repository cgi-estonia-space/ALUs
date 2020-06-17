#pragma once

#include <string_view>
#include <vector>

#include "i_data_tile_read_write_base.h"
#include "tile.h"

namespace alus {
class IDataTileWriter : public IDataTileReadWriteBase {
   public:
    IDataTileWriter() = delete;
    IDataTileWriter(const std::string_view file_name,
                    std::vector<int> band_map,
                    int &band_count,
                    const int &band_x_size,
                    const int &band_y_size,
                    int band_x_min,
                    int band_y_min,
                    std::vector<double> affine_geo_transform_out,
                    const std::string_view data_projection_out)
        : IDataTileReadWriteBase(file_name,
                                 band_map,
                                 band_count,
                                 band_x_size,
                                 band_y_size,
                                 band_x_min,
                                 band_y_min,
                                 data_projection_out,
                                 affine_geo_transform_out) {}
    virtual void WriteTile(const Tile &tile, void *tile_data) = 0;
    virtual ~IDataTileWriter() = default;
};
}  // namespace alus