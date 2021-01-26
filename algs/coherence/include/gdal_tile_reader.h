#pragma once

#include <string_view>
#include <vector>

#include <gdal_priv.h>

#include "gdal_util.h"
#include "i_data_tile_read_write_base.h"
#include "i_data_tile_reader.h"
#include "tile.h"

namespace alus {
class GdalTileReader : virtual public IDataTileReader {
   private:
    GDALDataset *dataset_{};
    float *data_{};

    void AllocateForTileData(const Tile &tile);

   public:
    GdalTileReader(std::string_view file_name, std::vector<int> band_map, int &band_count, bool has_transform);
    void ReadTile(const Tile &tile) override;
    void CleanBuffer();
    void CloseDataSet();
    [[nodiscard]] float *GetData() const override;
    [[nodiscard]] int GetBandXSize() const override { return band_x_size_; }
    [[nodiscard]] int GetBandYSize() const override { return band_y_size_; }
    [[nodiscard]] int GetBandXMin() const override { return band_x_min_; }
    [[nodiscard]] int GetBandYMin() const override { return band_y_min_; }

    [[nodiscard]] const std::string_view GetDataProjection() const override;
    [[nodiscard]] std::vector<double> GetGeoTransform() const override;
    [[nodiscard]] double GetValueAtXy(int x, int y) const override;
    ~GdalTileReader() override;
    // todo:  void ReadTileToTensors(const IDataTileIn &tile) override;
};
}  // namespace alus
