#pragma once

#include <string_view>
#include <vector>

#include <gdal_priv.h>

#include "i_data_tile_read_write_base.h"
#include "i_data_tile_reader.h"
#include "tile.h"

namespace alus {
class GdalTileReader : public IDataTileReader {
public:
    GdalTileReader(std::string_view file_name, std::vector<int> band_map, int band_count, bool has_transform);
    GdalTileReader(const GdalTileReader&) = delete;
    GdalTileReader& operator=(const GdalTileReader&) = delete;
    ~GdalTileReader() override;

    void ReadTile(const Tile& tile) override;
    void CleanBuffer();
    void CloseDataSet();
    [[nodiscard]] float* GetData() const override;
    [[nodiscard]] int GetBandXSize() const override { return band_x_size_; }
    [[nodiscard]] int GetBandYSize() const override { return band_y_size_; }
    [[nodiscard]] int GetBandXMin() const override { return band_x_min_; }
    [[nodiscard]] int GetBandYMin() const override { return band_y_min_; }

    [[nodiscard]] const std::string_view GetDataProjection() const override;
    [[nodiscard]] std::vector<double> GetGeoTransform() const override;
    [[nodiscard]] double GetValueAtXy(int x, int y) const override;
    // todo:  void ReadTileToTensors(const IDataTileIn &tile) override;
private:
    void AllocateForTileData(const Tile& tile);

    GDALDataset* dataset_{};
    float* data_{};
};
}  // namespace alus
