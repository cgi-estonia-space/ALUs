#pragma once

#include <string_view>
#include <vector>

#include "i_data_tile_read_write_base.h"
#include "tile.h"

/**
 * reads I tiles from tiles
 */
namespace alus {
class IDataTileReader : public IDataTileReadWriteBase {
   public:
    IDataTileReader() = delete;
    // todo:check if this works like expected
    IDataTileReader(std::string_view file_name, std::vector<int> band_map, int &band_count, bool has_transform)
        : IDataTileReadWriteBase(file_name, band_map, band_count) {}
    virtual void ReadTile(const Tile &tile) = 0;
    [[nodiscard]] virtual float *GetData() const = 0;
    [[nodiscard]] virtual int GetBandXSize() const = 0;
    [[nodiscard]] virtual int GetBandYSize() const = 0;
    [[nodiscard]] virtual int GetBandXMin() const = 0;
    [[nodiscard]] virtual int GetBandYMin() const = 0;
    [[nodiscard]] virtual const std::string_view GetDataProjection() const = 0;
    [[nodiscard]] virtual std::vector<double> GetGeoTransform() const = 0;
    [[nodiscard]] virtual double GetValueAtXy(int x, int y) const = 0;
    virtual ~IDataTileReader() = default;
};
}  // namespace alus