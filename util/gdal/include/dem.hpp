#pragma once

#include <vector>

#include "dataset.hpp"

namespace alus {
class Dem {
   public:
    Dem(Dataset ds);

    /**
     * This is a ripoff of a Sentinel 1 Toolbox's code from
     * DEMFactory::getLocalDem() with some functionality stripped off.
     *
     * Original function signature in Sentinel 1 toolbox is following:
     * @code{.java}
     * public static boolean getLocalDEM(final ElevationModel dem,
     *                            final double demNoDataValue,
     *                            final String demResamplingMethod,
     *                            final TileGeoreferencing tileGeoRef,
     *                            final int x_0, final int y_0,
     *                            final int tileWidth, final int tileHeight,
     *                            final Product sourceProduct,
     *                            final boolean nodataValueAtSea,
     *                            final double[][] localDEM) throws Exception
     * @endcode
     * Where:
     * ElevationModel dem is not used here per se, because this class
     * kinda represents that.
     * demNoDataValue ??
     * demResamplingMethod not used here since this class by default supports
     * only bilinear resampling method.
     * tileGeoRef ??
     * x_0 and y_0 ??
     * tileWidth and tileHeight ??
     * sourceProduct Not used.
     * nodataValueAtSea Not supporting this.
     * localDEM 2D array is a return product of this procedure.
     *
     * @image
     * @return 2D array of the elevation values for a specified area.
     */
    std::vector<double> GetLocalDemFor(Dataset& image, unsigned int x_0,
                                       unsigned int y_0, unsigned int width,
                                       unsigned int height);

    auto const& GetData() const { return m_ds_.GetDataBuffer(); }
    void FillGeoTransform(double& origin_lon, double& origin_lat,
                          double& pixel_size_lon, double& pixel_size_lat) const {
        origin_lon = m_ds_.GetOriginLon();
        origin_lat = m_ds_.GetOriginLat();
        pixel_size_lon = m_ds_.GetPixelSizeLon();
        pixel_size_lat = m_ds_.GetPixelSizeLat();
    }

    int GetRasterSizeX() const { return m_ds_.GetRasterSizeX(); }
    int GetRasterSizeY() const { return m_ds_.GetRasterSizeY(); }
    int GetColumnCount() const { return GetRasterSizeX(); }
    int GetRowCount() const { return GetRasterSizeY(); }

   private:
    Dataset m_ds_;

    double m_no_data_value_;
};
}  // namespace alus
