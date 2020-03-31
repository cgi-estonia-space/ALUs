#pragma once

#include <vector>

#include "dataset.hpp"

namespace slap {
class Dem {
   public:
    Dem(Dataset ds);

    void doWork();

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
     *                            final int x0, final int y0,
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
     * x0 and y0 ??
     * tileWidth and tileHeight ??
     * sourceProduct Not used.
     * nodataValueAtSea Not supporting this.
     * localDEM 2D array is a return product of this procedure.
     *
     * @image
     * @return 2D array of the elevation values for a specified area.
     */
    std::vector<double> getLocalDemFor(Dataset& image, unsigned int x0,
                                       unsigned int y0, unsigned int width,
                                       unsigned int height);

   private:
    Dataset m_ds;

    double m_noDataValue;
};
}  // namespace slap