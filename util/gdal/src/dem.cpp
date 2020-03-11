#include "dem.hpp"

#include <iostream>

namespace slap {

Dem::Dem(slap::Dataset ds) : m_ds{std::move(ds)} {
    //    auto gdalDs = m_ds.getGDALDataset();

    //    std::cout << "DEM" << std::endl;
    //    std::cout << "GDAL driver desc: " <<
    //    gdalDs->GetDriver()->GetDescription() << std::endl; std::cout <<
    //    "Raster dim " << gdalDs->GetRasterXSize() << "x" <<
    //    gdalDs->GetRasterYSize()
    //              << " count " << gdalDs->GetRasterCount() << std::endl;
    //    std::cout << "projection ref " << gdalDs->GetProjectionRef() <<
    //    std::endl;
    //
    //    auto rsb = gdalDs->GetRasterBand(1);
    //    auto const scale = rsb->GetScale();
    //    double minMax[2];
    //    rsb->ComputeRasterMinMax(0, minMax);
    //    std::cout << "Dem scale " << scale << " min " << minMax[0] << " max "
    //    << minMax[1] << std::endl;

    //    double adfGeoTransform[6];
    //    if (gdalDs->GetGeoTransform(adfGeoTransform) != CE_None)
    //    {
    //        throw DatasetError(
    //            CPLGetLastErrorMsg(),
    //            gdalDs->GetFileList()[0], CPLGetLastErrorNo());
    //    }
    //
    //    std::cout << "Origin lat, lon (" << adfGeoTransform[3] << " " <<
    //    adfGeoTransform[0] << ")" << std::endl; std::cout << "Pixel size (" <<
    //    adfGeoTransform[1] << "," << adfGeoTransform[5] << ")" << std::endl;
    //    std::cout << "Other items " << adfGeoTransform[2] << " " <<
    //    adfGeoTransform[4] << std::endl;

    //    m_noDataValue = rsb->GetNoDataValue();
}

void Dem::doWork() {
    std::cout << "Dem file origin " << m_ds.getOriginLon() << " "
              << m_ds.getOriginLat() << std::endl;
    auto const index =
        m_ds.getPixelIndexFromCoordinates(25.0004172, 54.9995839);
    std::cout
        << "Lower right pixel (25.0004172, 54.9995839) index from coordinates "
        << std::get<0>(index) << " " << std::get<1>(index) << std::endl;
    auto const coordinates = m_ds.getPixelCoordinatesFromIndex(6001, 6001);
    std::cout << "lower right pixel (6001, 6001) coordinates "
              << std::get<0>(coordinates) << " " << std::get<1>(coordinates)
              << std::endl;
}
}  // namespace slap