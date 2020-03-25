#include "terrain_correction.hpp"

#include <iostream>

#include "dem.hpp"

namespace slap {

//TerrainCorrection::TerrainCorrection() {}

void TerrainCorrection::doWork() {

    auto gdalDs = m_ds->getGDALDataset();

    //std::cout << "GDAL driver desc: " << gdalDs->GetDriver()->GetDescription() << std::endl;
    // This works not - std::cout << GDAL_DMD_LONGNAME << " " << gdalDs->GetDriver()->GetMetadata(GDAL_DMD_LONGNAME) << std::endl;
    // Same as above - std::cout << "GDAL driver name: " << gdalDs->GetDriverName() << std::endl;
    std::cout << "Input data size " << gdalDs->GetRasterXSize() << "x" << gdalDs->GetRasterYSize()
              << " count " << gdalDs->GetRasterCount() << std::endl;
    //std::cout << "projection ref " << gdalDs->GetProjectionRef() << std::endl;
    // Same as above - std::cout << "GCP proj " << gdalDs->GetGCPProjection() << std::endl;

    slap::Dem dem{{"/home/sven/.snap/auxdata/dem/SRTM 3Sec/srtm_41_01.tif"}};
    dem.doWork();
}

TerrainCorrection::~TerrainCorrection() {}
}

