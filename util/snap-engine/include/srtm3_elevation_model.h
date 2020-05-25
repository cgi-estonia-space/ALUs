#pragma once

#include <cstdio>
#include <vector>
#include <sstream>

#include "srtm3_elevation_model_constants.h"
#include "shapes.h"
#include "CudaFriendlyObject.hpp"
#include "earth_gravitational_model96.h"
#include "srtm3_formatter.cuh"
#include "pointer_holders.h"
#include "dataset.hpp"


namespace alus{
namespace snapengine{


/*
    Majority of this class is from SRTM3ElevationModel.java and BaseElevationModel.java.
    But this also encorporates parts from SRTM3GeoTiffTile.java, SRTM3GeoTiffFile.java and their base classes,
    as it made sense to pile all of this together due to having to be available from the gpu.

    This class is given the tile coordinates using a map and then it finds those tiles, formats them and
    makes them available on the gpu.
*/
class SRTM3ElevationModel: public alus::cuda::CudaFriendlyObject{
private:
    std::vector<Point> fileIndexes_;
    std::string filesDirectory_;
    std::string tifExtension_ = ".tif";
    std::string tfwExtension_ = ".tfw";
    std::vector<std::string> fileTemplates_;

    int nrOfTiles_;
    std::vector<Dataset> srtms_;
    //use this for cudaFree once you are done with image.
    std::vector<double *> deviceSrtms_;

    std::vector<DemFormatterData> datas_;

    void ResolveFileNames();
    std::string FormatName(Point coords);

public:
    SRTM3ElevationModel(std::vector<Point> fileCoords, std::string directory);
    ~SRTM3ElevationModel();
    PointerHolder *deviceSrtm3Tiles_{nullptr};
    std::vector<DemFormatterData> srtmPlaceholderData(EarthGravitationalModel96 *egm96);

    void ReadSrtmTiles(EarthGravitationalModel96 *egm96);

    void hostToDevice();
    void deviceToHost();
    void deviceFree();

};

}//namespace
}//namespace
