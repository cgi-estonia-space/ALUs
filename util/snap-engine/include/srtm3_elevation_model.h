#pragma once

#include <cstdio>
#include <vector>
#include <sstream>

#include "CudaFriendlyObject.h"
#include "dataset.hpp"
#include "earth_gravitational_model96.h"
#include "pointer_holders.h"
#include "shapes.h"
#include "srtm3_elevation_model_constants.h"
#include "srtm3_formatter.cuh"

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
    std::vector<Point> file_indexes_;
    std::string files_directory_;
    std::string tif_extension_ = ".tif";
    std::string tfw_extension_ = ".tfw";
    std::vector<std::string> file_templates_;

    int nr_of_tiles_;
    std::vector<Dataset> srtms_;
    //use this for cudaFree once you are done with image.
    std::vector<double *> device_srtms_;

    std::vector<DemFormatterData> datas_;

    void ResolveFileNames();
    std::string FormatName(Point coords);

public:
    SRTM3ElevationModel(std::vector<Point> file_indexes, std::string directory);
    ~SRTM3ElevationModel();
    PointerHolder *device_srtm3_tiles_{nullptr};
    std::vector<DemFormatterData> SrtmPlaceholderData(EarthGravitationalModel96 *egm96);

    void ReadSrtmTiles(EarthGravitationalModel96 *egm96);

    void HostToDevice() override ;
    void DeviceToHost() override ;
    void DeviceFree() override ;

};

}//namespace
}//namespace
