#include "dataset.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

void printHelp() {
    std::cout << "Usage:" << std::endl
              << "./map_cut {lon} {lat} {from file} {to file}" << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printHelp();
        return 1;
    }

    try {
        auto const lon = std::stod(argv[1]);
        auto const lat = std::stod(argv[2]);
        std::string const from{argv[3]};
        std::string const dest{argv[4]};
        alus::Dataset inDataset{from.c_str()};
        auto inGdalDataset = inDataset.GetGdalDataset();

        std::string const driverFormat{"GTiff"};
        GDALDriver *outDriver;
        char **papszMetadata;
        outDriver =
            GetGDALDriverManager()->GetDriverByName(driverFormat.c_str());
        if (outDriver == NULL) {
            std::cerr << "driver is null" << std::endl;
            return 2;
        }
        papszMetadata = outDriver->GetMetadata();
        if (!CSLFetchBoolean(papszMetadata, GDAL_DCAP_CREATE, FALSE)) {
            std::cerr << "driver does not support create()" << std::endl;
            return 3;
        }

        constexpr int OUT_WIDTH = 100;   // X size.
        constexpr int OUT_HEIGHT = 100;  // Y size.
        GDALDataset *outputDataset;
        char **outputOptions = NULL;
        outputDataset = outDriver->Create(dest.c_str(), OUT_WIDTH, OUT_HEIGHT,
                                          1, GDT_Float32, outputOptions);
        if (outputDataset == NULL) {
            std::cerr << "output dataset can not be formed" << std::endl;
            return 4;
        }

        auto const inGeoTransform = inDataset.GetTransform();
        std::array const geoTransform{
            lon, inGeoTransform[1], inGeoTransform[2],
            lat, inGeoTransform[4], inGeoTransform[5]};
        // If this tries to modify it, damned it be.
        outputDataset->SetGeoTransform(
            const_cast<double *>(geoTransform.data()));
        outputDataset->SetProjection(inGdalDataset->GetProjectionRef());

        std::array<float, OUT_WIDTH * OUT_HEIGHT> rasterBuffer;

        auto const origin = inDataset.GetPixelIndexFromCoordinates(lon, lat);
        auto const inError = inGdalDataset->GetRasterBand(1)->RasterIO(
            GF_Read, std::get<0>(origin), std::get<1>(origin), OUT_WIDTH,
            OUT_HEIGHT, rasterBuffer.data(), OUT_WIDTH, OUT_HEIGHT, GDT_Float32,
            0, 0);
        if (inError != CE_None) {
            std::cout << "Error when reading data in." << std::endl;
            return 5;
        }

        auto const outError = outputDataset->GetRasterBand(1)->RasterIO(
            GF_Write, 0, 0, OUT_WIDTH, OUT_HEIGHT, rasterBuffer.data(),
            OUT_WIDTH, OUT_HEIGHT, GDT_Float32, 0, 0);
        if (outError != CE_None) {
            std::cout << "Error when reading data in." << std::endl;
            return 5;
        }

        GDALClose((GDALDatasetH)outputDataset);

    } catch (std::runtime_error const &e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}