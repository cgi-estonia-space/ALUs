#pragma once
#include "gdal_priv.h"
#include "cpl_conv.h" // for CPLMalloc()

struct AlgoData{
    GDALRasterBand *outputBand;
    GDALRasterBand *inputBand;
    float min;
    float max;
    int tileX;
    int tileY;
    int tileXo;
    int tileYo;
    int tileXa;
    int tileYa;
    int rasterX;
    int rasterY;
    float *buffer;
};
