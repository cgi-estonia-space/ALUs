#pragma once

struct LocalDemKernelArgs {
    int demCols;
    int demRows;
    int targetCols;
    int targetRows;
    double demPixelSizeLon;
    double demPixelSizeLat;
    double demOriginLon;
    double demOriginLat;
    double targetPixelSizeLon;
    double targetPixelSizeLat;
    double targetOriginLon;
    double targetOriginLat;
};

void runElevationKernel(double const* dem, double* targetElevations,
                        LocalDemKernelArgs const args);