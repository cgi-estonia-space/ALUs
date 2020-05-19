#pragma once

#include <string>
#include <fstream>

#include "CudaFriendlyObject.hpp"
#include "allocators.hpp"
#include "cuda_util.hpp"


/** EXPLANATION FROM ESA SNAP
 * "WW15MGH.GRD"
 * <p>
 * This file contains 1038961 point values in grid form.  The first row of the file is the "header" of the file
 * and shows the south, north, west, and east limits of the file followed by the grid spacing in n-s and e-w.
 * All values in the "header" are in DECIMAL DEGREES.
 * <p>
 * The geoid undulation grid is computed at 15 arc minute spacings in north/south and east/west with the new
 * "EGM96" spherical harmonic potential coefficient set complete to degree and order 360 and a geoid height
 * correction value computed from a set of spherical harmonic coefficients ("CORRCOEF"), also to degree and
 * order 360.  The file is arranged from north to south, west to east (i.e., the data after the header is
 * the north most latitude band and is ordered from west to east).
 * <p>
 * The coverage of this file is:
 * <p>
 * 90.00 N  +------------------+
 * |                  |
 * | 15' spacing N/S  |
 * |                  |
 * |                  |
 * | 15' spacing E/W  |
 * |                  |
 * -90.00 N  +------------------+
 * 0.00 E           360.00 E
 */
namespace slap {

class EarthGravitationalModel96: public CudaFriendlyObject {
private:
    std::string gridFile = "../test/goods/ww15mgh_b.grd";

    void readGridFile();
public:
    const int NUM_LATS = 721; // 180*4 + 1  (cover 90 degree to -90 degree)
    const int NUM_LONS = 1441; // 360*4 + 1 (cover 0 degree to 360 degree)
    const int NUM_CHAR_PER_NORMAL_LINE = 74;
    const int NUM_CHAR_PER_SHORT_LINE = 11;
    const int NUM_CHAR_PER_EMPTY_LINE = 1;
    const int BLOCK_HEIGHT = 20;
    const int NUM_OF_BLOCKS_PER_LAT = 9;

    const int MAX_LATS = NUM_LATS - 1;
    const int MAX_LONS = NUM_LONS - 1;

    double **egm = nullptr;
    double *deviceEgm = nullptr;

    EarthGravitationalModel96(std::string gridFile);
    EarthGravitationalModel96();
    ~EarthGravitationalModel96();

    void hostToDevice();
    void deviceToHost();
    void deviceFree();

};

}//namespace
