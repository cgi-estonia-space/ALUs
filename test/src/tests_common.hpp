#pragma once

#include <string>

#include <cpl_error.h>

namespace slap::tests {

// DEM file that was downloaded by SNAP when doing range doppler terrain
// correction on Saaremaa test file.
std::string const DEM_PATH_1{"goods/srtm_41_01.tif"};

/**
 * Output from gdalinfo.
 * Upper Left  (  22.2362770,  58.3731210) ( 22d14'10.60"E, 58d22'23.24"N)
 * Lower Left  (  22.2362770,  58.3600208) ( 22d14'10.60"E, 58d21'36.08"N)
 * Upper Right (  22.2388538,  58.3731210) ( 22d14'19.87"E, 58d22'23.24"N)
 * Lower Right (  22.2388538,  58.3600208) ( 22d14'19.87"E, 58d21'36.08"N)
 * Center      (  22.2375654,  58.3665709) ( 22d14'15.24"E, 58d21'59.66"N)
 */
std::string const TIF_PATH_1{"./goods/karujarve_kallas.tif"};

// Test set consisting part of the Saaremaa.
std::string const COH_1_TIF{
    "./goods/"
    "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_"
    "Stack_coh_deb.tif"};
std::string const COH_1_DATA{
    "./goods/"
    "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_"
    "Stack_coh_deb.data"};

void silentGdalErrorHandler(CPLErr, CPLErrorNum, const char*);

}  // namespace slap::tests
