/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */

#include "gdal_management.h"

#include <string_view>

#include <gdal_frmts.h>
#include <gdal_priv.h>
#include <ogrsf_frmts.h>

#include "alus_log.h"

namespace {

const alus::gdalmanagement::ErrorCallback* custom_error_handle{nullptr}; // NOSONAR

void AssignErrorHandle(const alus::gdalmanagement::ErrorCallback* callback) { custom_error_handle = callback; }

void ClearErrorHandle(const alus::gdalmanagement::ErrorCallback* callback) {
    if (custom_error_handle == callback) {  // Do not disable if it is not the same callback active anymore.
        AssignErrorHandle(nullptr);
    }
}

void GdalErrorHandle(CPLErr level, CPLErrorNum errorNum, const char* message) {
    if (custom_error_handle && *custom_error_handle) {
        custom_error_handle->operator()(message);
        return;
    }

    switch (level) {
        case CE_None:
            LOGV << "GDAL message(" << errorNum << ") - " << message;
            break;
        case CE_Debug:
            LOGD << "GDAL debug(" << errorNum << ") - " << message;
            break;
        case CE_Warning:
            LOGW << "GDAL warning(" << errorNum << ") - " << message;
            break;
        case CE_Failure:
            LOGE << "GDAL failure(" << errorNum << ") - " << message;
            break;
        case CE_Fatal:
            LOGE << "GDAL fatal(" << errorNum << ") - " << message;
            break;
        default:
            LOGE << "GDAL unknown level(" << errorNum << ") - " << message;
            break;
    }
}
}  // namespace

namespace alus::gdalmanagement {

void Initialize() {
    CPLSetErrorHandler(GdalErrorHandle);
    // Driver registration example from https://github.com/OSGeo/gdal/blob/master/frmts/gdalallregister.cpp
    GDALRegister_GTiff();
    GDALRegister_VRT();
    GDALRegister_MEM();
    GDALRegister_ENVI();  // ENVI .img and .hdr
    GDALRegister_JP2OpenJPEG();
    GDALRegister_SENTINEL2();
    GDALRegister_netCDF();
    RegisterOGRSQLite();
    RegisterOGRShape();
    GetGDALDriverManager()->AutoSkipDrivers();
}

void Deinitialize() { CPLSetErrorHandler(CPLDefaultErrorHandler); }

void SetCacheMax(size_t bytes) { GDALSetCacheMax64(bytes); }

ErrorCallbackGuard SetErrorHandle(const ErrorCallback& handler) {
    AssignErrorHandle(&handler);
    // This is mistakenly interpreted by rule cpp:S936 in sonarcloud.
    return ErrorCallbackGuard(&handler, ClearErrorHandle); // NOSONAR
}

}  // namespace alus::gdalmanagement
