/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.Band.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
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
#include "band.h"

#include <stdexcept>
#include "../util/guardian.h"

#include "ceres-core/core/i_progress_monitor.h"
#include "snap-core/core/datamodel/flag_coding.h"
#include "snap-core/core/datamodel/index_coding.h"
#include "snap-core/core/datamodel/sample_coding.h"

namespace alus::snapengine {

Band::Band(std::string_view name, int data_type, int width, int height) : AbstractBand(name, data_type, width, height) {
    SetSpectralBandIndex(-1);
    SetModified(false);
}
void Band::SetSpectralBandIndex(int spectral_band_index) {
    if (spectral_band_index_ != spectral_band_index) {
        spectral_band_index_ = spectral_band_index;
        SetModified(true);
    }
}
std::shared_ptr<FlagCoding> Band::GetFlagCoding() {
    //        todo:check if this is same as in java and works like needed
    // return GetSampleCoding() instanceof FlagCoding ? (FlagCoding)GetSampleCoding() : nullptr;
    return std::dynamic_pointer_cast<FlagCoding>(GetSampleCoding());
}
std::shared_ptr<IndexCoding> Band::GetIndexCoding() {
    //        todo:check if this is same as in java and works like needed
    //        return GetSampleCoding() instanceof IndexCoding ? (IndexCoding)GetSampleCoding() : nullptr;
    return std::dynamic_pointer_cast<IndexCoding>(GetSampleCoding());
}
void Band::SetSampleCoding(const std::shared_ptr<SampleCoding>& sample_coding) {
    if (sample_coding != nullptr) {
        if (!HasIntPixels()) {
            throw std::invalid_argument("band does not contain integer pixels");
        }
    }
    if (sample_coding_ != sample_coding) {
        sample_coding_ = sample_coding;
        SetModified(true);
    }
}
void Band::SetSpectralWavelength(float spectral_wavelength) {
    if (spectral_wavelength_ != spectral_wavelength) {
        spectral_wavelength_ = spectral_wavelength;
        SetModified(true);
    }
}
void Band::SetSpectralBandwidth(float spectral_bandwidth) {
    if (spectral_bandwidth_ != spectral_bandwidth) {
        spectral_bandwidth_ = spectral_bandwidth;
        SetModified(true);
    }
}
void Band::SetSolarFlux(float solar_flux) {
    if (solar_flux_ != solar_flux) {
        solar_flux_ = solar_flux;
        SetModified(true);
    }
}
void Band::ReadRasterData([[maybe_unused]] int offset_x, [[maybe_unused]] int offset_y, [[maybe_unused]] int width,
                          [[maybe_unused]] int height, [[maybe_unused]] std::shared_ptr<ProductData> raster_data,
                          [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {
    throw std::runtime_error("Not implemented yet!");
    //    Guardian::AssertNotNull("rasterData", raster_data);
    //    if (IsProductReaderDirectlyUsable()) {
    //        // Don't go the long way round the source image.
    //        GetProductReader()->ReadBandRasterData(this, offset_x, offset_y, width, height, raster_rata, pm);
    //    } else {
    //        //        try {
    //        //            pm.beginTask("Reading raster data...", 100);
    //        std::shared_ptr<RenderedImage> source_image = GetSourceImage();
    //        int x = source_image->GetMinX() + offset_x;
    //        int y = source_image->GetMinY() + offset_y;
    //        std::shared_ptr<Raster> data = source_image->GetData(std::make_shared<Rectangle>(x, y, width, height));
    //        //            pm.worked(90);
    //        data->GetDataElements(x, y, width, height, raster_data->GetElems());
    //        //            pm.worked(10);
    //        //        } finally {
    //        //            pm.done();
    //        //        }
    //    }
}
void Band::ReadRasterDataFully([[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {}
void Band::WriteRasterData([[maybe_unused]] int offset_x, [[maybe_unused]] int offset_y, [[maybe_unused]] int width,
                           [[maybe_unused]] int height, [[maybe_unused]] std::shared_ptr<ProductData> raster_data,
                           [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {}
void Band::WriteRasterDataFully([[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {}
}  // namespace alus::snapengine