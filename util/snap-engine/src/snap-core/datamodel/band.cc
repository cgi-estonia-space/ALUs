#include "band.h"

#include <stdexcept>

namespace alus {
namespace snapengine {

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
}  // namespace snapengine
}  // namespace alus