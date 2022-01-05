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
#pragma once

#include <memory>
#include <string_view>

#include "snap-core/core/datamodel/abstract_band.h"

namespace alus {

namespace ceres {
// pre-declare
class IProgressMonitor;
}  // namespace ceres
namespace snapengine {

// pre-declare
class IndexCoding;
class FlagCoding;
class SampleCoding;

/**
 * A band contains the data for geophysical parameter in remote sensing data products. Bands are two-dimensional images
 * which hold their pixel values (samples) in a buffer of the type {@link ProductData}. The band class is just a
 * container for attached metadata of the band, currently: <ul> <li>the flag coding {@link FlagCoding}</li> <li>the band
 * index at which position the band is stored in the associated product</li> <li>the center wavelength of the band</li>
 * <li>the bandwidth of the band</li> <li>the solar spectral flux of the band</li> <li>the width and height of the
 * band</li> </ul> The band can contain a buffer to the real data, but this buffer must be read explicitely, to keep the
 * memory fingerprint small, the data is not read automatically.
 * <p>
 * The several <code>getPixel</code> and <code>readPixel</code> methods of this class do not necessarily return the
 * values contained in the data buffer of type {@link ProductData}. If the <code>scalingFactor</code>,
 * <code>scalingOffset</code> or <code>log10Scaled</code> are set a conversion of the form <code>scalingFactor *
 * rawSample + scalingOffset</code> is applied to the raw samples before the <code>getPixel</code> and @
 * <code>readPixel</code> methods return the actual pixel values. If the <code>log10Scaled</code> property is true then
 * the conversion is <code>pow(10, scalingFactor * rawSample + scalingOffset)</code>. The several <code>setPixel</code>
 * and <code>writePixel</code> perform the inverse operations in this case.
 *
 * original java version author Norman Fomferra
 * @see ProductData
 */
class Band : public AbstractBand {
private:
    int spectral_band_index_ = 0;
    float spectral_wavelength_ = 0.0F;
    float spectral_bandwidth_ = 0.0F;
    float solar_flux_ = 0.0F;

    /**
     * If this band contains flag data, this is the flag coding.
     */
    std::shared_ptr<SampleCoding> sample_coding_;

public:
    static constexpr std::string_view PROPERTY_NAME_SAMPLE_CODING = "sampleCoding";
    static constexpr std::string_view PROPERTY_NAME_SOLAR_FLUX = "solarFlux";
    static constexpr std::string_view PROPERTY_NAME_SPECTRAL_BAND_INDEX = "spectralBandIndex";
    static constexpr std::string_view PROPERTY_NAME_SPECTRAL_BANDWIDTH = "spectralBandwidth";
    static constexpr std::string_view PROPERTY_NAME_SPECTRAL_WAVELENGTH = "spectralWavelength";

    /**
     * Constructs a new <code>Band</code>.
     *
     * @param name     the name of the new object
     * @param dataType the raster data type, must be one of the multiple <code>ProductData.TYPE_<i>X</i></code>
     *                 constants, with the exception of <code>ProductData.TYPE_UINT32</code>
     * @param width    the width of the raster in pixels
     * @param height   the height of the raster in pixels
     */
    Band(std::string_view name, int data_type, int width, int height);

    /**
     * Gets the (zero-based) spectral band index.
     *
     * @return the (zero-based) spectral band index or <code>-1</code> if it is unknown
     */
    [[nodiscard]] int GetSpectralBandIndex() const { return spectral_band_index_; }

    /**
     * Sets the (zero-based) spectral band index.
     *
     * @param spectralBandIndex the (zero-based) spectral band index or <code>-1</code> if it is unknown
     */
    void SetSpectralBandIndex(int spectral_band_index);

    /**
     * Gets the spectral wavelength in <code>nm</code> (nanometer) units.
     *
     * @return the wave length in nanometers of this band, or zero if this is not a spectral band or the wave length is
     *         not known.
     */
    [[nodiscard]] float GetSpectralWavelength() const { return spectral_wavelength_; }
    /**
     * Sets the spectral wavelength in <code>nm</code> (nanometer) units.
     *
     * @param spectral_wavelength the wavelength in nanometers of this band, or zero if this is not a spectral band or
     *                           the wavelength is not known.
     */
    void SetSpectralWavelength(float spectral_wavelength);

    /**
     * Gets the solar flux in <code>mW/(m^2 nm)</code> (milli-watts per square metre per nanometer)
     * units for the wavelength of this band.
     *
     * @return the solar flux for the wavelength of this band, or zero if this is not a spectral band or the solar flux
     *         is not known.
     */
    [[nodiscard]] float GetSolarFlux() const { return solar_flux_; }

    /**
     * Sets the solar flux in <code>mW/(m^2 nm)</code> (milli-watts per square metre per nanometer)
     * units for the wavelength of this band.
     *
     * @param solar_flux the solar flux for the wavelength of this band, or zero if this is not a spectral band or the
     *                  solar flux is not known.
     */
    void SetSolarFlux(float solar_flux);

    /**
     * Gets the spectral bandwidth in <code>nm</code> (nanometer) units.
     *
     * @return the bandwidth in nanometers of this band, or zero if this is not a spectral band or the bandwidth is not
     *         known.
     */
    [[nodiscard]] float GetSpectralBandwidth() const { return spectral_bandwidth_; }

    /**
     * Sets the spectral bandwidth in <code>nm</code> (nanometer) units.
     *
     * @param spectral_bandwidth the spectral bandwidth in nanometers of this band, or zero if this is not a spectral
     * band or the spectral bandwidth is not known.
     */
    void SetSpectralBandwidth(float spectral_bandwidth);

    /**
     * Tests whether or not this band is a flag band (<code>getFlagCoding() != null</code>).
     *
     * @return <code>true</code> if so
     */
    bool IsFlagBand() { return GetFlagCoding() != nullptr; }

    /**
     * Gets the sample coding.
     *
     * @return the sample coding, or {@code null} if not set.
     */
    std::shared_ptr<SampleCoding> GetSampleCoding() { return sample_coding_; }

    /**
     * Sets the sample coding for this band.
     *
     * @param sampleCoding the sample coding
     * @throws IllegalArgumentException if this band does not contain integer pixels
     */
    void SetSampleCoding(const std::shared_ptr<SampleCoding>& sample_coding);

    /**
     * Gets the flag coding for this band.
     *
     * @return a non-null value if this band is a flag dataset, <code>null</code> otherwise
     */
    std::shared_ptr<FlagCoding> GetFlagCoding();

    /**
     * Gets the index coding for this band.
     *
     * @return a non-null value if this band is a flag dataset, <code>null</code> otherwise
     */
    std::shared_ptr<IndexCoding> GetIndexCoding();

    /**
     * Reads raster data from its associated data source into the given data buffer.
     *
     * @param offsetX    the X-offset in the band's pixel co-ordinates where reading starts
     * @param offsetY    the Y-offset in the band's pixel co-ordinates where reading starts
     * @param width      the width of the raster data buffer
     * @param height     the height of the raster data buffer
     * @param rasterData a raster data buffer receiving the pixels to be read
     * @param pm         a monitor to inform the user about progress
     * @throws java.io.IOException      if an I/O error occurs
     * @throws IllegalArgumentException if the raster is null
     * @throws IllegalStateException    if this product raster was not added to a product so far, or if the product to
     * which this product raster belongs to, has no associated product reader
     * @see ProductReader#readBandRasterData(Band, int, int, int, int, ProductData, com.bc.ceres.core.ProgressMonitor)
     */
    void ReadRasterData(int offset_x, int offset_y, int width, int height, std::shared_ptr<ProductData> raster_data,
                        std::shared_ptr<ceres::IProgressMonitor> pm) override;

    void ReadRasterDataFully(std::shared_ptr<ceres::IProgressMonitor> pm) override;

    void WriteRasterData(int offset_x, int offset_y, int width, int height, std::shared_ptr<ProductData> raster_data,
                         std::shared_ptr<ceres::IProgressMonitor> pm) override;

    void WriteRasterDataFully(std::shared_ptr<ceres::IProgressMonitor> pm) override;
};
}  // namespace snapengine
}  // namespace alus
