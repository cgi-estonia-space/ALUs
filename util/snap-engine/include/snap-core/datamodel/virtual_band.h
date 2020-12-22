/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.datamodel.VirtualBand.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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

#include <string>
#include <string_view>

#include "band.h"

namespace alus {
namespace snapengine {

/**
 * A band contains the data for geophysical parameter in remote sensing data products. Bands are two-dimensional images
 * which hold their pixel values (samples) in a buffer of the type {@link ProductData}. The band class is just a
 * container for attached metadata of the band, currently: <ul> <li>the flag coding {@link FlagCoding}</li> <li>the band
 * index at which position the band is stored in the associated product</li> <li>the center wavelength of the band</li>
 * <li>the bandwidth of the band</li> <li>the solar spectral flux of the band</li> <li>the width and height of the
 * band</li> </ul> The band can contain a buffer to the real data, but this buffer must be read explicitely, to keep the
 * memory fingerprint small, the data is not read automatically.
 * <p>
 * <p>
 * The several <code>getPixel</code> and <code>readPixel</code> methods of this class do not necessarily return the
 * values contained in the data buffer of type {@link ProductData}. If the <code>scalingFactor</code>,
 * <code>scalingOffset</code> or <code>log10Scaled</code> are set a conversion of the form <code>scalingFactor *
 * rawSample + scalingOffset</code> is applied to the raw samples before the <code>getPixel</code> and
 * <code>readPixel</code> methods return the actual pixel values. If the <code>log10Scaled</code> property is true then
 * the conversion is <code>pow(10, scalingFactor * rawSample + scalingOffset)</code>. The several <code>setPixel</code>
 * and <code>writePixel</code> perform the inverse operations in this case.
 *
 * original java version author Norman Fomferra
 * @see ProductData
 */
class VirtualBand : public Band {
private:
    static constexpr std::string_view PROPERTY_NAME_EXPRESSION = "expression";

    std::string expression_;

public:
    /**
     * Constructs a new <code>Band</code>.
     *
     * @param name       the name of the new object
     * @param dataType   the raster data type, must be one of the multiple <code>ProductData.TYPE_<i>X</i></code>
     *                   constants, with the exception of <code>ProductData.TYPE_UINT32</code>
     * @param width      the width of the raster in pixels
     * @param height     the height of the raster in pixels
     * @param expression the expression code
     */
    VirtualBand(std::string_view name, int data_type, int width, int height, std::string_view expression);

    std::string GetExpression() { return expression_; }

    void SetExpression(std::string_view expression);
};
}  // namespace snapengine
}  // namespace alus
