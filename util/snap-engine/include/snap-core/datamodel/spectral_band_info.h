/**
 * This file is a filtered duplicate of a SNAP's  org.esa.snap.core.datamodel.ProductData.java ported
 * for native code. Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
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

#include <optional>
#include <string>

namespace alus {
namespace snapengine {

/**
 * Struct representing Spectral_Band_Info tag of the BEAM-DIMAP format.
 *
 * @todo Should be replaced with correct product/metadata reader which is currently in development.
 */
struct SpectralBandInfo {
    int band_index;
    std::string band_name;
    int product_data_type;
    bool log_10_scaled;
    bool no_data_value_used;
    std::optional<std::string> band_description;
    std::optional<std::string> physical_unit;
    std::optional<double> solar_flux;
    std::optional<int> spectral_band_index;
    std::optional<double> band_wavelength;
    std::optional<double> bandwidth;
    std::optional<double> scaling_factor;
    std::optional<double> scaling_offset;
    std::optional<double> no_data_value;
    std::optional<std::string> valid_mask_term;
};
}  // namespace snapengine
}  // namespace alus