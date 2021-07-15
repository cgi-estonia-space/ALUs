/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.OperatorUtils.java
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
#include <string>
#include <string_view>
#include <vector>

#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/product.h"

#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/datamodel/tie_point_grid.h"

namespace alus {
namespace snapengine {

/**
 * Helper methods for working with Operators
 */
class OperatorUtils {
private:
public:
    static constexpr std::string_view TPG_SLANT_RANGE_TIME = "slant_range_time";
    static constexpr std::string_view TPG_INCIDENT_ANGLE = "incident_angle";
    static constexpr std::string_view TPG_ELEVATION_ANGLE = "elevation_angle";
    static constexpr std::string_view TPG_LATITUDE = "latitude";
    static constexpr std::string_view TPG_LONGITUDE = "longitude";

    static std::string GetPolarizationFromBandName(std::string_view band_name);

    /**
     * Get incidence angle tie point grid.
     *
     * @param sourceProduct The source product.
     * @return srcTPG The incidence angle tie point grid.
     */
    static std::shared_ptr<TiePointGrid> GetIncidenceAngle(const std::shared_ptr<snapengine::Product>& source_product) {
        return source_product->GetTiePointGrid(TPG_INCIDENT_ANGLE);
    }
    static std::vector<std::shared_ptr<Band>> GetSourceBands(std::shared_ptr<Product> source_product, std::vector<std::string> source_band_names, bool include_virtual_bands);
    static std::string GetSuffixFromBandName(std::string_view band_name);
    static std::string GetAcquisitionDate(std::shared_ptr<MetadataElement>& root);
};

}  // namespace snapengine
}  // namespace alus
