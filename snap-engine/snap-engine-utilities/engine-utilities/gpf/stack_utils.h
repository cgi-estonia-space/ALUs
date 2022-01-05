/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.StackUtils.java
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
#include <vector>

#include "snap-core/core/datamodel/product.h"

namespace alus::snapengine {

class StackUtils {
public:
    static std::string CreateBandTimeStamp(std::shared_ptr<Product>& product);
    static bool IsMasterBand(std::string_view band_name, const std::shared_ptr<Product>& product);
    static bool IsSlaveBand(std::string_view band_name, std::shared_ptr<Product> product);
    static bool IsSlaveBand(std::string_view band_name, const std::shared_ptr<Product>& product,
                            std::string_view slave_product_name);
    static std::vector<std::string> GetSlaveProductNames(const std::shared_ptr<Product>& product);
    static bool IsCoregisteredStack(const std::shared_ptr<Product>& product);

private:
    static constexpr std::string_view MST{"_mst"};
    static constexpr std::string_view SLV{"_slv"};
};

}  // namespace alus::snapengine