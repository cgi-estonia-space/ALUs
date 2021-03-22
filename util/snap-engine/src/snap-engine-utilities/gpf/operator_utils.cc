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
#include "snap-engine-utilities/gpf/operator_utils.h"

#include <boost/algorithm/string/case_conv.hpp>

#include <stdexcept>

namespace alus {
namespace snapengine {

std::string OperatorUtils::GetPolarizationFromBandName(std::string_view band_name) {
    // Account for possibilities like "x_HH_dB" or "x_HH_times_VV_conj"
    // where the last one will return an exception because it appears to contain
    // multiple polarizations
    std::string pol;
    std::string band_name_str(band_name);
    boost::algorithm::to_lower(band_name_str);

    if (band_name_str.find("_hh") != std::string::npos) {
        pol += "hh";
    }
    if (band_name_str.find("_vv") != std::string::npos) {
        pol += "vv";
    }
    if (band_name_str.find("_hv") != std::string::npos) {
        pol += "hv";
    }
    if (band_name_str.find("_vh") != std::string::npos) {
        pol += "vh";
    }

    // compact pol
    if (band_name_str.find("_rh") != std::string::npos) {
        pol += "rh";
    }
    if (band_name_str.find("_rv") != std::string::npos) {
        pol += "rv";
    }
    if (band_name_str.find("_rch") != std::string::npos) {
        pol += "rch";
    }
    if (band_name_str.find("_rcv") != std::string::npos) {
        pol += "rcv";
    }

    if (pol.length() >= 2 && pol.length() <= 3) {
        return pol;
    }
    if (pol.length() > 3) {
        throw std::runtime_error("Band name contains multiple polarizations: " + pol);
    }
    return "";
}
}  // namespace snapengine
}  // namespace alus
