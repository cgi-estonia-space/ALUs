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
