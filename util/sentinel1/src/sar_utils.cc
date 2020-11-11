#include "sar_utils.h"

#include "general_constants.h"
#include "meta_data_node_names.h"

namespace alus::s1tbx {

double SarUtils::GetRadarFrequency(std::shared_ptr<snapengine::MetadataElement> abs_root) {
    double radar_freq = snapengine::MetaDataNodeNames::GetAttributeDouble(abs_root, alus::snapengine::MetaDataNodeNames::RADAR_FREQUENCY) * snapengine::constants::oneMillion; // Hz
    if (radar_freq <= 0.0) {
        throw std::runtime_error("Invalid radar frequency: " +  std::to_string(radar_freq));
    }
    return snapengine::constants::lightSpeed / radar_freq;
}
} //namespace
