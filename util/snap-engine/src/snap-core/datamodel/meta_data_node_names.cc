#include "meta_data_node_names.h"

#include "parse_exception.h"

namespace alus {
namespace snapengine {

bool MetaDataNodeNames::GetAttributeBoolean(MetadataElement& element, std::string_view tag) {
    int val = element.GetAttributeInt(tag);
    if (val == NO_METADATA) {
        throw std::runtime_error("Metadata " + std::string(tag) + " has not been set");
    }
    return val != 0;
}
double MetaDataNodeNames::GetAttributeDouble(MetadataElement& element, std::string_view tag) {
    double val = element.GetAttributeDouble(tag);
    if (val == NO_METADATA) {
        throw std::runtime_error("Metadata " + std::string(tag) + " has not been set");
    }
    return val;
}
std::shared_ptr<Utc> MetaDataNodeNames::ParseUtc(std::string_view time_str) {
    try {
        if (time_str == nullptr) return NO_METADATA_UTC;
        return snapengine::Utc::Parse(time_str);
    } catch (alus::ParseException& e) {
        try {
            auto dot_pos = time_str.find_last_of(".");
            if (dot_pos != std::string::npos && dot_pos > 0) {
                std::string fraction_string{time_str.substr(dot_pos + 1, time_str.length())};
                // fix some ERS times
                boost::erase_all(fraction_string, "-");
                std::string new_time_str = std::string(time_str.substr(0, dot_pos)) + fraction_string;
                return snapengine::Utc::Parse(new_time_str);
            }
        } catch (ParseException& e2) {
            return NO_METADATA_UTC;
        }
    }
    return NO_METADATA_UTC;
}
std::vector<OrbitStateVector> MetaDataNodeNames::GetOrbitStateVectors(
    MetadataElement& abs_root) {
    auto elem_root = abs_root.GetElement(snapengine::MetaDataNodeNames::ORBIT_STATE_VECTORS);
    if (elem_root == nullptr) {
        return std::vector<OrbitStateVector>{};
    }
    const int num_elems = elem_root->GetNumElements();
    std::vector<OrbitStateVector> orbit_state_vectors;
    for (int i = 0; i < num_elems; i++) {
        auto sub_elem_root = elem_root->GetElement(std::string(MetaDataNodeNames::ORBIT_VECTOR) +
                                                   std::to_string(i + 1));
        auto vector = OrbitStateVector(
            sub_elem_root->GetAttributeUtc(snapengine::MetaDataNodeNames::ORBIT_VECTOR_TIME),
            sub_elem_root->GetAttributeDouble(snapengine::MetaDataNodeNames::ORBIT_VECTOR_X_POS),
            sub_elem_root->GetAttributeDouble(snapengine::MetaDataNodeNames::ORBIT_VECTOR_Y_POS),
            sub_elem_root->GetAttributeDouble(snapengine::MetaDataNodeNames::ORBIT_VECTOR_Z_POS),
            sub_elem_root->GetAttributeDouble(snapengine::MetaDataNodeNames::ORBIT_VECTOR_X_VEL),
            sub_elem_root->GetAttributeDouble(snapengine::MetaDataNodeNames::ORBIT_VECTOR_Y_VEL),
            sub_elem_root->GetAttributeDouble(snapengine::MetaDataNodeNames::ORBIT_VECTOR_Z_VEL));
        orbit_state_vectors.push_back(vector);
    }
    return orbit_state_vectors;
}

}  // namespace snapengine
}  // namespace alus