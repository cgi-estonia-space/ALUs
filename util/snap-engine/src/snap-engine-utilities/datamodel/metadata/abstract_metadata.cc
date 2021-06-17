/**
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
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"

#include <string>

#include <boost/algorithm/string.hpp>

#include "alus_log.h"
#include "metadata_attribute.h"
#include "parse_exception.h"
#include "snap-core/util/string_utils.h"

namespace alus {
namespace snapengine {

bool AbstractMetadata::GetAttributeBoolean(const std::shared_ptr<MetadataElement>& element, std::string_view tag) {
    int val = element->GetAttributeInt(tag);
    if (val == NO_METADATA) {
        throw std::runtime_error("Metadata " + std::string(tag) + " has not been set");
    }
    return val != 0;
}

double AbstractMetadata::GetAttributeDouble(const std::shared_ptr<MetadataElement>& element, std::string_view tag) {
    double val = element->GetAttributeDouble(tag);
    if (val == NO_METADATA) {
        throw std::runtime_error("Metadata " + std::string(tag) + " has not been set");
    }
    return val;
}

std::shared_ptr<Utc> AbstractMetadata::ParseUtc(std::string_view time_str) {
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

std::shared_ptr<MetadataElement> AbstractMetadata::GetSlaveMetadata(std::shared_ptr<MetadataElement> target_root) {
    std::shared_ptr<MetadataElement> target_slave_metadata_root = target_root->GetElement("Slave_Metadata");
    if (target_slave_metadata_root == nullptr) {
        target_slave_metadata_root = std::make_shared<MetadataElement>("Slave_Metadata");
        target_root->AddElement(target_slave_metadata_root);
    }

    return target_slave_metadata_root;
}

std::vector<OrbitStateVector> AbstractMetadata::GetOrbitStateVectors(const std::shared_ptr<MetadataElement>& abs_root) {
    auto elem_root = abs_root->GetElement(snapengine::AbstractMetadata::ORBIT_STATE_VECTORS);
    if (elem_root == nullptr) {
        return std::vector<OrbitStateVector>{};
    }
    const int num_elems = elem_root->GetNumElements();
    std::vector<OrbitStateVector> orbit_state_vectors;
    for (int i = 0; i < num_elems; i++) {
        auto sub_elem_root = elem_root->GetElement(std::string(AbstractMetadata::ORBIT_VECTOR) + std::to_string(i + 1));
        auto vector =
            OrbitStateVector(sub_elem_root->GetAttributeUtc(snapengine::AbstractMetadata::ORBIT_VECTOR_TIME),
                             sub_elem_root->GetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_X_POS),
                             sub_elem_root->GetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_Y_POS),
                             sub_elem_root->GetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_Z_POS),
                             sub_elem_root->GetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_X_VEL),
                             sub_elem_root->GetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_Y_VEL),
                             sub_elem_root->GetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_Z_VEL));
        orbit_state_vectors.push_back(vector);
    }
    return orbit_state_vectors;
}

std::shared_ptr<MetadataElement> AbstractMetadata::GetAbstractedMetadata(
    const std::shared_ptr<Product>& source_product) {
    auto root = source_product->GetMetadataRoot();
    if (root == nullptr) {
        return nullptr;
    }
    auto abstracted_metadata = root->GetElement(ABSTRACT_METADATA_ROOT);
    if (abstracted_metadata == nullptr) {
        abstracted_metadata = root->GetElement("Abstracted Metadata");  // legacy
        if (abstracted_metadata == nullptr) {
            bool is_modified = source_product->IsModified();
            abstracted_metadata = AddAbstractedMetadataHeader(root);
            DefaultToProduct(abstracted_metadata, source_product);
            source_product->SetModified(is_modified);
        }
    }
    MigrateToCurrentVersion(abstracted_metadata);
    PatchMissingMetadata(abstracted_metadata);

    return abstracted_metadata;
}

std::shared_ptr<MetadataAttribute> AbstractMetadata::AddAbstractedAttribute(
    const std::shared_ptr<MetadataElement>& dest, std::string_view tag, int data_type, std::string_view unit,
    std::string_view desc) {
    auto attribute = std::make_shared<MetadataAttribute>(tag, data_type, 1);
    if (data_type == ProductData::TYPE_ASCII) {
        attribute->GetData()->SetElems(NO_METADATA_STRING);
    } else if (data_type == ProductData::TYPE_INT8 || data_type == ProductData::TYPE_UINT8) {
        attribute->GetData()->SetElems(std::vector<std::string>{NO_METADATA_BYTE});
    } else if (data_type != ProductData::TYPE_UTC) {
        attribute->GetData()->SetElems(std::vector<std::string>{std::to_string(NO_METADATA)});
    }
    attribute->SetUnit(unit);
    attribute->SetDescription(desc);
    attribute->SetReadOnly(false);
    dest->AddAttribute(attribute);
    return attribute;
}

void AbstractMetadata::SetAttribute(const std::shared_ptr<MetadataElement>& dest, std::string_view tag, int value) {
    if (dest == nullptr) return;
    std::shared_ptr<MetadataAttribute> attrib = dest->GetAttribute(tag);
    if (attrib == nullptr) {
        LOGW << tag << " not found in metadata";
    } else {
        attrib->GetData()->SetElemInt(value);
    }
}

void AbstractMetadata::SetAttribute(const std::shared_ptr<MetadataElement>& dest, std::string_view tag,
                                    std::optional<std::string> value) {
    if (dest == nullptr) {
        return;
    }
    std::shared_ptr<MetadataAttribute> attrib = dest->GetAttribute(tag);
    if (attrib == nullptr) {
        attrib = std::make_shared<MetadataAttribute>(tag, ProductData::TYPE_ASCII);
        dest->AddAttribute(attrib);
    }
    if (!value || value.value().empty()) {
        attrib->GetData()->SetElems(NO_METADATA_STRING);
    } else {
        attrib->GetData()->SetElems(value.value());
    }
}

void AbstractMetadata::SetAttribute(const std::shared_ptr<MetadataElement>& dest, std::string_view tag,
                                    const double value) {
    if (dest == nullptr) {
        return;
    }
    const std::shared_ptr<MetadataAttribute> attrib = dest->GetAttribute(tag);
    if (attrib) {
        attrib->GetData()->SetElemDouble(value);
    } else {
        const auto new_attrib = std::make_shared<MetadataAttribute>(tag, ProductData::TYPE_FLOAT64);
        dest->AddAttribute(new_attrib);
        new_attrib->GetData()->SetElemDouble(value);
    }
}

void AbstractMetadata::SetAttribute(const std::shared_ptr<MetadataElement>& dest, std::string_view tag,
                                    const std::shared_ptr<Utc>& value) {
    if (dest == nullptr) {
        return;
    }
    std::shared_ptr<MetadataAttribute> attrib = dest->GetAttribute(tag);
    if (attrib != nullptr && value != nullptr) {
        attrib->GetData()->SetElems(value->GetArray());
    } else {
        if (attrib == nullptr) {
            LOGW << tag << " not found in metadata";
        }
        if (value == nullptr) {
            LOGW << tag << " metadata value is nullptr";
        }
    }
}

std::shared_ptr<MetadataElement> AbstractMetadata::AddAbstractedMetadataHeader(
    const std::shared_ptr<MetadataElement>& root) {
    std::shared_ptr<MetadataElement> abs_root;
    if (root == nullptr) {
        abs_root = std::make_shared<MetadataElement>(ABSTRACT_METADATA_ROOT);
    } else {
        abs_root = root->GetElement(ABSTRACT_METADATA_ROOT);
        if (abs_root == nullptr) {
            abs_root = std::make_shared<MetadataElement>(ABSTRACT_METADATA_ROOT);
            root->AddElementAt(abs_root, 0);
        }
    }

    // MPH
    AddAbstractedAttribute(abs_root, PRODUCT, ProductData::TYPE_ASCII, "", "Product name");
    AddAbstractedAttribute(abs_root, PRODUCT_TYPE, ProductData::TYPE_ASCII, "", "Product type");
    AddAbstractedAttribute(abs_root, SPH_DESCRIPTOR, ProductData::TYPE_ASCII, "", "Description");
    AddAbstractedAttribute(abs_root, MISSION, ProductData::TYPE_ASCII, "", "Satellite mission");
    AddAbstractedAttribute(abs_root, ACQUISITION_MODE, ProductData::TYPE_ASCII, "", "Acquisition mode");
    AddAbstractedAttribute(abs_root, ANTENNA_POINTING, ProductData::TYPE_ASCII, "", "Right or left facing");
    AddAbstractedAttribute(abs_root, BEAMS, ProductData::TYPE_ASCII, "", "Beams used");
    AddAbstractedAttribute(abs_root, SWATH, ProductData::TYPE_ASCII, "", "Swath name");
    AddAbstractedAttribute(abs_root, PROC_TIME, ProductData::TYPE_UTC, "utc", "Processed time");
    AddAbstractedAttribute(abs_root, ProcessingSystemIdentifier, ProductData::TYPE_ASCII, "",
                           "Processing system identifier");
    //    todo: check issue here
    AddAbstractedAttribute(abs_root, CYCLE, ProductData::TYPE_INT32, "", "Cycle");
    AddAbstractedAttribute(abs_root, REL_ORBIT, ProductData::TYPE_INT32, "", "Track");
    AddAbstractedAttribute(abs_root, ABS_ORBIT, ProductData::TYPE_INT32, "", "Orbit");
    AddAbstractedAttribute(abs_root, STATE_VECTOR_TIME, ProductData::TYPE_UTC, "utc", "Time of orbit state vector");
    AddAbstractedAttribute(abs_root, VECTOR_SOURCE, ProductData::TYPE_ASCII, "", "State vector source");

    AddAbstractedAttribute(abs_root, INCIDENCE_NEAR, ProductData::TYPE_FLOAT64, "deg", "");
    AddAbstractedAttribute(abs_root, INCIDENCE_FAR, ProductData::TYPE_FLOAT64, "deg", "");

    // SPH
    AddAbstractedAttribute(abs_root, SLICE_NUM, ProductData::TYPE_INT32, "", "Slice number");
    AddAbstractedAttribute(abs_root, DATA_TAKE_ID, ProductData::TYPE_INT32, "", "Data take identifier");
    AddAbstractedAttribute(abs_root, FIRST_LINE_TIME, ProductData::TYPE_UTC, "utc", "First zero doppler azimuth time");
    AddAbstractedAttribute(abs_root, LAST_LINE_TIME, ProductData::TYPE_UTC, "utc", "Last zero doppler azimuth time");
    AddAbstractedAttribute(abs_root, FIRST_NEAR_LAT, ProductData::TYPE_FLOAT64, "deg", "");
    AddAbstractedAttribute(abs_root, FIRST_NEAR_LONG, ProductData::TYPE_FLOAT64, "deg", "");
    AddAbstractedAttribute(abs_root, FIRST_FAR_LAT, ProductData::TYPE_FLOAT64, "deg", "");
    AddAbstractedAttribute(abs_root, FIRST_FAR_LONG, ProductData::TYPE_FLOAT64, "deg", "");
    AddAbstractedAttribute(abs_root, LAST_NEAR_LAT, ProductData::TYPE_FLOAT64, "deg", "");
    AddAbstractedAttribute(abs_root, LAST_NEAR_LONG, ProductData::TYPE_FLOAT64, "deg", "");
    AddAbstractedAttribute(abs_root, LAST_FAR_LAT, ProductData::TYPE_FLOAT64, "deg", "");
    AddAbstractedAttribute(abs_root, LAST_FAR_LONG, ProductData::TYPE_FLOAT64, "deg", "");

    AddAbstractedAttribute(abs_root, PASS, ProductData::TYPE_ASCII, "", "ASCENDING or DESCENDING");
    AddAbstractedAttribute(abs_root, SAMPLE_TYPE, ProductData::TYPE_ASCII, "", "DETECTED or COMPLEX");
    AddAbstractedAttribute(abs_root, MDS1_TX_RX_POLAR, ProductData::TYPE_ASCII, "", "Polarization");
    AddAbstractedAttribute(abs_root, MDS2_TX_RX_POLAR, ProductData::TYPE_ASCII, "", "Polarization");
    AddAbstractedAttribute(abs_root, MDS3_TX_RX_POLAR, ProductData::TYPE_ASCII, "", "Polarization");
    AddAbstractedAttribute(abs_root, MDS4_TX_RX_POLAR, ProductData::TYPE_ASCII, "", "Polarization");
    AddAbstractedAttribute(abs_root, POLSARDATA, ProductData::TYPE_UINT8, "flag", "Polarimetric Matrix");
    AddAbstractedAttribute(abs_root, ALGORITHM, ProductData::TYPE_ASCII, "", "Processing algorithm");
    AddAbstractedAttribute(abs_root, AZIMUTH_LOOKS, ProductData::TYPE_FLOAT64, "", "");
    AddAbstractedAttribute(abs_root, RANGE_LOOKS, ProductData::TYPE_FLOAT64, "", "");
    AddAbstractedAttribute(abs_root, RANGE_SPACING, ProductData::TYPE_FLOAT64, "m", "Range sample spacing");
    AddAbstractedAttribute(abs_root, AZIMUTH_SPACING, ProductData::TYPE_FLOAT64, "m", "Azimuth sample spacing");
    AddAbstractedAttribute(abs_root, PULSE_REPETITION_FREQUENCY, ProductData::TYPE_FLOAT64, "Hz", "PRF");
    AddAbstractedAttribute(abs_root, RADAR_FREQUENCY, ProductData::TYPE_FLOAT64, "MHz", "Radar frequency");
    AddAbstractedAttribute(abs_root, LINE_TIME_INTERVAL, ProductData::TYPE_FLOAT64, "s", "");

    AddAbstractedAttribute(abs_root, TOT_SIZE, ProductData::TYPE_UINT32, "MB", "Total product size");
    AddAbstractedAttribute(abs_root, NUM_OUTPUT_LINES, ProductData::TYPE_UINT32, "lines", "Raster height");
    AddAbstractedAttribute(abs_root, NUM_SAMPLES_PER_LINE, ProductData::TYPE_UINT32, "samples", "Raster width");

    AddAbstractedAttribute(abs_root, SUBSET_OFFSET_X, ProductData::TYPE_UINT32, "samples",
                           "X coordinate of UL corner of subset in original image");
    AddAbstractedAttribute(abs_root, SUBSET_OFFSET_Y, ProductData::TYPE_UINT32, "samples",
                           "Y coordinate of UL corner of subset in original image");
    SetAttribute(abs_root, SUBSET_OFFSET_X, 0);
    SetAttribute(abs_root, SUBSET_OFFSET_Y, 0);

    // SRGR
    AddAbstractedAttribute(abs_root, SRGR_FLAG, ProductData::TYPE_UINT8, "flag", "SRGR applied");
    auto att = AddAbstractedAttribute(abs_root, AVG_SCENE_HEIGHT, ProductData::TYPE_FLOAT64, "m",
                                      "Average scene height ellipsoid");
    att->GetData()->SetElemInt(0);
    AddAbstractedAttribute(abs_root, MAP_PROJECTION, ProductData::TYPE_ASCII, "", "Map projection applied");

    // orthorectification
    AddAbstractedAttribute(abs_root, IS_TERRAIN_CORRECTED, ProductData::TYPE_UINT8, "flag",
                           "orthorectification applied");
    AddAbstractedAttribute(abs_root, DEM, ProductData::TYPE_ASCII, "", "Digital Elevation Model used");
    AddAbstractedAttribute(abs_root, GEO_REF_SYSTEM, ProductData::TYPE_ASCII, "", "geographic reference system");
    AddAbstractedAttribute(abs_root, LAT_PIXEL_RES, ProductData::TYPE_FLOAT64, "deg",
                           "pixel resolution in geocoded image");
    AddAbstractedAttribute(abs_root, LON_PIXEL_RES, ProductData::TYPE_FLOAT64, "deg",
                           "pixel resolution in geocoded image");
    AddAbstractedAttribute(abs_root, SLANT_RANGE_TO_FIRST_PIXEL, ProductData::TYPE_FLOAT64, "m",
                           "Slant range to 1st data sample");

    // calibration
    AddAbstractedAttribute(abs_root, ANT_ELEV_CORR_FLAG, ProductData::TYPE_UINT8, "flag", "Antenna elevation applied");
    AddAbstractedAttribute(abs_root, RANGE_SPREAD_COMP_FLAG, ProductData::TYPE_UINT8, "flag",
                           "range spread compensation applied");
    AddAbstractedAttribute(abs_root, REPLICA_POWER_CORR_FLAG, ProductData::TYPE_UINT8, "flag",
                           "Replica pulse power correction applied");
    AddAbstractedAttribute(abs_root, ABS_CALIBRATION_FLAG, ProductData::TYPE_UINT8, "flag", "Product calibrated");
    AddAbstractedAttribute(abs_root, CALIBRATION_FACTOR, ProductData::TYPE_FLOAT64, "dB", "Calibration constant");
    AddAbstractedAttribute(abs_root, CHIRP_POWER, ProductData::TYPE_FLOAT64, "", "Chirp power");
    AddAbstractedAttribute(abs_root, INC_ANGLE_COMP_FLAG, ProductData::TYPE_UINT8, "flag",
                           "incidence angle compensation applied");
    AddAbstractedAttribute(abs_root, REF_INC_ANGLE, ProductData::TYPE_FLOAT64, "", "Reference incidence angle");
    AddAbstractedAttribute(abs_root, REF_SLANT_RANGE, ProductData::TYPE_FLOAT64, "", "Reference slant range");
    AddAbstractedAttribute(abs_root, REF_SLANT_RANGE_EXP, ProductData::TYPE_FLOAT64, "",
                           "Reference slant range exponent");
    AddAbstractedAttribute(abs_root, RESCALING_FACTOR, ProductData::TYPE_FLOAT64, "", "Rescaling factor");
    AddAbstractedAttribute(abs_root, BISTATIC_CORRECTION_APPLIED, ProductData::TYPE_UINT8, "flag", "");

    AddAbstractedAttribute(abs_root, RANGE_SAMPLING_RATE, ProductData::TYPE_FLOAT64, "MHz", "Range Sampling Rate");

    // range and azimuth bandwidths for InSAR
    AddAbstractedAttribute(abs_root, RANGE_BANDWIDTH, ProductData::TYPE_FLOAT64, "MHz", "Bandwidth total in range");
    AddAbstractedAttribute(abs_root, AZIMUTH_BANDWIDTH, ProductData::TYPE_FLOAT64, "Hz", "Bandwidth total in azimuth");

    // Multilook
    AddAbstractedAttribute(abs_root, MULTILOOK_FLAG, ProductData::TYPE_UINT8, "flag", "Multilook applied");

    // coregistration
    AddAbstractedAttribute(abs_root, COREGISTERED_STACK, ProductData::TYPE_UINT8, "flag", "Coregistration applied");

    AddAbstractedAttribute(abs_root, EXTERNAL_CALIBRATION_FILE, ProductData::TYPE_ASCII, "",
                           "External calibration file used");
    AddAbstractedAttribute(abs_root, ORBIT_STATE_VECTOR_FILE, ProductData::TYPE_ASCII, "", "Orbit file used");

    abs_root->AddElement(std::make_shared<MetadataElement>(ORBIT_STATE_VECTORS));
    abs_root->AddElement(std::make_shared<MetadataElement>(SRGR_COEFFICIENTS));
    abs_root->AddElement(std::make_shared<MetadataElement>(DOP_COEFFICIENTS));

    att = AddAbstractedAttribute(abs_root, ABSTRACTED_METADATA_VERSION, ProductData::TYPE_ASCII, "",
                                 "AbsMetadata version");
    att->GetData()->SetElems(METADATA_VERSION);

    return abs_root;
}

std::vector<std::shared_ptr<MetadataElement>> AbstractMetadata::GetBandAbsMetadataList(
    std::shared_ptr<MetadataElement> abs_root) {
    std::vector<std::shared_ptr<MetadataElement>> band_metadata_list;
    std::vector<std::shared_ptr<MetadataElement>> children = abs_root->GetElements();
    for (std::shared_ptr<MetadataElement> child : children) {
        if (child->GetName().find(BAND_PREFIX, 0) == 0) {
            band_metadata_list.push_back(child);
        }
    }
    return band_metadata_list;
}

void AbstractMetadata::SetOrbitStateVectors(const std::shared_ptr<MetadataElement>& abs_root,
                                            const std::vector<OrbitStateVector>& orbit_state_vectors) {
    std::shared_ptr<MetadataElement> elem_root = abs_root->GetElement(ORBIT_STATE_VECTORS);

    // remove old
    std::vector<std::shared_ptr<MetadataElement>> old_list = elem_root->GetElements();
    for (std::shared_ptr<MetadataElement> old : old_list) {
        elem_root->RemoveElement(old);
    }
    // add new
    int i = 1;
    for (const OrbitStateVector& vector : orbit_state_vectors) {
        std::shared_ptr<MetadataElement> sub_elem_root =
            std::make_shared<MetadataElement>(std::string{ORBIT_VECTOR} + std::to_string(i));
        elem_root->AddElement(sub_elem_root);
        ++i;
        sub_elem_root->SetAttributeUtc(ORBIT_VECTOR_TIME, vector.time_);
        sub_elem_root->SetAttributeDouble(ORBIT_VECTOR_X_POS, vector.x_pos_);
        sub_elem_root->SetAttributeDouble(ORBIT_VECTOR_Y_POS, vector.y_pos_);
        sub_elem_root->SetAttributeDouble(ORBIT_VECTOR_Z_POS, vector.z_pos_);
        sub_elem_root->SetAttributeDouble(ORBIT_VECTOR_X_VEL, vector.x_vel_);
        sub_elem_root->SetAttributeDouble(ORBIT_VECTOR_Y_VEL, vector.y_vel_);
        sub_elem_root->SetAttributeDouble(ORBIT_VECTOR_Z_VEL, vector.z_vel_);
    }
}

void AbstractMetadata::DefaultToProduct(const std::shared_ptr<MetadataElement>& abstracted_metadata,
                                        const std::shared_ptr<Product>& the_product) {
    SetAttribute(abstracted_metadata, PRODUCT, the_product->GetName());
    SetAttribute(abstracted_metadata, PRODUCT_TYPE, the_product->GetProductType());
    SetAttribute(abstracted_metadata, SPH_DESCRIPTOR, the_product->GetDescription());

    SetAttribute(abstracted_metadata, NUM_OUTPUT_LINES, the_product->GetSceneRasterHeight());
    SetAttribute(abstracted_metadata, NUM_SAMPLES_PER_LINE, the_product->GetSceneRasterWidth());

    SetAttribute(abstracted_metadata, FIRST_LINE_TIME, the_product->GetStartTime());
    SetAttribute(abstracted_metadata, LAST_LINE_TIME, the_product->GetEndTime());

    //        todo:add support
    //        if (product->GetProductReader() != nullptr && product->GetProductReader().getReaderPlugIn() !=
    //        nullptr) {
    //            SetAttribute(
    //                abstracted_metadata, MISSION,
    //                product->GetProductReader().getReaderPlugIn().getFormatNames()[0]);
    //        }
}

void AbstractMetadata::PatchMissingMetadata(const std::shared_ptr<MetadataElement>& abstracted_metadata) {
    std::string version = abstracted_metadata->GetAttributeString(ABSTRACTED_METADATA_VERSION, "");
    if (version == METADATA_VERSION) {
        return;
    }
    auto tmp_elem = std::make_shared<MetadataElement>("tmp");
    std::shared_ptr<MetadataElement> complete_metadata = AddAbstractedMetadataHeader(tmp_elem);

    auto attribs = complete_metadata->GetAttributes();
    for (std::shared_ptr<MetadataAttribute> at : attribs) {
        if (!abstracted_metadata->ContainsAttribute(at->GetName())) {
            abstracted_metadata->AddAttribute(at);
            abstracted_metadata->GetProduct()->SetModified(false);
        }
    }
}

void AbstractMetadata::MigrateToCurrentVersion(const std::shared_ptr<MetadataElement>& abstracted_metadata) {
    // check if version has changed
    std::string version = abstracted_metadata->GetAttributeString(ABSTRACTED_METADATA_VERSION, "");
    if (version == METADATA_VERSION) return;
    // todo
}

std::shared_ptr<Utc> AbstractMetadata::ParseUtc(std::string_view time_str, std::string_view date_format_pattern) {
    try {
        int dot_pos = time_str.find_last_of('.');
        if (dot_pos > 0) {
            std::string new_time_str =
                std::string(time_str).substr(0, std::min(dot_pos + 7, static_cast<int>(time_str.length())));
            try {
                return Utc::Parse(new_time_str, date_format_pattern);
            } catch (const std::exception& e) {
                auto time = Utc::Parse(new_time_str, date_format_pattern);
                return time;
            }
        }
        return Utc::Parse(time_str, date_format_pattern);
    } catch (const std::exception& e) {
        LOGW << "UTC parse error:" << time_str << ":" << e.what();
        return NO_METADATA_UTC;
    }
}

std::shared_ptr<MetadataElement> AbstractMetadata::AddOriginalProductMetadata(
    const std::shared_ptr<MetadataElement>& root) {
    std::shared_ptr<MetadataElement> orig_metadata = root->GetElement(ORIGINAL_PRODUCT_METADATA);
    if (orig_metadata) {
        return orig_metadata;
    }
    orig_metadata = std::make_shared<MetadataElement>(ORIGINAL_PRODUCT_METADATA);
    root->AddElement(orig_metadata);
    return orig_metadata;
}

std::shared_ptr<MetadataElement> AbstractMetadata::GetOriginalProductMetadata(const std::shared_ptr<Product>& p) {
    std::shared_ptr<MetadataElement> root = p->GetMetadataRoot();
    std::shared_ptr<MetadataElement> orig_metadata = root->GetElement(ORIGINAL_PRODUCT_METADATA);
    if (orig_metadata == nullptr) {
        return root;
    }
    return orig_metadata;
}

void AbstractMetadata::AddBandToBandMap(const std::shared_ptr<MetadataElement>& band_abs_root, std::string_view name) {
    std::string band_names = band_abs_root->GetAttributeString(BAND_NAMES);
    if (band_names == NO_METADATA_STRING) {
        band_names = "";
    }
    if (!band_names.empty()) {
        band_names += ' ';
    }
    band_names += name;
    band_abs_root->SetAttributeString(BAND_NAMES, band_names);
}

bool AbstractMetadata::IsNoData(const std::shared_ptr<MetadataElement>& elem, std::string_view tag) {
    std::string val = elem->GetAttributeString(tag, NO_METADATA_STRING);
    boost::algorithm::trim(val);
    return val == NO_METADATA_STRING || val.empty();
}

std::shared_ptr<MetadataElement> AbstractMetadata::AddBandAbstractedMetadata(
    const std::shared_ptr<snapengine::MetadataElement>& abs_root, std::string_view name) {
    std::shared_ptr<MetadataElement> band_root = abs_root->GetElement(name);
    if (band_root == nullptr) {
        band_root = std::make_shared<MetadataElement>(name);
        abs_root->AddElement(band_root);
    }

    AddAbstractedAttribute(band_root, SWATH, ProductData::TYPE_ASCII, "", "Swath name");
    AddAbstractedAttribute(band_root, POLARIZATION, ProductData::TYPE_ASCII, "", "Polarization");
    AddAbstractedAttribute(band_root, ANNOTATION, ProductData::TYPE_ASCII, "", "metadata file");
    AddAbstractedAttribute(band_root, BAND_NAMES, ProductData::TYPE_ASCII, "", "corresponding bands");

    AddAbstractedAttribute(band_root, FIRST_LINE_TIME, ProductData::TYPE_UTC, "utc", "First zero doppler azimuth time");
    AddAbstractedAttribute(band_root, LAST_LINE_TIME, ProductData::TYPE_UTC, "utc", "Last zero doppler azimuth time");
    AddAbstractedAttribute(band_root, LINE_TIME_INTERVAL, ProductData::TYPE_FLOAT64, "s", "Time per line");

    AddAbstractedAttribute(band_root, NUM_OUTPUT_LINES, ProductData::TYPE_UINT32, "lines", "Raster height");
    AddAbstractedAttribute(band_root, NUM_SAMPLES_PER_LINE, ProductData::TYPE_UINT32, "samples", "Raster width");
    AddAbstractedAttribute(band_root, SAMPLE_TYPE, ProductData::TYPE_ASCII, "", "DETECTED or COMPLEX");

    AddAbstractedAttribute(band_root, CALIBRATION_FACTOR, ProductData::TYPE_FLOAT64, "", "Calibration constant");

    return band_root;
}

std::shared_ptr<MetadataElement> AbstractMetadata::GetBandAbsMetadata(
    const std::shared_ptr<snapengine::MetadataElement>& abs_root, const std::shared_ptr<snapengine::Band>& band) {
    std::vector<std::shared_ptr<snapengine::MetadataElement>> children = abs_root->GetElements();
    for (const auto& child : children) {
        if (boost::algorithm::starts_with(child->GetName(), BAND_PREFIX)) {
            std::vector<std::string> band_name_array =
                snapengine::StringUtils::StringToVectorByDelimiter(child->GetAttributeString(BAND_NAMES), " ");
            for (const auto& band_name : band_name_array) {
                if (band_name == band->GetName()) {
                    return child;
                }
            }
        }
    }
    return nullptr;
}

}  // namespace snapengine
}  // namespace alus