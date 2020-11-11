/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.engine_utilities.datamodel.AbstractMetadata.java ported
 * for native code. Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2016 by Array Systems Computing Inc. http://www.array.ca"
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

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "metadata_element.h"
#include "orbit_state_vector.h"
#include "product.h"
#include "product_data_utc.h"

namespace alus::snapengine {

class MetaDataNodeNames {
private:
    static constexpr std::string_view METADATA_VERSION = "6.0";

    // todo:MigrateToCurrentVersion is half backed already inside snap, we need to decide what to do with that (future)
    static void MigrateToCurrentVersion(std::shared_ptr<MetadataElement> abstracted_metadata);

    static void PatchMissingMetadata(std::shared_ptr<MetadataElement> abstracted_metadata);

    static void DefaultToProduct(std::shared_ptr<MetadataElement> abstracted_metadata,
                                 std::shared_ptr<Product> product);

public:
    /**
     * Default no data values
     */
    static inline std::shared_ptr<Utc> NO_METADATA_UTC = std::make_shared<Utc>(0.0);

    static constexpr int NO_METADATA = 99999;
    static constexpr short NO_METADATA_BYTE = 0;
    static constexpr std::string_view NO_METADATA_STRING = "-";

    static constexpr std::string_view ABSTRACTED_METADATA_VERSION = "metadata_version";
    static constexpr std::string_view ABSTRACT_METADATA_ROOT = "Abstracted_Metadata";
    static constexpr std::string_view ORIGINAL_PRODUCT_METADATA = "Original_Product_Metadata";
    static constexpr std::string_view BAND_PREFIX = "Band_";

    /**
     * Abstracted metadata generic to most EO products
     */
    static constexpr std::string_view SLAVE_METADATA_ROOT = "Slave_Metadata";
    static constexpr std::string_view MASTER_BANDS = "Master_bands";
    static constexpr std::string_view SLAVE_BANDS = "Slave_bands";

    static constexpr std::string_view PRODUCT = "PRODUCT";
    static constexpr std::string_view PRODUCT_TYPE = "PRODUCT_TYPE";
    static constexpr std::string_view SPH_DESCRIPTOR = "SPH_DESCRIPTOR";
    static constexpr std::string_view PATH = "PATH";
    static constexpr std::string_view MISSION = "MISSION";
    static constexpr std::string_view ACQUISITION_MODE = "ACQUISITION_MODE";
    static constexpr std::string_view ANTENNA_POINTING = "antenna_pointing";
    static constexpr std::string_view BEAMS = "BEAMS";
    static constexpr std::string_view ANNOTATION = "annotation";
    static constexpr std::string_view BAND_NAMES = "band_names";
    static constexpr std::string_view SWATH = "SWATH";
    // Not following style policy because of the Metadata naming context.
    static constexpr std::string_view swath = "swath";
    static constexpr std::string_view PROC_TIME = "PROC_TIME";
    static constexpr std::string_view ProcessingSystemIdentifier = "Processing_system_identifier";
    static constexpr std::string_view CYCLE = "orbit_cycle";
    static constexpr std::string_view REL_ORBIT = "REL_ORBIT";
    static constexpr std::string_view ABS_ORBIT = "ABS_ORBIT";
    static constexpr std::string_view STATE_VECTOR_TIME = "STATE_VECTOR_TIME";
    static constexpr std::string_view VECTOR_SOURCE = "VECTOR_SOURCE";

    // SPH
    static constexpr std::string_view SLICE_NUM = "slice_num";
    static constexpr std::string_view DATA_TAKE_ID = "data_take_id";
    static constexpr std::string_view FIRST_LINE_TIME = "first_line_time";
    static constexpr std::string_view LAST_LINE_TIME = "last_line_time";
    static constexpr std::string_view FIRST_NEAR_LAT = "first_near_lat";
    static constexpr std::string_view FIRST_NEAR_LONG = "first_near_long";
    static constexpr std::string_view FIRST_FAR_LAT = "first_far_lat";
    static constexpr std::string_view FIRST_FAR_LONG = "first_far_long";
    static constexpr std::string_view LAST_NEAR_LAT = "last_near_lat";
    static constexpr std::string_view LAST_NEAR_LONG = "last_near_long";
    static constexpr std::string_view LAST_FAR_LAT = "last_far_lat";
    static constexpr std::string_view LAST_FAR_LONG = "last_far_long";

    static constexpr std::string_view PASS = "PASS";
    static constexpr std::string_view SAMPLE_TYPE = "SAMPLE_TYPE";
    // this is against our style policy, but what can I do ...
    static constexpr std::string_view sample_type = "sample_type";

    // SAR Specific

    static constexpr std::string_view INCIDENCE_NEAR = "incidence_near";
    static constexpr std::string_view INCIDENCE_FAR = "incidence_far";

    static constexpr std::string_view MDS1_TX_RX_POLAR = "mds1_tx_rx_polar";
    static constexpr std::string_view MDS2_TX_RX_POLAR = "mds2_tx_rx_polar";
    static constexpr std::string_view MDS3_TX_RX_POLAR = "mds3_tx_rx_polar";
    static constexpr std::string_view MDS4_TX_RX_POLAR = "mds4_tx_rx_polar";
    static constexpr std::string_view POLARIZATION = "polarization";
    static constexpr std::string_view POLSARDATA = "polsar_data";
    // might not be needed
    //    static constexpr String[] polarTags = {AbstractMetadata.mds1_tx_rx_polar,
    //                                           AbstractMetadata.mds2_tx_rx_polar,
    //                                           AbstractMetadata.mds3_tx_rx_polar,
    //                                           AbstractMetadata.mds4_tx_rx_polar};
    static constexpr std::string_view ALGORITHM = "algorithm";
    static constexpr std::string_view AZIMUTH_LOOKS = "azimuth_looks";
    static constexpr std::string_view RANGE_LOOKS = "range_looks";
    static constexpr std::string_view RANGE_SPACING = "range_spacing";
    static constexpr std::string_view AZIMUTH_SPACING = "azimuth_spacing";
    static constexpr std::string_view PULSE_REPETITION_FREQUENCY = "pulse_repetition_frequency";
    static constexpr std::string_view RADAR_FREQUENCY = "radar_frequency";
    static constexpr std::string_view LINE_TIME_INTERVAL = "line_time_interval";

    static constexpr std::string_view TOT_SIZE = "total_size";
    static constexpr std::string_view NUM_OUTPUT_LINES = "num_output_lines";
    static constexpr std::string_view NUM_SAMPLES_PER_LINE = "num_samples_per_line";

    static constexpr std::string_view SUBSET_OFFSET_X = "subset_offset_x";
    static constexpr std::string_view SUBSET_OFFSET_Y = "subset_offset_y";

    // SRGR
    static constexpr std::string_view SRGR_FLAG = "srgr_flag";
    static constexpr std::string_view MAP_PROJECTION = "map_projection";

    // Line times
    static constexpr std::string_view FIRST_VALID_LINE_TIME = "firstValidLineTime";
    static constexpr std::string_view LAST_VALID_LINE_TIME = "lastValidLineTime";

    // Valid pixel indexes
    static constexpr std::string_view FIRST_VALID_PIXEL = "firstValidPixel";
    static constexpr std::string_view LAST_VALID_PIXEL = "lastValidPixel";

    // calibration and flags
    static constexpr std::string_view ANT_ELEV_CORR_FLAG = "ant_elev_corr_flag";
    static constexpr std::string_view RANGE_SPREAD_COMP_FLAG = "range_spread_comp_flag";
    static constexpr std::string_view INC_ANGLE_COMP_FLAG = "inc_angle_comp_flag";
    static constexpr std::string_view ABS_CALIBRATION_FLAG = "abs_calibration_flag";
    static constexpr std::string_view CALIBRATION_FACTOR = "calibration_factor";
    static constexpr std::string_view CHIRP_POWER = "chirp_power";
    static constexpr std::string_view REPLICA_POWER_CORR_FLAG = "replica_power_corr_flag";
    static constexpr std::string_view RANGE_SAMPLING_RATE = "range_sampling_rate";
    static constexpr std::string_view AVG_SCENE_HEIGHT = "avg_scene_height";
    static constexpr std::string_view MULTILOOK_FLAG = "multilook_flag";
    static constexpr std::string_view BISTATIC_CORRECTION_APPLIED = "bistatic_correction_applied";

    // cosmo calibration
    static constexpr std::string_view REF_INC_ANGLE = "ref_inc_angle";
    static constexpr std::string_view REF_SLANT_RANGE = "ref_slant_range";
    static constexpr std::string_view REF_SLANT_RANGE_EXP = "ref_slant_range_exp";
    static constexpr std::string_view RESCALING_FACTOR = "rescaling_factor";

    static constexpr std::string_view COREGISTERED_STACK = "coregistered_stack";
    static constexpr std::string_view BISTATIC_STACK = "bistatic_stack";

    static constexpr std::string_view EXTERNAL_CALIBRATION_FILE = "external_calibration_file";
    static constexpr std::string_view ORBIT_STATE_VECTOR_FILE = "orbit_state_vector_file";
    static constexpr std::string_view TARGET_REPORT_FILE = "target_report_file";
    static constexpr std::string_view WIND_FIELD_REPORT_FILE = "wind_field_report_file";

    // orbit state vectors
    static constexpr std::string_view ORBIT_STATE_VECTORS = "Orbit_State_Vectors";
    static constexpr std::string_view ORBIT_VECTOR = "orbit_vector";
    static constexpr std::string_view ORBIT_VECTOR_TIME = "time";
    static constexpr std::string_view ORBIT_VECTOR_X_POS = "x_pos";
    static constexpr std::string_view ORBIT_VECTOR_Y_POS = "y_pos";
    static constexpr std::string_view ORBIT_VECTOR_Z_POS = "z_pos";
    static constexpr std::string_view ORBIT_VECTOR_X_VEL = "x_vel";
    static constexpr std::string_view ORBIT_VECTOR_Y_VEL = "y_vel";
    static constexpr std::string_view ORBIT_VECTOR_Z_VEL = "z_vel";

    // SRGR Coefficients
    static constexpr std::string_view SRGR_COEFFICIENTS = "SRGR_Coefficients";
    static constexpr std::string_view SRGR_COEF_LIST = "srgr_coef_list";
    static constexpr std::string_view SRGR_COEF_TIME = "zero_doppler_time";
    static constexpr std::string_view GROUND_RANGE_ORIGIN = "ground_range_origin";
    static constexpr std::string_view COEFFICIENT = "coefficient";
    static constexpr std::string_view SRGR_COEF = "srgr_coef";

    // Doppler Centroid Coefficients
    static constexpr std::string_view DOP_COEFFICIENTS = "Doppler_Centroid_Coefficients";
    static constexpr std::string_view DOP_COEF_LIST = "dop_coef_list";
    static constexpr std::string_view DOP_COEF_TIME = "zero_doppler_time";
    static constexpr std::string_view SLANT_RANGE_TIME = "slant_range_time";
    static constexpr std::string_view DOP_COEF = "dop_coef";

    // orthorectification
    static constexpr std::string_view IS_TERRAIN_CORRECTED = "is_terrain_corrected";
    static constexpr std::string_view DEM = "DEM";
    static constexpr std::string_view GEO_REF_SYSTEM = "geo_ref_system";
    static constexpr std::string_view LAT_PIXEL_RES = "lat_pixel_res";
    static constexpr std::string_view LON_PIXEL_RES = "lon_pixel_res";
    static constexpr std::string_view SLANT_RANGE_TO_FIRST_PIXEL = "slant_range_to_first_pixel";

    // bandwidths for insar
    static constexpr std::string_view RANGE_BANDWIDTH = "range_bandwidth";
    static constexpr std::string_view AZIMUTH_BANDWIDTH = "azimuth_bandwidth";

    static constexpr std::string_view COMPACT_MODE = "compact_mode";

    static bool GetAttributeBoolean(std::shared_ptr<MetadataElement> element, std::string_view view);

    static double GetAttributeDouble(std::shared_ptr<MetadataElement> element, std::string_view view);

    static std::shared_ptr<Utc> ParseUtc(std::string_view time_str);

    /**
     * Get orbit state vectors.
     *
     * @param absRoot Abstracted metadata root.
     * @return orbitStateVectors Array of orbit state vectors.
     */
    static std::vector<OrbitStateVector> GetOrbitStateVectors(std::shared_ptr<MetadataElement> abs_root);

    /**
     * Get abstracted metadata.
     *
     * @param sourceProduct the product
     * @return MetadataElement or null if no root found
     */
    static std::shared_ptr<MetadataElement> GetAbstractedMetadata(std::shared_ptr<Product> source_product);

    /**
     * Adds an attribute into dest
     *
     * @param dest     the destination element
     * @param tag      the name of the attribute
     * @param dataType the ProductData type
     * @param unit     The unit
     * @param desc     The description
     * @return the newly created attribute
     */
    static std::shared_ptr<MetadataAttribute> AddAbstractedAttribute(std::shared_ptr<MetadataElement> dest,
                                                                     std::string_view tag, int data_type,
                                                                     std::string_view unit, std::string_view desc);
    /**
     * Sets an attribute as an int
     *
     * @param dest  the destination element
     * @param tag   the name of the attribute
     * @param value the string value
     */
    static void SetAttribute(std::shared_ptr<MetadataElement> dest, std::string_view tag, int value);

    /**
     * Sets an attribute as a string
     *
     * @param dest  the destination element
     * @param tag   the name of the attribute
     * @param value the string value
     */
    static void SetAttribute(std::shared_ptr<MetadataElement> dest, std::string_view tag,
                             std::optional<std::string> value);
    /**
     * Sets an attribute as a UTC
     *
     * @param dest  the destination element
     * @param tag   the name of the attribute
     * @param value the UTC value
     */
    static void SetAttribute(std::shared_ptr<MetadataElement> dest, std::string_view tag, std::shared_ptr<Utc> value);

    /**
     * Abstract common metadata from products to be used uniformly by all operators
     *
     * @param root the product metadata root
     * @return abstracted metadata root
     */
    static std::shared_ptr<MetadataElement> AddAbstractedMetadataHeader(std::shared_ptr<MetadataElement> root);

    /**
     * Set orbit state vectors.
     *
     * @param absRoot           Abstracted metadata root.
     * @param orbitStateVectors The orbit state vectors.
     * @throws Exception if orbit state vector length is not correct
     */
    static void SetOrbitStateVectors(std::shared_ptr<MetadataElement> abs_root,
                                     std::vector<OrbitStateVector> orbit_state_vectors);
};

}  // namespace alus::snapengine
