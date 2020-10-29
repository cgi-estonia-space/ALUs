/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.dataio.dimap.DimapProductConstants.java ported
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

#include <string_view>

namespace alus::snapengine {

class DimapProductConstants {
   public:
    static constexpr std::string_view DIMAP_FORMAT_NAME = "BEAM-DIMAP";
    /**
     * BEAM-Dimap XML-File extension
     */
    static constexpr std::string_view DIMAP_HEADER_FILE_EXTENSION = ".dim";
    /**
     * BEAM-Dimap data directory extension
     */
    static constexpr std::string_view DIMAP_DATA_DIRECTORY_EXTENSION = ".data";
    static constexpr std::string_view IMAGE_FILE_EXTENSION = ".img"; /* ENVI specific */
    static constexpr std::string_view TIE_POINT_GRID_DIR_NAME = "tie_point_grids";

    /**
     * BEAM-DIMAP version number.
     * <p>
     * Important note: If you change this number, update the BEAM-DIMAP version history given at {@link
     * DimapProductWriterPlugIn}.
     */
    static constexpr std::string_view DIMAP_CURRENT_VERSION = "2.12.1";

    // BEAM-Dimap default text
    static constexpr std::string_view DIMAP_METADATA_PROFILE = "BEAM-DATAMODEL-V1";
    static constexpr std::string_view DIMAP_DATASET_SERIES = "BEAM-PRODUCT";
    static constexpr std::string_view DATASET_PRODUCER_NAME = " ";
    // final static String DATASET_PRODUCER_NAME = "Brockmann-Consult | Phone +49 (04152) 889 300";
    static constexpr std::string_view DATA_FILE_FORMAT = "ENVI";
    static constexpr std::string_view DATA_FILE_FORMAT_DESCRIPTION = "ENVI File Format";
    static constexpr std::string_view DATA_FILE_ORGANISATION = "BAND_SEPARATE";

    // BEAM-Dimap document root tag
    static constexpr std::string_view TAG_ROOT = "Dimap_Document";

    // BEAM-Dimap metadata ID tags
    static constexpr std::string_view TAG_METADATA_ID = "Metadata_Id";
    static constexpr std::string_view TAG_METADATA_FORMAT = "METADATA_FORMAT";
    static constexpr std::string_view TAG_METADATA_PROFILE = "METADATA_PROFILE";

    // BEAM-Dimap production tags
    static constexpr std::string_view TAG_PRODUCTION = "Production";
    static constexpr std::string_view TAG_DATASET_PRODUCER_NAME = "DATASET_PRODUCER_NAME";
    static constexpr std::string_view TAG_DATASET_PRODUCER_URL = "DATASET_PRODUCER_URL";
    static constexpr std::string_view TAG_DATASET_PRODUCTION_DATE = "DATASET_PRODUCTION_DATE";
    static constexpr std::string_view TAG_QUICKLOOK_BAND_NAME = "QUICKLOOK_BAND_NAME";
    static constexpr std::string_view TAG_JOB_ID = "JOB_ID";
    static constexpr std::string_view TAG_PRODUCT_TYPE = "PRODUCT_TYPE";
    static constexpr std::string_view TAG_PRODUCT_INFO = "PRODUCT_INFO";
    static constexpr std::string_view TAG_PROCESSING_REQUEST = "PROCESSING_REQUEST";
    static constexpr std::string_view TAG_REQUEST = "Request";
    static constexpr std::string_view TAG_PARAMETER = "Parameter";
    static constexpr std::string_view TAG_INPUTPRODUCT = "InputProduct";
    static constexpr std::string_view TAG_OUTPUTPRODUCT = "OutputProduct";
    static constexpr std::string_view TAG_PRODUCT_SCENE_RASTER_START_TIME = "PRODUCT_SCENE_RASTER_START_TIME";
    static constexpr std::string_view TAG_PRODUCT_SCENE_RASTER_STOP_TIME = "PRODUCT_SCENE_RASTER_STOP_TIME";
    static constexpr std::string_view TAG_OLD_SCENE_RASTER_START_TIME = "SENSING_START";
    static constexpr std::string_view TAG_OLD_SCENE_RASTER_STOP_TIME = "SENSING_STOP";

    // BEAM-Dimap geocoding tags
    static constexpr std::string_view TAG_COORDINATE_REFERENCE_SYSTEM = "Coordinate_Reference_System";
    static constexpr std::string_view TAG_GEOCODING_TIE_POINT_GRIDS = "Geocoding_Tie_Point_Grids";
    static constexpr std::string_view TAG_GEOPOSITION_POINTS = "Geoposition_Points";
    static constexpr std::string_view TAG_ORIGINAL_GEOCODING = "Original_Geocoding";
    static constexpr std::string_view TAG_INTERPOLATION_METHOD = "INTERPOLATION_METHOD";
    static constexpr std::string_view TAG_TIE_POINT_GRID_NAME_LAT = "TIE_POINT_GRID_NAME_LAT";
    static constexpr std::string_view TAG_TIE_POINT_GRID_NAME_LON = "TIE_POINT_GRID_NAME_LON";
    static constexpr std::string_view TAG_GEOCODING_MAP = "Geocoding_Map";
    static constexpr std::string_view TAG_GEOCODING_MAP_INFO = "MAP_INFO";
    static constexpr std::string_view TAG_LATITUDE_BAND = "LATITUDE_BAND";
    static constexpr std::string_view TAG_LONGITUDE_BAND = "LONGITUDE_BAND";
    static constexpr std::string_view TAG_VALID_MASK_EXPRESSION = "VALID_MASK_EXPRESSION";
    static constexpr std::string_view TAG_SEARCH_RADIUS = "SEARCH_RADIUS";
    static constexpr std::string_view TAG_PIXEL_POSITION_ESTIMATOR = "Pixel_Position_Estimator";
    static constexpr std::string_view TAG_WKT = "WKT";
    // This Tag is used for geo-coding support and multi size support
    static constexpr std::string_view TAG_IMAGE_TO_MODEL_TRANSFORM = "IMAGE_TO_MODEL_TRANSFORM";

    //  -since version 2.0.0
    static constexpr std::string_view TAG_HORIZONTAL_CS_TYPE = "HORIZONTAL_CS_TYPE";

    static constexpr std::string_view TAG_MAP_INFO_PIXEL_X = "PIXEL_X";
    static constexpr std::string_view TAG_MAP_INFO_PIXEL_Y = "PIXEL_Y";
    static constexpr std::string_view TAG_MAP_INFO_EASTING = "EASTING";
    static constexpr std::string_view TAG_MAP_INFO_NORTHING = "NORTHING";
    static constexpr std::string_view TAG_MAP_INFO_ORIENTATION = "ORIENTATION";
    static constexpr std::string_view TAG_MAP_INFO_PIXELSIZE_X = "PIXELSIZE_X";
    static constexpr std::string_view TAG_MAP_INFO_PIXELSIZE_Y = "PIXELSIZE_Y";
    static constexpr std::string_view TAG_MAP_INFO_NODATA_VALUE = "NODATA_VALUE";
    static constexpr std::string_view TAG_MAP_INFO_MAPUNIT = "MAPUNIT";
    static constexpr std::string_view TAG_MAP_INFO_ORTHORECTIFIED = "ORTHORECTIFIED";
    static constexpr std::string_view TAG_MAP_INFO_ELEVATION_MODEL = "ELEVATION_MODEL";
    static constexpr std::string_view TAG_MAP_INFO_SCENE_FITTED = "SCENE_FITTED";
    static constexpr std::string_view TAG_MAP_INFO_SCENE_WIDTH = "SCENE_WIDTH";
    static constexpr std::string_view TAG_MAP_INFO_SCENE_HEIGHT = "SCENE_HEIGHT";
    static constexpr std::string_view TAG_MAP_INFO_RESAMPLING = "RESAMPLING";

    static constexpr std::string_view TAG_GEOPOSITION = "Geoposition";
    static constexpr std::string_view TAG_GEOPOSITION_INSERT = "Geoposition_Insert";
    static constexpr std::string_view TAG_ULX_MAP = "ULXMAP";
    static constexpr std::string_view TAG_ULY_MAP = "ULYMAP";
    static constexpr std::string_view TAG_X_DIM = "XDIM";
    static constexpr std::string_view TAG_Y_DIM = "YDIM";
    static constexpr std::string_view TAG_SIMPLIFIED_LOCATION_MODEL = "Simplified_Location_Model";
    static constexpr std::string_view TAG_DIRECT_LOCATION_MODEL = "Direct_Location_Model";
    static constexpr std::string_view TAG_LC_LIST = "lc_List";
    static constexpr std::string_view TAG_LC = "lc";
    static constexpr std::string_view TAG_PC_LIST = "pc_List";
    static constexpr std::string_view TAG_PC = "pc";
    static constexpr std::string_view TAG_REVERSE_LOCATION_MODEL = "Reverse_Location_Model";
    static constexpr std::string_view TAG_IC_LIST = "ic_List";
    static constexpr std::string_view TAG_IC = "ic";
    static constexpr std::string_view TAG_JC_LIST = "jc_List";
    static constexpr std::string_view TAG_JC = "jc";

    //   - since version 1.4.0
    static constexpr std::string_view TAG_GEO_TABLES = "GEO_TABLES";
    static constexpr std::string_view TAG_HORIZONTAL_CS = "Horizontal_CS";
    static constexpr std::string_view TAG_HORIZONTAL_CS_NAME = "HORIZONTAL_CS_NAME";
    static constexpr std::string_view TAG_GEOGRAPHIC_CS = "Geographic_CS";
    static constexpr std::string_view TAG_GEOGRAPHIC_CS_NAME = "GEOGRAPHIC_CS_NAME";
    static constexpr std::string_view TAG_HORIZONTAL_DATUM = "Horizontal_Datum";
    static constexpr std::string_view TAG_HORIZONTAL_DATUM_NAME = "HORIZONTAL_DATUM_NAME";
    static constexpr std::string_view TAG_ELLIPSOID = "Ellipsoid";
    static constexpr std::string_view TAG_ELLIPSOID_NAME = "ELLIPSOID_NAME";
    static constexpr std::string_view TAG_ELLIPSOID_PARAMETERS = "Ellipsoid_Parameters";
    static constexpr std::string_view TAG_ELLIPSOID_MAJ_AXIS = "ELLIPSOID_MAJ_AXIS";
    static constexpr std::string_view TAG_ELLIPSOID_MIN_AXIS = "ELLIPSOID_MIN_AXIS";
    static constexpr std::string_view TAG_PROJECTION = "Projection";
    static constexpr std::string_view TAG_PROJECTION_NAME = "NAME";
    static constexpr std::string_view TAG_PROJECTION_CT_METHOD = "Projection_CT_Method";
    static constexpr std::string_view TAG_PROJECTION_CT_NAME = "PROJECTION_CT_NAME";
    static constexpr std::string_view TAG_PROJECTION_PARAMETERS = "Projection_Parameters";
    static constexpr std::string_view TAG_PROJECTION_PARAMETER = "Projection_Parameter";
    static constexpr std::string_view TAG_PROJECTION_PARAMETER_NAME = "PROJECTION_PARAMETER_NAME";
    static constexpr std::string_view TAG_PROJECTION_PARAMETER_VALUE = "PROJECTION_PARAMETER_VALUE";

    // BEAM-Dimap dataset id tags
    static constexpr std::string_view TAG_DATASET_ID = "Dataset_Id";
    static constexpr std::string_view TAG_DATASET_INDEX = "DATASET_INDEX";
    static constexpr std::string_view TAG_DATASET_SERIES = "DATASET_SERIES";
    static constexpr std::string_view TAG_DATASET_NAME = "DATASET_NAME";
    static constexpr std::string_view TAG_DATASET_DESCRIPTION = "DATASET_DESCRIPTION";
    static constexpr std::string_view TAG_DATASET_AUTO_GROUPING = "DATASET_AUTO_GROUPING";
    static constexpr std::string_view TAG_COPYRIGHT = "COPYRIGHT";
    static constexpr std::string_view TAG_COUNTRY_NAME = "COUNTRY_NAME";
    static constexpr std::string_view TAG_COUNTRY_CODE = "COUNTRY_CODE";
    static constexpr std::string_view TAG_DATASET_LOCATION = "DATASET_LOCATION";
    static constexpr std::string_view TAG_DATASET_TN_PATH = "DATASET_TN_PATH";
    static constexpr std::string_view TAG_DATASET_TN_FORMAT = "DATASET_TN_FORMAT";
    static constexpr std::string_view TAG_DATASET_QL_PATH = "DATASET_QL_PATH";
    static constexpr std::string_view TAG_DATASET_QL_FORMAT = "DATASET_QL_FORMAT";

    // BEAM_Dimap dataset use tags
    static constexpr std::string_view TAG_DATASET_USE = "Dataset_Use";
    static constexpr std::string_view TAG_DATASET_COMMENTS = "DATASET_COMMENTS";

    // BEAM-Dimap flag coding tags
    static constexpr std::string_view TAG_FLAG_CODING = "Flag_Coding";
    static constexpr std::string_view TAG_FLAG = "Flag";
    static constexpr std::string_view TAG_FLAG_NAME = "Flag_Name";
    static constexpr std::string_view TAG_FLAG_INDEX = "Flag_Index";
    static constexpr std::string_view TAG_FLAG_DESCRIPTION = "Flag_description";

    // BEAM-Dimap index coding tags
    static constexpr std::string_view TAG_INDEX_CODING = "Index_Coding";
    static constexpr std::string_view TAG_INDEX = "Index";
    static constexpr std::string_view TAG_INDEX_NAME = "INDEX_NAME";
    static constexpr std::string_view TAG_INDEX_VALUE = "INDEX_VALUE";
    static constexpr std::string_view TAG_INDEX_DESCRIPTION = "INDEX_DESCRIPTION";

    // BEAM-Dimap raster dimension tags
    static constexpr std::string_view TAG_RASTER_DIMENSIONS = "Raster_Dimensions";
    static constexpr std::string_view TAG_NCOLS = "NCOLS";
    static constexpr std::string_view TAG_NROWS = "NROWS";
    static constexpr std::string_view TAG_NBANDS = "NBANDS";

    // BEAM-Dimap tie point grid tags
    static constexpr std::string_view TAG_TIE_POINT_GRIDS = "Tie_Point_Grids";
    static constexpr std::string_view TAG_TIE_POINT_NUM_TIE_POINT_GRIDS = "NUM_TIE_POINT_GRIDS";
    static constexpr std::string_view TAG_TIE_POINT_GRID_INFO = "Tie_Point_Grid_Info";
    static constexpr std::string_view TAG_TIE_POINT_GRID_INDEX = "TIE_POINT_GRID_INDEX";
    static constexpr std::string_view TAG_TIE_POINT_DESCRIPTION = "TIE_POINT_DESCRIPTION";
    static constexpr std::string_view TAG_TIE_POINT_PHYSICAL_UNIT = "PHYSICAL_UNIT";
    static constexpr std::string_view TAG_TIE_POINT_GRID_NAME = "TIE_POINT_GRID_NAME";
    static constexpr std::string_view TAG_TIE_POINT_DATA_TYPE = "DATA_TYPE";
    static constexpr std::string_view TAG_TIE_POINT_NCOLS = "NCOLS";
    static constexpr std::string_view TAG_TIE_POINT_NROWS = "NROWS";
    static constexpr std::string_view TAG_TIE_POINT_OFFSET_X = "OFFSET_X";
    static constexpr std::string_view TAG_TIE_POINT_OFFSET_Y = "OFFSET_Y";
    static constexpr std::string_view TAG_TIE_POINT_STEP_X = "STEP_X";
    static constexpr std::string_view TAG_TIE_POINT_STEP_Y = "STEP_Y";
    static constexpr std::string_view TAG_TIE_POINT_CYCLIC = "CYCLIC";

    // BEAM-Dimap data access tags
    static constexpr std::string_view TAG_DATA_ACCESS = "Data_Access";
    static constexpr std::string_view TAG_DATA_FILE_FORMAT = "DATA_FILE_FORMAT";
    static constexpr std::string_view TAG_DATA_FILE_FORMAT_DESC = "DATA_FILE_FORMAT_DESC";
    static constexpr std::string_view TAG_DATA_FILE_ORGANISATION = "DATA_FILE_ORGANISATION";
    static constexpr std::string_view TAG_DATA_FILE = "Data_File";
    static constexpr std::string_view TAG_DATA_FILE_PATH = "DATA_FILE_PATH";
    static constexpr std::string_view TAG_BAND_INDEX = "BAND_INDEX";
    static constexpr std::string_view TAG_TIE_POINT_GRID_FILE = "Tie_Point_Grid_File";
    static constexpr std::string_view TAG_TIE_POINT_GRID_FILE_PATH = "TIE_POINT_GRID_FILE_PATH";

    // BEAM-Dimap image display tags
    static constexpr std::string_view TAG_IMAGE_DISPLAY = "Image_Display";
    static constexpr std::string_view TAG_BAND_STATISTICS = "Band_Statistics";
    static constexpr std::string_view TAG_STX_MIN = "STX_MIN";
    static constexpr std::string_view TAG_STX_MAX = "STX_MAX";
    static constexpr std::string_view TAG_STX_MEAN = "STX_MEAN";
    static constexpr std::string_view TAG_STX_STDDEV = "STX_STD_DEV";
    static constexpr std::string_view TAG_STX_LEVEL = "STX_RES_LEVEL";
    static constexpr std::string_view TAG_STX_LIN_MIN = "STX_LIN_MIN";
    static constexpr std::string_view TAG_STX_LIN_MAX = "STX_LIN_MAX";
    static constexpr std::string_view TAG_HISTOGRAM = "HISTOGRAM";
    static constexpr std::string_view TAG_NUM_COLORS = "NUM_COLORS";
    static constexpr std::string_view TAG_COLOR_PALETTE_POINT = "Color_Palette_Point";
    static constexpr std::string_view TAG_SAMPLE = "SAMPLE";
    static constexpr std::string_view TAG_LABEL = "LABEL";
    static constexpr std::string_view TAG_COLOR = "COLOR";
    static constexpr std::string_view TAG_GAMMA = "GAMMA";
    static constexpr std::string_view TAG_NO_DATA_COLOR = "NO_DATA_COLOR";
    static constexpr std::string_view TAG_HISTOGRAM_MATCHING = "HISTOGRAM_MATCHING";
    static constexpr std::string_view TAG_BITMASK_OVERLAY = "Bitmask_Overlay";
    static constexpr std::string_view TAG_BITMASK = "BITMASK";
    static constexpr std::string_view TAG_ROI_DEFINITION = "ROI_Definition";
    static constexpr std::string_view TAG_ROI_ONE_DIMENSIONS = "ROI_ONE_DIMENSIONS";
    static constexpr std::string_view TAG_VALUE_RANGE_MAX = "VALUE_RANGE_MAX";
    static constexpr std::string_view TAG_VALUE_RANGE_MIN = "VALUE_RANGE_MIN";
    static constexpr std::string_view TAG_BITMASK_ENABLED = "BITMASK_ENABLED";
    static constexpr std::string_view TAG_INVERTED = "INVERTED";
    static constexpr std::string_view TAG_OR_COMBINED = "OR_COMBINED";
    static constexpr std::string_view TAG_SHAPE_ENABLED = "SHAPE_ENABLED";
    static constexpr std::string_view TAG_SHAPE_FIGURE = "Shape_Figure";
    static constexpr std::string_view TAG_VALUE_RANGE_ENABLED = "VALUE_RANGE_ENABLED";
    static constexpr std::string_view TAG_PATH_SEG = "SEGMENT";
    static constexpr std::string_view TAG_PIN_USE_ENABLED = "PIN_USE_ENABLED";
    static constexpr std::string_view TAG_MASK_USAGE = "Mask_Usage";
    static constexpr std::string_view TAG_ROI = "ROI";
    static constexpr std::string_view TAG_OVERLAY = "OVERLAY";

    // BEAM-Dimap image interpretation tags
    static constexpr std::string_view TAG_IMAGE_INTERPRETATION = "Image_Interpretation";
    static constexpr std::string_view TAG_SPECTRAL_BAND_INFO = "Spectral_Band_Info";
    static constexpr std::string_view TAG_VIRTUAL_BAND_INFO = "Virtual_Band_Info";
    static constexpr std::string_view TAG_BAND_DESCRIPTION = "BAND_DESCRIPTION";
    static constexpr std::string_view TAG_PHYSICAL_GAIN = "PHYSICAL_GAIN";
    static constexpr std::string_view TAG_PHYSICAL_BIAS = "PHYSICAL_BIAS";
    static constexpr std::string_view TAG_PHYSICAL_UNIT = "PHYSICAL_UNIT";
    static constexpr std::string_view TAG_BAND_NAME = "BAND_NAME";
    static constexpr std::string_view TAG_BAND_RASTER_WIDTH = "BAND_RASTER_WIDTH";
    static constexpr std::string_view TAG_BAND_RASTER_HEIGHT = "BAND_RASTER_HEIGHT";
    static constexpr std::string_view TAG_DATA_TYPE = "DATA_TYPE";
    static constexpr std::string_view TAG_SOLAR_FLUX = "SOLAR_FLUX";
    static constexpr std::string_view TAG_SPECTRAL_BAND_INDEX = "SPECTRAL_BAND_INDEX";
    static constexpr std::string_view TAG_SOLAR_FLUX_UNIT = "SOLAR_FLUX_UNIT";
    static constexpr std::string_view TAG_BANDWIDTH = "BANDWIDTH";
    static constexpr std::string_view TAG_BAND_WAVELEN = "BAND_WAVELEN";
    static constexpr std::string_view TAG_WAVELEN_UNIT = "WAVELEN_UNIT";
    static constexpr std::string_view TAG_FLAG_CODING_NAME = "FLAG_CODING_NAME";
    static constexpr std::string_view TAG_INDEX_CODING_NAME = "INDEX_CODING_NAME";
    static constexpr std::string_view TAG_SCALING_FACTOR = "SCALING_FACTOR";
    static constexpr std::string_view TAG_SCALING_OFFSET = "SCALING_OFFSET";
    static constexpr std::string_view TAG_SCALING_LOG_10 = "LOG10_SCALED";
    static constexpr std::string_view TAG_VALID_MASK_TERM = "VALID_MASK_TERM";
    static constexpr std::string_view TAG_NO_DATA_VALUE_USED = "NO_DATA_VALUE_USED";
    static constexpr std::string_view TAG_NO_DATA_VALUE = "NO_DATA_VALUE";

    // Ancillary support
    static constexpr std::string_view TAG_ANCILLARY_RELATION = "ANCILLARY_RELATION";
    static constexpr std::string_view TAG_ANCILLARY_VARIABLE = "ANCILLARY_VARIABLE";

    // Virtual bands support
    static constexpr std::string_view TAG_VIRTUAL_BAND = "VIRTUAL_BAND";
    static constexpr std::string_view TAG_VIRTUAL_BAND_CHECK_INVALIDS = "CHECK_INVALIDS";
    static constexpr std::string_view TAG_VIRTUAL_BAND_EXPRESSION = "EXPRESSION";
    static constexpr std::string_view TAG_VIRTUAL_BAND_INVALID_VALUE = "INVALID_VALUE";
    static constexpr std::string_view TAG_VIRTUAL_BAND_USE_INVALID_VALUE = "USE_INVALID_VALUE";
    static constexpr std::string_view TAG_VIRTUAL_BAND_WRITE_DATA = "WRITE_DATA";

    // Filter bands support -- version 1.0
    //@Deprecated
    static constexpr std::string_view TAG_FILTER_SUB_WINDOW_WIDTH = "FILTER_SUB_WINDOW_WIDTH";
    //@Deprecated
    static constexpr std::string_view TAG_FILTER_SUB_WINDOW_HEIGHT = "FILTER_SUB_WINDOW_HEIGHT";

    // Filter bands support -- versions 1.0, 1.1
    static constexpr std::string_view TAG_FILTER_BAND_INFO = "Filter_Band_Info";
    static constexpr std::string_view TAG_FILTER_SOURCE = "FILTER_SOURCE";
    static constexpr std::string_view TAG_FILTER_KERNEL = "Filter_Kernel";
    static constexpr std::string_view TAG_FILTER_OP_TYPE = "FILTER_OP_TYPE";
    static constexpr std::string_view TAG_FILTER_SUB_WINDOW_SIZE = "FILTER_SUB_WINDOW_SIZE";
    static constexpr std::string_view TAG_FILTER_OPERATOR_CLASS_NAME = "FILTER_OPERATOR_CLASS_NAME";

    // Kernel support
    static constexpr std::string_view TAG_KERNEL_HEIGHT = "KERNEL_HEIGHT";
    static constexpr std::string_view TAG_KERNEL_WIDTH = "KERNEL_WIDTH";
    static constexpr std::string_view TAG_KERNEL_X_ORIGIN = "KERNEL_X_ORIGIN";  // new in 1.2
    static constexpr std::string_view TAG_KERNEL_Y_ORIGIN = "KERNEL_Y_ORIGIN";  // new in 1.2
    static constexpr std::string_view TAG_KERNEL_FACTOR = "KERNEL_FACTOR";
    static constexpr std::string_view TAG_KERNEL_DATA = "KERNEL_DATA";

    // BEAM-Dimap dataset sources tags
    static constexpr std::string_view TAG_DATASET_SOURCES = "Dataset_Sources";
    static constexpr std::string_view TAG_SOURCE_INFORMATION = "Source_Information";
    static constexpr std::string_view TAG_SOURCE_ID = "SOURCE_ID";
    static constexpr std::string_view TAG_SOURCE_TYPE = "SOURCE_TYPE";
    static constexpr std::string_view TAG_SOURCE_DESCRIPTION = "SOURCE_DESCRIPTION";
    static constexpr std::string_view TAG_SOURCE_FRAME = "Source_Frame";
    static constexpr std::string_view TAG_VERTEX = "Vertex";
    static constexpr std::string_view TAG_FRAME_LON = "FRAME_LON";
    static constexpr std::string_view TAG_FRAME_LAT = "FRAME_LAT";
    static constexpr std::string_view TAG_FRAME_X = "FRAME_X";
    static constexpr std::string_view TAG_FRAME_Y = "FRAME_Y";
    static constexpr std::string_view TAG_SCENE_SOURCE = "Scene_Source";
    static constexpr std::string_view TAG_MISSION = "MISSION";
    static constexpr std::string_view TAG_INSTRUMENT = "INSTRUMENT";
    static constexpr std::string_view TAG_IMAGING_MODE = "IMAGING_MODE";
    static constexpr std::string_view TAG_IMAGING_DATE = "IMAGING_DATE";
    static constexpr std::string_view TAG_IMAGING_TIME = "IMAGING_TIME";
    static constexpr std::string_view TAG_GRID_REFERENCE = "GRID_REFERENCE";
    static constexpr std::string_view TAG_SCENE_RECTIFICATION_ELEV = "SCENE_RECTIFICATION_ELEV";
    static constexpr std::string_view TAG_INCIDENCE_ANGLE = "INCIDENCE_ANGLE";
    static constexpr std::string_view TAG_THEORETICAL_RESOLUTION = "THEORETICAL_RESOLUTION";
    static constexpr std::string_view TAG_SUN_AZIMUTH = "SUN_AZIMUTH";
    static constexpr std::string_view TAG_SUN_ELEVATION = "SUN_ELEVATION";
    static constexpr std::string_view TAG_METADATA_ELEMENT = "MDElem";
    static constexpr std::string_view TAG_METADATA_VALUE = "VALUE";
//    static constexpr std::string_view TAG_METADATA_ATTRIBUTE = "MDATTR";
    static constexpr std::string_view TAG_METADATA_ATTRIBUTE = "MDATTR";

    // BEAM-Dimap mask definition tags
    static constexpr std::string_view TAG_MASKS = "Masks";
    static constexpr std::string_view TAG_MASK = "Mask";
    static constexpr std::string_view TAG_NAME = "NAME";
    static constexpr std::string_view TAG_DESCRIPTION = "DESCRIPTION";
    static constexpr std::string_view TAG_TRANSPARENCY = "TRANSPARENCY";
    static constexpr std::string_view TAG_MASK_RASTER_WIDTH = "MASK_RASTER_WIDTH";
    static constexpr std::string_view TAG_MASK_RASTER_HEIGHT = "MASK_RASTER_HEIGHT";

    // BandMathMask
    static constexpr std::string_view TAG_EXPRESSION = "EXPRESSION";
    // RangeMask
    static constexpr std::string_view TAG_MINIMUM = "MINIMUM";
    static constexpr std::string_view TAG_MAXIMUM = "MAXIMUM";
    static constexpr std::string_view TAG_RASTER = "RASTER";

    // BEAM-Dimap bitmask definition tags
    static constexpr std::string_view TAG_BITMASK_DEFINITIONS = "Bitmask_Definitions";
    static constexpr std::string_view TAG_BITMASK_DEFINITION = "Bitmask_Definition";
    static constexpr std::string_view TAG_BITMASK_DESCRIPTION = "TAG_DESCRIPTION";
    static constexpr std::string_view TAG_BITMASK_EXPRESSION = "TAG_EXPRESSION";
    static constexpr std::string_view TAG_BITMASK_COLOR = "TAG_COLOR";
    static constexpr std::string_view TAG_BITMASK_TRANSPARENCY = "TAG_TRANSPARENCY";

    // BEAM-Dimap placemark tags
    static constexpr std::string_view TAG_PLACEMARK = "Placemark";
    static constexpr std::string_view TAG_PLACEMARK_LABEL = "LABEL";
    static constexpr std::string_view TAG_PLACEMARK_DESCRIPTION = "DESCRIPTION";
    static constexpr std::string_view TAG_PLACEMARK_LATITUDE = "LATITUDE";
    static constexpr std::string_view TAG_PLACEMARK_LONGITUDE = "LONGITUDE";
    static constexpr std::string_view TAG_PLACEMARK_PIXEL_X = "PIXEL_X";
    static constexpr std::string_view TAG_PLACEMARK_PIXEL_Y = "PIXEL_Y";
    static constexpr std::string_view TAG_PLACEMARK_STYLE_CSS = "STYLE_CSS";
    /**
     * //@Deprecated since SNAP 2.0
     */
    //@Deprecated
    static constexpr std::string_view TAG_PLACEMARK_FILL_COLOR = "FillColor";
    /**
     * //@Deprecated since SNAP 2.0
     */
    //@Deprecated
    static constexpr std::string_view TAG_PLACEMARK_OUTLINE_COLOR = "OutlineColor";

    // BEAM-Dimap pin tags
    static constexpr std::string_view TAG_PIN_GROUP = "Pin_Group";
    static constexpr std::string_view TAG_PIN = "Pin";

    // BEAM-Dimap gcp tags
    static constexpr std::string_view TAG_GCP_GROUP = "Gcp_Group";

    // attribute
    static constexpr std::string_view ATTRIB_RED = "red";
    static constexpr std::string_view ATTRIB_GREEN = "green";
    static constexpr std::string_view ATTRIB_BLUE = "blue";
    static constexpr std::string_view ATTRIB_ALPHA = "alpha";
    static constexpr std::string_view ATTRIB_NAMES = "names";
    static constexpr std::string_view ATTRIB_DESCRIPTION = "desc";
    static constexpr std::string_view ATTRIB_UNIT = "unit";
    static constexpr std::string_view ATTRIB_MODE = "mode";
    static constexpr std::string_view ATTRIB_TYPE = "type";
    static constexpr std::string_view ATTRIB_ELEMS = "elems";
    static constexpr std::string_view ATTRIB_NAME = "name";
    static constexpr std::string_view ATTRIB_VERSION = "version";
    static constexpr std::string_view ATTRIB_HREF = "href";
    static constexpr std::string_view ATTRIB_VALUE = "value";
    static constexpr std::string_view ATTRIB_ORDER = "order";
    static constexpr std::string_view ATTRIB_INDEX = "index";
    static constexpr std::string_view ATTRIB_BAND_TYPE = "bandType";
};

}  // namespace alus::snapengine
