/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.sentinel1.Sentinel1Level1Directory.java
 * ported for native code.
 * Copied from(https://github.com/senbox-org/s1tbx). It was originally stated:
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
#include "s1tbx-io/sentinel1/sentinel1_level1_directory.h"

#include <stdexcept>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/tokenizer.hpp>

#include "alus_log.h"
#include "general_constants.h"
#include "s1tbx-commons/io/sar_reader.h"
#include "s1tbx-io/geotiffxml/geo_tiff_utils.h"
#include "s1tbx-io/sentinel1/sentinel1_constants.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/i_geo_coding.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/datamodel/product_data.h"
#include "snap-core/core/datamodel/tie_point_geo_coding.h"
#include "snap-core/core/datamodel/tie_point_grid.h"
#include "snap-core/core/util/io/file_utils.h"
#include "snap-core/core/util/math/math_utils.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata_i_o.h"
#include "snap-engine-utilities/engine-utilities/datamodel/unit.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"
#include "snap-engine-utilities/engine-utilities/gpf/operator_utils.h"
#include "snap-engine-utilities/engine-utilities/gpf/reader_utils.h"

namespace alus::s1tbx {

Sentinel1Level1Directory::Sentinel1Level1Directory(const boost::filesystem::path& input_file)
    : XMLProductDirectory(input_file) {}

void Sentinel1Level1Directory::AddBands(const std::shared_ptr<snapengine::Product>& product) {
    std::shared_ptr<snapengine::MetadataElement> abs_root =
        snapengine::AbstractMetadata::GetAbstractedMetadata(product);
    int cnt = 1;
    for (auto& string_image_i_o_file_entry : band_image_file_map_) {
        std::shared_ptr<ImageIOFile> img = string_image_i_o_file_entry.second;
        std::string img_name = img->GetName();
        boost::algorithm::to_lower(img_name);
        std::shared_ptr<snapengine::MetadataElement> band_metadata =
            abs_root->GetElement(img_band_metadata_map_.at(img_name));
        std::string swath = band_metadata->GetAttributeString(snapengine::AbstractMetadata::SWATH);
        std::string pol = band_metadata->GetAttributeString(snapengine::AbstractMetadata::POLARIZATION);
        int width = band_metadata->GetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE);
        int height = band_metadata->GetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES);
        int num_images = img->GetNumImages();

        std::string tpg_prefix;
        std::string suffix = pol;
        //        todo: java allows this from interface and abstract class which does not use interface, not sure if we
        //        get issues in c++
        if (AbstractProductDirectory::IsSLC()) {
            num_images *= 2;  // real + imaginary
            if (IsTOPSAR()) {
                (suffix = swath).append("_").append(pol);
                tpg_prefix = swath;
            } else if (acq_mode_ == "WV") {
                suffix.append("_").append(std::to_string(cnt));
                ++cnt;
            }
        }

        std::string band_name;
        bool real = true;
        std::shared_ptr<snapengine::Band> last_real_band = nullptr;
        for (int i = 0; i < num_images; ++i) {
            if (AbstractProductDirectory::IsSLC()) {
                std::string unit;

                for (int b = 0; b < img->GetNumBands(); ++b) {
                    if (real) {
                        band_name = std::string("i") + '_' + suffix;
                        unit = snapengine::Unit::REAL;
                    } else {
                        band_name = std::string("q") + '_' + suffix;
                        unit = snapengine::Unit::IMAGINARY;
                    }
                    auto band = std::make_shared<snapengine::Band>(band_name, snapengine::ProductData::TYPE_INT16,
                                                                   width, height);
                    band->SetUnit(std::make_optional(unit));
                    band->SetNoDataValueUsed(true);
                    band->SetNoDataValue(NO_DATA_VALUE);

                    product->AddBand(band);
                    std::shared_ptr<BandInfo> band_info = std::make_shared<BandInfo>(band, img, i, b);
                    band_map_.emplace(band, band_info);
                    snapengine::AbstractMetadata::AddBandToBandMap(band_metadata, band_name);

                    if (real) {
                        last_real_band = band;
                    } else {
                        snapengine::ReaderUtils::CreateVirtualIntensityBand(product, last_real_band, band,
                                                                            '_' + suffix);
                        band_info->SetRealBand(last_real_band);
                        band_map_.at(last_real_band)->SetImaginaryBand(band);
                    }
                    real = !real;

                    // add tiepointgrids and geocoding for band
                    AddTiePointGrids(product, band, img_name, tpg_prefix);

                    // reset to null so it doesn't adopt a geocoding from the bands
                    product->SetSceneGeoCoding(nullptr);
                }
            } else {
                for (int b = 0; b < img->GetNumBands(); ++b) {
                    band_name = std::string("Amplitude") + '_' + suffix;
                    std::shared_ptr<snapengine::Band> band = std::make_shared<snapengine::Band>(
                        band_name, snapengine::ProductData::TYPE_INT32, width, height);
                    band->SetUnit(snapengine::Unit::AMPLITUDE);
                    band->SetNoDataValueUsed(true);
                    band->SetNoDataValue(NO_DATA_VALUE);

                    product->AddBand(band);
                    band_map_.emplace(band, std::make_shared<BandInfo>(band, img, i, b));
                    snapengine::AbstractMetadata::AddBandToBandMap(band_metadata, band_name);

                    SARReader::CreateVirtualIntensityBand(product, band, '_' + suffix);

                    // add tiepointgrids and geocoding for band
                    AddTiePointGrids(product, band, img_name, tpg_prefix);
                }
            }
        }
    }
}
std::string Sentinel1Level1Directory::GetProductName() {
    std::string name = GetBaseName();
    boost::algorithm::to_upper(name);
    const int safe_extension_length{5};
    const int zip_extension_length{4};
    if (boost::algorithm::ends_with(name, ".SAFE")) {
        return name.substr(0, name.length() - safe_extension_length);
    }
    if (boost::algorithm::ends_with(name, ".ZIP")) {
        return name.substr(0, name.length() - zip_extension_length);
    }
    return name;
}

std::string Sentinel1Level1Directory::GetProductType() { return "Level-1"; }

std::shared_ptr<snapengine::Utc> Sentinel1Level1Directory::GetTime(
    const std::shared_ptr<snapengine::MetadataElement>& elem, std::string_view tag,
    std::string_view sentinel_date_format) {
    std::string start = elem->GetAttributeString(tag, snapengine::AbstractMetadata::NO_METADATA_STRING);
    boost::algorithm::replace_all(start, "T", "_");
    return snapengine::AbstractMetadata::ParseUtc(start, sentinel_date_format);
}
std::shared_ptr<snapengine::Product> Sentinel1Level1Directory::CreateProduct() {
    std::shared_ptr<snapengine::MetadataElement> new_root = AddMetaData();  // This reads all XML files.
    FindImages(new_root);                                                   // This reads all raster files.
    std::shared_ptr<snapengine::MetadataElement> abs_root =
        new_root->GetElement(snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);
    DetermineProductDimensions(abs_root);
    int scene_width = abs_root->GetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE);
    int scene_height = abs_root->GetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES);
    auto product = snapengine::Product::CreateProduct(GetProductName(), GetProductType(), scene_width, scene_height);
    UpdateProduct(product, new_root);
    AddBands(product);
    AddGeoCoding(product);
    product->SetName(GetProductName());
    // product.setProductType(getProductType());
    product->SetDescription(std::make_optional<std::string>(GetProductDescription()));
    snapengine::ReaderUtils::AddMetadataIncidenceAngles(product);
    snapengine::ReaderUtils::AddMetadataProductSize(product);
    return product;
}
void Sentinel1Level1Directory::DetermineProductDimensions(
    const std::shared_ptr<snapengine::MetadataElement>& abs_root) {
    int total_width = 0;
    int max_height = 0;
    int k = 0;
    std::string pol;
    for (auto& string_image_i_o_file_entry : band_image_file_map_) {
        auto img = string_image_i_o_file_entry.second;
        std::string img_name = img->GetName();
        boost::algorithm::to_lower(img_name);
        std::string band_metadata_name = img_band_metadata_map_.at(img_name);
        if (band_metadata_name.empty()) {
            throw std::runtime_error("Metadata for measurement dataset " + img_name + " not found");
        }

        if (k == 0) {
            pol = band_metadata_name.substr(band_metadata_name.find_last_of('_') + 1);
        } else if (!(band_metadata_name.substr(band_metadata_name.find_last_of('_') + 1) == pol)) {
            continue;
        }
        k++;

        std::shared_ptr<snapengine::MetadataElement> band_metadata = abs_root->GetElement(band_metadata_name);
        int width = band_metadata->GetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE);
        int height = band_metadata->GetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES);
        total_width += width;
        if (height > max_height) {
            max_height = height;
        }
    }

    if (AbstractProductDirectory::IsSLC() && IsTOPSAR()) {  // approximate does not account for overlap
        abs_root->SetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE, total_width);
        abs_root->SetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES, max_height);
    }
}
void Sentinel1Level1Directory::AddTiePointGrids([[maybe_unused]] const std::shared_ptr<snapengine::Product>& product) {
    // replaced by call to addTiePointGrids(band)
}

void Sentinel1Level1Directory::AddTiePointGrids(const std::shared_ptr<snapengine::Product>& product,
                                                const std::shared_ptr<snapengine::Band>& band,
                                                std::string_view img_x_m_l_name, std::string_view tpg_prefix) {
    std::string pre;
    if (!tpg_prefix.empty()) {
        pre = std::string(tpg_prefix) + '_';
    }

    std::shared_ptr<snapengine::TiePointGrid> existing_lat_t_p_g =
        product->GetTiePointGrid(pre + std::string(snapengine::OperatorUtils::TPG_LATITUDE));
    std::shared_ptr<snapengine::TiePointGrid> existing_lon_t_p_g =
        product->GetTiePointGrid(pre + std::string(snapengine::OperatorUtils::TPG_LONGITUDE));
    if (existing_lat_t_p_g && existing_lon_t_p_g) {
        if (band) {
            // reuse geocoding
            auto tp_geo_coding =
                std::make_shared<snapengine::TiePointGeoCoding>(existing_lat_t_p_g, existing_lon_t_p_g);
            band->SetGeoCoding(tp_geo_coding);
        }
        return;
    }
    // System.out.println("add new TPG for band = " + band.getName());
    std::string annotation = snapengine::FileUtils::ExchangeExtension(img_x_m_l_name, ".xml");
    std::shared_ptr<snapengine::MetadataElement> orig_prod_root =
        snapengine::AbstractMetadata::GetOriginalProductMetadata(product);
    std::shared_ptr<snapengine::MetadataElement> annotation_elem = orig_prod_root->GetElement("annotation");
    std::shared_ptr<snapengine::MetadataElement> img_elem = annotation_elem->GetElement(annotation);
    std::shared_ptr<snapengine::MetadataElement> product_elem = img_elem->GetElement("product");
    std::shared_ptr<snapengine::MetadataElement> geolocation_grid = product_elem->GetElement("geolocationGrid");
    std::shared_ptr<snapengine::MetadataElement> geolocation_grid_point_list =
        geolocation_grid->GetElement("geolocationGridPointList");

    auto geo_grid = geolocation_grid_point_list->GetElements();

    std::vector<double> lat_list(geo_grid.size());
    std::vector<double> lng_list(geo_grid.size());
    std::vector<double> incidence_angle_list(geo_grid.size());
    std::vector<double> elev_angle_list(geo_grid.size());
    std::vector<double> range_time_list(geo_grid.size());
    std::vector<int> x(geo_grid.size());
    std::vector<int> y(geo_grid.size());

    // Loop through the list of geolocation grid points, assuming that it represents a row-major rectangular grid.
    int grid_width = 0;
    int grid_height = 0;
    int i = 0;
    for (const auto& gg_point : geo_grid) {
        lat_list.at(i) = gg_point->GetAttributeDouble("latitude", 0);
        lng_list.at(i) = gg_point->GetAttributeDouble("longitude", 0);
        incidence_angle_list.at(i) = gg_point->GetAttributeDouble("incidenceAngle", 0);
        elev_angle_list.at(i) = gg_point->GetAttributeDouble("elevationAngle", 0);
        range_time_list.at(i) =
            gg_point->GetAttributeDouble("slantRangeTime", 0) * snapengine::eo::constants::ONE_BILLION;  // s to ns

        x.at(i) = static_cast<int>(gg_point->GetAttributeDouble("pixel", 0));
        y.at(i) = static_cast<int>(gg_point->GetAttributeDouble("line", 0));
        if (x.at(i) == 0) {
            // This means we are at the start of a new line
            if (grid_width == 0)  // Here we are implicitly assuming that the pixel horizontal spacing is assumed to be
                                  // the same from line to line.
                grid_width = i;
            ++grid_height;
        }
        ++i;
    }

    const int new_grid_width = grid_width;
    const int new_grid_height = grid_height;
    std::vector<float> new_lat_list(new_grid_width * new_grid_height);
    std::vector<float> new_lon_list(new_grid_width * new_grid_height);
    std::vector<float> new_inc_list(new_grid_width * new_grid_height);
    std::vector<float> new_elev_list(new_grid_width * new_grid_height);
    std::vector<float> new_slrt_list(new_grid_width * new_grid_height);
    int scene_raster_width = product->GetSceneRasterWidth();
    int scene_raster_height = product->GetSceneRasterHeight();
    if (band) {
        scene_raster_width = band->GetRasterWidth();
        scene_raster_height = band->GetRasterHeight();
    }

    const auto sub_sampling_x = static_cast<double>(scene_raster_width) / (new_grid_width - 1);
    const auto sub_sampling_y = static_cast<double>(scene_raster_height) / (new_grid_height - 1);

    GetListInEvenlySpacedGrid(scene_raster_width, scene_raster_height, grid_width, grid_height, x, y, lat_list,
                              new_grid_width, new_grid_height, sub_sampling_x, sub_sampling_y, new_lat_list);

    GetListInEvenlySpacedGrid(scene_raster_width, scene_raster_height, grid_width, grid_height, x, y, lng_list,
                              new_grid_width, new_grid_height, sub_sampling_x, sub_sampling_y, new_lon_list);

    GetListInEvenlySpacedGrid(scene_raster_width, scene_raster_height, grid_width, grid_height, x, y,
                              incidence_angle_list, new_grid_width, new_grid_height, sub_sampling_x, sub_sampling_y,
                              new_inc_list);

    GetListInEvenlySpacedGrid(scene_raster_width, scene_raster_height, grid_width, grid_height, x, y, elev_angle_list,
                              new_grid_width, new_grid_height, sub_sampling_x, sub_sampling_y, new_elev_list);

    GetListInEvenlySpacedGrid(scene_raster_width, scene_raster_height, grid_width, grid_height, x, y, range_time_list,
                              new_grid_width, new_grid_height, sub_sampling_x, sub_sampling_y, new_slrt_list);

    std::shared_ptr<snapengine::TiePointGrid> lat_grid =
        product->GetTiePointGrid(pre + std::string(snapengine::OperatorUtils::TPG_LATITUDE));
    if (lat_grid == nullptr) {
        lat_grid = std::make_shared<snapengine::TiePointGrid>(
            pre + std::string(snapengine::OperatorUtils::TPG_LATITUDE), new_grid_width, new_grid_height, 0.5F, 0.5F,
            sub_sampling_x, sub_sampling_y, new_lat_list);
        lat_grid->SetUnit(std::make_optional<std::string>(snapengine::Unit::DEGREES));
        product->AddTiePointGrid(lat_grid);
    }

    std::shared_ptr<snapengine::TiePointGrid> lon_grid =
        product->GetTiePointGrid(pre + std::string(snapengine::OperatorUtils::TPG_LONGITUDE));
    if (lon_grid == nullptr) {
        lon_grid = std::make_shared<snapengine::TiePointGrid>(
            pre + std::string(snapengine::OperatorUtils::TPG_LONGITUDE), new_grid_width, new_grid_height, 0.5F, 0.5F,
            sub_sampling_x, sub_sampling_y, new_lon_list, snapengine::TiePointGrid::DISCONT_AT_180);
        lon_grid->SetUnit(std::make_optional<std::string>(snapengine::Unit::DEGREES));
        product->AddTiePointGrid(lon_grid);
    }

    if (product->GetTiePointGrid(pre + std::string(snapengine::OperatorUtils::TPG_INCIDENT_ANGLE)) == nullptr) {
        std::shared_ptr<snapengine::TiePointGrid> incident_angle_grid = std::make_shared<snapengine::TiePointGrid>(
            pre + std::string(snapengine::OperatorUtils::TPG_INCIDENT_ANGLE), new_grid_width, new_grid_height, 0.5F,
            0.5F, sub_sampling_x, sub_sampling_y, new_inc_list);
        incident_angle_grid->SetUnit(std::make_optional<std::string>(snapengine::Unit::DEGREES));
        product->AddTiePointGrid(incident_angle_grid);
    }

    if (product->GetTiePointGrid(pre + std::string(snapengine::OperatorUtils::TPG_ELEVATION_ANGLE)) == nullptr) {
        std::shared_ptr<snapengine::TiePointGrid> elev_angle_grid = std::make_shared<snapengine::TiePointGrid>(
            pre + std::string(snapengine::OperatorUtils::TPG_ELEVATION_ANGLE), new_grid_width, new_grid_height, 0.5F,
            0.5F, sub_sampling_x, sub_sampling_y, new_elev_list);
        elev_angle_grid->SetUnit(std::make_optional<std::string>(snapengine::Unit::DEGREES));
        product->AddTiePointGrid(elev_angle_grid);
    }

    if (product->GetTiePointGrid(pre + std::string(snapengine::OperatorUtils::TPG_SLANT_RANGE_TIME)) == nullptr) {
        std::shared_ptr<snapengine::TiePointGrid> slant_range_grid = std::make_shared<snapengine::TiePointGrid>(
            pre + std::string(snapengine::OperatorUtils::TPG_SLANT_RANGE_TIME), new_grid_width, new_grid_height, 0.5F,
            0.5F, sub_sampling_x, sub_sampling_y, new_slrt_list);
        slant_range_grid->SetUnit(std::make_optional<std::string>(snapengine::Unit::NANOSECONDS));
        product->AddTiePointGrid(slant_range_grid);
    }

    std::shared_ptr<snapengine::TiePointGeoCoding> tp_geo_coding =
        std::make_shared<snapengine::TiePointGeoCoding>(lat_grid, lon_grid);

    if (band) {
        band_geocoding_map_.emplace(band, tp_geo_coding);
    }
}
void Sentinel1Level1Directory::GetListInEvenlySpacedGrid(const int scene_raster_width, const int scene_raster_height,
                                                         const int source_grid_width, const int source_grid_height,
                                                         const std::vector<int>& x, const std::vector<int>& y,
                                                         const std::vector<double>& source_point_list,
                                                         const int target_grid_width, const int target_grid_height,
                                                         const double sub_sampling_x, const double sub_sampling_y,
                                                         std::vector<float>& target_point_list) {
    if (source_point_list.size() != static_cast<size_t>(source_grid_width) * source_grid_height) {
        throw std::invalid_argument(
            "Original tie point array size does not match 'sourceGridWidth' x 'sourceGridHeight'");
    }

    if (target_point_list.size() != static_cast<size_t>(target_grid_width) * target_grid_height) {
        throw std::invalid_argument(
            "Target tie point array size does not match 'targetGridWidth' x 'targetGridHeight'");
    }

    int k = 0;
    for (int r = 0; r < target_grid_height; r++) {
        double new_y = r * sub_sampling_y;
        if (new_y > scene_raster_height - 1) {
            new_y = scene_raster_height - 1;
        }
        double old_y_0 = 0;
        double old_y_1 = 0;
        int j0 = 0;
        int j1 = 0;
        for (int rr = 1; rr < source_grid_height; rr++) {
            j0 = rr - 1;
            j1 = rr;
            old_y_0 = y.at(j0 * source_grid_width);
            old_y_1 = y.at(j1 * source_grid_width);
            if (old_y_1 > new_y) {
                break;
            }
        }

        const double wj = (new_y - old_y_0) / (old_y_1 - old_y_0);

        for (int c = 0; c < target_grid_width; c++) {
            double new_x = c * sub_sampling_x;
            if (new_x > scene_raster_width - 1) {
                new_x = scene_raster_width - 1;
            }
            double old_x_0 = 0;
            double old_x_1 = 0;
            int i0 = 0;
            int i1 = 0;
            for (int cc = 1; cc < source_grid_width; cc++) {
                i0 = cc - 1;
                i1 = cc;
                old_x_0 = x.at(i0);
                old_x_1 = x.at(i1);
                if (old_x_1 > new_x) {
                    break;
                }
            }
            const double wi = (new_x - old_x_0) / (old_x_1 - old_x_0);

            target_point_list.at(k++) = static_cast<float>(snapengine::MathUtils::Interpolate2D(
                wi, wj, source_point_list.at(i0 + j0 * source_grid_width),
                source_point_list.at(i1 + j0 * source_grid_width), source_point_list.at(i0 + j1 * source_grid_width),
                source_point_list.at(i1 + j1 * source_grid_width)));
        }
    }
}

void Sentinel1Level1Directory::AddImageFile(std::string_view img_path,
                                            const std::shared_ptr<snapengine::MetadataElement>& new_root) {
    std::string name = GetBandFileNameFromImage(img_path);

    if (boost::algorithm::ends_with(name, "tiff")) {
        try {
            std::shared_ptr<snapengine::custom::Dimension> band_dimensions =
                GetBandDimensions(new_root, img_band_metadata_map_.at(name));
            //            todo think through
            //            auto in_stream = GetInputStream(img_path);
            // unlike esa snap we construct reader directly here (not providing stream)

            // todo: not 100% sure it is a good port
            //            if (in_stream.rdbuf()->in_avail() > 0) {
            //                auto img_stream = ImageIOFile::CreateImageInputStream(in_stream, band_dimensions);

            ////!!NB current solution is shortcut to make it work (starting from here) and does not use stream directly!
            /// this will probably change todo: investigate C++ API solution for partial file reading from cloud:
            /// https://gdal.org/user/virtual_file_systems.html

            //            todo:img_path should be solved similar to stream path solving using directory and getfile

            //            todo: check get_file for zip files
            auto img = std::make_shared<ImageIOFile>(name, band_dimensions, img_path, 1, 1,
                                                     snapengine::ProductData::TYPE_INT32, product_input_file_);
            //                if (AbstractProductDirectory::IsSLC()) {
            //                    img = std::make_shared<ImageIOFile>(name, img_stream,
            //                    GeoTiffUtils::GetTiffIIOReader(img_stream), 1,
            //                                                        1, snapengine::ProductData::TYPE_INT32,
            //                                                        product_input_file_);
            //                } else {
            //                    img = std::make_shared<ImageIOFile>(name, img_stream,
            //                    GeoTiffUtils::GetTiffIIOReader(img_stream), 1,
            //                                                        1, snapengine::ProductData::TYPE_INT32,
            //                                                        product_input_file_);
            //                }
            band_image_file_map_.emplace(img->GetName(), img);
            //            }
        } catch (const std::exception& e) {
            LOGE << img_path << " not found";
        }
    }
}
void Sentinel1Level1Directory::AddGeoCoding(const std::shared_ptr<snapengine::Product>& product) {
    std::shared_ptr<snapengine::TiePointGrid> lat_grid =
        product->GetTiePointGrid(snapengine::OperatorUtils::TPG_LATITUDE);
    std::shared_ptr<snapengine::TiePointGrid> lon_grid =
        product->GetTiePointGrid(snapengine::OperatorUtils::TPG_LONGITUDE);
    if (lat_grid != nullptr && lon_grid != nullptr) {
        SetLatLongMetadata(product, lat_grid, lon_grid);

        const std::shared_ptr<snapengine::TiePointGeoCoding> tp_geo_coding =
            std::make_shared<snapengine::TiePointGeoCoding>(lat_grid, lon_grid);
        product->SetSceneGeoCoding(tp_geo_coding);
        return;
    }

    const std::shared_ptr<snapengine::MetadataElement> abs_root =
        snapengine::AbstractMetadata::GetAbstractedMetadata(product);
    std::string acquisition_mode = abs_root->GetAttributeString(snapengine::AbstractMetadata::ACQUISITION_MODE);
    int num_of_sub_swath;
    if (acquisition_mode == "IW") {
        num_of_sub_swath = 3;  // NOLINT
    } else if (acquisition_mode == "EW") {
        num_of_sub_swath = 5;  // NOLINT
    } else {
        num_of_sub_swath = 1;  // NOLINT
    }

    std::vector<std::string> band_names = product->GetBandNames();
    std::shared_ptr<snapengine::Band> first_s_w_band;
    std::shared_ptr<snapengine::Band> last_s_w_band;
    bool first_s_w_band_found = false;
    bool last_s_w_band_found = false;

    for (auto& band_name : band_names) {
        if (!first_s_w_band_found && band_name.find(acquisition_mode + std::to_string(1)) != std::string::npos) {
            first_s_w_band = product->GetBand(band_name);
            first_s_w_band_found = true;
        }

        if (!last_s_w_band_found &&
            band_name.find(acquisition_mode + std::to_string(num_of_sub_swath)) != std::string::npos) {
            last_s_w_band = product->GetBand(band_name);
            last_s_w_band_found = true;
        }
    }
    if (first_s_w_band != nullptr && last_s_w_band != nullptr) {
        const std::shared_ptr<snapengine::IGeoCoding> first_s_w_band_geo_coding =
            band_geocoding_map_.at(first_s_w_band);
        const int first_s_w_band_height = first_s_w_band->GetRasterHeight();

        const std::shared_ptr<snapengine::IGeoCoding> last_s_w_band_geo_coding = band_geocoding_map_.at(last_s_w_band);
        const int last_s_w_band_width = last_s_w_band->GetRasterWidth();
        const int last_s_w_band_height = last_s_w_band->GetRasterHeight();

        const auto ul_pix = std::make_shared<snapengine::PixelPos>(0, 0);
        const auto ll_pix = std::make_shared<snapengine::PixelPos>(0, first_s_w_band_height - 1);
        auto ul_geo = std::make_shared<snapengine::GeoPos>();
        auto ll_geo = std::make_shared<snapengine::GeoPos>();
        first_s_w_band_geo_coding->GetGeoPos(ul_pix, ul_geo);
        first_s_w_band_geo_coding->GetGeoPos(ll_pix, ll_geo);

        const auto ur_pix = std::make_shared<snapengine::PixelPos>(last_s_w_band_width - 1, 0);
        const auto lr_pix = std::make_shared<snapengine::PixelPos>(last_s_w_band_width - 1, last_s_w_band_height - 1);
        auto ur_geo = std::make_shared<snapengine::GeoPos>();
        auto lr_geo = std::make_shared<snapengine::GeoPos>();
        last_s_w_band_geo_coding->GetGeoPos(ur_pix, ur_geo);
        last_s_w_band_geo_coding->GetGeoPos(lr_pix, lr_geo);

        const std::vector<float> lat_corners = {
            static_cast<float>(ul_geo->GetLat()), static_cast<float>(ur_geo->GetLat()),
            static_cast<float>(ll_geo->GetLat()), static_cast<float>(lr_geo->GetLat())};
        const std::vector<float> lon_corners = {
            static_cast<float>(ul_geo->GetLon()), static_cast<float>(ur_geo->GetLon()),
            static_cast<float>(ll_geo->GetLon()), static_cast<float>(lr_geo->GetLon())};

        snapengine::ReaderUtils::AddGeoCoding(product, lat_corners, lon_corners);

        snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::FIRST_NEAR_LAT,
                                                   ul_geo->GetLat());
        snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::FIRST_NEAR_LONG,
                                                   ul_geo->GetLon());
        snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::FIRST_FAR_LAT,
                                                   ur_geo->GetLat());
        snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::FIRST_FAR_LONG,
                                                   ur_geo->GetLon());

        snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::LAST_NEAR_LAT,
                                                   ll_geo->GetLat());
        snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::LAST_NEAR_LONG,
                                                   ll_geo->GetLon());
        snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::LAST_FAR_LAT,
                                                   lr_geo->GetLat());
        snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::LAST_FAR_LONG,
                                                   lr_geo->GetLon());

        std::vector<std::shared_ptr<snapengine::Band>> bands = product->GetBands();
        for (const auto& band : bands) {
            try {
                band->SetGeoCoding(band_geocoding_map_.at(band));
            } catch (const std::out_of_range& e) {
                band->SetGeoCoding(nullptr);
            }
        }
    } else {
        try {
            const std::string annot_folder = GetRootFolder() + "annotation";
            const std::vector<std::string> filenames = ListFiles(annot_folder);

            AddTiePointGrids(product, nullptr, filenames.at(0), "");

            lat_grid = product->GetTiePointGrid(snapengine::OperatorUtils::TPG_LATITUDE);
            lon_grid = product->GetTiePointGrid(snapengine::OperatorUtils::TPG_LONGITUDE);
            if (lat_grid && lon_grid) {
                SetLatLongMetadata(product, lat_grid, lon_grid);

                const auto tp_geo_coding = std::make_shared<snapengine::TiePointGeoCoding>(lat_grid, lon_grid);
                product->SetSceneGeoCoding(tp_geo_coding);
            }
        } catch (const std::exception& e) {
            LOGE << "Unable to add tpg geocoding " << e.what();
        }
    }
}

void Sentinel1Level1Directory::SetLatLongMetadata(const std::shared_ptr<snapengine::Product>& product,
                                                  const std::shared_ptr<snapengine::TiePointGrid>& lat_grid,
                                                  const std::shared_ptr<snapengine::TiePointGrid>& lon_grid) {
    const std::shared_ptr<snapengine::MetadataElement> abs_root =
        snapengine::AbstractMetadata::GetAbstractedMetadata(product);

    const int w = product->GetSceneRasterWidth();
    const int h = product->GetSceneRasterHeight();

    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::FIRST_NEAR_LAT,
                                               lat_grid->GetPixelDouble(0, 0));
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::FIRST_NEAR_LONG,
                                               lon_grid->GetPixelDouble(0, 0));
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::FIRST_FAR_LAT,
                                               lat_grid->GetPixelDouble(w, 0));
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::FIRST_FAR_LONG,
                                               lon_grid->GetPixelDouble(w, 0));

    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::LAST_NEAR_LAT,
                                               lat_grid->GetPixelDouble(0, h));
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::LAST_NEAR_LONG,
                                               lon_grid->GetPixelDouble(0, h));
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::LAST_FAR_LAT,
                                               lat_grid->GetPixelDouble(w, h));
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::LAST_FAR_LONG,
                                               lon_grid->GetPixelDouble(w, h));
}
void Sentinel1Level1Directory::AddAbstractedMetadataHeader(const std::shared_ptr<snapengine::MetadataElement>& root) {
    const std::shared_ptr<snapengine::MetadataElement> abs_root =
        snapengine::AbstractMetadata::AddAbstractedMetadataHeader(root);
    const std::shared_ptr<snapengine::MetadataElement> orig_prod_root =
        snapengine::AbstractMetadata::AddOriginalProductMetadata(root);
    AddManifestMetadata(GetProductName(), abs_root, orig_prod_root, false);
    acq_mode_ = abs_root->GetAttributeString(snapengine::AbstractMetadata::ACQUISITION_MODE);
    SetSLC(abs_root->GetAttributeString(snapengine::AbstractMetadata::SAMPLE_TYPE) == "COMPLEX");

    AddProductInfoJSON(orig_prod_root);

    // get metadata for each band
    AddBandAbstractedMetadata(abs_root, orig_prod_root);
    AddCalibrationAbstractedMetadata(orig_prod_root);
    AddNoiseAbstractedMetadata(orig_prod_root);
}

std::shared_ptr<snapengine::MetadataElement> Sentinel1Level1Directory::FindElement(
    const std::shared_ptr<snapengine::MetadataElement>& elem, std::string_view name) {
    const std::shared_ptr<snapengine::MetadataElement> metadata_wrap = elem->GetElement("metadataWrap");
    const std::shared_ptr<snapengine::MetadataElement> xml_data = metadata_wrap->GetElement("xmlData");
    return xml_data->GetElement(name);
}
void Sentinel1Level1Directory::AddManifestMetadata(std::string_view product_name,
                                                   const std::shared_ptr<snapengine::MetadataElement>& abs_root,
                                                   const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root,
                                                   bool is_o_c_n) {
    const std::string def_str(snapengine::AbstractMetadata::NO_METADATA_STRING);
    const int def_int = snapengine::AbstractMetadata::NO_METADATA;

    const std::shared_ptr<snapengine::MetadataElement> x_f_d_u = orig_prod_root->GetElement("XFDU");
    const std::shared_ptr<snapengine::MetadataElement> information_package_map =
        x_f_d_u->GetElement("informationPackageMap");
    const std::shared_ptr<snapengine::MetadataElement> content_unit =
        information_package_map->GetElement("contentUnit");

    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::PRODUCT,
                                               std::string(product_name));
    const std::string descriptor = content_unit->GetAttributeString("textInfo", def_str);
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::SPH_DESCRIPTOR, descriptor);
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::ANTENNA_POINTING, "right");

    const std::shared_ptr<snapengine::MetadataElement> metadata_section = x_f_d_u->GetElement("metadataSection");
    const std::vector<std::shared_ptr<snapengine::MetadataElement>> metadata_object_list =
        metadata_section->GetElements();
    //    const auto sentinel_date_format = snapengine::Utc::CreateDateFormatIn("yyyy-MM-dd_HH:mm:ss");
    //    todo: check if we need to use some special format (check utc class for examples)
    //    https://www.boost.org/doc/libs/1_75_0/doc/html/date_time/date_time_io.html#date_time.format_flags
    const auto* const sentinel_date_format("%Y-%m-%d_%H:%M:%S");

    for (const auto& metadata_object : metadata_object_list) {
        const std::string id = metadata_object->GetAttributeString("ID", def_str);

        if (boost::algorithm::ends_with(id, "Annotation") || boost::algorithm::ends_with(id, "Schema") ||
            id == "measurementFrameSet") {
            // continue;
        } else if (id == "processing") {
            const std::shared_ptr<snapengine::MetadataElement> processing = FindElement(metadata_object, "processing");
            const std::shared_ptr<snapengine::MetadataElement> facility = processing->GetElement("facility");
            const std::shared_ptr<snapengine::MetadataElement> software = facility->GetElement("software");
            const std::string org = facility->GetAttributeString("organisation");
            const std::string name = software->GetAttributeString("name");
            const std::string version = software->GetAttributeString("version");
            snapengine::AbstractMetadata::SetAttribute(
                abs_root, snapengine::AbstractMetadata::PROCESSING_SYSTEM_IDENTIFIER,
                std::string(org).append(" ").append(name).append(" ").append(version));

            const std::shared_ptr<snapengine::Utc> start = GetTime(processing, "start", sentinel_date_format);
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::PROC_TIME, start);
        } else if (id == "acquisitionPeriod") {
            const std::shared_ptr<snapengine::MetadataElement> acquisition_period =
                FindElement(metadata_object, "acquisitionPeriod");
            const std::shared_ptr<snapengine::Utc> start_time =
                GetTime(acquisition_period, "startTime", sentinel_date_format);
            const std::shared_ptr<snapengine::Utc> stop_time =
                GetTime(acquisition_period, "stopTime", sentinel_date_format);
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::FIRST_LINE_TIME,
                                                       start_time);
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::LAST_LINE_TIME,
                                                       stop_time);

        } else if (id == "platform") {
            const std::shared_ptr<snapengine::MetadataElement> platform = FindElement(metadata_object, "platform");
            std::string mission_name = platform->GetAttributeString("familyName", "Sentinel-1");
            const std::string number = platform->GetAttributeString("number", def_str);
            if (mission_name != "ENVISAT") {
                mission_name += number;
            }
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::MISSION, mission_name);

            const std::shared_ptr<snapengine::MetadataElement> instrument = platform->GetElement("instrument");
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::SWATH,
                                                       instrument->GetAttributeString("swath", def_str));
            std::string acq_mode = instrument->GetAttributeString("mode", def_str);
            if (acq_mode.empty() || acq_mode == def_str) {
                const std::shared_ptr<snapengine::MetadataElement> extension_elem = instrument->GetElement("extension");
                if (extension_elem) {
                    const std::shared_ptr<snapengine::MetadataElement> instrument_mode_elem =
                        extension_elem->GetElement("instrumentMode");
                    if (instrument_mode_elem) {
                        acq_mode = instrument_mode_elem->GetAttributeString("mode", def_str);
                    }
                }
            }
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::ACQUISITION_MODE,
                                                       acq_mode);
        } else if (id == "measurementOrbitReference") {
            const std::shared_ptr<snapengine::MetadataElement> orbit_reference =
                FindElement(metadata_object, "orbitReference");
            const std::shared_ptr<snapengine::MetadataElement> orbit_number =
                FindElementContaining(orbit_reference, "OrbitNumber", "type", "start");
            const std::shared_ptr<snapengine::MetadataElement> relative_orbit_number =
                FindElementContaining(orbit_reference, "relativeOrbitNumber", "type", "start");
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::ABS_ORBIT,
                                                       orbit_number->GetAttributeInt("orbitNumber", def_int));
            snapengine::AbstractMetadata::SetAttribute(
                abs_root, snapengine::AbstractMetadata::REL_ORBIT,
                relative_orbit_number->GetAttributeInt("relativeOrbitNumber", def_int));
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::CYCLE,
                                                       orbit_reference->GetAttributeInt("cycleNumber", def_int));

            std::string pass = orbit_reference->GetAttributeString("pass", def_str);
            if (pass == def_str) {
                const std::shared_ptr<snapengine::MetadataElement> extension = orbit_reference->GetElement("extension");
                const std::shared_ptr<snapengine::MetadataElement> orbit_properties =
                    extension->GetElement("orbitProperties");
                pass = orbit_properties->GetAttributeString("pass", def_str);
            }
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::PASS, pass);
        } else if (id == "generalProductInformation") {
            std::shared_ptr<snapengine::MetadataElement> general_product_information =
                FindElement(metadata_object, "generalProductInformation");
            if (general_product_information == nullptr) {
                general_product_information = FindElement(metadata_object, "standAloneProductInformation");
            }
            std::string product_type = "unknown";
            if (is_o_c_n) {
                product_type = "OCN";
            } else {
                if (general_product_information) {
                    product_type = general_product_information->GetAttributeString("productType", def_str);
                }
            }
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::PRODUCT_TYPE,
                                                       product_type);
            if (product_type.find("SLC") != std::string::npos) {
                snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::SAMPLE_TYPE,
                                                           "COMPLEX");
            } else {
                snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::SAMPLE_TYPE,
                                                           "DETECTED");
                snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::SRGR_FLAG, 1);
            }
        }
    }
}

std::shared_ptr<snapengine::MetadataElement> Sentinel1Level1Directory::FindElementContaining(
    const std::shared_ptr<snapengine::MetadataElement>& parent, std::string_view elem_name,
    std::string_view attrib_name, std::string_view att_value) {
    const std::vector<std::shared_ptr<snapengine::MetadataElement>> elems = parent->GetElements();
    for (const auto& elem : elems) {
        if (boost::algorithm::iequals(elem->GetName(), elem_name) && elem->ContainsAttribute(attrib_name)) {
            auto value = elem->GetAttributeString(attrib_name);
            if (!value.empty() && boost::algorithm::iequals(value, att_value)) {
                return elem;
            }
        }
    }
    return nullptr;
}
void Sentinel1Level1Directory::AddProductInfoJSON(const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root) {
    if (product_dir_->Exists("productInfo.json")) {
        try {
            // THIS MAKES A SMALL ROUNDTRIP BETWEEN DIFFERENT MODELS AND FORMATS, BUT STARTING OUT LIKE ESA SNAP... CAN
            // ALWAYS IMPROVE IF IT WORKS
            boost::filesystem::path product_info_file = product_dir_->GetFile("productInfo.json");
            boost::property_tree::ptree tree;
            boost::property_tree::read_json(product_info_file.filename().string(), tree);
            std::stringstream ss;
            boost::property_tree::write_xml(ss, tree);
            pugi::xml_document doc;
            pugi::xml_parse_result result = doc.load(ss);
            if (result) {
                pugi::xml_node root = doc.document_element();
                pugi::xml_document doc2;
                doc2.append_child("ProductInfo");
                doc2.document_element().append_move(root);
                snapengine::AbstractMetadataIO::AddXMLMetadata(doc2.document_element(), orig_prod_root);
            }
            //            todo: delete comments after this has been tested
            //            if (product_info_file.length() > 0) {
            //                final BufferedReader streamReader = new BufferedReader(new
            //                FileReader(productInfoFile.getPath())); final JSONParser parser = new JSONParser(); final
            //                JSONObject json = (JSONObject)parser.parse(streamReader); json.remove("filenameMap");
            //                snapengine::AbstractMetadataIO::AddXMLMetadata(JSONProductDirectory.jsonToXML("ProductInfo",
            //                json),
            //                                                               orig_prod_root);
            //            }
        } catch (const std::exception& e) {
            //            todo: not sure why snap had throw commented out, I will just log atm
            LOGE << "Unable to read productInfo " << e.what();
        }
    }
}
void Sentinel1Level1Directory::AddBandAbstractedMetadata(
    const std::shared_ptr<snapengine::MetadataElement>& abs_root,
    const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root) {
    std::shared_ptr<snapengine::MetadataElement> annotation_element = orig_prod_root->GetElement("annotation");
    if (annotation_element == nullptr) {
        annotation_element = std::make_shared<snapengine::MetadataElement>("annotation");
        orig_prod_root->AddElement(annotation_element);
    }

    // collect range and azimuth spacing
    double range_spacing_total = 0;
    double azimuth_spacing_total = 0;
    bool common_metadata_retrieved = false;
    //    todo: use format string
    //    final DateFormat sentinelDateFormat = ProductData.UTC.createDateFormat("yyyy-MM-dd_HH:mm:ss");
    const auto* const sentinel_date_format("%Y-%m-%d_%H:%M:%S");

    double height_sum = 0.0;

    int num_bands = 0;
    const std::string annot_folder = GetRootFolder() + "annotation";

    const std::vector<std::string> filenames = ListFiles(annot_folder);
    //    if (!filenames.empty()) {
    for (const auto& metadata_file : filenames) {
        std::fstream is;
        GetInputStream(std::string().append(annot_folder).append("/").append(metadata_file), is);
        pugi::xml_document xml_doc;
        xml_doc.load(is);
        if (is.is_open()) {
            is.close();
        }
        const pugi::xml_node root_element = xml_doc.document_element();
        const auto name_elem = std::make_shared<snapengine::MetadataElement>(metadata_file);
        annotation_element->AddElement(name_elem);
        snapengine::AbstractMetadataIO::AddXMLMetadata(root_element, name_elem);

        const std::shared_ptr<snapengine::MetadataElement> prod_elem = name_elem->GetElement("product");
        const std::shared_ptr<snapengine::MetadataElement> ads_header = prod_elem->GetElement("adsHeader");
        const std::string swath = ads_header->GetAttributeString("swath");
        const std::string pol = ads_header->GetAttributeString("polarisation");

        const std::shared_ptr<snapengine::Utc> start_time = GetTime(ads_header, "startTime", sentinel_date_format);
        const std::shared_ptr<snapengine::Utc> stop_time = GetTime(ads_header, "stopTime", sentinel_date_format);

        const std::string band_root_name =
            std::string(snapengine::AbstractMetadata::BAND_PREFIX).append(swath).append("_").append(pol);
        const std::shared_ptr<snapengine::MetadataElement> band_abs_root =
            snapengine::AbstractMetadata::AddBandAbstractedMetadata(abs_root, band_root_name);

        const std::string img_name = snapengine::FileUtils::ExchangeExtension(metadata_file, ".tiff");
        img_band_metadata_map_.emplace(img_name, band_root_name);

        snapengine::AbstractMetadata::SetAttribute(band_abs_root, snapengine::AbstractMetadata::SWATH, swath);
        snapengine::AbstractMetadata::SetAttribute(band_abs_root, snapengine::AbstractMetadata::POLARIZATION, pol);
        snapengine::AbstractMetadata::SetAttribute(band_abs_root, snapengine::AbstractMetadata::ANNOTATION,
                                                   metadata_file);
        snapengine::AbstractMetadata::SetAttribute(band_abs_root, snapengine::AbstractMetadata::FIRST_LINE_TIME,
                                                   start_time);
        snapengine::AbstractMetadata::SetAttribute(band_abs_root, snapengine::AbstractMetadata::LAST_LINE_TIME,
                                                   stop_time);

        if (snapengine::AbstractMetadata::IsNoData(abs_root, snapengine::AbstractMetadata::MDS1_TX_RX_POLAR)) {
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::MDS1_TX_RX_POLAR, pol);
        } else if (abs_root->GetAttributeString(snapengine::AbstractMetadata::MDS1_TX_RX_POLAR,
                                                snapengine::AbstractMetadata::NO_METADATA_STRING) != pol) {
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::MDS2_TX_RX_POLAR, pol);
        }

        const std::shared_ptr<snapengine::MetadataElement> image_annotation = prod_elem->GetElement("imageAnnotation");
        const std::shared_ptr<snapengine::MetadataElement> image_information =
            image_annotation->GetElement("imageInformation");

        snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::DATA_TAKE_ID,
                                                   std::stoi(ads_header->GetAttributeString("missionDataTakeId")));
        snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::SLICE_NUM,
                                                   std::stoi(image_information->GetAttributeString("sliceNumber")));

        range_spacing_total += image_information->GetAttributeDouble("rangePixelSpacing");
        azimuth_spacing_total += image_information->GetAttributeDouble("azimuthPixelSpacing");

        snapengine::AbstractMetadata::SetAttribute(band_abs_root, snapengine::AbstractMetadata::LINE_TIME_INTERVAL,
                                                   image_information->GetAttributeDouble("azimuthTimeInterval"));
        snapengine::AbstractMetadata::SetAttribute(band_abs_root, snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE,
                                                   image_information->GetAttributeInt("numberOfSamples"));
        snapengine::AbstractMetadata::SetAttribute(band_abs_root, snapengine::AbstractMetadata::NUM_OUTPUT_LINES,
                                                   image_information->GetAttributeInt("numberOfLines"));
        std::string pixel_value = image_information->GetAttributeString("pixelValue");
        boost::to_upper(pixel_value);
        snapengine::AbstractMetadata::SetAttribute(band_abs_root, snapengine::AbstractMetadata::SAMPLE_TYPE,
                                                   pixel_value);

        height_sum += GetBandTerrainHeight(prod_elem);

        if (!common_metadata_retrieved) {
            // these should be the same for all swaths
            // set to absRoot

            const std::shared_ptr<snapengine::MetadataElement> general_annotation =
                prod_elem->GetElement("generalAnnotation");
            const std::shared_ptr<snapengine::MetadataElement> product_information =
                general_annotation->GetElement("productInformation");
            const std::shared_ptr<snapengine::MetadataElement> processing_information =
                image_annotation->GetElement("processingInformation");
            const std::shared_ptr<snapengine::MetadataElement> swath_proc_params_list =
                processing_information->GetElement("swathProcParamsList");
            const std::shared_ptr<snapengine::MetadataElement> swath_proc_params =
                swath_proc_params_list->GetElement("swathProcParams");
            const std::shared_ptr<snapengine::MetadataElement> range_processing =
                swath_proc_params->GetElement("rangeProcessing");
            const std::shared_ptr<snapengine::MetadataElement> azimuth_processing =
                swath_proc_params->GetElement("azimuthProcessing");

            snapengine::AbstractMetadata::SetAttribute(
                abs_root, snapengine::AbstractMetadata::RANGE_SAMPLING_RATE,
                product_information->GetAttributeDouble("rangeSamplingRate") / snapengine::eo::constants::ONE_MILLION);
            snapengine::AbstractMetadata::SetAttribute(
                abs_root, snapengine::AbstractMetadata::RADAR_FREQUENCY,
                product_information->GetAttributeDouble("radarFrequency") / snapengine::eo::constants::ONE_MILLION);
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::LINE_TIME_INTERVAL,
                                                       image_information->GetAttributeDouble("azimuthTimeInterval"));

            snapengine::AbstractMetadata::SetAttribute(
                abs_root, snapengine::AbstractMetadata::SLANT_RANGE_TO_FIRST_PIXEL,
                image_information->GetAttributeDouble("slantRangeTime") * snapengine::eo::constants::HALF_LIGHT_SPEED);

            const std::shared_ptr<snapengine::MetadataElement> downlink_information_list =
                general_annotation->GetElement("downlinkInformationList");
            const std::shared_ptr<snapengine::MetadataElement> downlink_information =
                downlink_information_list->GetElement("downlinkInformation");

            snapengine::AbstractMetadata::SetAttribute(abs_root,
                                                       snapengine::AbstractMetadata::PULSE_REPETITION_FREQUENCY,
                                                       downlink_information->GetAttributeDouble("prf"));

            snapengine::AbstractMetadata::SetAttribute(
                abs_root, snapengine::AbstractMetadata::RANGE_BANDWIDTH,
                range_processing->GetAttributeDouble("processingBandwidth") / snapengine::eo::constants::ONE_MILLION);
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::AZIMUTH_BANDWIDTH,
                                                       azimuth_processing->GetAttributeDouble("processingBandwidth"));

            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::RANGE_LOOKS,
                                                       range_processing->GetAttributeDouble("numberOfLooks"));
            snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::AZIMUTH_LOOKS,
                                                       azimuth_processing->GetAttributeDouble("numberOfLooks"));

            if (!IsTOPSAR() || !IsSLC()) {
                snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::NUM_OUTPUT_LINES,
                                                           image_information->GetAttributeInt("numberOfLines"));
                snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE,
                                                           image_information->GetAttributeInt("numberOfSamples"));
            }
            AddOrbitStateVectors(abs_root, general_annotation->GetElement("orbitList"));
            AddSRGRCoefficients(abs_root, prod_elem->GetElement("coordinateConversion"));
            AddDopplerCentroidCoefficients(abs_root, prod_elem->GetElement("dopplerCentroid"));

            common_metadata_retrieved = true;
        }
        ++num_bands;
        //        }
    }
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::RANGE_SPACING,
                                               range_spacing_total / num_bands);
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::AZIMUTH_SPACING,
                                               azimuth_spacing_total / num_bands);
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::AVG_SCENE_HEIGHT,
                                               height_sum / filenames.size());
    snapengine::AbstractMetadata::SetAttribute(abs_root, snapengine::AbstractMetadata::BISTATIC_CORRECTION_APPLIED, 1);
}

void Sentinel1Level1Directory::AddOrbitStateVectors(const std::shared_ptr<snapengine::MetadataElement>& abs_root,
                                                    const std::shared_ptr<snapengine::MetadataElement>& orbit_list) {
    const std::shared_ptr<snapengine::MetadataElement> orbit_vector_list_elem =
        abs_root->GetElement(snapengine::AbstractMetadata::ORBIT_STATE_VECTORS);
    const std::vector<std::shared_ptr<snapengine::MetadataElement>> state_vector_elems = orbit_list->GetElements();
    for (std::size_t i = 1; i <= state_vector_elems.size(); ++i) {
        AddVector(snapengine::AbstractMetadata::ORBIT_VECTOR, orbit_vector_list_elem, state_vector_elems.at(i - 1), i);
    }
    // set state vector time
    if (abs_root
            ->GetAttributeUtc(snapengine::AbstractMetadata::STATE_VECTOR_TIME,
                              snapengine::AbstractMetadata::NO_METADATA_UTC)
            ->EqualElems(snapengine::AbstractMetadata::NO_METADATA_UTC)) {
        snapengine::AbstractMetadata::SetAttribute(
            abs_root, snapengine::AbstractMetadata::STATE_VECTOR_TIME,
            snapengine::ReaderUtils::GetTime(state_vector_elems.at(0), "time", SENTINEL_DATE_FORMAT_PATTERN));
    }
}

void Sentinel1Level1Directory::AddVector(std::string_view name,
                                         const std::shared_ptr<snapengine::MetadataElement>& orbit_vector_list_elem,
                                         const std::shared_ptr<snapengine::MetadataElement>& orbit_elem,
                                         std::size_t num) {
    const auto orbit_vector_elem =
        std::make_shared<snapengine::MetadataElement>(std::string(name) + std::to_string(num));

    const std::shared_ptr<snapengine::MetadataElement> position_elem = orbit_elem->GetElement("position");
    const std::shared_ptr<snapengine::MetadataElement> velocity_elem = orbit_elem->GetElement("velocity");

    orbit_vector_elem->SetAttributeUtc(
        snapengine::AbstractMetadata::ORBIT_VECTOR_TIME,
        snapengine::ReaderUtils::GetTime(orbit_elem, "time", SENTINEL_DATE_FORMAT_PATTERN));

    orbit_vector_elem->SetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_X_POS,
                                          position_elem->GetAttributeDouble("x", 0));
    orbit_vector_elem->SetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_Y_POS,
                                          position_elem->GetAttributeDouble("y", 0));
    orbit_vector_elem->SetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_Z_POS,
                                          position_elem->GetAttributeDouble("z", 0));
    orbit_vector_elem->SetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_X_VEL,
                                          velocity_elem->GetAttributeDouble("x", 0));
    orbit_vector_elem->SetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_Y_VEL,
                                          velocity_elem->GetAttributeDouble("y", 0));
    orbit_vector_elem->SetAttributeDouble(snapengine::AbstractMetadata::ORBIT_VECTOR_Z_VEL,
                                          velocity_elem->GetAttributeDouble("z", 0));
    orbit_vector_list_elem->AddElement(orbit_vector_elem);
}

void Sentinel1Level1Directory::AddSRGRCoefficients(
    const std::shared_ptr<snapengine::MetadataElement>& abs_root,
    const std::shared_ptr<snapengine::MetadataElement>& coordinate_conversion) {
    if (coordinate_conversion == nullptr) {
        return;
    }
    const std::shared_ptr<snapengine::MetadataElement> coordinate_conversion_list =
        coordinate_conversion->GetElement("coordinateConversionList");
    if (coordinate_conversion_list == nullptr) {
        return;
    }

    const std::shared_ptr<snapengine::MetadataElement> srgr_coefficients_elem =
        abs_root->GetElement(snapengine::AbstractMetadata::SRGR_COEFFICIENTS);

    int list_cnt = 1;
    for (const auto& elem : coordinate_conversion_list->GetElements()) {
        const auto srgr_list_elem = std::make_shared<snapengine::MetadataElement>(
            std::string(snapengine::AbstractMetadata::SRGR_COEF_LIST) + '.' + std::to_string(list_cnt));
        srgr_coefficients_elem->AddElement(srgr_list_elem);
        ++list_cnt;

        const std::shared_ptr<snapengine::Utc> utc_time =
            snapengine::ReaderUtils::GetTime(elem, "azimuthTime", SENTINEL_DATE_FORMAT_PATTERN);
        srgr_list_elem->SetAttributeUtc(snapengine::AbstractMetadata::SRGR_COEF_TIME, utc_time);

        const double gr_origin = elem->GetAttributeDouble("gr0", 0);
        snapengine::AbstractMetadata::AddAbstractedAttribute(
            srgr_list_elem, snapengine::AbstractMetadata::GROUND_RANGE_ORIGIN, snapengine::ProductData::TYPE_FLOAT64,
            "m", "Ground Range Origin");
        snapengine::AbstractMetadata::SetAttribute(srgr_list_elem, snapengine::AbstractMetadata::GROUND_RANGE_ORIGIN,
                                                   gr_origin);

        const std::string coeff_str = elem->GetElement("grsrCoefficients")->GetAttributeString("grsrCoefficients", "");
        if (!coeff_str.empty()) {
            boost::char_separator<char> sep{" \t\n\r\f"};
            boost::tokenizer<boost::char_separator<char>> st{coeff_str, sep};
            int cnt = 1;
            for (auto it = st.begin(); it != st.end(); ++it) {
                const double coef_value = std::stod(*it);
                const auto coef_elem = std::make_shared<snapengine::MetadataElement>(
                    std::string(snapengine::AbstractMetadata::COEFFICIENT) + '.' + std::to_string(cnt));
                ++cnt;
                snapengine::AbstractMetadata::AddAbstractedAttribute(coef_elem, snapengine::AbstractMetadata::SRGR_COEF,
                                                                     snapengine::ProductData::TYPE_FLOAT64, "",
                                                                     "SRGR Coefficient");
                snapengine::AbstractMetadata::SetAttribute(coef_elem, snapengine::AbstractMetadata::SRGR_COEF,
                                                           coef_value);
            }
        }
    }
}

void Sentinel1Level1Directory::AddDopplerCentroidCoefficients(
    const std::shared_ptr<snapengine::MetadataElement>& abs_root,
    const std::shared_ptr<snapengine::MetadataElement>& doppler_centroid) {
    if (doppler_centroid == nullptr) {
        return;
    }
    const std::shared_ptr<snapengine::MetadataElement> dc_estimate_list =
        doppler_centroid->GetElement("dcEstimateList");
    if (dc_estimate_list == nullptr) {
        return;
    }

    const std::shared_ptr<snapengine::MetadataElement> doppler_centroid_coefficients_elem =
        abs_root->GetElement(snapengine::AbstractMetadata::DOP_COEFFICIENTS);

    int list_cnt = 1;
    for (const auto& elem : dc_estimate_list->GetElements()) {
        const auto doppler_list_elem = std::make_shared<snapengine::MetadataElement>(
            std::string(snapengine::AbstractMetadata::DOP_COEF_LIST) + '.' + std::to_string(list_cnt));
        doppler_centroid_coefficients_elem->AddElement(doppler_list_elem);
        ++list_cnt;

        const std::shared_ptr<snapengine::Utc> utc_time =
            snapengine::ReaderUtils::GetTime(elem, "azimuthTime", SENTINEL_DATE_FORMAT_PATTERN);
        doppler_list_elem->SetAttributeUtc(snapengine::AbstractMetadata::DOP_COEF_TIME, utc_time);

        const double ref_time = elem->GetAttributeDouble("t0", 0) * 1e9;  // s to ns
        snapengine::AbstractMetadata::AddAbstractedAttribute(
            doppler_list_elem, snapengine::AbstractMetadata::SLANT_RANGE_TIME, snapengine::ProductData::TYPE_FLOAT64,
            "ns", "Slant Range Time");
        snapengine::AbstractMetadata::SetAttribute(doppler_list_elem, snapengine::AbstractMetadata::SLANT_RANGE_TIME,
                                                   ref_time);

        const std::string coeff_str =
            elem->GetElement("geometryDcPolynomial")->GetAttributeString("geometryDcPolynomial", "");
        if (!coeff_str.empty()) {
            boost::char_separator<char> sep{" \t\n\r\f"};
            boost::tokenizer<boost::char_separator<char>> st{coeff_str, sep};
            int cnt = 1;
            for (auto it = st.begin(); it != st.end(); ++it) {
                const double coef_value = std::stod(*it);
                const auto coef_elem = std::make_shared<snapengine::MetadataElement>(
                    std::string(snapengine::AbstractMetadata::COEFFICIENT) + '.' + std::to_string(cnt));
                doppler_list_elem->AddElement(coef_elem);
                ++cnt;
                snapengine::AbstractMetadata::AddAbstractedAttribute(coef_elem, snapengine::AbstractMetadata::DOP_COEF,
                                                                     snapengine::ProductData::TYPE_FLOAT64, "",
                                                                     "Doppler Centroid Coefficient");
                snapengine::AbstractMetadata::SetAttribute(coef_elem, snapengine::AbstractMetadata::DOP_COEF,
                                                           coef_value);
            }
        }
    }
}
double Sentinel1Level1Directory::GetBandTerrainHeight(const std::shared_ptr<snapengine::MetadataElement>& prod_elem) {
    const std::shared_ptr<snapengine::MetadataElement> general_annotation = prod_elem->GetElement("generalAnnotation");
    const std::shared_ptr<snapengine::MetadataElement> terrain_height_list =
        general_annotation->GetElement("terrainHeightList");

    double height_sum = 0.0;

    const std::vector<std::shared_ptr<snapengine::MetadataElement>> height_list = terrain_height_list->GetElements();
    int cnt = 0;
    for (const auto& terrain_height : height_list) {
        height_sum += terrain_height->GetAttributeDouble("value");
        ++cnt;
    }
    return height_sum / cnt;
}

void Sentinel1Level1Directory::AddCalibrationAbstractedMetadata(
    const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root) {
    std::shared_ptr<snapengine::MetadataElement> calibration_element = orig_prod_root->GetElement("calibration");
    if (calibration_element == nullptr) {
        calibration_element = std::make_shared<snapengine::MetadataElement>("calibration");
        orig_prod_root->AddElement(calibration_element);
    }
    const std::string calib_folder = GetRootFolder() + "annotation" + '/' + "calibration";
    const std::vector<std::string> filenames = ListFiles(calib_folder);

    if (!filenames.empty()) {
        for (const auto& metadata_file : filenames) {
            if (boost::algorithm::starts_with(metadata_file, "calibration")) {
                std::fstream is;
                GetInputStream(std::string(calib_folder).append("/").append(metadata_file), is);
                if (is) {
                    pugi::xml_document xml_doc;
                    xml_doc.load(is);
                    const pugi::xml_node root_element = xml_doc.document_element();
                    std::string name = metadata_file;
                    boost::algorithm::replace_all(name, "calibration-", "");
                    const auto name_elem = std::make_shared<snapengine::MetadataElement>(name);
                    calibration_element->AddElement(name_elem);
                    snapengine::AbstractMetadataIO::AddXMLMetadata(root_element, name_elem);
                }
            }
        }
    }
}

void Sentinel1Level1Directory::AddNoiseAbstractedMetadata(
    const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root) {
    std::shared_ptr<snapengine::MetadataElement> noise_element = orig_prod_root->GetElement("noise");
    if (noise_element == nullptr) {
        noise_element = std::make_shared<snapengine::MetadataElement>("noise");
        orig_prod_root->AddElement(noise_element);
    }
    const std::string calib_folder = GetRootFolder() + "annotation" + '/' + "calibration";
    const std::vector<std::string> filenames = ListFiles(calib_folder);

    if (!filenames.empty()) {
        for (const auto& metadata_file : filenames) {
            if (boost::algorithm::starts_with(metadata_file, "noise")) {
                std::fstream is;
                GetInputStream(std::string(calib_folder).append("/").append(metadata_file), is);
                if (is) {
                    pugi::xml_document xml_doc;
                    xml_doc.load(is);
                    const pugi::xml_node root_element = xml_doc.document_element();
                    std::string name = metadata_file;
                    boost::algorithm::replace_all(name, "noise-", "");
                    const auto name_elem = std::make_shared<snapengine::MetadataElement>(name);
                    noise_element->AddElement(name_elem);
                    snapengine::AbstractMetadataIO::AddXMLMetadata(root_element, name_elem);
                }
            }
        }
    }
}
std::string Sentinel1Level1Directory::GetHeaderFileName() {
    return std::string(Sentinel1Constants::PRODUCT_HEADER_NAME);
}

std::string Sentinel1Level1Directory::GetRelativePathToImageFolder() { return GetRootFolder() + "measurement" + '/'; }

}  // namespace alus::s1tbx
