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
#include "topsar_split.h"

#include <memory>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/geometry.hpp>

#include "algorithm_exception.h"
#include "alus_log.h"
#include "aoi_burst_extract.h"
#include "c16_dataset.h"
#include "ceres-core/core/zip.h"
#include "general_constants.h"
#include "s1tbx-commons/subswath_info.h"
#include "s1tbx-io/sentinel1/sentinel1_product_reader_plug_in.h"
#include "snap-core/core/dataio/product_subset_def.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/datamodel/product_data_utc.h"
#include "snap-core/core/datamodel/pugixml_meta_data_reader.h"
#include "snap-core/core/datamodel/tie_point_geo_coding.h"
#include "snap-core/core/datamodel/tie_point_grid.h"
#include "snap-core/core/subset/pixel_subset_region.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"
#include "snap-engine-utilities/engine-utilities/gpf/input_product_validator.h"
#include "split_product_subset_builder.h"
#include "zip_util.h"

namespace {
constexpr std::string_view ALG_NAME{"TOPSAR-SPLIT"};
}

namespace alus::topsarsplit {

TopsarSplit::TopsarSplit(std::string_view filename, std::string_view selected_subswath,
                         std::string_view selected_polarisation, size_t first_burst, size_t last_burst)
    : TopsarSplit(filename, selected_subswath, selected_polarisation) {
    first_burst_index_ = static_cast<int>(first_burst);
    last_burst_index_ = static_cast<int>(last_burst);
}

TopsarSplit::TopsarSplit(std::string_view filename, std::string_view selected_subswath,
                         std::string_view selected_polarisation)
    : subswath_(selected_subswath), selected_polarisations_({std::string(selected_polarisation)}) {
    LoadInputDataset(filename);
}

TopsarSplit::TopsarSplit(std::string_view filename, std::string_view selected_subswath,
                         std::string_view selected_polarisation, std::string_view aoi_polygon_wkt)
    : subswath_{selected_subswath},
      selected_polarisations_{std::string(selected_polarisation)},
      burst_aoi_wkt_{aoi_polygon_wkt} {
    LoadInputDataset(filename);
}

TopsarSplit::TopsarSplit(std::shared_ptr<snapengine::Product> source_product, std::string_view selected_subswath,
                         std::string_view selected_polarisation)
    : source_product_(std::move(source_product)),
      subswath_{selected_subswath},
      selected_polarisations_{std::string(selected_polarisation)} {}

TopsarSplit::TopsarSplit(std::shared_ptr<snapengine::Product> source_product, std::string_view selected_subswath,
                         std::string_view selected_polarisation, std::string_view aoi_polygon_wkt)

    : source_product_(std::move(source_product)),
      subswath_{selected_subswath},
      selected_polarisations_{std::string(selected_polarisation)},
      burst_aoi_wkt_{aoi_polygon_wkt} {}

TopsarSplit::TopsarSplit(std::shared_ptr<snapengine::Product> source_product, std::string_view selected_subswath,
                         std::string_view selected_polarisation, size_t first_burst, size_t last_burst)
    : TopsarSplit(std::move(source_product), selected_subswath, selected_polarisation) {
    first_burst_index_ = static_cast<int>(first_burst);
    last_burst_index_ = static_cast<int>(last_burst);
}

void TopsarSplit::LoadInputDataset(std::string_view filename) {
    boost::filesystem::path path = std::string(filename);

    auto reader_plug_in = std::make_shared<alus::s1tbx::Sentinel1ProductReaderPlugIn>();
    auto reader = reader_plug_in->CreateReaderInstance();
    source_product_ = reader->ReadProductNodes(boost::filesystem::canonical(path), nullptr);
}

void TopsarSplit::OpenPixelReader(std::string_view filename) {
    bool found_it = false;
    boost::filesystem::path path = std::string(filename);
    boost::filesystem::path measurement = path.string() + "/measurement";
    boost::filesystem::directory_iterator end_itr;
    std::string low_subswath = boost::to_lower_copy(std::string(subswath_));
    std::string low_polarisation = boost::to_lower_copy(std::string(selected_polarisations_.front()));
    std::string input_file{};

    if (common::zip::IsFileAnArchive(filename)) {
        if (!boost::filesystem::exists(filename.data())) {
            std::string error_message{"No file " + std::string(filename) + " found"};
            THROW_ALGORITHM_EXCEPTION(ALG_NAME, error_message);
        }

        ceres::Zip dir(path);
        // Convert path to SAFE
        auto leaf = path.leaf();
        leaf.replace_extension("SAFE");

        std::shared_ptr<snapengine::PugixmlMetaDataReader> xml_reader;
        const auto file_list = dir.List(leaf.string() + "/measurement");
        const auto image_file =
            std::find_if(std::begin(file_list), std::end(file_list), [&low_subswath, &low_polarisation](auto& file) {
                return file.find(low_subswath) != std::string::npos && file.find(low_polarisation) != std::string::npos;
            });
        if (image_file == std::end(file_list)) {
            THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Image file for " + low_subswath + " and " + low_polarisation +
                                                    " does not exist in dataset.");
        }
        input_file = leaf.string() + "/measurement/" + *image_file;
        pixel_reader_ = std::make_shared<C16Dataset<int16_t>>(gdal::constants::GDAL_ZIP_PREFIX.data() + path.string() +
                                                              "/" + input_file);
        pixel_reader_->SetReadingArea(split_reading_area_);
        pixel_reader_->TryToCacheImage();
        found_it = true;
    } else {
        std::shared_ptr<snapengine::PugixmlMetaDataReader> xml_reader;
        for (boost::filesystem::directory_iterator itr(measurement); itr != end_itr; itr++) {
            if (is_regular_file(itr->path())) {
                std::string current_file = itr->path().string();
                if (current_file.find(low_subswath) != std::string::npos &&
                    current_file.find(low_polarisation) != std::string::npos) {
                    LOGV << "Selecting tif for reading: " << current_file;
                    pixel_reader_ = std::make_shared<C16Dataset<int16_t>>(current_file);
                    pixel_reader_->SetReadingArea(split_reading_area_);
                    pixel_reader_->TryToCacheImage();
                    found_it = true;
                    break;
                }
            }
        }
    }

    if (!found_it) {
        throw common::AlgorithmException(ALG_NAME, "SAFE file does not contain GeoTIFF file for subswath '" +
                                                       subswath_ + "' and polarisation '" +
                                                       selected_polarisations_.front() + "'");
    }
}

/**
 * Initializes this operator and sets the one and only target product.
 * <p>The target product can be either defined by a field of type {@link Product} annotated with the
 * {@link TargetProduct TargetProduct} annotation or
 * by calling {@link #setTargetProduct} method.</p>
 * <p>The framework calls this method after it has created this operator.
 * Any client code that must be performed before computation of tile data
 * should be placed here.</p>
 *
 * @throws OperatorException If an error occurs during operator initialisation.
 * @see #getTargetProduct()
 */
void TopsarSplit::Initialize() {
    snapengine::InputProductValidator validator(source_product_);
    validator.CheckIfSARProduct();
    validator.CheckIfSentinel1Product();
    validator.CheckProductType({"SLC"});
    validator.CheckAcquisitionMode({"IW", "EW"});

    std::shared_ptr<snapengine::MetadataElement> abs_root =
        snapengine::AbstractMetadata::GetAbstractedMetadata(source_product_);
    if (subswath_.empty()) {
        subswath_ = abs_root->GetAttributeString(snapengine::AbstractMetadata::ACQUISITION_MODE) + "1";
    }

    // TODO(unknown): forget the index, find the pointer.
    s1_utils_ = std::make_unique<s1tbx::Sentinel1Utils>(source_product_);
    const std::vector<std::shared_ptr<s1tbx::SubSwathInfo>>& subswath_info = s1_utils_->GetSubSwath();
    for (size_t i = 0; i < subswath_info.size(); i++) {
        if (subswath_info.at(i)->subswath_name_.find(subswath_) != std::string::npos) {
            selected_subswath_info_ = subswath_info.at(i).get();
            break;
        }
    }
    if (selected_subswath_info_ == nullptr) {
        throw common::AlgorithmException(ALG_NAME, "No subswath named '" + subswath_ + "' found");
    }

    if (selected_polarisations_.empty()) {
        selected_polarisations_ = s1_utils_->GetPolarizations();
    }

    std::vector<std::shared_ptr<snapengine::Band>> selected_bands;
    std::vector<std::shared_ptr<snapengine::Band>> le_bands = source_product_->GetBands();
    for (const auto& src_band : le_bands) {
        if (src_band->GetName().find(subswath_) != std::string::npos) {
            for (std::string pol : selected_polarisations_) {
                if (src_band->GetName().find(pol) != std::string::npos) {
                    selected_bands.push_back(src_band);
                }
            }
        }
    }
    // TODO(unknown): Why is this here? Does the first one trigger some init as it goes through?
    if (selected_bands.empty()) {
        // try again
        selected_polarisations_ = s1_utils_->GetPolarizations();

        for (const auto& src_band : source_product_->GetBands()) {
            if (src_band->GetName().find(subswath_) != std::string::npos) {
                for (const auto& pol : selected_polarisations_) {
                    if (src_band->GetName().find(pol) != std::string::npos) {
                        selected_bands.push_back(src_band);
                    }
                }
            }
        }
    }

    const auto max_bursts = selected_subswath_info_->num_of_bursts_;
    if (last_burst_index_ > max_bursts) {
        last_burst_index_ = max_bursts;
    }

    if (!burst_aoi_wkt_.empty()) {
        Aoi aoi_polygon;
        try {
            boost::geometry::read<boost::geometry::format_wkt>(aoi_polygon, burst_aoi_wkt_);
        } catch (const boost::geometry::exception& e) {
            throw common::AlgorithmException(ALG_NAME, e.what());
        }

        // Tie point grid's latitude and longitude points are along the burst north and south edges
        const auto burst_count = selected_subswath_info_->num_of_geo_lines_ - 1;
        // If data is not malformed, then first index is the edge's most west point and last the most east point along
        // the burst edge.
        const auto last_coordinate_index = selected_subswath_info_->num_of_geo_points_per_line_ - 1;
        // Assemble reasonable container for coordinates first.
        std::vector<std::vector<Coordinates>> burst_edge_line_coordinates;
        for (int line_i = 0; line_i <= burst_count; ++line_i) {
            std::vector<Coordinates> coordinates;
            for (int coordinate_i = 0; coordinate_i <= last_coordinate_index; ++coordinate_i) {
                coordinates.push_back({selected_subswath_info_->longitude_[line_i][coordinate_i],
                                       selected_subswath_info_->latitude_[line_i][coordinate_i]});
            }
            burst_edge_line_coordinates.push_back(std::move(coordinates));
        }

        auto bursts = DetermineBurstIndexesCoveredBy(aoi_polygon, burst_edge_line_coordinates);
        if (bursts.empty()) {
            throw common::AlgorithmException(ALG_NAME, "Given AOI '" + burst_aoi_wkt_ +
                                                           "' does not cover any bursts in the selected subswath '" +
                                                           subswath_ + "'");
        }
        first_burst_index_ = bursts.front();
        last_burst_index_ = bursts.back();

        if (first_burst_index_ > last_burst_index_ || last_burst_index_ > max_bursts) {
            THROW_ALGORITHM_EXCEPTION(ALG_NAME, "First burst(" + std::to_string(first_burst_index_) +
                                                    ") and last burst(" + std::to_string(last_burst_index_) +
                                                    ") indexes do not align");
        }
        LOGI << subswath_ << " AOI - Burst indexes " << first_burst_index_ << " " << last_burst_index_;
    }

    if (first_burst_index_ < BURST_INDEX_OFFSET) {
        throw common::AlgorithmException(
            ALG_NAME, ("First burst index (" + std::to_string(first_burst_index_) + ") not valid - starting from " +
                       std::to_string(BURST_INDEX_OFFSET)));
    }

    subset_builder_ = std::make_unique<snapengine::SplitProductSubsetBuilder>();
    auto subset_def = std::make_shared<snapengine::ProductSubsetDef>();

    std::vector<std::string> selected_tpg_list;
    for (const auto& src_tpg : source_product_->GetTiePointGrids()) {
        if (src_tpg->GetName().find(subswath_) != std::string::npos) {
            selected_tpg_list.push_back(src_tpg->GetName());
        }
    }
    subset_def->AddNodeNames(selected_tpg_list);
    int x = 0;
    int y = (first_burst_index_ - BURST_INDEX_OFFSET) * selected_subswath_info_->lines_per_burst_;
    int w = selected_bands.at(0)->GetRasterWidth();
    int h = (last_burst_index_ - first_burst_index_ + BURST_INDEX_OFFSET) * selected_subswath_info_->lines_per_burst_;
    subset_def->SetSubsetRegion(std::make_shared<snapengine::PixelSubsetRegion>(x, y, w, h, 0));

    subset_def->SetSubSampling(1, 1);
    subset_def->SetIgnoreMetadata(false);
    split_reading_area_ = {x, y, w, h};

    std::vector<std::string> selected_band_names(selected_bands.size());
    for (size_t i = 0; i < selected_bands.size(); i++) {
        selected_band_names.at(i) = selected_bands.at(i)->GetName();
    }
    subset_def->AddNodeNames(selected_band_names);

    target_product_ = subset_builder_->ReadProductNodes(source_product_, subset_def);

    // target_product_->RemoveTiePointGrid(target_product_->GetTiePointGrid("latitude"));
    // target_product_->RemoveTiePointGrid(target_product_->GetTiePointGrid("longitude"));

    for (std::shared_ptr<snapengine::TiePointGrid> tpg : target_product_->GetTiePointGrids()) {
        tpg->SetName(boost::replace_all_copy(tpg->GetName(), subswath_ + "_", ""));
    }

    // std::shared_ptr<snapengine::IGeoCoding> geoCoding =
    // std::make_shared<snapengine::TiePointGeoCoding>(target_product_->GetTiePointGrid("latitude"),
    //                                            target_product_->GetTiePointGrid("longitude"));
    // target_product_->SetSceneGeoCoding(geoCoding);

    UpdateTargetProductMetadata();
}

void TopsarSplit::UpdateTargetProductMetadata() {
    UpdateAbstractedMetadata();
    UpdateOriginalMetadata();
}

void TopsarSplit::UpdateAbstractedMetadata() {
    std::shared_ptr<snapengine::MetadataElement> abs_src =
        snapengine::AbstractMetadata::GetAbstractedMetadata(source_product_);
    std::shared_ptr<snapengine::MetadataElement> abs_tgt =
        snapengine::AbstractMetadata::GetAbstractedMetadata(target_product_);

    abs_tgt->SetAttributeUtc(
        snapengine::AbstractMetadata::FIRST_LINE_TIME,
        std::make_shared<snapengine::Utc>(
            selected_subswath_info_->burst_first_line_time_[first_burst_index_ - BURST_INDEX_OFFSET] /
            snapengine::eo::constants::SECONDS_IN_DAY));

    abs_tgt->SetAttributeUtc(
        snapengine::AbstractMetadata::LAST_LINE_TIME,
        std::make_shared<snapengine::Utc>(
            selected_subswath_info_->burst_last_line_time_[last_burst_index_ - BURST_INDEX_OFFSET] /
            snapengine::eo::constants::SECONDS_IN_DAY));

    abs_tgt->SetAttributeDouble(snapengine::AbstractMetadata::LINE_TIME_INTERVAL,
                                selected_subswath_info_->azimuth_time_interval_);

    abs_tgt->SetAttributeDouble(
        snapengine::AbstractMetadata::SLANT_RANGE_TO_FIRST_PIXEL,
        selected_subswath_info_->slr_time_to_first_pixel_ * snapengine::eo::constants::LIGHT_SPEED);

    abs_tgt->SetAttributeDouble(snapengine::AbstractMetadata::RANGE_SPACING,
                                selected_subswath_info_->range_pixel_spacing_);

    abs_tgt->SetAttributeDouble(snapengine::AbstractMetadata::AZIMUTH_SPACING,
                                selected_subswath_info_->azimuth_pixel_spacing_);

    abs_tgt->SetAttributeInt(
        snapengine::AbstractMetadata::NUM_OUTPUT_LINES,
        selected_subswath_info_->lines_per_burst_ * (last_burst_index_ - first_burst_index_ + BURST_INDEX_OFFSET));

    abs_tgt->SetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE,
                             selected_subswath_info_->num_of_samples_);

    int cols = selected_subswath_info_->num_of_geo_points_per_line_;

    snapengine::AbstractMetadata::SetAttribute(
        abs_tgt, snapengine::AbstractMetadata::FIRST_NEAR_LAT,
        selected_subswath_info_->latitude_[first_burst_index_ - BURST_INDEX_OFFSET][0]);

    snapengine::AbstractMetadata::SetAttribute(
        abs_tgt, snapengine::AbstractMetadata::FIRST_NEAR_LONG,
        selected_subswath_info_->longitude_[first_burst_index_ - BURST_INDEX_OFFSET][0]);

    snapengine::AbstractMetadata::SetAttribute(
        abs_tgt, snapengine::AbstractMetadata::FIRST_FAR_LAT,
        selected_subswath_info_->latitude_[first_burst_index_ - BURST_INDEX_OFFSET][cols - 1]);

    snapengine::AbstractMetadata::SetAttribute(
        abs_tgt, snapengine::AbstractMetadata::FIRST_FAR_LONG,
        selected_subswath_info_->longitude_[first_burst_index_ - BURST_INDEX_OFFSET][cols - 1]);

    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::LAST_NEAR_LAT,
                                               selected_subswath_info_->latitude_[last_burst_index_][0]);

    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::LAST_NEAR_LONG,
                                               selected_subswath_info_->longitude_[last_burst_index_][0]);

    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::LAST_FAR_LAT,
                                               selected_subswath_info_->latitude_[last_burst_index_][cols - 1]);

    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::LAST_FAR_LONG,
                                               selected_subswath_info_->longitude_[last_burst_index_][cols - 1]);
    // TODO: Tie point grids
    /*double incidenceNear = target_product_->GetTiePointGrid(snapengine::OperatorUtils::TPG_INCIDENT_ANGLE)
                               ->GetPixelDouble(0, target_product_->GetSceneRasterHeight() / 2);*/
    /*double incidenceNear = snapengine::OperatorUtils::GetIncidenceAngle(targetProduct).getPixelDouble(
        0, targetProduct.getSceneRasterHeight() / 2);*/

    // snapengine::AbstractMetadata::SetAttribute(absTgt, snapengine::AbstractMetadata::INCIDENCE_NEAR, incidenceNear);

    /*double incidenceFar =
        target_product_->GetTiePointGrid(snapengine::OperatorUtils::TPG_INCIDENT_ANGLE)
            ->GetPixelDouble(target_product_->GetSceneRasterWidth() - 1, target_product_->GetSceneRasterHeight() / 2);*/
    /*double incidenceFar = snapengine::OperatorUtils::GetIncidenceAngle(targetProduct).getPixelDouble(
        targetProduct.getSceneRasterWidth() - 1, targetProduct.getSceneRasterHeight() / 2);*/

    // snapengine::AbstractMetadata::SetAttribute(absTgt, snapengine::AbstractMetadata::INCIDENCE_FAR, incidenceFar);

    abs_tgt->SetAttributeString(snapengine::AbstractMetadata::swath, subswath_);

    for (size_t i = 0; i < selected_polarisations_.size(); i++) {
        if (i == 0) {
            abs_tgt->SetAttributeString(snapengine::AbstractMetadata::MDS1_TX_RX_POLAR, selected_polarisations_.at(i));
        } else if (i == 1) {
            abs_tgt->SetAttributeString(snapengine::AbstractMetadata::MDS2_TX_RX_POLAR, selected_polarisations_.at(i));
        } else if (i == 2) {
            abs_tgt->SetAttributeString(snapengine::AbstractMetadata::MDS3_TX_RX_POLAR, selected_polarisations_.at(i));
        } else {
            abs_tgt->SetAttributeString(snapengine::AbstractMetadata::MDS4_TX_RX_POLAR, selected_polarisations_.at(i));
        }
    }

    auto band_metadata_list = snapengine::AbstractMetadata::GetBandAbsMetadataList(abs_tgt);
    for (const auto& band_meta : band_metadata_list) {
        bool include = false;

        if (band_meta->GetName().find(subswath_) != std::string::npos) {
            for (const auto& pol : selected_polarisations_) {
                if (band_meta->GetName().find(pol) != std::string::npos) {
                    include = true;
                    break;
                }
            }
        }
        if (!include) {
            // remove band metadata if polarization or subswath is not included
            abs_tgt->RemoveElement(band_meta);
        }
    }

    // Do not delete the following lines because the orbit state vector time in the target product could be wrong.
    std::shared_ptr<snapengine::MetadataElement> tgt_orbit_vectors_elem =
        abs_tgt->GetElement(snapengine::AbstractMetadata::ORBIT_STATE_VECTORS);
    std::shared_ptr<snapengine::MetadataElement> src_orbit_vectors_elem =
        abs_src->GetElement(snapengine::AbstractMetadata::ORBIT_STATE_VECTORS);

    int num_orbit_vectors = src_orbit_vectors_elem->GetNumElements();
    for (int i = 1; i <= num_orbit_vectors; ++i) {
        const auto elem_name = std::string(snapengine::AbstractMetadata::ORBIT_VECTOR) + std::to_string(i);
        std::shared_ptr<snapengine::MetadataElement> orb_elem = src_orbit_vectors_elem->GetElement(elem_name);
        std::shared_ptr<snapengine::Utc> time =
            orb_elem->GetAttributeUtc(snapengine::AbstractMetadata::ORBIT_VECTOR_TIME);
        tgt_orbit_vectors_elem->GetElement(elem_name)->SetAttributeUtc(snapengine::AbstractMetadata::ORBIT_VECTOR_TIME,
                                                                       time);
    }
}

void TopsarSplit::UpdateOriginalMetadata() {
    std::shared_ptr<snapengine::MetadataElement> orig_meta =
        snapengine::AbstractMetadata::GetOriginalProductMetadata(target_product_);
    RemoveElements(orig_meta, "annotation");
    RemoveElements(orig_meta, "calibration");
    RemoveElements(orig_meta, "noise");
    RemoveBursts(orig_meta);
    UpdateImageInformation(orig_meta);
}

void TopsarSplit::RemoveElements(std::shared_ptr<snapengine::MetadataElement>& orig_meta, std::string parent) {
    std::shared_ptr<snapengine::MetadataElement> parent_elem = orig_meta->GetElement(parent);
    if (parent_elem != nullptr) {
        auto elem_list = parent_elem->GetElements();
        for (const auto& elem : elem_list) {
            if (boost::to_upper_copy<std::string>(elem->GetName()).find(subswath_) == std::string::npos) {
                parent_elem->RemoveElement(elem);
            } else {
                bool is_selected = false;
                for (const auto& pol : selected_polarisations_) {
                    if (boost::to_upper_copy<std::string>(elem->GetName()).find(pol) != std::string::npos) {
                        is_selected = true;
                        break;
                    }
                }
                if (!is_selected) {
                    parent_elem->RemoveElement(elem);
                }
            }
        }
    }
}

void TopsarSplit::RemoveBursts(std::shared_ptr<snapengine::MetadataElement>& orig_meta) {
    std::shared_ptr<snapengine::MetadataElement> annotation = orig_meta->GetElement("annotation");
    if (annotation == nullptr) {
        throw common::AlgorithmException(ALG_NAME, "Annotation Metadata not found");
    }

    auto elems = annotation->GetElements();
    for (const auto& elem : elems) {
        std::shared_ptr<snapengine::MetadataElement> product = elem->GetElement("product");
        std::shared_ptr<snapengine::MetadataElement> swath_timing = product->GetElement("swathTiming");
        std::shared_ptr<snapengine::MetadataElement> burst_list = swath_timing->GetElement("burstList");
        burst_list->SetAttributeString("count",
                                       std::to_string(last_burst_index_ - first_burst_index_ + BURST_INDEX_OFFSET));
        auto burst_list_elem = burst_list->GetElements();
        auto size = static_cast<int>(burst_list_elem.size());
        for (int i = 0; i < size; i++) {
            if (i < first_burst_index_ - BURST_INDEX_OFFSET || i > last_burst_index_ - BURST_INDEX_OFFSET) {
                burst_list->RemoveElement(burst_list_elem[i]);
            }
        }
    }
}

void TopsarSplit::UpdateImageInformation(std::shared_ptr<snapengine::MetadataElement>& orig_meta) {
    std::shared_ptr<snapengine::MetadataElement> annotation = orig_meta->GetElement("annotation");
    if (annotation == nullptr) {
        throw common::AlgorithmException(ALG_NAME, "Annotation Metadata not found");
    }

    auto elems = annotation->GetElements();
    for (const auto& elem : elems) {
        std::shared_ptr<snapengine::MetadataElement> product = elem->GetElement("product");
        std::shared_ptr<snapengine::MetadataElement> image_annotation = product->GetElement("imageAnnotation");
        std::shared_ptr<snapengine::MetadataElement> image_information =
            image_annotation->GetElement("imageInformation");

        image_information->SetAttributeString(
            "numberOfLines", std::to_string(selected_subswath_info_->lines_per_burst_ *
                                            (last_burst_index_ - first_burst_index_ + BURST_INDEX_OFFSET)));

        auto first_line_time_utc = std::make_shared<snapengine::Utc>(
            selected_subswath_info_->burst_first_line_time_[first_burst_index_ - BURST_INDEX_OFFSET] /
            snapengine::eo::constants::SECONDS_IN_DAY);

        image_information->SetAttributeString("productFirstLineUtcTime",
                                              first_line_time_utc->Format("%Y-%m-%d %H:%M:%S"));

        auto last_line_time_utc = std::make_shared<snapengine::Utc>(
            selected_subswath_info_->burst_last_line_time_[last_burst_index_ - BURST_INDEX_OFFSET] /
            snapengine::eo::constants::SECONDS_IN_DAY);

        image_information->SetAttributeString("productLastLineUtcTime",
                                              last_line_time_utc->Format("%Y-%m-%d %H:%M:%S"));
    }
}

}  // namespace alus::topsarsplit
