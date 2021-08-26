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

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "alus_log.h"
#include "c16_dataset.h"
#include "general_constants.h"
#include "s1tbx-io/sentinel1/sentinel1_product_reader_plug_in.h"
#include "snap-core/dataio/product_subset_def.h"
#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/product_data_utc.h"
#include "snap-core/datamodel/pugixml_meta_data_reader.h"
#include "snap-core/datamodel/tie_point_geo_coding.h"
#include "snap-core/datamodel/tie_point_grid.h"
#include "snap-core/subset/pixel_subset_region.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/eo/constants.h"
#include "snap-engine-utilities/gpf/input_product_validator.h"
#include "split_product_subset_builder.h"
#include "subswath_info.h"

namespace alus::topsarsplit {

TopsarSplit::TopsarSplit(std::string_view filename, std::string_view selected_subswath, std::string_view selected_polarisation,
                         size_t first_burst, size_t last_burst)
    :  TopsarSplit(filename, selected_subswath, selected_polarisation) {
    first_burst_index_ = static_cast<int>(first_burst);
    last_burst_index_ = static_cast<int>(last_burst);

    if(first_burst_index_ == 0){
        throw std::runtime_error("Numbers to select burst in TOPSAR split start from 1, you chose 0.");
    }
}

TopsarSplit::TopsarSplit(std::string_view filename, std::string_view selected_subswath, std::string_view selected_polarisation)
    : subswath_(selected_subswath), selected_polarisations_({std::string(selected_polarisation)}) {
    boost::filesystem::path path = std::string(filename);
    boost::filesystem::path measurement = path.string() + "/measurement";
    boost::filesystem::directory_iterator end_itr;
    std::string low_subswath = boost::to_lower_copy(std::string(selected_subswath));
    std::string low_polarisation = boost::to_lower_copy(std::string(selected_polarisation));

    auto reader_plug_in = std::make_shared<alus::s1tbx::Sentinel1ProductReaderPlugIn>();
    reader_ = reader_plug_in->CreateReaderInstance();
    source_product_ = reader_->ReadProductNodes(boost::filesystem::canonical("manifest.safe", path), nullptr);

    bool found_it = false;
    std::shared_ptr<snapengine::PugixmlMetaDataReader> xml_reader;
    for (boost::filesystem::directory_iterator itr(measurement); itr != end_itr; itr++) {
        if (is_regular_file(itr->path())) {
            std::string current_file = itr->path().string();
            if (current_file.find(low_subswath) != std::string::npos &&
                current_file.find(low_polarisation) != std::string::npos) {
                LOGV << "Selecting tif for reading: " << current_file;
                pixel_reader_ = std::make_shared<C16Dataset<double>>(current_file);
                pixel_reader_->TryToCacheImage();
                found_it = true;
                break;
            }
        }
    }

    if (!found_it) {
        throw std::runtime_error("SAFE file does not contain GeoTIFF file for subswath '" + std::string(selected_subswath) +
                                 "' and polarisation '" + std::string(selected_polarisation) + "'");
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
void TopsarSplit::initialize() {
    snapengine::InputProductValidator validator(source_product_);
    validator.CheckIfSARProduct();
    validator.CheckIfSentinel1Product();
    validator.CheckIfMultiSwathTOPSARProduct();
    validator.CheckProductType({"SLC"});
    validator.CheckAcquisitionMode({"IW", "EW"});

    std::shared_ptr<snapengine::MetadataElement> abs_root =
        snapengine::AbstractMetadata::GetAbstractedMetadata(source_product_);
    if (subswath_.empty()) {
        subswath_ = abs_root->GetAttributeString(snapengine::AbstractMetadata::ACQUISITION_MODE) + "1";
    }

    // TODO: forget the index, find the pointer.
    s1_utils_ = std::make_unique<s1tbx::Sentinel1Utils>(source_product_);
    const std::vector<std::unique_ptr<s1tbx::SubSwathInfo>>& subswath_info = s1_utils_->GetSubSwath();
    for (size_t i = 0; i < subswath_info.size(); i++) {
        if (subswath_info.at(i)->subswath_name_.find(subswath_) != std::string::npos) {
            selected_subswath_info_ = subswath_info.at(i).get();
            break;
        }
    }
    if (selected_subswath_info_ == nullptr) {
        throw std::runtime_error("Topsar split did not find subswath named " + subswath_);
    }

    if (selected_polarisations_.empty()) {
        selected_polarisations_ = s1_utils_->GetPolarizations();
    }

    std::vector<std::shared_ptr<snapengine::Band>> selected_bands;
    std::vector<std::shared_ptr<snapengine::Band>> le_bands = source_product_->GetBands();
    for (std::shared_ptr<snapengine::Band> src_band : le_bands) {
        if (src_band->GetName().find(subswath_) != std::string::npos) {
            for (std::string pol : selected_polarisations_) {
                if (src_band->GetName().find(pol) != std::string::npos) {
                    selected_bands.push_back(src_band);
                }
            }
        }
    }
    // TODO: Why is this here? Does the first one trigger some init as it goes through?
    if (selected_bands.size() < 1) {
        // try again
        selected_polarisations_ = s1_utils_->GetPolarizations();

        for (std::shared_ptr<snapengine::Band> src_band : source_product_->GetBands()) {
            if (src_band->GetName().find(subswath_) != std::string::npos) {
                for (std::string pol : selected_polarisations_) {
                    if (src_band->GetName().find(pol) != std::string::npos) {
                        selected_bands.push_back(src_band);
                    }
                }
            }
        }
    }

    int max_bursts = selected_subswath_info_->num_of_bursts_;
    if (last_burst_index_ > max_bursts) {
        last_burst_index_ = max_bursts;
    }

    // TODO: switch this on, when using WKT.
    /*if(wktAoi != null) {
        findValidBurstsBasedOnWkt();
    }*/

    subset_builder_ = std::make_unique<snapengine::SplitProductSubsetBuilder>();
    std::shared_ptr<snapengine::ProductSubsetDef> subset_def = std::make_shared<snapengine::ProductSubsetDef>();

    std::vector<std::string> selected_tpg_list;
    for (std::shared_ptr<snapengine::TiePointGrid> src_tpg : source_product_->GetTiePointGrids()) {
        if (src_tpg->GetName().find(subswath_) != std::string::npos) {
            selected_tpg_list.push_back(src_tpg->GetName());
        }
    }
    subset_def->AddNodeNames(selected_tpg_list);

    int x = 0;
    int y = (first_burst_index_ - 1) * selected_subswath_info_->lines_per_burst_;
    int w = selected_bands.at(0)->GetRasterWidth();
    int h = (last_burst_index_ - first_burst_index_ + 1) * selected_subswath_info_->lines_per_burst_;
    subset_def->SetSubsetRegion(std::make_shared<snapengine::PixelSubsetRegion>(x, y, w, h, 0));

    subset_def->SetSubSampling(1, 1);
    subset_def->SetIgnoreMetadata(false);
    pixel_reader_->SetReadingArea({x,y,w,h});

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
        std::make_shared<snapengine::Utc>(selected_subswath_info_->burst_first_line_time_[first_burst_index_ - 1] /
                                          snapengine::eo::constants::SECONDS_IN_DAY));

    abs_tgt->SetAttributeUtc(
        snapengine::AbstractMetadata::LAST_LINE_TIME,
        std::make_shared<snapengine::Utc>(selected_subswath_info_->burst_last_line_time_[last_burst_index_ - 1] /
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

    abs_tgt->SetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES,
                             selected_subswath_info_->lines_per_burst_ * (last_burst_index_ - first_burst_index_ + 1));

    abs_tgt->SetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE,
                             selected_subswath_info_->num_of_samples_);

    int cols = selected_subswath_info_->num_of_geo_points_per_line_;

    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::FIRST_NEAR_LAT,
                                               selected_subswath_info_->latitude_[first_burst_index_ - 1][0]);

    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::FIRST_NEAR_LONG,
                                               selected_subswath_info_->longitude_[first_burst_index_ - 1][0]);

    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::FIRST_FAR_LAT,
                                               selected_subswath_info_->latitude_[first_burst_index_ - 1][cols - 1]);

    snapengine::AbstractMetadata::SetAttribute(abs_tgt, snapengine::AbstractMetadata::FIRST_FAR_LONG,
                                               selected_subswath_info_->longitude_[first_burst_index_ - 1][cols - 1]);

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
    for (std::shared_ptr<snapengine::MetadataElement> band_meta : band_metadata_list) {
        bool include = false;

        if (band_meta->GetName().find(subswath_) != std::string::npos) {
            for (std::string pol : selected_polarisations_) {
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
        for (std::shared_ptr<snapengine::MetadataElement> elem : elem_list) {
            if (boost::to_upper_copy<std::string>(elem->GetName()).find(subswath_) == std::string::npos) {
                parent_elem->RemoveElement(elem);
            } else {
                bool is_selected = false;
                for (std::string pol : selected_polarisations_) {
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
        throw std::runtime_error("Annotation Metadata not found");
    }

    auto elems = annotation->GetElements();
    for (std::shared_ptr<snapengine::MetadataElement> elem : elems) {
        std::shared_ptr<snapengine::MetadataElement> product = elem->GetElement("product");
        std::shared_ptr<snapengine::MetadataElement> swath_timing = product->GetElement("swathTiming");
        std::shared_ptr<snapengine::MetadataElement> burst_list = swath_timing->GetElement("burstList");
        burst_list->SetAttributeString("count", std::to_string(last_burst_index_ - first_burst_index_ + 1));
        auto burst_list_elem = burst_list->GetElements();
        int size = burst_list_elem.size();
        for (int i = 0; i < size; i++) {
            if (i < first_burst_index_ - 1 || i > last_burst_index_ - 1) {
                burst_list->RemoveElement(burst_list_elem[i]);
            }
        }
    }
}

void TopsarSplit::UpdateImageInformation(std::shared_ptr<snapengine::MetadataElement>& orig_meta) {
    std::shared_ptr<snapengine::MetadataElement> annotation = orig_meta->GetElement("annotation");
    if (annotation == nullptr) {
        throw std::runtime_error("Annotation Metadata not found");
    }

    auto elems = annotation->GetElements();
    for (std::shared_ptr<snapengine::MetadataElement> elem : elems) {
        std::shared_ptr<snapengine::MetadataElement> product = elem->GetElement("product");
        std::shared_ptr<snapengine::MetadataElement> image_annotation = product->GetElement("imageAnnotation");
        std::shared_ptr<snapengine::MetadataElement> image_information =
            image_annotation->GetElement("imageInformation");

        image_information->SetAttributeString(
            "numberOfLines",
            std::to_string(selected_subswath_info_->lines_per_burst_ * (last_burst_index_ - first_burst_index_ + 1)));

        std::shared_ptr<snapengine::Utc> first_line_time_utc =
            std::make_shared<snapengine::Utc>(selected_subswath_info_->burst_first_line_time_[first_burst_index_ - 1] /
                                              snapengine::eo::constants::SECONDS_IN_DAY);

        image_information->SetAttributeString("productFirstLineUtcTime",
                                              first_line_time_utc->Format("%Y-%m-%d %H:%M:%S"));

        std::shared_ptr<snapengine::Utc> last_line_time_utc =
            std::make_shared<snapengine::Utc>(selected_subswath_info_->burst_last_line_time_[last_burst_index_ - 1] /
                                              snapengine::eo::constants::SECONDS_IN_DAY);

        image_information->SetAttributeString("productLastLineUtcTime",
                                              last_line_time_utc->Format("%Y-%m-%d %H:%M:%S"));
    }
}

}  // namespace alus::topsarsplit
