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

#include "c16_dataset.h"
#include "general_constants.h"
#include "s1tbx-io/sentinel1/sentinel1_product_reader_plug_in.h"
#include "snap-core/dataio/product_subset_def.h"
#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/i_geo_coding.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/product_data_utc.h"
#include "snap-core/datamodel/pugixml_meta_data_reader.h"
#include "snap-core/datamodel/tie_point_geo_coding.h"
#include "snap-core/datamodel/tie_point_grid.h"
#include "snap-core/subset/pixel_subset_region.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/gpf/input_product_validator.h"
#include "snap-engine-utilities/gpf/operator_utils.h"
#include "split_product_subset_builder.h"
#include "subswath_info.h"

namespace alus::topsarsplit {

TopsarSplit::TopsarSplit(std::string filename, std::string selected_subswath, std::string selected_polarisation)
    : subswath_(selected_subswath), selected_polarisations_({selected_polarisation}) {
    boost::filesystem::path path = std::string(filename);
    /*std::cout << "Reading manifest at: " << path.string() + "/manifest.safe" << std::endl;
    snapengine::PugixmlMetaDataReader xml_reader(path.string() + "/manifest.safe");
    std::shared_ptr<snapengine::MetadataElement> info_pack = xml_reader.Read("informationPackageMap");
    for(std::shared_ptr<snapengine::MetadataElement> elem : info_pack->GetElements()){
        std::cout << elem->GetName() << std::endl;
    }*/
    boost::filesystem::path measurement = path.string() + "/measurement";
    boost::filesystem::directory_iterator end_itr;
    std::string low_subswath = boost::to_lower_copy(selected_subswath);
    std::string low_polarisation = boost::to_lower_copy(selected_polarisation);

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
                std::cout << "Selecting tif for reading: " << current_file << std::endl;
                pixel_reader_ = std::make_shared<C16Dataset<double>>(current_file);
                // xml_reader = std::make_shared<snapengine::PugixmlMetaDataReader>(current_file);
                found_it = true;
                break;
            }
        }
    }
    // source_product_->SetMetadataReader(xml_reader);
    // xml_reader->SetProduct(source_product_);

    if (!found_it) {
        std::stringstream stream;
        stream << "SAFE file does not contain a tif file for subswath " << selected_subswath << " and polarisation "
               << selected_polarisation << std::endl;
        throw std::runtime_error(stream.str());
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

    std::shared_ptr<snapengine::MetadataElement> absRoot =
        snapengine::AbstractMetadata::GetAbstractedMetadata(source_product_);
    if (subswath_.empty()) {
        std::stringstream ss;
        ss << absRoot->GetAttributeString(snapengine::AbstractMetadata::ACQUISITION_MODE) << "1";
        subswath_ = ss.str();
    }

    // TODO: forget the index, find the pointer.
    s1_utils_ = std::make_unique<s1tbx::Sentinel1Utils>(source_product_);
    const std::vector<std::unique_ptr<s1tbx::SubSwathInfo>>* subSwathInfo = &s1_utils_->GetSubSwath();
    for (size_t i = 0; i < subSwathInfo->size(); i++) {
        if (subSwathInfo->at(i)->subswath_name_.find(subswath_) != std::string::npos) {
            selected_subswath_info_ = subSwathInfo->at(i).get();
            break;
        }
    }
    if (selected_subswath_info_ == nullptr) {
        std::stringstream ss;
        ss << "Topsar split did not find your subswath named " << subswath_ << std::endl;
        throw std::runtime_error(ss.str());
    }

    if (selected_polarisations_.empty()) {
        selected_polarisations_ = s1_utils_->GetPolarizations();
    }

    std::vector<std::shared_ptr<snapengine::Band>> selectedBands;
    std::vector<std::shared_ptr<snapengine::Band>> le_bands = source_product_->GetBands();
    for (std::shared_ptr<snapengine::Band> srcBand : le_bands) {
        if (srcBand->GetName().find(subswath_) != std::string::npos) {
            for (std::string pol : selected_polarisations_) {
                if (srcBand->GetName().find(pol) != std::string::npos) {
                    selectedBands.push_back(srcBand);
                }
            }
        }
    }
    // TODO: Why is this here? Does the first one trigger some init as it goes through?
    if (selectedBands.size() < 1) {
        // try again
        selected_polarisations_ = s1_utils_->GetPolarizations();

        for (std::shared_ptr<snapengine::Band> srcBand : source_product_->GetBands()) {
            if (srcBand->GetName().find(subswath_) != std::string::npos) {
                for (std::string pol : selected_polarisations_) {
                    if (srcBand->GetName().find(pol) != std::string::npos) {
                        selectedBands.push_back(srcBand);
                    }
                }
            }
        }
    }

    int maxBursts = selected_subswath_info_->num_of_bursts_;
    if (last_burst_index_ > maxBursts) {
        last_burst_index_ = maxBursts;
    }

    // TODO: switch this on, when using WKT.
    /*if(wktAoi != null) {
        findValidBurstsBasedOnWkt();
    }*/

    subset_builder_ = std::make_unique<snapengine::SplitProductSubsetBuilder>();
    std::shared_ptr<snapengine::ProductSubsetDef> subset_def = std::make_shared<snapengine::ProductSubsetDef>();

    std::vector<std::string> selectedTPGList;
    for (std::shared_ptr<snapengine::TiePointGrid> srcTPG : source_product_->GetTiePointGrids()) {
        if (srcTPG->GetName().find(subswath_) != std::string::npos) {
            selectedTPGList.push_back(srcTPG->GetName());
        }
    }
    subset_def->AddNodeNames(selectedTPGList);

    int x = 0;
    int y = (first_burst_index_ - 1) * selected_subswath_info_->lines_per_burst_;
    int w = selectedBands.at(0)->GetRasterWidth();
    int h = (last_burst_index_ - first_burst_index_ + 1) * selected_subswath_info_->lines_per_burst_;
    subset_def->SetSubsetRegion(std::make_shared<snapengine::PixelSubsetRegion>(x, y, w, h, 0));

    subset_def->SetSubSampling(1, 1);
    subset_def->SetIgnoreMetadata(false);

    std::vector<std::string> selected_band_names(selectedBands.size());
    for (size_t i = 0; i < selectedBands.size(); i++) {
        selected_band_names.at(i) = selectedBands.at(i)->GetName();
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
    std::shared_ptr<snapengine::MetadataElement> absSrc =
        snapengine::AbstractMetadata::GetAbstractedMetadata(source_product_);
    std::shared_ptr<snapengine::MetadataElement> absTgt =
        snapengine::AbstractMetadata::GetAbstractedMetadata(target_product_);

    absTgt->SetAttributeUtc(
        snapengine::AbstractMetadata::FIRST_LINE_TIME,
        std::make_shared<snapengine::Utc>(selected_subswath_info_->burst_first_line_time_[first_burst_index_ - 1] /
                                          snapengine::constants::secondsInDay));

    absTgt->SetAttributeUtc(
        snapengine::AbstractMetadata::LAST_LINE_TIME,
        std::make_shared<snapengine::Utc>(selected_subswath_info_->burst_last_line_time_[last_burst_index_ - 1] /
                                          snapengine::constants::secondsInDay));

    absTgt->SetAttributeDouble(snapengine::AbstractMetadata::LINE_TIME_INTERVAL,
                               selected_subswath_info_->azimuth_time_interval_);

    absTgt->SetAttributeDouble(snapengine::AbstractMetadata::SLANT_RANGE_TO_FIRST_PIXEL,
                               selected_subswath_info_->slr_time_to_first_pixel_ * snapengine::constants::lightSpeed);

    absTgt->SetAttributeDouble(snapengine::AbstractMetadata::RANGE_SPACING,
                               selected_subswath_info_->range_pixel_spacing_);

    absTgt->SetAttributeDouble(snapengine::AbstractMetadata::AZIMUTH_SPACING,
                               selected_subswath_info_->azimuth_pixel_spacing_);

    absTgt->SetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES,
                            selected_subswath_info_->lines_per_burst_ * (last_burst_index_ - first_burst_index_ + 1));

    absTgt->SetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE,
                            selected_subswath_info_->num_of_samples_);

    int cols = selected_subswath_info_->num_of_geo_points_per_line_;

    snapengine::AbstractMetadata::SetAttribute(absTgt, snapengine::AbstractMetadata::FIRST_NEAR_LAT,
                                               selected_subswath_info_->latitude_[first_burst_index_ - 1][0]);

    snapengine::AbstractMetadata::SetAttribute(absTgt, snapengine::AbstractMetadata::FIRST_NEAR_LONG,
                                               selected_subswath_info_->longitude_[first_burst_index_ - 1][0]);

    snapengine::AbstractMetadata::SetAttribute(absTgt, snapengine::AbstractMetadata::FIRST_FAR_LAT,
                                               selected_subswath_info_->latitude_[first_burst_index_ - 1][cols - 1]);

    snapengine::AbstractMetadata::SetAttribute(absTgt, snapengine::AbstractMetadata::FIRST_FAR_LONG,
                                               selected_subswath_info_->longitude_[first_burst_index_ - 1][cols - 1]);

    snapengine::AbstractMetadata::SetAttribute(absTgt, snapengine::AbstractMetadata::LAST_NEAR_LAT,
                                               selected_subswath_info_->latitude_[last_burst_index_][0]);

    snapengine::AbstractMetadata::SetAttribute(absTgt, snapengine::AbstractMetadata::LAST_NEAR_LONG,
                                               selected_subswath_info_->longitude_[last_burst_index_][0]);

    snapengine::AbstractMetadata::SetAttribute(absTgt, snapengine::AbstractMetadata::LAST_FAR_LAT,
                                               selected_subswath_info_->latitude_[last_burst_index_][cols - 1]);

    snapengine::AbstractMetadata::SetAttribute(absTgt, snapengine::AbstractMetadata::LAST_FAR_LONG,
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

    absTgt->SetAttributeString(snapengine::AbstractMetadata::swath, subswath_);

    for (size_t i = 0; i < selected_polarisations_.size(); i++) {
        if (i == 0) {
            absTgt->SetAttributeString(snapengine::AbstractMetadata::MDS1_TX_RX_POLAR, selected_polarisations_.at(i));
        } else if (i == 1) {
            absTgt->SetAttributeString(snapengine::AbstractMetadata::MDS2_TX_RX_POLAR, selected_polarisations_.at(i));
        } else if (i == 2) {
            absTgt->SetAttributeString(snapengine::AbstractMetadata::MDS3_TX_RX_POLAR, selected_polarisations_.at(i));
        } else {
            absTgt->SetAttributeString(snapengine::AbstractMetadata::MDS4_TX_RX_POLAR, selected_polarisations_.at(i));
        }
    }

    auto bandMetadataList = snapengine::AbstractMetadata::GetBandAbsMetadataList(absTgt);
    for (std::shared_ptr<snapengine::MetadataElement> bandMeta : bandMetadataList) {
        bool include = false;

        if (bandMeta->GetName().find(subswath_) != std::string::npos) {
            for (std::string pol : selected_polarisations_) {
                if (bandMeta->GetName().find(pol) != std::string::npos) {
                    include = true;
                    break;
                }
            }
        }
        if (!include) {
            // remove band metadata if polarization or subswath is not included
            absTgt->RemoveElement(bandMeta);
        }
    }

    // Do not delete the following lines because the orbit state vector time in the target product could be wrong.
    std::shared_ptr<snapengine::MetadataElement> tgtOrbitVectorsElem =
        absTgt->GetElement(snapengine::AbstractMetadata::ORBIT_STATE_VECTORS);
    std::shared_ptr<snapengine::MetadataElement> srcOrbitVectorsElem =
        absSrc->GetElement(snapengine::AbstractMetadata::ORBIT_STATE_VECTORS);

    int numOrbitVectors = srcOrbitVectorsElem->GetNumElements();
    for (int i = 1; i <= numOrbitVectors; ++i) {
        std::stringstream ss;
        ss << snapengine::AbstractMetadata::ORBIT_VECTOR << i;
        std::string elem_name = ss.str();
        std::shared_ptr<snapengine::MetadataElement> orbElem = srcOrbitVectorsElem->GetElement(elem_name);
        std::shared_ptr<snapengine::Utc> time =
            orbElem->GetAttributeUtc(snapengine::AbstractMetadata::ORBIT_VECTOR_TIME);
        tgtOrbitVectorsElem->GetElement(elem_name)->SetAttributeUtc(snapengine::AbstractMetadata::ORBIT_VECTOR_TIME,
                                                                    time);
    }
}

void TopsarSplit::UpdateOriginalMetadata() {
    std::shared_ptr<snapengine::MetadataElement> origMeta =
        snapengine::AbstractMetadata::GetOriginalProductMetadata(target_product_);
    RemoveElements(origMeta, "annotation");
    RemoveElements(origMeta, "calibration");
    RemoveElements(origMeta, "noise");
    RemoveBursts(origMeta);
    UpdateImageInformation(origMeta);
}

void TopsarSplit::RemoveElements(std::shared_ptr<snapengine::MetadataElement>& origMeta, std::string parent) {
    std::shared_ptr<snapengine::MetadataElement> parentElem = origMeta->GetElement(parent);
    if (parentElem != nullptr) {
        auto elemList = parentElem->GetElements();
        for (std::shared_ptr<snapengine::MetadataElement> elem : elemList) {
            if (boost::to_upper_copy<std::string>(elem->GetName()).find(subswath_) == std::string::npos) {
                parentElem->RemoveElement(elem);
            } else {
                bool isSelected = false;
                for (std::string pol : selected_polarisations_) {
                    if (boost::to_upper_copy<std::string>(elem->GetName()).find(pol) != std::string::npos) {
                        isSelected = true;
                        break;
                    }
                }
                if (!isSelected) {
                    parentElem->RemoveElement(elem);
                }
            }
        }
    }
}

void TopsarSplit::RemoveBursts(std::shared_ptr<snapengine::MetadataElement>& origMeta) {
    std::shared_ptr<snapengine::MetadataElement> annotation = origMeta->GetElement("annotation");
    if (annotation == nullptr) {
        throw std::runtime_error("Annotation Metadata not found");
    }

    auto elems = annotation->GetElements();
    for (std::shared_ptr<snapengine::MetadataElement> elem : elems) {
        std::shared_ptr<snapengine::MetadataElement> product = elem->GetElement("product");
        std::shared_ptr<snapengine::MetadataElement> swathTiming = product->GetElement("swathTiming");
        std::shared_ptr<snapengine::MetadataElement> burstList = swathTiming->GetElement("burstList");
        burstList->SetAttributeString("count", std::to_string(last_burst_index_ - first_burst_index_ + 1));
        auto burstListElem = burstList->GetElements();
        int size = burstListElem.size();
        for (int i = 0; i < size; i++) {
            if (i < first_burst_index_ - 1 || i > last_burst_index_ - 1) {
                burstList->RemoveElement(burstListElem[i]);
            }
        }
    }
}

void TopsarSplit::UpdateImageInformation(std::shared_ptr<snapengine::MetadataElement>& origMeta) {
    std::shared_ptr<snapengine::MetadataElement> annotation = origMeta->GetElement("annotation");
    if (annotation == nullptr) {
        throw std::runtime_error("Annotation Metadata not found");
    }

    auto elems = annotation->GetElements();
    for (std::shared_ptr<snapengine::MetadataElement> elem : elems) {
        std::shared_ptr<snapengine::MetadataElement> product = elem->GetElement("product");
        std::shared_ptr<snapengine::MetadataElement> imageAnnotation = product->GetElement("imageAnnotation");
        std::shared_ptr<snapengine::MetadataElement> imageInformation = imageAnnotation->GetElement("imageInformation");

        imageInformation->SetAttributeString(
            "numberOfLines",
            std::to_string(selected_subswath_info_->lines_per_burst_ * (last_burst_index_ - first_burst_index_ + 1)));

        std::shared_ptr<snapengine::Utc> firstLineTimeUTC =
            std::make_shared<snapengine::Utc>(selected_subswath_info_->burst_first_line_time_[first_burst_index_ - 1] /
                                              snapengine::constants::secondsInDay);

        imageInformation->SetAttributeString("productFirstLineUtcTime", firstLineTimeUTC->Format("%Y-%m-%d %H:%M:%S"));

        std::shared_ptr<snapengine::Utc> lastLineTimeUTC =
            std::make_shared<snapengine::Utc>(selected_subswath_info_->burst_last_line_time_[last_burst_index_ - 1] /
                                              snapengine::constants::secondsInDay);

        imageInformation->SetAttributeString("productLastLineUtcTime", lastLineTimeUTC->Format("%Y-%m-%d %H:%M:%S"));
    }
}

}  // namespace alus::topsarsplit
