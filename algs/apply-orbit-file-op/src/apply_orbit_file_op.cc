#include "apply_orbit_file_op.h"

#include <cstddef>
#include <exception>
#include <iostream>

#include "gdal_data_copy.h"
#include "general_constants.h"
#include "io/orbits/sentinel1/sentinel_p_o_d_orbit_file.h"
#include "meta_data_node_names.h"
#include "orbit_state_vector.h"
#include "product_data_utc.h"
#include "product_utils.h"

namespace alus {
namespace s1tbx {

ApplyOrbitFileOp::ApplyOrbitFileOp(const std::shared_ptr<snapengine::Product>& source_product)
    : source_product_(source_product) {}

void ApplyOrbitFileOp::GetSourceMetadata() {
    // original snap version creates Product when it is opened inside snap, it uses DimapProductHelpers and calls createProduct which
    // calls addAnnotationDataset which attaches metadata to root element source_product_ already has everything set,
    // snap version is absRoot = AbstractMetadata.getAbstractedMetadata(sourceProduct);  which uses a bit different
    // logic
    // copy is using root element which is metadata
    // todo: check if source product already gets respective elements attached before (this logic is deviation from
    // original snap logic)
    source_product_->GetMetadataRoot()->AddElement(
        source_product_->GetMetadataReader()->Read(alus::snapengine::MetaDataNodeNames::ABSTRACT_METADATA_ROOT));
    // get abstracted metadata from source product
    abs_root_ = snapengine::MetaDataNodeNames::GetAbstractedMetadata(source_product_);
    mission_ = abs_root_->GetAttributeString(snapengine::MetaDataNodeNames::MISSION);
    // todo:check these, took from metadata vs from product dimensions in java (if something is missing)
    source_image_width_ = abs_root_->GetAttributeInt(snapengine::MetaDataNodeNames::NUM_SAMPLES_PER_LINE);
    source_image_height_ = abs_root_->GetAttributeInt(snapengine::MetaDataNodeNames::NUM_OUTPUT_LINES);
}

void ApplyOrbitFileOp::Initialize() {
    try {
        GetSourceMetadata();
        // todo: we need to check over later if snap's logic suits our project logic
        if (orbit_type_.empty()) {
            if (mission_.rfind("SENTINEL", 0) == 0) {
                orbit_type_ = SentinelPODOrbitFile::PRECISE;
            } else {
                throw std::runtime_error("Please select an orbit file type");
            }
        }

        if (mission_.rfind("SENTINEL", 0) == 0) {
            if (!orbit_type_.rfind("Sentinel", 0)) {
                orbit_type_ = SentinelPODOrbitFile::PRECISE;
            }
        } else {
            throw std::runtime_error(orbit_type_ + " is not suitable for a " + mission_ + " product");
        }

        if (orbit_type_.find("Sentinel") != std::string::npos) {
            orbit_provider_ = std::make_shared<SentinelPODOrbitFile>(poly_degree_, abs_root_);
        }

        // skip until we have more core datamodel ported
        //        GetTiePointGrid();
        CreateTargetProduct();

        if (!product_updated_) {
            try {
                UpdateOrbits();
            } catch (const std::exception& e) {
                if (continue_on_fail_) {
                    //                    LOG(WARNING) << "ApplyOrbit ignoring error and continuing: " << e.what();
                    std::cerr << "ApplyOrbit ignoring error and continuing: " << e.what() << std::endl;
                    product_updated_ = true;
                } else {
                    throw e;
                }
            }
        }

    } catch (std::exception& e) {
        throw std::runtime_error("ApplyOrbitFileOp exception: " + std::string(e.what()));
    }
}

// todo: snapimplementation had race condition, check if we have this problem
void ApplyOrbitFileOp::UpdateOrbits() {
    if (product_updated_) {
        return;
    }
    try {
        orbit_provider_->RetrieveOrbitFile(orbit_type_);
    } catch (std::exception& e) {
        //        LOG(WARNING) << e.what();
        std::cerr << e.what() << std::endl;
        // try other orbit file types
        bool try_another_type = false;
        for (std::string type : orbit_provider_->GetAvailableOrbitTypes()) {
            if (!orbit_type_.rfind(type, 0)) {
                try {
                    orbit_provider_->RetrieveOrbitFile(type);
                } catch (std::exception& e2) {
                    throw e;
                }
                //                LOG(WARNING) << "Using " + type + ' ' << orbit_provider_->GetOrbitFile() << "
                //                instead";
                std::cerr << "Using " << type << ' ' << orbit_provider_->GetOrbitFile() << " instead" << std::endl;
                try_another_type = true;
            }
        }
        if (!try_another_type) {
            throw e;
        }
    }
    UpdateOrbitStateVectors();
    product_updated_ = true;
}

void ApplyOrbitFileOp::UpdateOrbitStateVectors() {
    double first_line_utc = snapengine::MetaDataNodeNames::ParseUtc(
                                abs_root_->GetAttributeString(snapengine::MetaDataNodeNames::FIRST_LINE_TIME))
                                ->GetMjd();  // in days

    double last_line_utc = snapengine::MetaDataNodeNames::ParseUtc(
                               abs_root_->GetAttributeString(snapengine::MetaDataNodeNames::LAST_LINE_TIME))
                               ->GetMjd();  // in days

    double delta = 1.0 / snapengine::constants::secondsInDay;  // time interval = 1s

    int num_extra_vectors = 10;  // # of vectors before and after acquisition period

    double first_vector_time = first_line_utc - num_extra_vectors * delta;  // in days

    double last_vector_time = last_line_utc + num_extra_vectors * delta;  // in days

    int num_vectors = (int)((last_vector_time - first_vector_time) / delta);

    std::vector<snapengine::OrbitStateVector> orbit_state_vectors(num_vectors);

    // compute new orbit state vectors
    for (int i = 0; i < num_vectors; ++i) {
        double time = first_vector_time + i * delta;
        auto orbit_data = orbit_provider_->GetOrbitData(time);
        auto utc = std::make_shared<snapengine::Utc>(time);
        orbit_state_vectors.at(i) =
            snapengine::OrbitStateVector(utc, orbit_data->x_pos_, orbit_data->y_pos_, orbit_data->z_pos_,
                                         orbit_data->x_vel_, orbit_data->y_vel_, orbit_data->z_vel_);
    }

    auto tgt_abs_root = snapengine::MetaDataNodeNames::GetAbstractedMetadata(target_product_);
    snapengine::MetaDataNodeNames::SetOrbitStateVectors(tgt_abs_root, orbit_state_vectors);
    // save orbit file name
    std::string orb_type = orbit_type_;
    std::size_t pos = orb_type.find('(');
    if (pos != std::string::npos) {
        orb_type = orbit_type_.substr(0, pos);
    }
    auto orbit_file = orbit_provider_->GetOrbitFile();
    tgt_abs_root->SetAttributeString(snapengine::MetaDataNodeNames::ORBIT_STATE_VECTOR_FILE,
                                     orb_type + " " + orbit_file.filename().string());
}
void ApplyOrbitFileOp::CreateTargetProduct() {
    target_product_ = std::make_shared<snapengine::Product>(
        std::string(source_product_->GetName()) + std::string(PRODUCT_SUFFIX), source_product_->GetProductType(),
        source_product_->GetSceneRasterWidth(), source_product_->GetSceneRasterHeight());
    // todo::workaround should probably hide similar functionality inside this function
    snapengine::ProductUtils::CopyProductNodes(source_product_, target_product_);
    //    todo: naming logic is currently half baked on the fly changes (snap used different systems for that)
    target_product_->SetFileLocation(
        source_product_->GetFileLocation().parent_path().parent_path().generic_path().string() +
        boost::filesystem::path::preferred_separator + target_product_->GetName() +
        boost::filesystem::path::preferred_separator + target_product_->GetName() + ".tif");

    //todo: look this over when we decide on jai replacement
    /*
    for (std::shared_ptr<Band> src_band : source_product_->GetBands()) {
        if (src_band instanceof VirtualBand) {
            ProductUtils::CopyVirtualBand(target_product_, (VirtualBand)src_band, src_band->GetName());
        } else {
            ProductUtils::CopyBand(src_band->GetName(), source_product_, target_product_, true);
        }
    }*/
}
void ApplyOrbitFileOp::WriteProductFiles(std::shared_ptr<snapengine::IMetaDataWriter> metadata_writer) {
    boost::filesystem::create_directories(target_product_->GetFileLocation().parent_path().generic_path());
    // copy data
    alus::GdalDataCopy(source_product_->GetFileLocation().c_str(), target_product_->GetFileLocation().c_str());
    // temporary workaround to forward tie_point_grids directory...
    boost::filesystem::copy(source_product_->GetFileLocation().parent_path().generic_path().string() +
                                boost::filesystem::path::preferred_separator + "tie_point_grids",
                            target_product_->GetFileLocation().parent_path().generic_path().string() +
                                boost::filesystem::path::preferred_separator + "tie_point_grids",
                            boost::filesystem::copy_options::recursive);
    // temporary workaround to forward vector directory...
    boost::filesystem::copy(source_product_->GetFileLocation().parent_path().generic_path().string() +
                                boost::filesystem::path::preferred_separator + "vector_data",
                            target_product_->GetFileLocation().parent_path().generic_path().string() +
                                boost::filesystem::path::preferred_separator + "vector_data",
                            boost::filesystem::copy_options::recursive);
    // write metadata file
    target_product_->SetMetadataWriter(metadata_writer);
    target_product_->GetMetadataWriter()->Write();
}

}  // namespace s1tbx
}  // namespace alus