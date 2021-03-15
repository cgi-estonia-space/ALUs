#include "abstract_product_directory.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <boost/algorithm/string.hpp>

#include "zipper/unzipper.h"
#include "zipper/zipper.h"

#include "ceres-core/i_virtual_dir.h"
#include "guardian.h"
#include "s1tbx-commons/io/band_info.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/gpf/input_product_validator.h"
#include "snap-engine-utilities/gpf/reader_utils.h"
#include "snap-engine-utilities/util/zip_utils.h"

namespace alus::s1tbx {

AbstractProductDirectory::AbstractProductDirectory(const boost::filesystem::path& input_file) {
    //    todo: check over what is wrong with assert
    //    snapengine::Guardian::AssertNotNull("inputFile", input_file);
    product_input_file_ = input_file;
    CreateProductDir(input_file);
}

void AbstractProductDirectory::CreateProductDir(const boost::filesystem::path& input_file) {
    if (snapengine::ZipUtils::IsZip(input_file)) {
        base_dir_ = input_file;
        product_dir_ = ceres::IVirtualDir::Create(base_dir_);
        base_name_ = base_dir_.filename().string();
        if (boost::algorithm::ends_with(base_name_, ".zip")) {
            base_name_ = base_name_.substr(0, base_name_.rfind(".zip"));
        }
    } else {
        if (boost::filesystem::is_directory(input_file)) {
            base_dir_ = input_file;
        } else {
            base_dir_ = input_file.parent_path();
        }
        product_dir_ = ceres::IVirtualDir::Create(base_dir_);
        base_name_ = base_dir_.filename().string();
    }
}

std::string AbstractProductDirectory::GetRootFolder() {
    if (root_folder_.empty()) {
        root_folder_ = FindRootFolder();
    }
    return root_folder_;
}

std::string AbstractProductDirectory::FindRootFolder() {
    std::string root_folder;
    try {
        if (product_dir_ != nullptr && product_dir_->IsCompressed()) {
            root_folder = snapengine::ZipUtils::GetRootFolder(base_dir_, GetHeaderFileName());
        }
    } catch (const std::exception& e) {
        std::cerr << "Unable to get root path from zip file " << e.what() << std::endl;
    }
    return root_folder;
}

std::string AbstractProductDirectory::GetRelativePathToImageFolder() { return GetRootFolder(); }

void AbstractProductDirectory::FindImages(std::string_view parent_path,
                                          const std::shared_ptr<snapengine::MetadataElement>& new_root) {
    std::vector<std::string> listing;
    try {
        listing = GetProductDir()->List(parent_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        // just prevent exception to be thrown and continue
    }
    if (!listing.empty()) {
        std::sort(listing.begin(), listing.end());
        for (auto const& file_name : listing) {
            AddImageFile(std::string(parent_path) + file_name, new_root);
        }
    }
}

std::string AbstractProductDirectory::GetBandFileNameFromImage(std::string_view img_path) {
    if (img_path.find("/") != std::string::npos) {
        std::string output(img_path.substr(img_path.find_last_of('/') + 1));
        boost::algorithm::to_lower(output);
        return output;
    }
    return std::string(img_path);
}

bool AbstractProductDirectory::IsCompressed() { return product_dir_->IsCompressed(); }
std::string AbstractProductDirectory::GetHeaderFileName() { return product_input_file_.filename().string(); }
std::shared_ptr<snapengine::custom::Dimension> AbstractProductDirectory::GetBandDimensions(
    const std::shared_ptr<snapengine::MetadataElement>& new_root, std::string_view band_metadata_name) {
    std::shared_ptr<snapengine::MetadataElement> abs_root =
        new_root->GetElement(snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);
    std::shared_ptr<snapengine::MetadataElement> band_metadata = abs_root->GetElement(band_metadata_name);
    int width;
    int height;
    if (band_metadata) {
        width = band_metadata->GetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE);
        height = band_metadata->GetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES);
    } else {
        width = abs_root->GetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE);
        height = abs_root->GetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES);
    }
    return std::make_shared<snapengine::custom::Dimension>(width, height);
}
void AbstractProductDirectory::FindImages(const std::shared_ptr<snapengine::MetadataElement>& new_root) {
    std::string parent_path = GetRelativePathToImageFolder();
    FindImages(parent_path, new_root);
}
std::vector<std::string> AbstractProductDirectory::ListFiles(std::string_view path) {
    try {
        std::vector<std::string> listing = product_dir_->List(path);
        std::sort(listing.begin(), listing.end());
        std::vector<std::string> files;
        for (auto const& list_entry : listing) {
            std::string entry_path = list_entry;
            if (!path.empty()) {
                entry_path = std::string(path) + "/" + list_entry;
            }
            //            todo: this condition should exclude calibration directory!
            if (!IsDirectory(entry_path)) {
                files.emplace_back(list_entry);
            }
        }
        return files;
    } catch (const std::exception& e) {
        throw std::runtime_error("Product is corrupt or incomplete\n" + std::string(e.what()));
    }
}

std::vector<std::string> AbstractProductDirectory::FindFilesContaining(std::string_view path,
                                                                       std::string_view search_string) {
    std::vector<std::string> list;
    try {
        std::vector<std::string> files = ListFiles(path);
        for (auto const& file : files) {
            if (file.find(search_string) != std::string::npos) {
                list.emplace_back(file);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error listing files in " << path << std::endl;
    }
    return list;
}
bool AbstractProductDirectory::IsDirectory(std::string_view path) {
    if (product_dir_->IsCompressed()) {
        if (path.find(".") != std::string::npos) {
            int sep_index = path.find_last_of('/');
            int dot_index = path.find_last_of('.');
            return dot_index < sep_index;
        }
        zipper::Unzipper unzipper(base_dir_.filename().string());
        std::vector<zipper::ZipEntry> entries = unzipper.entries();
        auto entry = std::find_if(entries.begin(), entries.end(), [path](const zipper::ZipEntry& entry) {
            return boost::ends_with(entry.name, "/") && entry.name == path;
        });
        return entry != entries.end();
    }
    return boost::filesystem::is_directory(product_dir_->GetFile(path));
}
boost::filesystem::path AbstractProductDirectory::GetFile(std::string_view path) {
    return GetProductDir()->GetFile(path);
}
bool AbstractProductDirectory::Exists(std::string_view path) { return GetProductDir()->Exists(path); }

void AbstractProductDirectory::GetInputStream(std::string_view path, std::fstream& stream) {
    GetProductDir()->GetInputStream(path, stream);
    if (!stream.good()) {
        throw std::runtime_error("Product is corrupt or incomplete: unreadable " + std::string(path));
    }
}
std::shared_ptr<snapengine::Product> AbstractProductDirectory::CreateProduct() {
    std::shared_ptr<snapengine::MetadataElement> new_root = AddMetaData();
    FindImages(new_root);
    std::shared_ptr<snapengine::custom::Dimension> dim = GetProductDimensions(new_root);

    //    todo: remove if static CreateProduct works like expected
    //    std::shared_ptr<snapengine::Product> product =
    //        std::make_shared<snapengine::Product>(GetProductName(), GetProductType(), dim->width, dim->height);
    std::shared_ptr<snapengine::Product> product =
        snapengine::Product::CreateProduct(GetProductName(), GetProductType(), dim->width, dim->height);
    UpdateProduct(product, new_root);

    AddBands(product);

    //    todo: add support only if used
    //    is_map_projected_ = snapengine::InputProductValidator::IsMapProjected(product);

    AddGeoCoding(product);
    AddTiePointGrids(product);

    product->SetName(GetProductName());
    product->SetProductType(GetProductType());
    product->SetDescription(std::make_optional<std::string>(GetProductDescription()));

    snapengine::ReaderUtils::AddMetadataIncidenceAngles(product);
    snapengine::ReaderUtils::AddMetadataProductSize(product);

    return product;
}
std::shared_ptr<snapengine::custom::Dimension> AbstractProductDirectory::GetProductDimensions(
    const std::shared_ptr<snapengine::MetadataElement>& new_root) {
    std::shared_ptr<snapengine::MetadataElement> abs_root =
        new_root->GetElement(snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);
    int scene_width = abs_root->GetAttributeInt(snapengine::AbstractMetadata::NUM_SAMPLES_PER_LINE);
    int scene_height = abs_root->GetAttributeInt(snapengine::AbstractMetadata::NUM_OUTPUT_LINES);
    return std::make_shared<snapengine::custom::Dimension>(scene_width, scene_height);
}

void AbstractProductDirectory::UpdateProduct(const std::shared_ptr<snapengine::Product>& product,
                                             const std::shared_ptr<snapengine::MetadataElement>& new_root) {
    std::shared_ptr<snapengine::MetadataElement> root = product->GetMetadataRoot();
    for (auto const& elem : new_root->GetElements()) {
        root->AddElement(elem);
    }

    std::shared_ptr<snapengine::MetadataElement> abs_root =
        snapengine::AbstractMetadata::GetAbstractedMetadata(product);

    product->SetStartTime(abs_root->GetAttributeUtc(snapengine::AbstractMetadata::FIRST_LINE_TIME));
    product->SetEndTime(abs_root->GetAttributeUtc(snapengine::AbstractMetadata::LAST_LINE_TIME));

    product->SetProductType(abs_root->GetAttributeString(snapengine::AbstractMetadata::PRODUCT_TYPE));
    product->SetDescription(
        std::make_optional<std::string>(abs_root->GetAttributeString(snapengine::AbstractMetadata::SPH_DESCRIPTOR)));
}

std::shared_ptr<BandInfo> AbstractProductDirectory::GetBandInfo(const std::shared_ptr<snapengine::Band>& dest_band) {
    auto band_info = band_map_.at(dest_band);
    if (band_info == nullptr) {
        for (const auto& src_band : band_map_) {
            if (src_band.first->GetName() == dest_band->GetName()) {
                band_info = band_map_.at(src_band.first);
            }
        }
    }
    return band_info;
}

void AbstractProductDirectory::Close() {
    std::unordered_set<std::string> keys;
    for (const auto& kv : band_image_file_map_) {
        band_image_file_map_.at(kv.first)->Close();
    }
    product_dir_->Close();
}

}  // namespace alus::s1tbx
