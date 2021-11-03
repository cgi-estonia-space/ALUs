/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.io.AbstractProductDirectory.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
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

#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include <boost/filesystem/path.hpp>

#include "ceres-core/core/i_virtual_dir.h"
#include "s1tbx-commons/io/band_info.h"
#include "s1tbx-commons/io/image_i_o_file.h"
#include "s1tbx-io/sentinel1/i_sentinel1_directory.h"

namespace alus::snapengine {
namespace custom {
struct Dimension;
}  // namespace custom
class MetadataElement;
class Band;
class Product;
}  // namespace alus::snapengine

namespace alus::s1tbx {

class BandInfo;

/**
 * This class represents a product directory.
 */
class AbstractProductDirectory : public ISentinel1Directory {
public:
    std::string GetRootFolder();
    virtual void ReadProductDirectory() override = 0;
    [[nodiscard]] bool IsSLC() override { return is_s_l_c_; }
    [[nodiscard]] bool IsMapProjected() const { return is_map_projected_; }
    std::shared_ptr<BandInfo> GetBandInfo(const std::shared_ptr<snapengine::Band>& dest_band) override;
    void Close() override;
    std::vector<std::string> ListFiles(std::string_view path);
    std::vector<std::string> FindFilesContaining(std::string_view path, std::string_view search_string);
    virtual std::shared_ptr<snapengine::Product> CreateProduct() override;
    boost::filesystem::path GetFile(std::string_view path);
    bool Exists(std::string_view path);

protected:
    std::shared_ptr<ceres::IVirtualDir> product_dir_ = nullptr;
    std::string base_name_;
    boost::filesystem::path base_dir_;
    std::string root_folder_;
    boost::filesystem::path product_input_file_;
    std::unordered_map<std::string, std::shared_ptr<ImageIOFile>> band_image_file_map_;
    std::unordered_map<std::shared_ptr<snapengine::Band>, std::shared_ptr<BandInfo>> band_map_;

    explicit AbstractProductDirectory(const boost::filesystem::path& input_file);

    void CreateProductDir(const boost::filesystem::path& input_file);

    std::string FindRootFolder();

    std::shared_ptr<ceres::IVirtualDir> GetProductDir() { return product_dir_; }

    virtual std::string GetRelativePathToImageFolder();

    virtual std::string GetHeaderFileName();

    virtual void AddImageFile(std::string_view img_path,
                              const std::shared_ptr<snapengine::MetadataElement>& new_root) = 0;
    bool IsCompressed();

    void FindImages(std::string_view parent_path, const std::shared_ptr<snapengine::MetadataElement>& new_root);

    std::string GetBandFileNameFromImage(std::string_view img_path);

    std::shared_ptr<snapengine::custom::Dimension> GetBandDimensions(
        const std::shared_ptr<snapengine::MetadataElement>& new_root, std::string_view band_metadata_name);

    void FindImages(const std::shared_ptr<snapengine::MetadataElement>& new_root);

    virtual void AddBands(const std::shared_ptr<snapengine::Product>& product) = 0;

    virtual void AddGeoCoding(const std::shared_ptr<snapengine::Product>& product) = 0;

    virtual void AddTiePointGrids(const std::shared_ptr<snapengine::Product>& product) = 0;

    virtual void AddAbstractedMetadataHeader(const std::shared_ptr<snapengine::MetadataElement>& root) = 0;

    virtual std::string GetProductName() = 0;

    virtual std::string GetProductType() = 0;

    std::string GetProductDescription() { return ""; }

    virtual std::shared_ptr<snapengine::MetadataElement> AddMetaData() = 0;

    boost::filesystem::path GetBaseDir() { return base_dir_; }

    std::string GetBaseName() { return base_name_; }

    std::shared_ptr<snapengine::custom::Dimension> GetProductDimensions(
        const std::shared_ptr<snapengine::MetadataElement>& new_root);

    static void UpdateProduct(const std::shared_ptr<snapengine::Product>& product,
                              const std::shared_ptr<snapengine::MetadataElement>& new_root);

    void GetInputStream(std::string_view path, std::fstream& stream);

    void SetSLC(bool flag) { is_s_l_c_ = flag; }

private:
    bool is_s_l_c_ = false;
    bool is_map_projected_;

    bool IsDirectory(std::string_view path);
};

}  // namespace alus::s1tbx
