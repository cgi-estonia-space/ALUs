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
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include <boost/filesystem.hpp>

#include "s1tbx-commons/io/x_m_l_product_directory.h"

namespace alus::snapengine {
class Band;
class TiePointGeoCoding;
class Product;
class MetadataElement;
}  // namespace alus::snapengine

namespace alus::s1tbx {
/**
 * This class represents a product directory.
 */
class Sentinel1Level1Directory : public XMLProductDirectory {
private:
    std::unordered_map<std::shared_ptr<snapengine::Band>, std::shared_ptr<snapengine::TiePointGeoCoding>>
        band_geocoding_map_;
    std::unordered_map<std::string, std::string> img_band_metadata_map_;
    std::string acq_mode_ = "";
    static constexpr double NO_DATA_VALUE = 0.0;  //-9999.0;

    static std::shared_ptr<snapengine::MetadataElement> FindElement(
        const std::shared_ptr<snapengine::MetadataElement>& elem, std::string_view name);
    static std::shared_ptr<snapengine::MetadataElement> FindElementContaining(
        const std::shared_ptr<snapengine::MetadataElement>& parent, std::string_view elem_name,
        std::string_view attrib_name, std::string_view att_value);
    static void SetLatLongMetadata(const std::shared_ptr<snapengine::Product>& product,
                                   const std::shared_ptr<snapengine::TiePointGrid>& lat_grid,
                                   const std::shared_ptr<snapengine::TiePointGrid>& lon_grid);

    bool IsTOPSAR() { return acq_mode_ == "IW" || acq_mode_ == "EW"; }
    void DetermineProductDimensions(const std::shared_ptr<snapengine::MetadataElement>& abs_root);
    void AddTiePointGrids(const std::shared_ptr<snapengine::Product>& product,
                          const std::shared_ptr<snapengine::Band>& band, std::string_view img_x_m_l_name,
                          std::string_view tpg_prefix);
    //    todo: check if this works like in snap...
    void AddProductInfoJSON(const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root);
    void AddBandAbstractedMetadata(const std::shared_ptr<snapengine::MetadataElement>& abs_root,
                                   const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root);
    static void AddOrbitStateVectors(const std::shared_ptr<snapengine::MetadataElement>& abs_root,
                                     const std::shared_ptr<snapengine::MetadataElement>& orbit_list);
    static void AddSRGRCoefficients(const std::shared_ptr<snapengine::MetadataElement>& abs_root,
                                    const std::shared_ptr<snapengine::MetadataElement>& coordinate_conversion);
    static void AddDopplerCentroidCoefficients(const std::shared_ptr<snapengine::MetadataElement>& abs_root,
                                               const std::shared_ptr<snapengine::MetadataElement>& doppler_centroid);
    static void AddVector(std::string_view name,
                          const std::shared_ptr<snapengine::MetadataElement>& orbit_vector_list_elem,
                          const std::shared_ptr<snapengine::MetadataElement>& orbit_elem, std::size_t num);
    static double GetBandTerrainHeight(const std::shared_ptr<snapengine::MetadataElement>& prod_elem);
    void AddCalibrationAbstractedMetadata(const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root);
    void AddNoiseAbstractedMetadata(const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root);

protected:
    static void AddManifestMetadata(std::string_view product_name,
                                    const std::shared_ptr<snapengine::MetadataElement>& abs_root,
                                    const std::shared_ptr<snapengine::MetadataElement>& orig_prod_root, bool is_o_c_n);
    std::string GetHeaderFileName() override;
    std::string GetRelativePathToImageFolder() override;
    void AddImageFile(std::string_view img_path, const std::shared_ptr<snapengine::MetadataElement>& new_root) override;
    void AddBands(const std::shared_ptr<snapengine::Product>& product) override;
    void AddAbstractedMetadataHeader(const std::shared_ptr<snapengine::MetadataElement>& root) override;
    void AddGeoCoding(const std::shared_ptr<snapengine::Product>& product) override;
    void AddTiePointGrids([[maybe_unused]] const std::shared_ptr<snapengine::Product>& product) override;
    std::string GetProductType() override;

public:
    static std::shared_ptr<snapengine::Utc> GetTime(const std::shared_ptr<snapengine::MetadataElement>& elem,
                                                    std::string_view tag, std::string_view sentinel_date_format);
    static void GetListInEvenlySpacedGrid(int scene_raster_width, int scene_raster_height, int source_grid_width,
                                          int source_grid_height, const std::vector<int>& x, const std::vector<int>& y,
                                          const std::vector<double>& source_point_list, int target_grid_width,
                                          int target_grid_height, double sub_sampling_x, double sub_sampling_y,
                                          std::vector<float>& target_point_list);

    explicit Sentinel1Level1Directory(const boost::filesystem::path& input_file);

    std::string GetProductName() override;
    std::shared_ptr<snapengine::Product> CreateProduct() override;
};

}  // namespace alus::s1tbx
