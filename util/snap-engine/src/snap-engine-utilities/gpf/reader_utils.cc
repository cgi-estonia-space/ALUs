/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.ReaderUtils.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
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
#include "snap-engine-utilities/gpf/reader_utils.h"

#include <algorithm>
#include <cctype>
#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/product.h"
#include "snap-core/datamodel/product_data_utc.h"
#include "snap-core/datamodel/tie_point_geo_coding.h"
#include "snap-core/datamodel/tie_point_grid.h"
#include "snap-core/datamodel/virtual_band.h"
#include "snap-core/util/guardian.h"
#include "snap-core/util/math/math_utils.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/datamodel/unit.h"
#include "snap-engine-utilities/gpf/operator_utils.h"

namespace alus {
namespace snapengine {

void ReaderUtils::AddMetadataIncidenceAngles(const std::shared_ptr<Product>& product) {
    std::shared_ptr<TiePointGrid> tpg = product->GetTiePointGrid(OperatorUtils::TPG_INCIDENT_ANGLE);
    if (tpg == nullptr) {
        return;
    }

    int mid_az = product->GetSceneRasterHeight() / 2;
    double inc1 = tpg->GetPixelDouble(0, mid_az);
    double inc2 = tpg->GetPixelDouble(product->GetSceneRasterWidth(), mid_az);

    std::shared_ptr<MetadataElement> abs_root = AbstractMetadata::GetAbstractedMetadata(product);
    AbstractMetadata::SetAttribute(abs_root, AbstractMetadata::INCIDENCE_NEAR, std::min(inc1, inc2));
    AbstractMetadata::SetAttribute(abs_root, AbstractMetadata::INCIDENCE_FAR, std::max(inc1, inc2));
}

void ReaderUtils::AddMetadataProductSize(const std::shared_ptr<Product>& product) {
    std::shared_ptr<MetadataElement> abs_root = AbstractMetadata::GetAbstractedMetadata(product);
    if (abs_root) {
        AbstractMetadata::SetAttribute(abs_root, AbstractMetadata::TOT_SIZE, ReaderUtils::GetTotalSize(product));
    }
}
// todo: this might need rethinking for port
int ReaderUtils::GetTotalSize(const std::shared_ptr<Product>& product) {
    return static_cast<int>(product->ProductNode::GetRawStorageSize() / (1024.0f * 1024.0f));
}

std::shared_ptr<Band> ReaderUtils::CreateVirtualIntensityBand(const std::shared_ptr<Product>& product,
                                                              const std::shared_ptr<Band>& band_i,
                                                              const std::shared_ptr<Band>& band_q,
                                                              std::string_view suffix) {
    return CreateVirtualIntensityBand(product, band_i, band_q, CreateName(band_i->GetName(), "Intensity"), suffix);
}

std::shared_ptr<Band> ReaderUtils::CreateVirtualIntensityBand(const std::shared_ptr<Product>& product,
                                                              const std::shared_ptr<Band>& band_i,
                                                              const std::shared_ptr<Band>& band_q,
                                                              std::string_view band_name, std::string_view suffix) {
    const std::string band_name_i(band_i->GetName());
    const double nodatavalue_i = band_i->GetNoDataValue();
    const std::string band_name_q(band_q->GetName());
    const std::string expression(band_name_i + " == " + std::to_string(nodatavalue_i) + " ? " +
                                 std::to_string(nodatavalue_i) + " : " + band_name_i + " * " + band_name_i + " + " +
                                 band_name_q + " * " + band_name_q);

    std::string name(band_name);
    if (!boost::algorithm::ends_with(name, suffix)) {
        name += suffix;
    }
    const auto virt_band = std::make_shared<VirtualBand>(name, ProductData::TYPE_FLOAT32, band_i->GetRasterWidth(),
                                                         band_i->GetRasterHeight(), expression);
    virt_band->SetUnit(Unit::INTENSITY);
    virt_band->SetDescription(std::make_optional<std::string>("Intensity from complex data"));
    virt_band->SetNoDataValueUsed(true);
    virt_band->SetNoDataValue(nodatavalue_i);
    virt_band->SetOwner(product);
    product->AddBand(virt_band);

    if (band_i->GetGeoCoding() != product->GetSceneGeoCoding()) {
        virt_band->SetGeoCoding(band_i->GetGeoCoding());
    }
    // set as band to use for quicklook
    product->SetQuicklookBandName(virt_band->GetName());

    return virt_band;
}

std::string ReaderUtils::CreateName(std::string_view orig_name, std::string_view new_prefix) {
    auto sep_pos = orig_name.find('_');
    return std::string(new_prefix) + std::string(orig_name.substr(sep_pos));
}

void ReaderUtils::AddGeoCoding(const std::shared_ptr<Product>& product, std::vector<float> lat_corners,
                               std::vector<float> lon_corners) {
    if (lat_corners.empty() || lon_corners.empty()) {
        return;
    }

    const int grid_width = 10;
    const int grid_height = 10;

    const std::vector<float> fine_lat_tie_points(grid_width * grid_height);
    ReaderUtils::CreateFineTiePointGrid(2, 2, grid_width, grid_height, lat_corners, fine_lat_tie_points);

    double sub_sampling_x = product->GetSceneRasterWidth() / (grid_width - 1);
    double sub_sampling_y = product->GetSceneRasterHeight() / (grid_height - 1);
    if (sub_sampling_x == 0 || sub_sampling_y == 0) return;

    const auto lat_grid = std::make_shared<TiePointGrid>(OperatorUtils::TPG_LATITUDE, grid_width, grid_height, 0.5f,
                                                         0.5f, sub_sampling_x, sub_sampling_y, fine_lat_tie_points);
    lat_grid->SetUnit(Unit::DEGREES);

    std::vector<float> fine_lon_tie_points(grid_width * grid_height);
    ReaderUtils::CreateFineTiePointGrid(2, 2, grid_width, grid_height, lon_corners, fine_lon_tie_points);

    const auto lon_grid = std::make_shared<TiePointGrid>(OperatorUtils::TPG_LONGITUDE, grid_width, grid_height, 0.5f,
                                                         0.5f, sub_sampling_x, sub_sampling_y, fine_lon_tie_points,
                                                         TiePointGrid::DISCONT_AT_180);
    lon_grid->SetUnit(Unit::DEGREES);

    const auto tp_geo_coding = std::make_shared<TiePointGeoCoding>(lat_grid, lon_grid);

    product->AddTiePointGrid(lat_grid);
    product->AddTiePointGrid(lon_grid);
    product->SetSceneGeoCoding(tp_geo_coding);
}

void ReaderUtils::CreateFineTiePointGrid(int coarse_grid_width, int coarse_grid_height, int fine_grid_width,
                                         int fine_grid_height, std::vector<float> coarse_tie_points,
                                         std::vector<float> fine_tie_points) {
    if (coarse_tie_points.empty() ||
        coarse_tie_points.size() != static_cast<std::size_t>(coarse_grid_width * coarse_grid_height)) {
        throw std::invalid_argument(
            "coarse tie point array size does not match 'coarseGridWidth' x 'coarseGridHeight'");
    }

    if (fine_tie_points.empty() ||
        fine_tie_points.size() != static_cast<std::size_t>(fine_grid_width * fine_grid_height)) {
        throw std::invalid_argument("fine tie point array size does not match 'fineGridWidth' x 'fineGridHeight'");
    }

    int k = 0;
    for (int r = 0; r < fine_grid_height; r++) {
        const double lambda_r = r / static_cast<double>(fine_grid_height - 1);
        const double beta_r = lambda_r * (coarse_grid_height - 1);
        const int j0 = static_cast<int>(beta_r);
        const int j1 = std::min(j0 + 1, coarse_grid_height - 1);
        const double wj = beta_r - j0;

        for (int c = 0; c < fine_grid_width; c++) {
            const double lambda_c = c / (double)(fine_grid_width - 1);
            const double beta_c = lambda_c * (coarse_grid_width - 1);
            const int i0 = static_cast<int>(beta_c);
            const int i1 = std::min(i0 + 1, coarse_grid_width - 1);
            const double wi = beta_c - i0;

            fine_tie_points.at(k++) = static_cast<float>(MathUtils::Interpolate2D(
                wi, wj, coarse_tie_points.at(i0 + j0 * coarse_grid_width),
                coarse_tie_points.at(i1 + j0 * coarse_grid_width), coarse_tie_points.at(i0 + j1 * coarse_grid_width),
                coarse_tie_points.at(i1 + j1 * coarse_grid_width)));
        }
    }
}

std::shared_ptr<Utc> ReaderUtils::GetTime(const std::shared_ptr<snapengine::MetadataElement>& elem,
                                          std::string_view tag, std::string_view time_format) {
    if (elem == nullptr) {
        return AbstractMetadata::NO_METADATA_UTC;
    }
    std::string tag_local(elem->GetAttributeString(tag, " "));
    boost::algorithm::to_upper(tag_local);
    std::string time_str = CreateValidUTCString(tag_local, std::vector<char>{':', '.', '-'}, ' ');
    boost::algorithm::trim(time_str);

    return AbstractMetadata::ParseUtc(time_str, time_format);
}

std::string ReaderUtils::CreateValidUTCString(std::string_view name, std::vector<char> valid_chars, char replace_char) {
    Guardian::AssertNotNull("name", name);
    std::vector<char> sorted_valid_chars;
    if (valid_chars.empty()) {
        sorted_valid_chars.resize(5);
    } else {
        sorted_valid_chars.resize(valid_chars.size());
        std::copy(valid_chars.begin(), valid_chars.end(), sorted_valid_chars.begin());
    }
    // todo: check if this is sorted correctly for binary search to work
    std::sort(sorted_valid_chars.begin(), sorted_valid_chars.end());

    std::stringstream valid_name;
    for (char ch : name) {
        if (std::isdigit(ch)) {
            valid_name << ch;
        } else if (std::binary_search(sorted_valid_chars.begin(), sorted_valid_chars.end(), ch)) {
            valid_name << ch;
        } else {
            valid_name << replace_char;
        }
    }
    return valid_name.str();
}

std::optional<boost::filesystem::path> ReaderUtils::GetPathFromInput(const std::any& input) {
    if (input.type() == typeid(boost::filesystem::path)) {
        return std::make_optional<boost::filesystem::path>(std::any_cast<boost::filesystem::path>(input));
    }
    if (input.type() == typeid(std::string)) {
        auto str = std::any_cast<std::string>(input);
        return std::make_optional<boost::filesystem::path>(boost::filesystem::path(str));
    }
    if (input.type() == typeid(std::string_view)) {
        auto str = std::string(std::any_cast<std::string_view>(input));
        return std::make_optional<boost::filesystem::path>(boost::filesystem::path(str));
    }
    std::cerr << "Unable to convert std::any to path. input typeid:" << input.type().name() << std::endl;
    return std::nullopt;
}

}  // namespace snapengine
}  // namespace alus
