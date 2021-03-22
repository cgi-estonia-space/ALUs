/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.io.SARReader.java
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
#include "s1tbx-commons/io/sar_reader.h"

#include <algorithm>
#include <utility>

#include <boost/algorithm/string.hpp>

#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/geo_pos.h"
#include "snap-core/datamodel/pixel_pos.h"
#include "snap-core/datamodel/product.h"
#include "snap-core/datamodel/product_data.h"
#include "snap-core/datamodel/product_node_group.h"
#include "snap-core/datamodel/quicklooks/quicklook.h"
#include "snap-core/datamodel/virtual_band.h"
#include "snap-core/util/geo_utils.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/datamodel/unit.h"
#include "snap-engine-utilities/gpf/reader_utils.h"

namespace alus::s1tbx {

void SARReader::CreateVirtualIntensityBand(const std::shared_ptr<snapengine::Product>& product,
                                           const std::shared_ptr<snapengine::Band>& band, std::string_view count_str) {
    std::string band_name = band->GetName();
    double nodatavalue = band->GetNoDataValue();
    std::string expression = band_name + " == " + std::to_string(nodatavalue) + " ? " + std::to_string(nodatavalue) +
                             " : " + band_name + " * " + band_name;

    std::shared_ptr<snapengine::VirtualBand> virt_band = std::make_shared<snapengine::VirtualBand>(
        "Intensity" + std::string(count_str), snapengine::ProductData::TYPE_FLOAT32, band->GetRasterWidth(),
        band->GetRasterHeight(), expression);
    virt_band->SetUnit(snapengine::Unit::INTENSITY);
    virt_band->SetDescription(std::make_optional<std::string>("Intensity from complex data"));
    virt_band->SetNoDataValueUsed(true);
    virt_band->SetNoDataValue(nodatavalue);

    if (band->GetGeoCoding() != product->GetSceneGeoCoding()) {
        virt_band->SetGeoCoding(band->GetGeoCoding());
    }
    product->AddBand(virt_band);
}

std::string SARReader::FindPolarizationInBandName(std::string_view band_name) {
    std::string id = std::string(band_name);
    std::transform(id.begin(), id.end(), id.begin(), ::toupper);
    if ((id.find("HH") != std::string::npos) || (id.find("H/H") != std::string::npos) ||
        (id.find("H-H") != std::string::npos)) {
        return "HH";
    }
    if ((id.find("VV") != std::string::npos) || (id.find("V/V") != std::string::npos) ||
        (id.find("V-V") != std::string::npos)) {
        return "VV";
    }
    if ((id.find("HV") != std::string::npos) || (id.find("H/V") != std::string::npos) ||
        (id.find("H-V") != std::string::npos)) {
        return "HV";
    }
    if ((id.find("VH") != std::string::npos) || (id.find("V/H") != std::string::npos) ||
        (id.find("V-H") != std::string::npos)) {
        return "VH";
    }
    return "";
}

void SARReader::DiscardUnusedMetadata(std::shared_ptr<snapengine::Product> product) {
    //    todo::use config file solution to be made
    //    std::string dicard_unused_metadata = RuntimeContext.getModuleContext().getRuntimeConfig().
    //        GetContextProperty("discard.unused.metadata");
    std::string dicard_unused_metadata{"true"};
    if (boost::iequals(dicard_unused_metadata, "true")) {
        RemoveUnusedMetadata(snapengine::AbstractMetadata::GetOriginalProductMetadata(std::move(product)));
    }
}

void SARReader::HandleReaderException(const std::exception& e) {
    std::string message = std::string(typeid(*this).name()) + ":\n";
    boost::replace_all(message, "[input", "\n[input");
    if (e.what()) {
        message += e.what();
    }
    // todo: replace if we get logging system decided
    std::cerr << message << std::endl;
    throw std::runtime_error(message);
}

bool SARReader::CheckIfCrossMeridian(std::vector<float> longitude_list) {
    std::sort(longitude_list.begin(), longitude_list.end());
    return (longitude_list.at(longitude_list.size() - 1) - longitude_list.at(0) > 270.0f);
}

boost::filesystem::path SARReader::GetPathFromInput(const std::any& input) {
    //    todo::implement readerutils to finish this if needed
    std::optional<boost::filesystem::path> path = snapengine::ReaderUtils::GetPathFromInput(input);
    if (path->empty()) {
        throw std::runtime_error("Unable to get path from provided input");
    }
    return path.value();
}

void SARReader::AddCommonSARMetadata(const std::shared_ptr<snapengine::Product>& product) {
    if (product->GetSceneGeoCoding() == nullptr) {
        return;
    }
    //    todo: check if this works like expected
    std::shared_ptr<snapengine::GeoPos> empty_nullptr;
    std::shared_ptr<snapengine::GeoPos> geo_pos = product->GetSceneGeoCoding()->GetGeoPos(
        std::make_shared<snapengine::PixelPos>(product->GetSceneRasterWidth() / 2, product->GetSceneRasterHeight() / 2),
        empty_nullptr);

    std::shared_ptr<snapengine::GeoPos> geo_pos2 = product->GetSceneGeoCoding()->GetGeoPos(
        std::make_shared<snapengine::PixelPos>(product->GetSceneRasterWidth() / 2,
                                               (product->GetSceneRasterHeight() / 2) + 100),
        empty_nullptr);
    std::shared_ptr<snapengine::DistanceHeading> heading = snapengine::GeoUtils::VincentyInverse(geo_pos, geo_pos2);

    std::shared_ptr<snapengine::MetadataElement> abs_root =
        snapengine::AbstractMetadata::GetAbstractedMetadata(product);
    snapengine::AbstractMetadata::SetAttribute(abs_root, "centre_lat", geo_pos->GetLat());
    snapengine::AbstractMetadata::SetAttribute(abs_root, "centre_lon", geo_pos->GetLon());
    snapengine::AbstractMetadata::SetAttribute(abs_root, "centre_heading", heading->heading1);
    snapengine::AbstractMetadata::SetAttribute(abs_root, "centre_heading2", heading->heading2);
}

void SARReader::RemoveUnusedMetadata(const std::shared_ptr<snapengine::MetadataElement>& root) {
    std::vector<std::shared_ptr<snapengine::MetadataElement>> elems = root->GetElements();
    for (const std::shared_ptr<snapengine::MetadataElement>& elem : elems) {
        std::string name = elem->GetName();
        bool keep = false;
        for (auto to_keep : ELEMS_TO_KEEP) {
            if (name == std::string(to_keep)) {
                keep = true;
                break;
            }
        }
        if (!keep) {
            root->RemoveElement(elem);
            elem->Dispose();
        }
    }
}

void SARReader::SetQuicklookBandName(const std::shared_ptr<snapengine::Product>& product) {
    std::vector<std::shared_ptr<snapengine::Band>> bands = product->GetBands();
    for (const auto& band : bands) {
        auto unit = band->GetUnit();
        if (unit.has_value() &&
            ((unit->find("intensity") != std::string::npos) || (unit->find("amplitude") != std::string::npos))) {
            product->SetQuicklookBandName(band->GetName());
            return;
        }
    }
    // if not intensity bands found find first amplitude
    for (const auto& band : bands) {
        auto unit = band->GetUnit();
        if (unit.has_value() && (unit->find("amplitude") != std::string::npos)) {
            product->SetQuicklookBandName(band->GetName());
            return;
        }
    }
}

void SARReader::AddQuicklook(const std::shared_ptr<snapengine::Product>& product, std::string_view name,
                             const boost::filesystem::path& ql_file) {
    if (boost::filesystem::is_empty(ql_file)) {
        product->GetQuicklookGroup()->Add(std::make_shared<snapengine::Quicklook>(product, name, ql_file));
    }
}
SARReader::SARReader(const std::shared_ptr<snapengine::IProductReaderPlugIn>& reader_plug_in)
    : AbstractProductReader(reader_plug_in) {}

}  // namespace alus::s1tbx
