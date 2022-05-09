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

#include "sentinel2_dataset.h"

#include <gdal_priv.h>

#include "alus_log.h"
#include "gdal_util.h"
#include "sentinel2_constants.h"
#include "sentinel2_tools.h"

namespace alus::resample {

Sentinel2Dataset::Sentinel2Dataset(std::string_view dataset_path) {
    auto* in_ds = GDALDataset::Open(dataset_path.data(), GA_ReadOnly);
    CHECK_GDAL_PTR(in_ds);

    auto** subdatasets = in_ds->GetMetadata(gdal::constants::SUBDATASET_KEY.data());
    auto subdataset_list = GetSubdatasetList(subdatasets, static_cast<size_t>(CSLCount(subdatasets)));
    const auto* product_uri = in_ds->GetMetadataItem(sentinel2::PRODUCT_URI_KEY.data());
    granule_img_filename_stem_ = GetGranuleImgFilenameStem(product_uri);
    GDALClose(in_ds);

    if (subdataset_list.size() != sentinel2::SUBDATASET_COUNT) {
        LOGW << "Sentinel 2 archives usually have " << sentinel2::SUBDATASET_COUNT << " subdatasets, this one has "
             << subdataset_list.size();
    }

    RemoveTrueColorImageDatasetFrom(subdataset_list);
    FetchBands(subdataset_list);
}

std::vector<std::string> Sentinel2Dataset::GetSubdatasetList(char** entries, size_t count) {
    constexpr size_t NAME_IDENTIFIER_NAME_LENGTH_SKIP{6};
    std::vector<std::string> subdataset_list;
    for (size_t i = 0; i < count; i++) {
        const std::string_view subdataset_info(entries[i]);
        /* SUBDATASET_1_NAME=SENTINEL2_L1C:/home/sven/Downloads/
           S2B_MSIL1C_20211102T093049_N0301_R136_T35VNE_20211102T114211.SAFE/MTD_MSIL1C.xml:10m:EPSG_32635
           SUBDATASET_1_DESC=Bands B2, B3, B4, B8 with 10m resolution, UTM 35N
           SUBDATASET_2_NAME=......
         */
        const auto pos = subdataset_info.find(sentinel2::SUBDATASET_NAME_IDENTIFIER);
        if (pos != std::string::npos) {
            /* SENTINEL2_L1C:/foo/
               S2B_MSIL1C_20211102T093049_N0301_R136_T35VNE_20211102T114211.SAFE/MTD_MSIL1C.xml:10m:EPSG_32635
             */
            subdataset_list.emplace_back(
                subdataset_info.substr(pos + NAME_IDENTIFIER_NAME_LENGTH_SKIP, subdataset_info.length()));
        }
    }

    return subdataset_list;
}

void Sentinel2Dataset::FetchBands(const std::vector<std::string>& subdataset_list) {
    for (const auto& sd : subdataset_list) {
        datasets_.emplace_back(sd);
        auto* gdal_ds = datasets_.back().GetGdalDataset();
        const auto band_count = gdal_ds->GetBands().size();
        for (size_t band_index = 0; band_index < band_count; band_index++) {
            auto** metadata = gdal_ds->GetBands()[band_index]->GetMetadata();
            const auto& band_metadata = ParseBandMetadata(metadata, CSLCount(metadata));
            BandHandle handle{datasets_.size() - 1, band_index + 1, band_metadata};
            band_table_.try_emplace(band_metadata.at(sentinel2::BANDNAME_METADATA_KEY.data()), handle);
        }
    }
}

void Sentinel2Dataset::RemoveTrueColorImageDatasetFrom(std::vector<std::string>& subdataset_list) {
    for (auto it = subdataset_list.begin(); it < subdataset_list.end(); it++) {
        if (it->find(sentinel2::TRUE_COLOR_IMAGE_IDENTIFIER) == std::string::npos) {
            continue;
        }

        subdataset_list.erase(it);
        break;
    }
}

const std::vector<uint16_t>& Sentinel2Dataset::GetBandData(std::string_view band_id) {
    const auto& band_entry = band_table_.at(std::string(band_id));
    auto& ds = datasets_.at(band_entry.dataset_index);
    ds.LoadRasterBand(band_entry.band_index_in_dataset);
    return ds.GetHostDataBuffer();
}

std::string Sentinel2Dataset::GetGranuleImgFilenameStem(const char* metadata_value) {
    const auto fields = TryParseS2Fields(metadata_value);
    std::string name_stem = fields.tile;
    return name_stem.append("_").append(fields.sensing_start).append("_");
}

size_t Sentinel2Dataset::GetBandNoInDataset(size_t index) {
    const auto band_id = sentinel2::BANDS.at(index);
    const auto& band_entry = band_table_.at(std::string(band_id));
    return band_entry.band_index_in_dataset;
}

}  // namespace alus::resample