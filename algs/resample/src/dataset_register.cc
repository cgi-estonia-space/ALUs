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

#include "dataset_register.h"

#include <array>
#include <filesystem>
#include <string_view>

#include "alus_log.h"
#include "gdal_util.h"
#include "sentinel2_tools.h"

namespace alus::resample {

DatasetRegister::DatasetRegister(std::string_view dataset_path) : input_path_{dataset_path} {
    try {
        const auto& gdal_input = TryCreateSentinel2GdalDatasetName(dataset_path);
        is_sentinel2_dataset_ = true;
        s2_ds_ = std::make_unique<Sentinel2Dataset>(gdal_input);
    } catch (const Sentinel2DatasetNameParseException&) {
        LOGD << dataset_path << " - not a Sentinel 2 dataset, will be opened as a generic one";
        generic_ds_ = static_cast<GDALDataset*>(GDALOpen(dataset_path.data(), GA_ReadOnly));
        CHECK_GDAL_PTR(generic_ds_);
    }
}

RasterDimension DatasetRegister::GetBandDimension(size_t band_index) {
    if (is_sentinel2_dataset_) {
        return s2_ds_->GetBandDimension(band_index);
    }

    const auto band_no = band_index + gdal::constants::GDAL_DEFAULT_RASTER_BAND;
    return {static_cast<int>(generic_ds_->GetRasterBand(band_no)->GetXSize()),
            static_cast<int>(generic_ds_->GetRasterBand(band_no)->GetYSize())};
}

size_t DatasetRegister::GetBandCount() {
    return is_sentinel2_dataset_ ? s2_ds_->GetBandCount() : static_cast<size_t>(generic_ds_->GetRasterCount());
}

std::string DatasetRegister::GetBandDescription(size_t band_index) const {
    if (is_sentinel2_dataset_) {
        return std::string(sentinel2::BANDS.at(band_index));
    }

    return generic_ds_->GetRasterBand(static_cast<int>(band_index) + gdal::constants::GDAL_DEFAULT_RASTER_BAND)
        ->GetDescription();
}

std::unique_ptr<uint8_t[]> DatasetRegister::GetBandData(size_t band_index, size_t& buffer_size_bytes) {
    GDALRasterBand* band{};
    if (is_sentinel2_dataset_) {
        band = s2_ds_->GetDataset(band_index).GetGdalDataset()->GetRasterBand(s2_ds_->GetBandNoInDataset(band_index));
    } else {
        band = generic_ds_->GetRasterBand(static_cast<int>(band_index) + gdal::constants::GDAL_DEFAULT_RASTER_BAND);
    }

    const auto gdal_datatype = band->GetRasterDataType();
    const auto datatype_size = GDALGetDataTypeSizeBytes(gdal_datatype);
    const auto band_dim = GetBandDimension(band_index);
    buffer_size_bytes = band_dim.columnsX * band_dim.rowsY * datatype_size;
    auto buffer = std::unique_ptr<uint8_t[]>(new uint8_t[buffer_size_bytes]);
    CHECK_GDAL_ERROR(band->RasterIO(GF_Read, 0, 0, band_dim.columnsX, band_dim.rowsY, buffer.get(), band_dim.columnsX,
                                    band_dim.rowsY, gdal_datatype, 0, 0, nullptr));

    return buffer;
}

size_t DatasetRegister::GetRasterDataTypeSize(size_t band_index) {
    return GDALGetDataTypeSizeBytes(GetDataType(band_index));
}

GeoTransformParameters DatasetRegister::GetGeoTransform(size_t band_index) {
    GDALDataset* gdal_ds{};
    if (is_sentinel2_dataset_) {
        gdal_ds = s2_ds_->GetDataset(band_index).GetGdalDataset();
    } else {
        gdal_ds = generic_ds_;
    }

    std::array<double, gdal::constants::GDAL_GEOTRANSFORM_PARAMETER_COUNT> gt;
    gdal_ds->GetGeoTransform(gt.data());
    return GeoTransformConstruct::BuildFromGdal(gt.data());
}

OGRSpatialReference DatasetRegister::GetSrs() const {
    return is_sentinel2_dataset_ ? *s2_ds_->GetDataset(0).GetGdalDataset()->GetSpatialRef()
                                 : *generic_ds_->GetSpatialRef();
}

std::string DatasetRegister::GetGranuleImageFilenameStemFor(size_t band_index) const {
    if (is_sentinel2_dataset_) {
        return s2_ds_->GetGranuleImgFilenameStemFor(band_index);
    }

    return std::filesystem::path(input_path_).stem().string() + "_B" +
           std::to_string(band_index + gdal::constants::GDAL_DEFAULT_RASTER_BAND);
}

GDALDataType DatasetRegister::GetDataType(size_t band_index) {
    if (is_sentinel2_dataset_) {
        return s2_ds_->GetDataset(band_index)
            .GetGdalDataset()
            ->GetRasterBand(s2_ds_->GetBandNoInDataset(band_index))
            ->GetRasterDataType();
    }

    return generic_ds_->GetRasterBand(static_cast<int>(band_index) + gdal::constants::GDAL_DEFAULT_RASTER_BAND)
        ->GetRasterDataType();
}

GDALDriver* DatasetRegister::GetGdalDriver() {
    return is_sentinel2_dataset_ ? s2_ds_->GetDataset(0).GetGdalDataset()->GetDriver() : generic_ds_->GetDriver();
}

std::vector<std::pair<std::string, std::pair<std::string, std::string>>> DatasetRegister::GetBandMetadata(
    size_t band_index) {
    GDALRasterBand* band{};
    if (is_sentinel2_dataset_) {
        band = s2_ds_->GetDataset(band_index).GetGdalDataset()->GetRasterBand(s2_ds_->GetBandNoInDataset(band_index));
    } else {
        band = generic_ds_->GetRasterBand(static_cast<int>(band_index) + gdal::constants::GDAL_DEFAULT_RASTER_BAND);
    }

    std::vector<std::pair<std::string, std::pair<std::string, std::string>>> metadata;

    auto release_domain = [](char** domain){ CSLDestroy(domain); };
    auto domains = std::unique_ptr<char*, decltype(release_domain)>(band->GetMetadataDomainList(), release_domain);
    const auto domain_count = CSLCount(domains.get());
    for (int i{}; i < domain_count; i++) {
        std::string domain(domains.get()[i]);
        auto* band_md = band->GetMetadata(domain.c_str());
        const auto md_items = CSLCount(band_md);
        for (int j{}; j < md_items; j++) {
            std::string md_item_raw(band_md[j]);
            const auto pos = md_item_raw.find('=');
            if (pos != std::string::npos) {
                metadata.emplace_back(domain, std::make_pair(md_item_raw.substr(0, pos), md_item_raw.substr(pos + 1)));
            } else {
                metadata.emplace_back(domain, std::make_pair(md_item_raw, ""));
            }
        }
    }

    return metadata;
}

bool DatasetRegister::GetBandNoDataValue(size_t band_index, double& no_data_value) {
    GDALRasterBand* band{};
    if (is_sentinel2_dataset_) {
        band = s2_ds_->GetDataset(band_index).GetGdalDataset()->GetRasterBand(s2_ds_->GetBandNoInDataset(band_index));
    } else {
        band = generic_ds_->GetRasterBand(static_cast<int>(band_index) + gdal::constants::GDAL_DEFAULT_RASTER_BAND);
    }

    int has_no_data{};
    no_data_value = band->GetNoDataValue(&has_no_data);

    return has_no_data == 1;
}

DatasetRegister::~DatasetRegister() {
    if (generic_ds_ != nullptr) {
        GDALClose(generic_ds_);
    }
}

}  // namespace alus::resample