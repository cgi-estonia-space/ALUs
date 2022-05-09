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

#include "execute.h"

#include <algorithm>
#include <filesystem>
#include <list>
#include <string>

#include "../../../VERSION"
#include "algorithm_exception.h"
#include "alus_log.h"
#include "cuda_device_init.h"
#include "dataset_register.h"
#include "gdal_management.h"
#include "gdal_util.h"
#include "nppi_resample.h"
#include "output_factory.h"
#include "projection.h"
#include "tyler_the_creator.h"
#include "type_parameter.h"

namespace alus::resample {

Execute::Execute(Parameters params) : params_{std::move(params)} { gdalmanagement::Initialize(); }

void Execute::Run(alus::cuda::CudaInit& cuda_init, size_t gpu_mem_percentage) {
    for (const auto& input : params_.inputs) {
        LOGI << "Resampling input - " << input;
        DatasetRegister ds_register(input);

        if (params_.resample_dimension_band.has_value()) {
            TryCertifyResampleDimensions(ds_register.GetBandDimension(params_.resample_dimension_band.value() -
                                                                      gdal::constants::GDAL_DEFAULT_RASTER_BAND));
        } else {
            TryCertifyResampleDimensions(params_.resample_dimension);
        }

        const auto band_count = ds_register.GetBandCount();
        for (size_t band_index = 0; band_index < band_count; band_index++) {
            const auto band_dim = ds_register.GetBandDimension(band_index);
            if (!DoesRequireResampling(band_index, band_dim)) {
                continue;
            }

            LOGI << "Band - " << ds_register.GetBandDescription(band_index);

            size_t input_buffer_bytes_size{};
            auto raster_data = ds_register.GetBandData(band_index, input_buffer_bytes_size);

            WaitForCudaInit(cuda_init);

            const size_t resample_output_buffer_size_bytes =
                static_cast<size_t>(resample_dim_.columnsX * resample_dim_.rowsY) *
                ds_register.GetRasterDataTypeSize(band_index);
            auto output_buffer = std::unique_ptr<uint8_t[]>(new uint8_t[resample_output_buffer_size_bytes]);
            if (!CanDeviceFit(resample_output_buffer_size_bytes + input_buffer_bytes_size, gpu_mem_percentage)) {
                THROW_ALGORITHM_EXCEPTION(APP_NAME, "Cannot resample - not enough GPU memory");
            }

            const auto gdal_data_type = ds_register.GetDataType(band_index);
            const auto data_type_parameters = CreateTypeParametersFrom(gdal_data_type);
            NppiResampleArguments args{raster_data.get(),
                                       input_buffer_bytes_size,
                                       band_dim,
                                       output_buffer.get(),
                                       resample_output_buffer_size_bytes,
                                       resample_dim_,
                                       params_.resample_method,
                                       data_type_parameters};
            NppiResample(args);

            alus::GeoTransformParameters resampled_gt = ds_register.GetGeoTransform(band_index);
            resampled_gt.pixelSizeLon =
                CalculatePixelSize(band_dim.columnsX, resample_dim_.columnsX, resampled_gt.pixelSizeLon);
            resampled_gt.pixelSizeLat =
                CalculatePixelSize(band_dim.rowsY, resample_dim_.rowsY, resampled_gt.pixelSizeLat);

            OGRSpatialReference srs = ds_register.GetSrs();
            OGRSpatialReference dest_srs;
            if (params_.crs.has_value()) {
                auto gt = alus::GeoTransformConstruct::ConvertToGdal(resampled_gt);
                Reprojection(&srs, &dest_srs, gt.data(), params_.crs.value());
                resampled_gt = alus::GeoTransformConstruct::BuildFromGdal(gt.data());
            } else {
                dest_srs = srs;
            }

            TileConstruct tile_params{resample_dim_, resampled_gt, params_.tile_dimension, params_.pixel_overlap};

            auto* driver = ds_register.GetGdalDriver();
            if (params_.output_format.has_value()) {
                driver = GetGDALDriverManager()->GetDriverByName(params_.output_format.value().c_str());
                CHECK_GDAL_PTR(driver);
            }

            auto metadata = ds_register.GetBandMetadata(band_index);
            AddMetadata(metadata);
            double no_data_value{};
            const auto has_no_data = ds_register.GetBandNoDataValue(band_index, no_data_value);

            ReviseOutputFactories();

            output_factories_.push_back(
                StoreResampled(CreateTiles(tile_params), resample_dim_, std::move(output_buffer),
                               params_.output_path + std::filesystem::path::preferred_separator +
                                   ds_register.GetGranuleImageFilenameStemFor(band_index),
                               {data_type_parameters, driver, gdal_data_type, dest_srs}, std::move(metadata),
                               has_no_data ? std::make_optional(no_data_value) : std::nullopt));
        }
    }
    LOGI << "Waiting I/O operations to be finished...";
    while (!output_factories_.empty()) {
        ReviseOutputFactories();
    }
    LOGI << "Done";
}

void Execute::WaitForCudaInit(const alus::cuda::CudaInit& cuda_init) {
    if (cuda_init_done_) {
        return;
    }

    while (!cuda_init.IsFinished()) {
        // Wait until GPU devices have been inited.
    }

    cuda_init_done_ = true;

    const auto& cuda_devices = cuda_init.GetDevices();
    if (cuda_devices.empty()) {
        THROW_ALGORITHM_EXCEPTION(APP_NAME, "No Nvidia CUDA GPUs detected.");
    }

    gpu_device_ = &cuda_devices.front();
    gpu_device_->Set();
    LOGI << "Using '" << gpu_device_->GetName() << "' device nr " << gpu_device_->GetDeviceNr() << " for resamplings";
}

bool Execute::CanDeviceFit(size_t bytes, size_t percentage_available) const {
    constexpr size_t KERNEL_OVERHEAD{10000};
    constexpr size_t BYTES_SHIFT_MIB{20};

    const auto memory_free = gpu_device_->GetFreeGlobalMemory();
    const auto total_memory = gpu_device_->GetTotalGlobalMemory();
    const auto usable_memory = static_cast<size_t>(total_memory * (static_cast<double>(percentage_available) / 100.0));

    const auto bytes_required = bytes + KERNEL_OVERHEAD;
    if (bytes_required > usable_memory) {
        LOGW << "Processing requires " << (bytes_required >> BYTES_SHIFT_MIB) << " MiB, but "
             << (usable_memory >> BYTES_SHIFT_MIB) << " MiB are allowed to be used for GPU memory";
        return false;
    }

    if (memory_free < bytes_required) {
        LOGW << "Processing requires " << (bytes_required >> BYTES_SHIFT_MIB)
             << " MiB, but not enough free GPU memory - " << (memory_free >> BYTES_SHIFT_MIB);
        return false;
    }

    return true;
}

void Execute::TryCertifyResampleDimensions(RasterDimension dim) {
    if (resample_dim_certified_) {
        return;
    }

    resample_dim_ = dim;

    if (resample_dim_.columnsX < 1 || resample_dim_.rowsY < 1) {
        THROW_ALGORITHM_EXCEPTION(APP_NAME, "Given resampling dimensions(" + std::to_string(resample_dim_.columnsX) +
                                                "x" + std::to_string(resample_dim_.rowsY) + ") not valid.");
    }
    resample_dim_certified_ = true;
}

void Execute::ReviseOutputFactories() {
    auto iter = output_factories_.begin();
    while (iter != output_factories_.end()) {
        if (iter->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            iter->get();
            iter = output_factories_.erase(iter);
        } else {
            iter++;
        }
    }
}

bool Execute::IsBandInExcludeList(size_t band_index) const {
    return std::find(params_.excluded_bands.cbegin(), params_.excluded_bands.cend(),
                     band_index + gdal::constants::GDAL_DEFAULT_RASTER_BAND) != params_.excluded_bands.cend();
}

bool Execute::DoesBandNeedResampling(alus::RasterDimension band_dim) const {
    return band_dim.columnsX != resample_dim_.columnsX || band_dim.rowsY != resample_dim_.rowsY;
}

bool Execute::DoesRequireResampling(size_t band_index, RasterDimension band_dim) const {
    if (IsBandInExcludeList(band_index)) {
        return false;
    }

    if (!DoesBandNeedResampling(band_dim)) {
        LOGI << "Band " << band_index + gdal::constants::GDAL_DEFAULT_RASTER_BAND
             << " equals resampling dimension, no need to process";
        return false;
    }

    return true;
}

void Execute::AddMetadata(std::vector<std::pair<std::string, std::pair<std::string, std::string>>>& from_ds) const {
    from_ds.emplace_back(
        DATASET_DOMAIN_METADATA_HINT,
        std::make_pair("RESAMPLING_METHOD",
                       METHOD_STRINGS.at(static_cast<std::underlying_type_t<Method>>(params_.resample_method))));
    from_ds.emplace_back(
        DATASET_DOMAIN_METADATA_HINT,
        std::make_pair("ALUs_VERSION", std::to_string(VERSION_MAJOR) + "." + std::to_string(VERSION_MINOR) + "." +
                                           std::to_string(VERSION_PATCH)));
}

Execute::~Execute() {
    for (const auto& f : output_factories_) {
        if (f.valid()) {
            f.wait_for(std::chrono::milliseconds(0));
        }
    }
    gdalmanagement::Deinitialize();
}
}  // namespace alus::resample