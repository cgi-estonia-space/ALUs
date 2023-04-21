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
#include "thermal_noise_remover.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

#include <gdal.h>
#include <boost/algorithm/string.hpp>

#include "algorithm_exception.h"
#include "alus_log.h"
#include "gdal_util.h"
#include "general_utils.h"
#include "operator_utils.h"
#include "shapes.h"
#include "shapes_util.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/util/product_utils.h"
#include "snap-engine-utilities/engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/engine-utilities/datamodel/unit.h"
#include "snap-engine-utilities/engine-utilities/gpf/input_product_validator.h"
#include "thermal_noise_kernel.h"
#include "thermal_noise_utils.h"

namespace alus::tnr {

void InitThreadContext(ThreadData* context, const SharedData* tnr_data) {
    const auto tile_size = static_cast<size_t>(tnr_data->max_width) * static_cast<size_t>(tnr_data->max_height);
    const int average_burst_count{9};
    const int padding_coefficient{20};

    CHECK_CUDA_ERR(cudaStreamCreate(&context->stream));
    context->h_tile_buffer.Allocate(tnr_data->use_pinned_memory, tile_size);
    size_t dev_mem_bytes{0};
    dev_mem_bytes += sizeof(double) * tile_size;                              // Noise matrix
    dev_mem_bytes += sizeof(IntensityData) * tile_size;                       // Pixel buffer
    dev_mem_bytes += sizeof(s1tbx::DeviceNoiseVector) * average_burst_count;  // Maximum number of bursts
    dev_mem_bytes += dev_mem_bytes / padding_coefficient;
    context->dev_mem_arena.ReserveMemory(dev_mem_bytes);
}

void FreeContexts(std::vector<ThreadData>& thread_contexts) {
    for (auto& context : thread_contexts) {
        cudaStreamDestroy(context.stream);
    }
}

void ThermalNoiseRemover::ComputeTileImage(ThreadData* context, SharedData* tnr_data) {
    try {
        InitThreadContext(context, tnr_data);
        while (true) {
            {
                std::unique_lock lock(tnr_data->exception_mutex);
                if (tnr_data->exception) {
                    return;
                }
            }
            Rectangle target_tile;
            if (!tnr_data->tile_queue.PopFront(target_tile)) {
                return;
            }

            if (is_complex_data_) {
                ComputeComplexTile(target_tile, context, tnr_data);
            } else {
                ComputeAmplitudeTile(target_tile, context, tnr_data);
            }
        }
    } catch (const std::exception& e) {
        std::unique_lock lock(tnr_data->exception_mutex);
        if (!tnr_data->exception) {
            tnr_data->exception = std::current_exception();
        } else {
            LOGE << e.what();
        }
    }
}
void ThermalNoiseRemover::ComputeComplexTile(alus::Rectangle target_tile, ThreadData* context, SharedData* tnr_data) {
    const auto d_noise_block = BuildNoiseLutForTOPSSLC(target_tile, thermal_noise_info_, context);
//    LOGI << "Tile size " << target_tile.width << "x" << target_tile.height << " "
//         << target_tile.height * target_tile.width << " matrix size " << d_noise_block.size;
    static_assert(sizeof(alus::Iq16) == sizeof(*context->h_tile_buffer.Get()));
    auto* buffer_ptr = reinterpret_cast<alus::Iq16*>(context->h_tile_buffer.Get());

    {
        std::unique_lock l(tnr_data->dataset_mutex);
        CHECK_GDAL_ERROR(tnr_data->src_dataset->GetRasterBand(1)->RasterIO(
            GF_Read, target_tile.x + tnr_data->src_area.x, target_tile.y + tnr_data->src_area.y, target_tile.width,
            target_tile.height, buffer_ptr, target_tile.width, target_tile.height, GDT_CInt16, 0, 0));
    }

    const auto tile_size = static_cast<size_t>(target_tile.width) * static_cast<size_t>(target_tile.height);
    cuda::KernelArray<IntensityData> d_pixel_buffer{nullptr, tile_size};
    d_pixel_buffer.array = context->dev_mem_arena.AllocArray<IntensityData>(tile_size);

    CHECK_CUDA_ERR(cudaMemcpyAsync(d_pixel_buffer.array, context->h_tile_buffer.Get(), d_pixel_buffer.ByteSize(),
                                   cudaMemcpyHostToDevice, context->stream));

    // There is a check in SNAP for a case when TNR is performed after calibration. However, ALUs does not allow such
    // use case so this check was omitted.

    LaunchComputeComplexTileKernel(target_tile, target_no_data_value_, target_floor_value_, d_pixel_buffer,
                                   d_noise_block, context->stream);
    CHECK_CUDA_ERR(cudaMemcpyAsync(context->h_tile_buffer.Get(), d_pixel_buffer.array, d_pixel_buffer.ByteSize(),
                                   cudaMemcpyDeviceToHost, context->stream));
    CHECK_CUDA_ERR(cudaStreamSynchronize(context->stream));
    CHECK_CUDA_ERR(cudaGetLastError());

    device::DestroyKernelMatrix(d_noise_block);
    context->dev_mem_arena.Reset();

    {
        std::unique_lock lock(tnr_data->dst_mutex);
        CHECK_GDAL_ERROR(tnr_data->dst_band->RasterIO(GF_Write, target_tile.x, target_tile.y, target_tile.width,
                                                      target_tile.height, context->h_tile_buffer.Get(),
                                                      target_tile.width, target_tile.height, GDT_Float32, 0, 0));
    }
}

void ThermalNoiseRemover::ComputeAmplitudeTile(alus::Rectangle target_tile, ThreadData* context, SharedData* tnr_data) {
    const auto d_noise_block = BuildNoiseLutForTOPSGRD(target_tile, thermal_noise_info_, context);
    static_assert(sizeof(IntensityData::input_amplitude) == sizeof(*context->h_tile_buffer.Get()));
    auto* buffer_ptr = reinterpret_cast<uint32_t*>(context->h_tile_buffer.Get());
    {
        std::unique_lock l(tnr_data->dataset_mutex);
        CHECK_GDAL_ERROR(tnr_data->src_dataset->GetRasterBand(1)->RasterIO(
            GF_Read, target_tile.x + tnr_data->src_area.x, target_tile.y + tnr_data->src_area.y, target_tile.width,
            target_tile.height, buffer_ptr, target_tile.width, target_tile.height, GDT_UInt32, 0, 0));
    }

    const auto tile_size = static_cast<size_t>(target_tile.width) * static_cast<size_t>(target_tile.height);
    cuda::KernelArray<IntensityData> d_pixel_buffer{nullptr, tile_size};
    d_pixel_buffer.array = context->dev_mem_arena.AllocArray<IntensityData>(tile_size);

    CHECK_CUDA_ERR(cudaMemcpyAsync(d_pixel_buffer.array, context->h_tile_buffer.Get(), d_pixel_buffer.ByteSize(),
                                   cudaMemcpyHostToDevice, context->stream));

    // There is a check in SNAP for a case when TNR is performed after calibration. However, ALUs does not allow such
    // use case so this check was omitted.

    LaunchComputeAmplitudeTileKernel(target_tile, target_no_data_value_, target_floor_value_, d_pixel_buffer,
                                     d_noise_block, context->stream);
    CHECK_CUDA_ERR(cudaMemcpyAsync(context->h_tile_buffer.Get(), d_pixel_buffer.array, d_pixel_buffer.ByteSize(),
                                   cudaMemcpyDeviceToHost, context->stream));
    CHECK_CUDA_ERR(cudaStreamSynchronize(context->stream));
    CHECK_CUDA_ERR(cudaGetLastError());

    device::DestroyKernelMatrix(d_noise_block);
    context->dev_mem_arena.Reset();

    {
        std::unique_lock lock(tnr_data->dst_mutex);
        CHECK_GDAL_ERROR(tnr_data->dst_band->RasterIO(GF_Write, target_tile.x, target_tile.y, target_tile.width,
                                                      target_tile.height, context->h_tile_buffer.Get(),
                                                      target_tile.width, target_tile.height, GDT_Float32, 0, 0));
    }
}

void ThermalNoiseRemover::Initialise() {
    // CHECK INPUT PRODUCT
    snapengine::InputProductValidator input_product_validator{source_product_};
    input_product_validator.CheckIfSentinel1Product();
    input_product_validator.CheckAcquisitionMode(
        std::vector<std::string>(std::begin(SUPPORTED_ACQUISITION_MODES), std::end(SUPPORTED_ACQUISITION_MODES)));
    input_product_validator.CheckProductType(
        std::vector<std::string>(std::begin(SUPPORTED_PRODUCT_TYPES), std::end(SUPPORTED_PRODUCT_TYPES)));
    abstract_metadata_root_ = snapengine::AbstractMetadata::GetAbstractedMetadata(source_product_);

    GetSubsetOffset();
    // There is no need for these as only IW SLC product is currently supported
    // GetProductType();
    // GetAcquisitionMode();
    GetThermalNoiseCorrectionFlag();
    GetCalibrationFlag();
    CreateTargetProduct();
    AddSelectedBands();
    snapengine::ProductUtils::CopyProductNodes(source_product_, target_product_);

    CreateTargetDatasetFromProduct();

    if (is_complex_data_) {
        thermal_noise_info_ = GetThermalNoiseInfoForBursts(
            polarisation_, subswath_,
            source_product_->GetMetadataRoot()->GetElement(snapengine::AbstractMetadata::ORIGINAL_PRODUCT_METADATA));
    } else {
        thermal_noise_info_ = GetThermalNoiseInfoForGrd(
            polarisation_,
            source_product_->GetMetadataRoot()->GetElement(snapengine::AbstractMetadata::ORIGINAL_PRODUCT_METADATA));
        FillTimeMapsWithT0AndDeltaTS(
            std::filesystem::path(source_ds_->GetDescription()).stem().string(),
            source_product_->GetMetadataRoot()->GetElement(snapengine::AbstractMetadata::ORIGINAL_PRODUCT_METADATA),
            thermal_noise_info_.time_maps);
    }
}

void ThermalNoiseRemover::GetSubsetOffset() {
    subset_offset_x_ = abstract_metadata_root_->GetAttributeInt(snapengine::AbstractMetadata::SUBSET_OFFSET_X);
    subset_offset_y_ = abstract_metadata_root_->GetAttributeInt(snapengine::AbstractMetadata::SUBSET_OFFSET_Y);
}
void ThermalNoiseRemover::GetThermalNoiseCorrectionFlag() {
    const auto processing_information_element =
        source_product_->GetMetadataRoot()
            ->GetElement(snapengine::AbstractMetadata::ORIGINAL_PRODUCT_METADATA)
            ->GetElement(snapengine::AbstractMetadata::ANNOTATION)
            ->GetElements()
            .at(0)
            ->GetElement(snapengine::AbstractMetadata::product)
            ->GetElement(snapengine::AbstractMetadata::IMAGE_ANNOTATION)
            ->GetElement(snapengine::AbstractMetadata::PROCESSING_INFORMATION);
    is_tnr_done_ = processing_information_element->GetAttributeBool(
        snapengine::AbstractMetadata::THERMAL_NOISE_CORRECTION_PERFORMED);

    if (is_tnr_done_) {
        THROW_ALGORITHM_EXCEPTION(
            ALG_NAME,
            "Thermal Noise Removal was previously performed and noise data reintroduction is not currently supported. "
            "If You have interest in such functionality, please contact the developers team.");
    }
}
void ThermalNoiseRemover::GetCalibrationFlag() {
    was_absolute_calibration_performed_ =
        abstract_metadata_root_->GetAttributeBool(snapengine::AbstractMetadata::ABS_CALIBRATION_FLAG);

    if (was_absolute_calibration_performed_) {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME,
                                  "Absolute calibration was already performed and currently thermal noise removal of "
                                  "calibrated products is not supported. If You have interest in such functionality, "
                                  "please contact the developers team.");
    }
}

void ThermalNoiseRemover::CreateTargetProduct() {
    target_product_ = snapengine::Product::CreateProduct(
        source_product_->GetName() + std::string(PRODUCT_SUFFIX), source_product_->GetProductType(),
        source_product_->GetSceneRasterWidth(), source_product_->GetSceneRasterHeight());
}

void ThermalNoiseRemover::AddSelectedBands() {
    std::vector<std::string> source_band_names;
    const auto source_bands = snapengine::OperatorUtils::GetSourceBands(source_product_, source_band_names, false);
    for (size_t i = 0; i < source_bands.size(); i++) {
        const auto& source_band = source_bands.at(i);
        const auto unit = source_band->GetUnit();
        if (!unit.has_value()) {
            THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Band " + source_band->GetName() + " requires a unit.");
        }

        if (!utils::general::DoesStringContain(unit->data(), snapengine::Unit::REAL) &&
            !utils::general::DoesStringContain(unit->data(), snapengine::Unit::AMPLITUDE) &&
            !utils::general::DoesStringContain(unit->data(), snapengine::Unit::INTENSITY)) {
            continue;
        }

        if (utils::general::DoesStringContain(unit->data(), snapengine::Unit::REAL)) {  // SLC
            if (i + 1 >= source_bands.size()) {
                THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Real and imaginary bands are not in pairs.");
            }
            if (const auto next_unit = source_bands.at(i + 1)->GetUnit();
                !next_unit.has_value() ||
                !utils::general::DoesStringContain(next_unit->data(), snapengine::Unit::IMAGINARY)) {
                THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Real and imaginary bands are not in pairs.");
            }

            source_band_names = {source_band->GetName(), source_bands.at(i + 1)->GetName()};
            ++i;
        } else {  // GRD product
            source_band_names = {source_band->GetName()};
        }

        const auto pol_location = source_band_names.at(0).rfind('_');
        const auto polarisation = source_band_names.at(0).substr(pol_location + 1);
        if (!boost::iequals(polarisation, polarisation_)) {
            continue;
        }

        const auto target_band_name = source_band_names.at(0) + "_" + polarisation;
        if (!target_product_->GetBand(target_band_name)) {
            target_band_name_to_source_band_names_.try_emplace(target_band_name, source_band_names);

            auto target_band =
                std::make_shared<snapengine::Band>(target_band_name, snapengine::ProductData::TYPE_FLOAT32,
                                                   source_band->GetRasterWidth(), source_band->GetRasterHeight());

            target_band->SetUnit(snapengine::Unit::INTENSITY);
            target_band->SetDescription(source_band->GetDescription());
            target_band->SetNoDataValue(source_band->GetNoDataValue());
            target_band->SetNoDataValueUsed(true);
            target_product_->AddBand(target_band);
        }
    }
    // SNAP adds noise band but this functionality is not yet implemented.
}
void ThermalNoiseRemover::SetTargetImages() {
    const auto target_bands = target_product_->GetBands();
    LOGI << "Target bands count: " << target_bands.size();
    SharedData tnr_data{};

    tnr_data.max_height = tile_height_;
    tnr_data.max_width = tile_width_;
    for (const auto& band : target_bands) {
        if (const auto band_name = band->GetName(); utils::general::DoesStringContain(band_name, subswath_)) {
            LOGI << "Processing band " << band_name;
            auto source_band_name = target_band_name_to_source_band_names_.at(band_name).at(0);

            tnr_data.dst_band = target_dataset_->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND);
            tnr_data.src_dataset = source_ds_;
            tnr_data.src_area = source_ds_area_;

            std::vector<Rectangle> tiles = CalculateTiles(source_ds_area_);

            // If bound by GPU, then more than 2 give almost no gains
            // If bound by gdal, then locking for both input and output means more than 3 should give almost no benefit
            constexpr size_t THREAD_LIMIT{3};
            constexpr size_t TILE_THREAD_RATIO{5};

            const size_t n_threads = (tiles.size() / THREAD_LIMIT) > 0 ? THREAD_LIMIT : 1;
            tnr_data.use_pinned_memory = (tiles.size() / n_threads) > TILE_THREAD_RATIO;
            LOGD << "S1 TNR tiles: " << tiles.size() << "\tthreads = " << n_threads
                 << "\ttransfer mode = " << (tnr_data.use_pinned_memory ? "pinned" : "paged");

            tnr_data.tile_queue.InsertData(std::move(tiles));
            std::vector<ThreadData> thread_contexts(n_threads);
            std::vector<std::thread> threads;
            for (size_t i = 0; i < n_threads; ++i) {
                //                auto t = std::thread(&ThermalNoiseRemover::ComputeTileImage, this,
                //                &thread_contexts.at(i), &tnr_data); t.join();
                threads.emplace_back(&ThermalNoiseRemover::ComputeTileImage, this, &thread_contexts.at(i), &tnr_data);
            }

            for (auto& thread : threads) {
                thread.join();
            }

            FreeContexts(thread_contexts);
            if (tnr_data.exception) {
                std::rethrow_exception(tnr_data.exception);
            }
        }
    }
}
std::vector<Rectangle> ThermalNoiseRemover::CalculateTiles(Rectangle in_raster_area) const {
    std::vector<Rectangle> output_tiles;
    int x_max = in_raster_area.width;
    int y_max = in_raster_area.height;
    int x_count = x_max / tile_width_ + 1;
    int y_count = y_max / tile_height_ + 1;

    for (int y_index = 0; y_index < y_count; ++y_index) {
        for (int x_index = 0; x_index < x_count; ++x_index) {
            Rectangle rectangle{x_index * tile_width_, y_index * tile_height_, tile_width_, tile_height_};
            if (rectangle.x > x_max || rectangle.y > y_max) {
                continue;
            }
            Rectangle intersection = shapeutils::GetIntersection(rectangle, {0, 0, x_max, y_max});
            if (intersection.width != 0 && intersection.height != 0) {
                output_tiles.push_back(intersection);
            }
        }
    }
    return output_tiles;
}
void ThermalNoiseRemover::CreateTargetDatasetFromProduct() {
    auto get_band_sub_swath = [](std::string_view band_name) {
        auto delimiter_pos = band_name.find('_');
        const auto first_sub_string = band_name.substr(delimiter_pos + 1);
        delimiter_pos = first_sub_string.find('_');
        const auto sub_swath = first_sub_string.substr(0, delimiter_pos);
        return std::string(sub_swath);
    };

    char** dataset_options = nullptr;
    GDALDriver* driver = GetGdalMemDriver();
    if (!driver) {
        THROW_ALGORITHM_EXCEPTION(
            ALG_NAME, "could not create GDAL driver for " + std::string(gdal::constants::GDAL_MEM_DRIVER) + " format");
    }

    if (!CSLFetchBoolean(driver->GetMetadata(), GDAL_DCAP_CREATE, FALSE)) {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, "GDAL driver for " + std::string(gdal::constants::GDAL_MEM_DRIVER) +
                                                " format does not support creating datasets.");
    }
    // End of placeholder

    const auto target_bands = target_product_->GetBands();
    for (int i = 1; i <= static_cast<int>(target_bands.size()); i++) {
        const auto& band = target_bands.at(i - 1);
        const auto sub_swath = get_band_sub_swath(band->GetName());
        if (boost::iequals(subswath_, sub_swath)) {
            // Create dataset
            const auto output_file = output_path_ + target_product_->GetName() + "_" + sub_swath;

            const int band_count{1};

            GDALDataset* gdal_dataset =
                driver->Create(output_file.data(), source_ds_area_.width, source_ds_area_.height, band_count,
                               GDT_Float32, dataset_options);

            std::array<double, gdal::constants::GDAL_GEOTRANSFORM_PARAMETER_COUNT> geo_transform;
            source_ds_->GetGeoTransform(geo_transform.data());
            gdal_dataset->SetGeoTransform(geo_transform.data());

            std::shared_ptr<GDALDataset> dataset(gdal_dataset, [](auto dataset_arg) { GDALClose(dataset_arg); });
            target_dataset_ = dataset;
            target_path_ = output_file;
            return;  // There shouldn't be more than one output dataset for now.
        }
    }
}

ThermalNoiseRemover::ThermalNoiseRemover(std::shared_ptr<snapengine::Product> source_product,
                                         GDALDataset* source_dataset, Rectangle source_ds_area,
                                         std::string_view subswath, std::string_view polarisation,
                                         std::string_view output_path, int tile_width, int tile_height)
    : source_product_(source_product),
      source_ds_{source_dataset},
      source_ds_area_{source_ds_area},
      subswath_(subswath),
      polarisation_(polarisation),
      output_path_(output_path),
      tile_width_(tile_width),
      tile_height_(tile_height),
      is_complex_data_{GDALDataTypeIsComplex(
                           source_ds_->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetRasterDataType()) ==
                       1} {
    Initialise();
}
const std::shared_ptr<snapengine::Product>& ThermalNoiseRemover::GetTargetProduct() const { return target_product_; }
void ThermalNoiseRemover::Execute() { SetTargetImages(); }

std::pair<std::shared_ptr<GDALDataset>, std::string> ThermalNoiseRemover::GetOutputDataset() const {
    return {target_dataset_, target_path_};
}
}  // namespace alus::tnr
