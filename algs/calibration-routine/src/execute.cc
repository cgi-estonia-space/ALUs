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
#include <memory>
#include <string>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "abstract_metadata.h"
#include "algorithm_exception.h"
#include "alus_log.h"
#include "aoi_burst_extract.h"
#include "constants.h"
#include "dem_assistant.h"
#include "gdal_image_reader.h"
#include "gdal_image_writer.h"
#include "gdal_management.h"
#include "gdal_util.h"
#include "metadata_record.h"
#include "sentinel1_calibrate.h"
#include "sentinel1_product_reader_plug_in.h"
#include "terrain_correction.h"
#include "terrain_correction_metadata.h"
#include "thermal_noise_remover.h"
#include "topsar_deburst_op.h"
#include "topsar_merge.h"
#include "topsar_split.h"

namespace {
template <typename T>
bool EqualsAnyOf(std::string_view comparand, T begin, T end) {
    return std::any_of(begin, end, [&comparand](std::string_view value) { return boost::iequals(comparand, value); });
}

constexpr size_t TILE_SIZE_DIMENSION{2048};
constexpr size_t GDAL_CACHE_SIZE{static_cast<size_t>(4e9)};
constexpr size_t FULL_SUBSWATH_BURST_INDEX_START{alus::topsarsplit::TopsarSplit::BURST_INDEX_OFFSET};
constexpr size_t FULL_SUBSWATH_BURST_INDEX_END{9999};
}  // namespace

namespace alus::calibrationroutine {

Execute::Execute(Parameters params, const std::vector<std::string>& dem_files)
    : params_{std::move(params)}, dem_files_{dem_files} {
    alus::gdalmanagement::Initialize();
    params_.aoi = ConditionAoi(params_.aoi);
}

void Execute::Run(alus::cuda::CudaInit& cuda_init, size_t) {
    ParseCalibrationType(params_.calibration_type);
    PrintProcessingParameters();
    ValidateParameters();

    std::string output_dir;
    std::string final_path;
    auto output_path = boost::filesystem::path(params_.output);

    if (boost::filesystem::is_directory(output_path)) {
        // final name comes from input
        output_dir = params_.output + "/";
    } else {
        output_dir = output_path.parent_path().string() + "/";
        final_path = params_.output;
    }

    // SLC input x 1 = ~1.25GB
    // TC ouput = ~1GB
    alus::gdalmanagement::SetCacheMax(GDAL_CACHE_SIZE);

    auto dem_assistant = dem::Assistant::CreateFormattedDemTilesOnGpuFrom(dem_files_);

    auto reader_plug_in = std::make_shared<s1tbx::Sentinel1ProductReaderPlugIn>();
    auto reader = reader_plug_in->CreateReaderInstance();
    auto product = reader->ReadProductNodes(boost::filesystem::canonical(params_.input), nullptr);
    const auto pt = product->GetProductType();
    (void)pt;  // "GRD" or "SLC"

    // split
    std::vector<std::shared_ptr<alus::topsarsplit::TopsarSplit>> splits;
    std::vector<std::string> swath_selection;

    if (pt == "SLC") {
        const auto split_start = std::chrono::steady_clock::now();
        Split(product, params_.burst_first_index, params_.burst_last_index, splits, swath_selection);

        LOGI << "Sentinel 1 split done - "
             << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - split_start)
                    .count()
             << "ms";
    } else if (pt == "GRD") {
        swath_selection.emplace_back("IW");
    } else {
        throw std::invalid_argument("Expected 'GRD' or 'SLC' product type - '" + pt +
                                    "' not supported. Please check the input dataset.");
    }

    while (!cuda_init.IsFinished())
        ;
    cuda_init.CheckErrors();
    const auto cuda_device = cuda_init.GetDevices().front();
    cuda_device.Set();

    LOGI << "Using '" << cuda_device.GetName() << "' device nr " << cuda_device.GetDeviceNr() << " for calculations";

    // thermal noise removal
    const auto tnr_start = std::chrono::steady_clock::now();

    std::vector<std::shared_ptr<snapengine::Product>> tnr_products(splits.size());
    std::vector<std::shared_ptr<GDALDataset>> tnr_datasets(splits.size());
    ThermalNoiseRemoval(splits, swath_selection, output_dir, tnr_products, tnr_datasets);
    LOGI << "Thermal noise removal done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tnr_start).count()
         << "ms";

    metadata_.Add(common::metadata::sentinel1::SENSING_START,
                  snapengine::AbstractMetadata::GetAbstractedMetadata(splits.front()->GetTargetProduct())
                      ->GetAttributeUtc(alus::snapengine::AbstractMetadata::FIRST_LINE_TIME)
                      ->ToString());
    metadata_.Add(common::metadata::sentinel1::SENSING_END,
                  snapengine::AbstractMetadata::GetAbstractedMetadata(splits.back()->GetTargetProduct())
                      ->GetAttributeUtc(alus::snapengine::AbstractMetadata::LAST_LINE_TIME)
                      ->ToString());

    // calibration
    std::vector<std::shared_ptr<snapengine::Product>> calib_products(tnr_products.size());
    std::vector<std::shared_ptr<GDALDataset>> calib_datasets(tnr_datasets.size());
    std::vector<std::string> output_names(tnr_products.size());

    Calibration(tnr_products, tnr_datasets, swath_selection, output_dir, output_names, calib_products, calib_datasets);
    tnr_products.clear();
    tnr_datasets.clear();

    if (params_.wif) {
        for (size_t i = 0; i < output_names.size(); i++) {
            LOGI << "Calibration output @ " << output_names.at(i) << ".tif";
            GeoTiffWriteFile(calib_datasets.at(i).get(), output_names.at(i));
        }
    }

    // start thread for DEM calculations parallel with deburst which does not use GPU
    // rethink this part if we add support for larger GPUs with full chain on GPU processing
    dem_assistant->GetElevationManager()->TransferToDevice();

    // deburst
    std::vector<std::shared_ptr<snapengine::Product>> deburst_products(tnr_products.size());

    Deburst(calib_products, calib_datasets, output_names, deburst_products);
    for (auto& ds : calib_datasets) {
        ds.reset();
    }
    calib_datasets.clear();

    // Merge
    std::shared_ptr<snapengine::Product> merge_output;
    Merge(deburst_products, output_names, merge_output);

    // TC
    const auto output_file =
        TerrainCorrection(merge_output, deburst_products.size(), output_names.front(), dem_assistant, final_path);

    LOGI << "Algorithm completed, output file @ " << output_file;
}

void Execute::PrintProcessingParameters() const {
    LOGI << "Processing parameters:" << std::endl
         << "Input product - " << params_.input << std::endl
         << "Subswath - " << params_.subswath << std::endl
         << "Polarisation - " << params_.polarisation << std::endl
         << "Calibration type - " << params_.calibration_type << std::endl
         << "First burst index - " << params_.burst_first_index << std::endl
         << "Last burst index - " << params_.burst_last_index << std::endl
         << "AOI - " << params_.aoi << std::endl
         << "Write intermediate files - " << (params_.wif ? "YES" : "NO") << std::endl;
}

void Execute::ParseCalibrationType(std::string_view type) {
    if (boost::iequals(CALIBRATION_TYPE_BETA, type)) {
        calibration_types_selected_.get_beta_lut = true;
    } else if (boost::iequals(CALIBRATION_TYPE_GAMMA, type)) {
        calibration_types_selected_.get_gamma_lut = true;
    } else if (boost::iequals(CALIBRATION_TYPE_SIGMA, type)) {
        calibration_types_selected_.get_sigma_lut = true;
    } else if (boost::iequals(CALIBRATION_TYPE_DN, type)) {
        calibration_types_selected_.get_dn_lut = true;
    } else {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Unknown calibration type specified - " + std::string(type) + ".");
    }
}

void Execute::ValidateParameters() const { ValidatePolarisation(); }

void Execute::ValidateSubSwath() const {
    if (!EqualsAnyOf(params_.subswath, SUBSWATHS.cbegin(), SUBSWATHS.cend())) {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Subswath value not supported - " + params_.subswath);
    }
}

void Execute::ValidatePolarisation() const {
    if (!EqualsAnyOf(params_.polarisation, POLARISATIONS.cbegin(), POLARISATIONS.cend())) {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Polarisation value not supported - " + params_.polarisation);
    }
}

Execute::~Execute() { alus::gdalmanagement::Deinitialize(); }

void Execute::Split(std::shared_ptr<snapengine::Product> product, size_t burst_index_start, size_t burst_index_end,
                    std::vector<std::shared_ptr<topsarsplit::TopsarSplit>>& splits,
                    std::vector<std::string>& swath_selection) {
    if (!params_.subswath.empty()) {
        swath_selection = {params_.subswath};
        std::unique_ptr<topsarsplit::TopsarSplit> split_op{};
        if (!params_.aoi.empty()) {
            split_op = std::make_unique<topsarsplit::TopsarSplit>(product, params_.subswath, params_.polarisation,
                                                                  params_.aoi);
            metadata_.AddWhenMissing(common::metadata::sentinel1::AREA_SELECTION, params_.aoi);
        } else {
            if (burst_index_start == burst_index_end && burst_index_start < FULL_SUBSWATH_BURST_INDEX_START) {
                burst_index_start = FULL_SUBSWATH_BURST_INDEX_START;
                burst_index_end = FULL_SUBSWATH_BURST_INDEX_END;
            }
            split_op = std::make_unique<topsarsplit::TopsarSplit>(product, params_.subswath, params_.polarisation,
                                                                  burst_index_start, burst_index_end);
            metadata_.AddWhenMissing(common::metadata::sentinel1::AREA_SELECTION, params_.subswath);
        }
        split_op->Initialize();
        splits.push_back(std::move(split_op));
    } else if (params_.aoi.empty()) {
        swath_selection = {"IW1", "IW2", "IW3"};
        metadata_.AddWhenMissing(common::metadata::sentinel1::AREA_SELECTION, "IW1 IW2 IW3");
        for (const auto& swath : swath_selection) {
            splits.push_back(std::make_unique<topsarsplit::TopsarSplit>(product, swath, params_.polarisation));
            splits.back()->Initialize();
        }
    } else {
        // search for valid swaths with AOI
        topsarsplit::Aoi aoi_poly;
        boost::geometry::read_wkt(params_.aoi, aoi_poly);
        metadata_.AddWhenMissing(common::metadata::sentinel1::AREA_SELECTION, params_.aoi);
        for (const auto& swath : {"IW1", "IW2", "IW3"}) {
            auto swath_split = std::make_unique<topsarsplit::TopsarSplit>(product, swath, params_.polarisation);
            swath_split->Initialize();

            auto swath_poly = topsarsplit::ExtractSwathPolygon(swath_split->GetTargetProduct());

            if (topsarsplit::IsWithinSwath(aoi_poly, swath_poly)) {
                splits.clear();
                splits.push_back(
                    std::make_unique<topsarsplit::TopsarSplit>(product, swath, params_.polarisation, params_.aoi));
                splits.back()->Initialize();
                swath_selection = {std::string(swath)};
                break;
            }
            if (topsarsplit::IsCovered(swath_poly, aoi_poly)) {
                splits.push_back(
                    std::make_unique<topsarsplit::TopsarSplit>(product, swath, params_.polarisation, params_.aoi));
                splits.back()->Initialize();
                swath_selection.emplace_back(swath);
            }
        }

        LOGI << product->GetName() << " swaths to be processed = " << boost::algorithm::join(swath_selection, " ");
    }

    for (const auto& split : splits) {
        auto product_path = std::filesystem::path(product->GetFileLocation().string());
        // Remove manifest.safe
        if (product_path.has_filename() && product_path.extension() != ".zip") {
            split->OpenPixelReader(product_path.remove_filename().string());
        } else {
            split->OpenPixelReader(product_path.string());
        }
    }
}
void Execute::ThermalNoiseRemoval(const std::vector<std::shared_ptr<topsarsplit::TopsarSplit>>& splits,
                                  const std::vector<std::string>& subswaths, std::string_view output_dir,
                                  std::vector<std::shared_ptr<snapengine::Product>>& tnr_products,
                                  std::vector<std::shared_ptr<GDALDataset>>& tnr_datasets) const {
    if (tnr_products.empty()) {
        tnr_products.resize(splits.size());
    }
    if (tnr_datasets.empty()) {
        tnr_datasets.resize(splits.size());
    }
    if (tnr_datasets.size() != splits.size() || tnr_products.size() != splits.size()) {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Split and TNR product count mismatch!.");
    }
    for (size_t i = 0; i < splits.size(); i++) {
        const auto split = splits.at(i);
        const auto swath = subswaths.at(i);
        tnr::ThermalNoiseRemover thermal_noise_remover{split->GetTargetProduct(),
                                                       split->GetPixelReader()->GetDataset(),
                                                       swath,
                                                       params_.polarisation,
                                                       output_dir,
                                                       TILE_SIZE_DIMENSION,
                                                       TILE_SIZE_DIMENSION};
        thermal_noise_remover.Execute();
        tnr_products.at(i) = thermal_noise_remover.GetTargetProduct();
        tnr_datasets.at(i) = thermal_noise_remover.GetOutputDataset().first;

        if (params_.wif) {
            const auto tnr_tmp_file = thermal_noise_remover.GetOutputDataset().second;
            LOGI << "Thermal Noise Removal output @ " << tnr_tmp_file << ".tif";
            GeoTiffWriteFile(thermal_noise_remover.GetOutputDataset().first.get(), tnr_tmp_file);
        }
    }
}

void Execute::Calibration(const std::vector<std::shared_ptr<snapengine::Product>>& tnr_products,
                          const std::vector<std::shared_ptr<GDALDataset>>& tnr_datasets,
                          const std::vector<std::string>& subswaths, std::string_view output_dir,
                          std::vector<std::string>& output_names,
                          std::vector<std::shared_ptr<snapengine::Product>>& calib_products,
                          std::vector<std::shared_ptr<GDALDataset>>& calib_datasets) const {
    const auto cal_start = std::chrono::steady_clock::now();
    if (calib_products.empty()) {
        calib_products.resize(tnr_products.size());
    }
    if (calib_datasets.empty()) {
        calib_datasets.resize(tnr_datasets.size());
    }
    if (tnr_datasets.size() != calib_datasets.size() || tnr_products.size() != calib_products.size()) {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, "TNR and Calibration product count mismatch!.");
    }
    for (size_t i = 0; i < tnr_products.size(); i++) {
        sentinel1calibrate::Sentinel1Calibrator calibrator{tnr_products.at(i),
                                                           tnr_datasets.at(i).get(),
                                                           {subswaths.at(i)},
                                                           {params_.polarisation},
                                                           calibration_types_selected_,
                                                           output_dir,
                                                           false,
                                                           TILE_SIZE_DIMENSION,
                                                           TILE_SIZE_DIMENSION};
        calibrator.Execute();
        calib_products.at(i) = calibrator.GetTargetProduct();
        calib_datasets.at(i) = calibrator.GetOutputDatasets().begin()->second;

        output_names.at(i) = calibrator.GetTargetPath(subswaths.at(i));
    }

    LOGI << "Sentinel1 calibration done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - cal_start).count()
         << "ms";
}
void Execute::Deburst(const std::vector<std::shared_ptr<snapengine::Product>>& calib_products,
                      std::vector<std::shared_ptr<GDALDataset>>& calib_datasets, std::vector<std::string>& output_names,
                      std::vector<std::shared_ptr<snapengine::Product>>& deburst_products) const {
    const auto deburst_start = std::chrono::steady_clock::now();
    if (deburst_products.empty()) {
        deburst_products.resize(calib_products.size());
    }
    if (calib_products.size() != deburst_products.size()) {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, "Calibration and Deburst product count mismatch!.");
    }
    for (size_t i = 0; i < calib_products.size(); i++) {
        auto data_reader = std::make_shared<snapengine::custom::GdalImageReader>();
        data_reader->TakeExternalDataset(calib_datasets.at(i).get());

        calib_products.at(i)->SetImageReader(data_reader);
        auto data_writer = std::make_shared<snapengine::custom::GdalImageWriter>();
        auto deburst_op = s1tbx::TOPSARDeburstOp::CreateTOPSARDeburstOp(calib_products.at(i));
        deburst_products.at(i) = deburst_op->GetTargetProduct();

        auto& output_name = output_names.at(i);
        output_name = boost::filesystem::change_extension(output_name, "").string() + "_deb.tif";
        data_writer->Open(output_name, deburst_products.at(i)->GetSceneRasterWidth(),
                          deburst_products.at(i)->GetSceneRasterHeight(), data_reader->GetGeoTransform(),
                          data_reader->GetDataProjection(), true);
        deburst_products.at(i)->SetImageWriter(data_writer);
        deburst_op->Compute();
        calib_datasets.at(i).reset();

        if (params_.wif) {
            LOGI << "Deburst output @ " << output_name;
            GeoTiffWriteFile(data_writer->GetDataset(), output_name);
        }

        auto deb_reader = std::make_shared<snapengine::custom::GdalImageReader>();
        deb_reader->TakeExternalDataset(data_writer->GetDataset());
        data_writer->ReleaseDataset();
        deburst_products.at(i)->SetImageReader(deb_reader);
        data_reader->ReleaseDataset();
    }
    LOGI << "TOPSAR Deburst done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - deburst_start)
                .count()
         << "ms";
}
void Execute::Merge(const std::vector<std::shared_ptr<snapengine::Product>>& deburst_products,
                    std::vector<std::string>& output_names, std::shared_ptr<snapengine::Product>& merge_output) const {
    auto merge_start = std::chrono::steady_clock::now();
    if (deburst_products.size() == 1) {
        merge_output = deburst_products.front();
        output_names.resize(1);
    } else {
        output_names.resize(1);
        std::string sw_names{};
        for (size_t i{}; i < deburst_products.size(); i++) {
            const auto sw = s1tbx::Sentinel1Utils(deburst_products.at(i)).GetSubSwathNames().front();
            sw_names += sw;
            if (i + 1 < deburst_products.size()) {
                sw_names += "_";
            }
        }

        auto product_name_stem = boost::filesystem::change_extension(output_names.front(), "").string();
        {
            const auto find_it = product_name_stem.find("Cal_IW");
            if (find_it != std::string::npos) {
                product_name_stem = product_name_stem.erase(find_it + 3, 4);  // Plus subswath no.
            }
        }
        output_names.front() = product_name_stem + "_mrg_" + sw_names + ".tif";
        std::vector<std::string> merge_polarisations(deburst_products.size(), params_.polarisation);

        auto data_writer = std::make_shared<snapengine::custom::GdalImageWriter>();
        // TODO NB! merge not tile size independent - it has been decreased from 1024 since larger tile sizes introduce
        // a bug - https://github.com/cgi-estonia-space/ALUs/issues/18
        const size_t merge_tile_size = 256;  // Up until 256 merge operation processing time got smaller significantly.
        topsarmerge::TopsarMergeOperator merge_op(deburst_products, merge_polarisations, merge_tile_size,
                                                  merge_tile_size, output_names.front());

        auto target = merge_op.GetTargetProduct();

        data_writer->Open(target->GetFileLocation().generic_path().string(), target->GetSceneRasterWidth(),
                          target->GetSceneRasterHeight(), {}, {}, true);
        target->SetImageWriter(data_writer);
        merge_op.Compute();

        if (params_.wif) {
            LOGI << "Merge output @ " << output_names.front();
            GeoTiffWriteFile(data_writer->GetDataset(), output_names.front());
        }
        merge_output = merge_op.GetTargetProduct();
    }
    LOGI
        << "Merge done - "
        << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - merge_start).count()
        << "ms";
}
std::string Execute::TerrainCorrection(const std::shared_ptr<snapengine::Product>& merge_product,
                                       size_t deb_product_count, std::string_view output_name,
                                       std::shared_ptr<dem::Assistant> dem_assistant,
                                       std::string_view predefined_output_name) const {
    const auto tc_start = std::chrono::steady_clock::now();

    terraincorrection::Metadata metadata(merge_product);

    const auto* d_dem_tiles = dem_assistant->GetElevationManager()->GetBuffers();
    const size_t dem_tiles_length = dem_assistant->GetElevationManager()->GetTileCount();
    const int selected_band{1};

    // unfortunately have to assume this downcast is valid, because TC does not use the ImageWriter interface
    GDALDataset* tc_in_dataset = nullptr;
    if (deb_product_count > 1U) {
        tc_in_dataset = std::dynamic_pointer_cast<snapengine::custom::GdalImageWriter>(merge_product->GetImageWriter())
                            ->GetDataset();
    } else {
        tc_in_dataset = std::dynamic_pointer_cast<snapengine::custom::GdalImageReader>(merge_product->GetImageReader())
                            ->GetDataset();
    }

    const auto total_dimension_edge = 4096;
    const auto x_tile_size =
        static_cast<int>((tc_in_dataset->GetRasterXSize() /
                          static_cast<double>(tc_in_dataset->GetRasterXSize() + tc_in_dataset->GetRasterYSize())) *
                         total_dimension_edge);
    const auto y_tile_size = total_dimension_edge - x_tile_size;

    terraincorrection::TerrainCorrection tc(
        tc_in_dataset, metadata.GetMetadata(), metadata.GetLatTiePointGrid(), metadata.GetLonTiePointGrid(),
        d_dem_tiles, dem_tiles_length, dem_assistant->GetElevationManager()->GetProperties(), dem_assistant->GetType(),
        dem_assistant->GetElevationManager()->GetPropertiesValue(), selected_band);
    std::string tc_output_file = predefined_output_name.empty()
                                     ? boost::filesystem::change_extension(output_name.data(), "").string() + "_tc.tif"
                                     : std::string(predefined_output_name);
    tc.RegisterMetadata(metadata_);
    tc.ExecuteTerrainCorrection(tc_output_file, x_tile_size, y_tile_size);
    LOGI << "Terrain correction done - "
         << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tc_start).count()
         << "ms";

    return tc_output_file;
}

std::string Execute::ConditionAoi(const std::string& aoi) const {
    if (aoi.empty() or !std::filesystem::exists(aoi)) {
        return aoi;
    }

    return alus::ConvertToWkt(aoi);
}

}  // namespace alus::calibrationroutine
