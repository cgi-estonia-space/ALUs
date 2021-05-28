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
#include "coherence_execute.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

#include <gdal_priv.h>
#include <boost/algorithm/string.hpp>

#include "algorithm_parameters.h"
#include "band_params.h"
#include "coh_tiles_generator.h"
#include "coh_window.h"
#include "coherence_calc.h"
#include "gdal_tile_reader.h"
#include "gdal_tile_writer.h"
#include "meta_data.h"
#include "pugixml_meta_data_reader.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "tf_algorithm_runner.h"
#include "s1tbx-commons/s_a_r_geocoding.h"


namespace {
constexpr std::string_view PARAMETER_ID_SRP_NUMBER_POINTS{"srp_number_points"};
constexpr std::string_view PARAMETER_ID_SRP_POLYNOMIAL_DEGREE{"srp_polynomial_degree"};
constexpr std::string_view PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE{"subtract_flat_earth_phase"};
constexpr std::string_view PARAMETER_ID_RG_WINDOW{"rg_window"};
constexpr std::string_view PARAMETER_ID_AZ_WINDOW{"az_window"};
constexpr std::string_view PARAMETER_ID_ORBIT_DEGREE{"orbit_degree"};
constexpr std::string_view PARAMETER_FETCH_TRANSFORM{"fetch_geo_transform"};
constexpr std::string_view PARAMETER_PER_PROCESS_GPU_MEMORY_FRACTION{"per_process_gpu_memory_fraction"};
constexpr std::string_view PARAMETER_INPUT_BAND_MAP{"in_band_map"};
constexpr size_t NUMBER_OF_INPUT_BANDS{4};
}  // namespace

namespace alus {

    [[nodiscard]] int CoherenceExecuter::Execute() {
        try {
            PrintProcessingParameters();

            if (aux_location_.empty()) {
                ExecuteSafe();
            } else {
                ExecuteBeamDimap();
            }
        } catch (const std::exception& e) {
            std::cerr << "Caught exception while running coherence operation - " << e.what() << std::endl;
            return 3;
        } catch (...) {
            std::cerr << "Caught unknown exception while running coherence operation." << std::endl;
            return 2;
        }

        return 0;
    }

    void CoherenceExecuter::ExecuteBeamDimap() {
        const auto dimap_dim_file = aux_location_.at(0);
        const auto dimap_data_folder = dimap_dim_file.substr(0, dimap_dim_file.length() - 4) + ".data";  // Strip ".dim"
        auto const FILE_NAME_IA = dimap_data_folder + "/tie_point_grids/incident_angle.img";
        std::vector<int> band_map_ia{1};
        int band_count_ia = 1;
        alus::GdalTileReader ia_data_reader{FILE_NAME_IA, band_map_ia, band_count_ia, false};
        // small dataset as single tile
        alus::Tile incidence_angle_data_set{ia_data_reader.GetBandXSize() - 1, ia_data_reader.GetBandYSize() - 1,
                                            ia_data_reader.GetBandXMin(), ia_data_reader.GetBandYMin()};
        ia_data_reader.ReadTile(incidence_angle_data_set);
        alus::snapengine::PugixmlMetaDataReader xml_reader{dimap_dim_file};
        auto master_root = xml_reader.Read(alus::snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);
        auto slave_root = xml_reader.Read(alus::snapengine::AbstractMetadata::SLAVE_METADATA_ROOT)->GetElements().at(0);

        alus::MetaData meta_master{&ia_data_reader, master_root, orbit_degree_};
        alus::MetaData meta_slave{&ia_data_reader, slave_root, orbit_degree_};

        // todo:need some better thought through logic to map inputs from gdal
        std::vector<int> band_map_out{1};
        int band_count_out = 1;

        std::unique_ptr<GdalTileReader> coh_data_reader{};
        if (input_dataset_.size() == 1 && input_dataset_.at(0) != nullptr) {
            coh_data_reader =
                std::make_unique<GdalTileReader>(input_dataset_.at(0), band_map_in_, band_map_in_.size(), fetch_transform_);
        } else {
            coh_data_reader =
                std::make_unique<GdalTileReader>(input_name_.at(0), band_map_in_, band_map_in_.size(), fetch_transform_);
        }

        std::unique_ptr<GdalTileWriter> coh_data_writer{};
        BandParams band_params{band_map_out,
                               band_count_out,
                               coh_data_reader->GetBandXSize(),
                               coh_data_reader->GetBandYSize(),
                               coh_data_reader->GetBandXMin(),
                               coh_data_reader->GetBandYMin()};
        if (output_driver_ != nullptr) {
            coh_data_writer = std::make_unique<GdalTileWriter>(
                output_driver_, band_params, coh_data_reader->GetGeoTransform(), coh_data_reader->GetDataProjection());
        } else {
            coh_data_writer = std::make_unique<GdalTileWriter>(
                output_name_, band_params, coh_data_reader->GetGeoTransform(), coh_data_reader->GetDataProjection());
        }

        alus::CohTilesGenerator tiles_generator{
            coh_data_reader->GetBandXSize(), coh_data_reader->GetBandYSize(), static_cast<int>(tile_width_), static_cast<int>(tile_height_),
            coherence_window_range_,         coherence_window_azimuth_};
        CohWindow coh_window{coherence_window_range_, coherence_window_azimuth_};
        alus::Coh coherence{srp_number_points_, srp_polynomial_degree_, subtract_flat_earth_phase_,
                            coh_window,         orbit_degree_,          meta_master,
                            meta_slave};

        // create session for tensorflow
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        auto options = tensorflow::SessionOptions();
        options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(per_process_fpu_memory_fraction_);
        options.config.mutable_gpu_options()->set_allow_growth(true);
        tensorflow::ClientSession session(root, options);

        // run the algorithm
        alus::TFAlgorithmRunner tf_algo_runner{
            coh_data_reader.get(), coh_data_writer.get(), &tiles_generator, &coherence, &session, &root};
        tf_algo_runner.Run();

        output_dataset_ = coh_data_writer->GetGdalDataset();
    }

    void CoherenceExecuter::ExecuteSafe() {
        const auto near_range_on_left = s1tbx::SARGeocoding::IsNearRangeOnLeft(
            main_product_->GetTiePointGrid("incident_angle"), main_product_->GetSceneRasterWidth());
        alus::MetaData meta_master{near_range_on_left,
                                   snapengine::AbstractMetadata::GetAbstractedMetadata(main_product_), orbit_degree_};
        alus::MetaData meta_slave{
            near_range_on_left, snapengine::AbstractMetadata::GetAbstractedMetadata(secondary_product_), orbit_degree_};

        // todo:need some better thought through logic to map inputs from gdal
        std::vector<int> band_map_out{1};
        int band_count_out = 1;

        std::unique_ptr<GdalTileReader> coh_data_reader =
            std::make_unique<GdalTileReader>(input_name_.at(0), band_map_in_, band_map_in_.size(), fetch_transform_);

        std::unique_ptr<GdalTileWriter> coh_data_writer{};
        BandParams band_params{band_map_out,
                               band_count_out,
                               coh_data_reader->GetBandXSize(),
                               coh_data_reader->GetBandYSize(),
                               coh_data_reader->GetBandXMin(),
                               coh_data_reader->GetBandYMin()};
        coh_data_writer = std::make_unique<GdalTileWriter>(
            output_name_, band_params, coh_data_reader->GetGeoTransform(), coh_data_reader->GetDataProjection());

        alus::CohTilesGenerator tiles_generator{coh_data_reader->GetBandXSize(), coh_data_reader->GetBandYSize(),
                                                static_cast<int>(tile_width_),   static_cast<int>(tile_height_),
                                                coherence_window_range_,         coherence_window_azimuth_};
        CohWindow coh_window{coherence_window_range_, coherence_window_azimuth_};
        alus::Coh coherence{srp_number_points_, srp_polynomial_degree_, subtract_flat_earth_phase_,
                            coh_window,         orbit_degree_,          meta_master,
                            meta_slave};

        // create session for tensorflow
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        auto options = tensorflow::SessionOptions();
        options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(per_process_fpu_memory_fraction_);
        options.config.mutable_gpu_options()->set_allow_growth(true);
        tensorflow::ClientSession session(root, options);

        // run the algorithm
        alus::TFAlgorithmRunner tf_algo_runner{
            coh_data_reader.get(), coh_data_writer.get(), &tiles_generator, &coherence, &session, &root};
        tf_algo_runner.Run();

    }

    void CoherenceExecuter::SetParameters(const app::AlgorithmParameters::Table& param_values) {
        if (param_values.count(std::string(PARAMETER_ID_ORBIT_DEGREE))) {
            orbit_degree_ = std::stoi(param_values.at(std::string(PARAMETER_ID_ORBIT_DEGREE)));
        }
        if (param_values.count(std::string(PARAMETER_ID_AZ_WINDOW))) {
            coherence_window_azimuth_ = std::stoi(param_values.at(std::string(PARAMETER_ID_AZ_WINDOW)));
        }
        if (param_values.count(std::string(PARAMETER_ID_RG_WINDOW))) {
            coherence_window_range_ = std::stoi(param_values.at(std::string(PARAMETER_ID_RG_WINDOW)));
        }
        if (param_values.count(std::string(PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE))) {
            const auto& value = param_values.at(std::string(PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE));
            if (boost::iequals(value, subtract_flat_earth_phase_ ? "false" : "true")) {
                subtract_flat_earth_phase_ = !subtract_flat_earth_phase_;
            }
        }
        if (param_values.count(std::string(PARAMETER_ID_SRP_POLYNOMIAL_DEGREE))) {
            srp_polynomial_degree_ = std::stoi(param_values.at(std::string(PARAMETER_ID_SRP_POLYNOMIAL_DEGREE)));
        }
        if (param_values.count(std::string(PARAMETER_ID_SRP_NUMBER_POINTS))) {
            srp_number_points_ = std::stoi(param_values.at(std::string(PARAMETER_ID_SRP_NUMBER_POINTS)));
        }
        if (param_values.count(std::string(PARAMETER_FETCH_TRANSFORM))) {
            const auto& value = param_values.at(std::string(PARAMETER_FETCH_TRANSFORM));
            if (boost::iequals(value, fetch_transform_ ? "false" : "true")) {
                fetch_transform_ = !fetch_transform_;
            }
        }

        if (const auto mem_fraction = param_values.find(std::string(PARAMETER_PER_PROCESS_GPU_MEMORY_FRACTION));
            mem_fraction != param_values.end()) {
            const auto value = std::stod(mem_fraction->second);
            if (std::isless(value, 0.0) || std::isgreater(value, 1.0) || std::isnan(value)) {
                std::cerr << PARAMETER_PER_PROCESS_GPU_MEMORY_FRACTION << " value (" << value
                          << ") out of range [0.0...1.0]" << std::endl;
            } else {
                per_process_fpu_memory_fraction_ = value;
            }
        }

        const auto band_map_in = param_values.find(std::string(PARAMETER_INPUT_BAND_MAP));
        if (band_map_in != param_values.end()) {
            const auto value = band_map_in->second;
            if (value.length() != 4) {
                std::cerr << PARAMETER_INPUT_BAND_MAP << " value (" << value
                          << ") not in correct format - expecting 4 integers." << std::endl;
            } else {
                for (size_t i = 0; i < NUMBER_OF_INPUT_BANDS; i++) {
                    band_map_in_.at(i) = boost::lexical_cast<int>(value.at(i));
                }
            }
        }
    }

    [[nodiscard]] std::string CoherenceExecuter::GetArgumentsHelp() const {
        std::stringstream help_stream;
        help_stream << "Coherence configuration options:" << std::endl
                    << PARAMETER_ID_SRP_NUMBER_POINTS << " - unsigned integer (default:" << srp_number_points_ << ")"
                    << std::endl
                    << PARAMETER_ID_SRP_POLYNOMIAL_DEGREE << " - unsigned integer (default:" << srp_polynomial_degree_
                    << ")" << std::endl
                    << PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE
                    << " - true/false (default:" << (subtract_flat_earth_phase_ ? "true" : "false") << ")" << std::endl
                    << PARAMETER_ID_RG_WINDOW << " - range window size in pixels (default:" << coherence_window_range_
                    << ")" << std::endl
                    << PARAMETER_ID_AZ_WINDOW
                    << " - azimuth window size in pixels (default:" << coherence_window_azimuth_ << ")" << std::endl
                    << PARAMETER_ID_ORBIT_DEGREE << " - unsigned integer (default:" << orbit_degree_ << ")" << std::endl
                    << PARAMETER_FETCH_TRANSFORM << " - true/false (default:" << (fetch_transform_ ? "true" : "false")
                    << ")" << std::endl
                    << PARAMETER_INPUT_BAND_MAP
                    << " - 4 digit band map where each constitutes bands to read from. Example '1367' - to read from "
                       "band 1, 3, 6 and 7. (default:'1234')"
                    << std::endl
                    << PARAMETER_PER_PROCESS_GPU_MEMORY_FRACTION
                    << " - amount of total GPU memory to use, value between [0.0...1.0] (default:"
                    << per_process_fpu_memory_fraction_ << ")" << std::endl;

        return help_stream.str();
    }

    void CoherenceExecuter::PrintProcessingParameters() const {
        std::cout << "Coherence processing parameters:" << std::endl
                  << PARAMETER_ID_SRP_NUMBER_POINTS << " " << srp_number_points_ << std::endl
                  << PARAMETER_ID_SRP_POLYNOMIAL_DEGREE << " " << srp_polynomial_degree_ << std::endl
                  << PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE << " " << (subtract_flat_earth_phase_ ? "true" : "false")
                  << std::endl
                  << PARAMETER_ID_RG_WINDOW << " " << coherence_window_range_ << std::endl
                  << PARAMETER_ID_AZ_WINDOW << " " << coherence_window_azimuth_ << std::endl
                  << PARAMETER_ID_ORBIT_DEGREE << " " << orbit_degree_ << std::endl
                  << PARAMETER_FETCH_TRANSFORM << " " << (fetch_transform_ ? "true" : "false") << std::endl
                  << PARAMETER_PER_PROCESS_GPU_MEMORY_FRACTION << " " << per_process_fpu_memory_fraction_ << std::endl
                  << PARAMETER_INPUT_BAND_MAP << " ";

        for (const auto& band : band_map_in_) {
            std::cout << band << " ";
        }
        std::cout << std::endl;
    }
}  // namespace alus
