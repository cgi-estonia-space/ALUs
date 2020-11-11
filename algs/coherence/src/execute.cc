#include <iostream>
#include <string_view>

#include <boost/algorithm/string.hpp>

#include "algorithm_parameters.h"
#include "alg_bond.h"
#include "coh_tiles_generator.h"
#include "coherence_calc.h"
#include "gdal_tile_reader.h"
#include "gdal_tile_writer.h"
#include "meta_data.h"
#include "meta_data_node_names.h"
#include "pugixml_meta_data_reader.h"
#include "tf_algorithm_runner.h"

namespace {
    constexpr std::string_view PARAMETER_ID_SRP_NUMBER_POINTS{"srp_number_points"};
    constexpr std::string_view PARAMETER_ID_SRP_POLYNOMIAL_DEGREE{"srp_polynomial_degree"};
    constexpr std::string_view PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE{"subtract_flat_earth_phase"};
    constexpr std::string_view PARAMETER_ID_RG_WINDOW{"rg_window"};
    constexpr std::string_view PARAMETER_ID_AZ_WINDOW{"az_window"};
    constexpr std::string_view PARAMETER_ID_ORBIT_DEGREE{"orbit_degree"};
}

namespace alus {
class CoherenceExecuter final : public AlgBond {
   public:
    CoherenceExecuter() { std::cout << __FUNCTION__ << std::endl; };

    int Execute() override {
        PrintProcessingParameters();

        auto const FILE_NAME_IA = aux_location_ + "/tie_point_grids/incident_angle.img";
        std::vector<int> band_map_ia{1};
        int band_count_ia = 1;
        alus::GdalTileReader ia_data_reader{FILE_NAME_IA, band_map_ia, band_count_ia, false};
        // small dataset as single tile
        alus::Tile incidence_angle_data_set{ia_data_reader.GetBandXSize() - 1,
                                            ia_data_reader.GetBandYSize() - 1,
                                            ia_data_reader.GetBandXMin(),
                                            ia_data_reader.GetBandYMin()};
        ia_data_reader.ReadTile(incidence_angle_data_set);
        const auto metadata_file = aux_location_.substr(0, aux_location_.length() - 5); // Strip ".data"
        alus::snapengine::PugixmlMetaDataReader xml_reader{metadata_file + ".dim"};
        auto master_root = xml_reader.Read(alus::snapengine::MetaDataNodeNames::ABSTRACT_METADATA_ROOT);
        auto slave_root =
            xml_reader.Read(alus::snapengine::MetaDataNodeNames::SLAVE_METADATA_ROOT)->GetElements().at(0);

        alus::MetaData meta_master{&ia_data_reader, master_root, orbit_degree_};
        alus::MetaData meta_slave{&ia_data_reader, slave_root, orbit_degree_};

        // todo:check if bandmap works correctly (e.g if input has 8 bands and we use 1,2,5,6)
        // todo:need some better thought through logic to map inputs from gdal
        std::vector<int> band_map{1, 2, 3, 4};
        std::vector<int> band_map_out{1};
        // might want to take count from coherence?
        int band_count_in = 4;
        int band_count_out = 1;

        alus::GdalTileReader coh_data_reader{input_name_, band_map, band_count_in, true};
        alus::GdalTileWriter coh_data_writer{output_name_,
                                             band_map_out,
                                             band_count_out,
                                             coh_data_reader.GetBandXSize(),
                                             coh_data_reader.GetBandYSize(),
                                             coh_data_reader.GetBandXMin(),
                                             coh_data_reader.GetBandYMin(),
                                             coh_data_reader.GetGeoTransform(),
                                             coh_data_reader.GetDataProjection()};
        alus::CohTilesGenerator tiles_generator{coh_data_reader.GetBandXSize(),
                                                coh_data_reader.GetBandYSize(),
                                                tile_width_,
                                                tile_height_,
                                                coherence_window_range_,
                                                coherence_window_azimuth_};
        alus::Coh coherence{srp_number_points_,
                            srp_polynomial_degree_,
                            subtract_flat_earth_phase_,
                            coherence_window_range_,
                            coherence_window_azimuth_,
                            tile_width_,
                            tile_height_,
                            orbit_degree_,
                            meta_master,
                            meta_slave};

        // create session for tensorflow
        tensorflow::Scope root = tensorflow::Scope::NewRootScope();
        auto options = tensorflow::SessionOptions();
        options.config.mutable_gpu_options()->set_allow_growth(true);
        tensorflow::ClientSession session(root, options);

        // run the algorithm
        alus::TFAlgorithmRunner tf_algo_runner{
            &coh_data_reader, &coh_data_writer, &tiles_generator, &coherence, &session, &root};
        tf_algo_runner.Run();

        return 0;
    }
    [[nodiscard]] RasterDimension CalculateInputTileFrom(RasterDimension output) const override { return output; }

    void SetInputs(const std::string& input_dataset, const std::string& metadata_path) override {
        input_name_ = input_dataset;
        aux_location_ = metadata_path;
    }

    void SetParameters(const app::AlgorithmParameters::Table& param_values) override {
        if (param_values.count(std::string(PARAMETER_ID_ORBIT_DEGREE))) {
            orbit_degree_ = std::stoul(param_values.at(std::string(PARAMETER_ID_ORBIT_DEGREE)));
        }
        if (param_values.count(std::string(PARAMETER_ID_AZ_WINDOW))) {
            coherence_window_azimuth_ = std::stoul(param_values.at(std::string(PARAMETER_ID_AZ_WINDOW)));
        }
        if (param_values.count(std::string(PARAMETER_ID_RG_WINDOW))) {
            coherence_window_range_ = std::stoul(param_values.at(std::string(PARAMETER_ID_RG_WINDOW)));
        }
        if (param_values.count(std::string(PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE))) {
            const auto& value = param_values.at(std::string(PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE));
            if (boost::iequals(value, subtract_flat_earth_phase_ ? "false" : "true")) {
                subtract_flat_earth_phase_ = !subtract_flat_earth_phase_;
            }
        }
        if (param_values.count(std::string(PARAMETER_ID_SRP_POLYNOMIAL_DEGREE))) {
            srp_polynomial_degree_ = std::stoul(param_values.at(std::string(PARAMETER_ID_SRP_POLYNOMIAL_DEGREE)));
        }
        if (param_values.count(std::string(PARAMETER_ID_SRP_NUMBER_POINTS))) {
            srp_number_points_ = std::stoul(param_values.at(std::string(PARAMETER_ID_SRP_NUMBER_POINTS)));
        }
    }

    void SetTileSize(size_t width, size_t height) override {
        tile_width_ = width;
        tile_height_ = height;
    }

    void SetOutputFilename(const std::string& output_name) override { output_name_ = output_name; }

    [[nodiscard]] std::string GetArgumentsHelp() const override {
        std::stringstream help_stream;
        help_stream << "Coherence configuration options:" << std::endl
                    << PARAMETER_ID_SRP_NUMBER_POINTS << " - unsigned integer (default:" << srp_number_points_ << ")"
                    << std::endl
                    << PARAMETER_ID_SRP_POLYNOMIAL_DEGREE << " - unsigned integer (default:" << srp_polynomial_degree_
                    << ")" << std::endl
                    << PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE
                    << " - true/false (default:"
                    << (subtract_flat_earth_phase_ ? "true" : "false") << ")" << std::endl
                    << PARAMETER_ID_RG_WINDOW << " - range window size in pixels (default:" << coherence_window_range_
                    << ")"
                    << std::endl
                    << PARAMETER_ID_AZ_WINDOW << " - azimuth window size in pixels (default:" << coherence_window_azimuth_ << ")"
                    << std::endl
                    << PARAMETER_ID_ORBIT_DEGREE << " - unsigned integer (default:" << orbit_degree_ << ")" << std::endl;

        return help_stream.str();
    }

    ~CoherenceExecuter() override { std::cout << __FUNCTION__ << std::endl; };

   private:

    void PrintProcessingParameters() const override {
        std::cout << "Coherence processing parameters:" << std::endl
            << PARAMETER_ID_SRP_NUMBER_POINTS << " " << srp_number_points_ << std::endl
            << PARAMETER_ID_SRP_POLYNOMIAL_DEGREE << " " << srp_polynomial_degree_ << std::endl
            << PARAMETER_ID_SUBTRACT_FLAT_EARTH_PHASE << " " << (subtract_flat_earth_phase_ ? "true" : "false") << std::endl
            << PARAMETER_ID_RG_WINDOW << " " << coherence_window_range_ << std::endl
            << PARAMETER_ID_AZ_WINDOW << " " << coherence_window_azimuth_ << std::endl
            << PARAMETER_ID_ORBIT_DEGREE << " " << orbit_degree_ << std::endl;
    }

    std::string input_name_{};
    std::string output_name_{};
    std::string aux_location_{};
    size_t tile_width_{};
    size_t tile_height_{};
    int srp_number_points_{501};
    int srp_polynomial_degree_{5};
    bool subtract_flat_earth_phase_{true};
    int coherence_window_range_{15};
    int coherence_window_azimuth_{3};
    // orbit interpolation degree
    int orbit_degree_{3};

};
}  // namespace alus

extern "C" {
alus::AlgBond* CreateAlgorithm() { return new alus::CoherenceExecuter(); }

void DeleteAlgorithm(alus::AlgBond* instance) { delete (alus::CoherenceExecuter*)instance; }
}
