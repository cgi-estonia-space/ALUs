#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <boost/program_options.hpp>

#include "alg_load.h"
#include "algorithm_parameters.h"
#include "dem_assistant.h"

#include "../../VERSION"

namespace po = boost::program_options;

namespace {
void PrintHelp(const po::options_description& options_help) {
    std::cout << "ALUS - EO processing on steroids" << std::endl;
    std::cout << "Version " << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
    std::cout << std::endl;
    std::cout << options_help << std::endl;
    std::cout << std::endl;
    std::cout << "https://bitbucket.org/cgi-ee-space/alus" << std::endl;
}
}  // namespace

int main(int argc, char* argv[]) {
    std::string alg_to_run{};
    std::vector<std::string> input_files{};
    std::string output_path{};
    std::vector<std::string> aux_locations{};
    size_t tile_width{};
    size_t tile_height{};
    std::string alg_params{};
    std::string params_file_path{};
    std::vector<std::string> dem_files_param{};

    po::options_description visible_options("Arguments");
    po::options_description hidden_options("Hidden arguments");
    // clang-format off
    visible_options.add_options()("help,h", "Print help")(
        "alg_name", po::value<std::string>(&alg_to_run), "Specify algorithm to run")(
        "alg_help", "Print algorithm configurable parameters")(
        "input,i", po::value<std::vector<std::string>>(&input_files), "Input dataset path.")(
        "output,o", po::value<std::string>(&output_path), "Output dataset location")(
        "tile_width,x", po::value<size_t>(&tile_width)->default_value(1000), "Tile width.")(
        "tile_height,y", po::value<size_t>(&tile_height)->default_value(1000), "Tile height.")
        ("parameters,p", po::value<std::string>(&alg_params)->default_value(""),
                        "Algorithm specific configuration. Must be supplied as key=value "
                        "pairs "
                        "separated by comma','.\n"
                        "Example 1: 'orbit_degree=3,rg_window=15'\n"
                        "Example 2: 'coherence:az_window=3;backgeocoding:master=...'\nParameters specified "
         "here will overrule ones in the configuration file supplied as '--parameters_file' option.")
        ("parameters_file", po::value<std::string>(&params_file_path)->default_value(""), "Algorithm specific "
             "configurations file path. File contents shall follow same syntax as '--parameters' option.")(
        "dem", po::value<std::vector<std::string>>(&dem_files_param), "Dem files with full path. Can be specified "
                "multiple times for multiple DEM files or a space separated files in a single argument. Only SRTM3 is "
                "supported.")(
        "list_algs,l", "Print available algorithms");

    hidden_options.add_options()(
        "aux", po::value<std::vector<std::string>>(&aux_locations), "Auxiliary/metadata .dim file"
                                                                    "(metadata, incident angle, etc).");
    // clang-format on
    po::options_description all_options("All arguments");
    all_options.add(visible_options).add(hidden_options);

    int alg_execute_status{};
    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all_options), vm);
        po::notify(vm);

        if (vm.count("help") || argc == 1 || vm.count("alg_name") == 0) {
            PrintHelp(visible_options);
            return 0;
        }

        std::cout << "Algorithm to run: " << alg_to_run << std::endl;
        const alus::AlgorithmLoadGuard alg_guard{alg_to_run};

        alus::app::AlgorithmParameters::AlgParamTables command_line_parameters;
        if (!alg_params.empty()) {
            command_line_parameters = alus::app::AlgorithmParameters::TryCreateFrom(alg_params);
        }

        alus::app::AlgorithmParameters::AlgParamTables file_conf_parameters;
        if (!params_file_path.empty()) {
            file_conf_parameters = alus::app::AlgorithmParameters::TryCreateFromFile(params_file_path);
        }

        std::string warnings{};
        const auto& merged_parameters =
            alus::app::AlgorithmParameters::MergeAndWarn(file_conf_parameters, command_line_parameters, warnings);

        std::cout << warnings << std::endl;
        if (merged_parameters.count(alg_to_run)) {
            alg_guard.GetInstanceHandle()->SetParameters(merged_parameters.at(alg_to_run));
        } else if (!merged_parameters.empty() &&
                   merged_parameters.count("")) {  // Parameters without algorithm specification
            alg_guard.GetInstanceHandle()->SetParameters(merged_parameters.at(""));
        }

        if (vm.count("alg_help")) {
            std::cout << alg_guard.GetInstanceHandle()->GetArgumentsHelp();
        } else {
            std::shared_ptr<alus::app::DemAssistant> dem_assistant{};
            if (!dem_files_param.empty()) {
                std::cout << "Processing DEM files and creating EGM96." << std::endl;
                dem_assistant = alus::app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(std::move(dem_files_param));
            }

            alg_guard.GetInstanceHandle()->SetInputFilenames(input_files, aux_locations);
            alg_guard.GetInstanceHandle()->SetTileSize(tile_width, tile_height);
            alg_guard.GetInstanceHandle()->SetOutputFilename(output_path);

            if (dem_assistant != nullptr) {
                alg_guard.GetInstanceHandle()->SetSrtm3Manager(dem_assistant->GetSrtm3Manager());
                alg_guard.GetInstanceHandle()->SetEgm96Manager(dem_assistant->GetEgm96Manager());
            }

            alg_execute_status = alg_guard.GetInstanceHandle()->Execute();
        }
    } catch (const po::error& e) {
        std::cerr << e.what() << std::endl;
        PrintHelp(visible_options);
        return 1;  // TODO: Create constants for exit codes.
    } catch (const std::fstream::failure& e) {
        std::cerr << "Exception opening/reading file - " << e.what() << std::endl;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "An exception was caught - " << e.what() << std::endl << "Exciting" << std::endl;
        return 3;
    } catch (...) {
        std::cerr << "Exception of unknown type was caught." << std::endl
                  << "ERRNO:" << std::endl
                  << strerror(errno) << std::endl;
        return 4;
    }

    return alg_execute_status;
}
