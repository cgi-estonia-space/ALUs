#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <boost/program_options.hpp>

#include "alg_bond.h"
#include "algorithm_parameters.h"
#include "dem_assistant.h"

namespace po = boost::program_options;

int main(int argc, char* argv[]) {

    std::string alg_to_run{};
    std::string input_file{};
    std::string output_path{};
    std::string aux_location{};
    size_t tile_width{};
    size_t tile_height{};
    std::string alg_params{};
    std::string params_file_path{};
    std::vector<std::string> dem_files_param{};

    po::options_description options("Alus options");
    options.add_options()("help,h", "Print help")(
        "alg_name", po::value<std::string>(&alg_to_run), "Specify algorithm to run")(
        "alg_help", "Print algorithm configurable parameters")(
        "input,i", po::value<std::string>(&input_file), "Input dataset path/name GeoTIFF files only.")(
        "output,o", po::value<std::string>(&output_path), "Output dataset path/name")(
        "tile_width,x", po::value<size_t>(&tile_width)->default_value(500), "Tile width.")(
        "tile_height,y", po::value<size_t>(&tile_height)->default_value(500), "Tile height.")(
        "aux", po::value<std::string>(&aux_location), "Auxiliary file locations "
        "(metadata, incident angle, etc).")
        ("parameters,p", po::value<std::string>(&alg_params)->default_value(""),
                        "Algorithm specific configuration. Must be supplied as key=value "
                        "pairs "
                        "separated by comma','.\n"
                        "Example: 'algorithm1:points=14,height=84;"
                        "algorithm2:subtract=true;algorithm3:key=value,interpolation=bilinear'\nParameters specified "
         "here will overrule ones in the configuration file supplied as '--parameters_file' option.")
        ("parameters_file", po::value<std::string>(&params_file_path)->default_value(""), "Algorithm specific "
             "configurations file path. File contents shall follow same syntax as '--parameters' option.")(
        "dem", po::value<std::vector<std::string>>(&dem_files_param), "Dem files with full path. Can be specified"
                "multiple times for multiple DEM files or a space separated files in a single argument. Only SRTM3 is"
                "supported.")(
        "list_algs,l", "Print available algorithms");

    void* alg_lib{nullptr};

    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, options), vm);
        po::notify(vm);

        if (vm.count("help") || argc == 1 || vm.count("alg_name") == 0) {
            std::cout << options << std::endl;
            return 0;
        }

        std::cout << "Algorithm to run: " << alg_to_run << std::endl;

        auto const lib_file_path = "./lib" + alg_to_run + ".so";
        alg_lib = dlopen(lib_file_path.c_str(), RTLD_LAZY);
        if (alg_lib == nullptr) {
            std::cerr << "Cannot load library " << dlerror() << std::endl;
            return 1;
        }

        auto tc_creator = (AlgorithmBondEntry)dlsym(alg_lib, "CreateAlgorithm");
        if (tc_creator == nullptr) {
            std::cerr << "Cannot load symbol - " << dlerror() << std::endl;
            dlclose(alg_lib);
            return 3;
        }

        auto tc_deleter = (AlgorithmBondEnd)dlsym(alg_lib, "DeleteAlgorithm");
        if (tc_deleter == nullptr) {
            std::cerr << "Cannot load symbol - " << dlerror() << std::endl;
            dlclose(alg_lib);
            return 4;
        }

        {
            auto alg = std::unique_ptr<alus::AlgBond, AlgorithmBondEnd>(tc_creator(), tc_deleter);

            if (vm.count("alg_help")) {
                std::cout << alg->GetArgumentsHelp();
            }
            else {

                std::shared_ptr<alus::app::DemAssistant> dem_assistant{};
                if (!dem_files_param.empty()) {
                    std::cout << "Processing DEM files and moving to GPU." << std::endl;
                    dem_assistant =
                        alus::app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(std::move(dem_files_param));
                }

                alg->SetInputs(input_file, aux_location);
                alg->SetTileSize(tile_width, tile_height);
                alg->SetOutputFilename(output_path);

                alus::app::AlgorithmParameters::AlgParamTables command_line_parameters;
                if (!alg_params.empty()) {
                    command_line_parameters = alus::app::AlgorithmParameters::TryCreateFrom(alg_params);
                }

                alus::app::AlgorithmParameters::AlgParamTables file_conf_parameters;
                if (!params_file_path.empty()) {
                    file_conf_parameters = alus::app::AlgorithmParameters::TryCreateFromFile(params_file_path);
                }

                std::string warnings{};
                const auto& merged_parameters = alus::app::AlgorithmParameters::MergeAndWarn(file_conf_parameters,
                                                                                             command_line_parameters,
                                                                                             warnings);
                std::cout << warnings << std::endl;
                if (merged_parameters.count(alg_to_run)) {
                    alg->SetParameters(merged_parameters.at(alg_to_run));
                }

                if (dem_assistant != nullptr) {
                    alg->SetSrtm3Buffers(dem_assistant->GetSrtm3ValuesOnGpu());
                }

                alg->Execute();
            }
        }

        dlclose(alg_lib); // TODO: Guard this!

    } catch (const po::error& e) {
        std::cerr << e.what() << std::endl;
        std::cout << options << std::endl;
        return 1; //TODO: Create constants for exit codes.
    } catch (const std::fstream::failure& e) {
        std::cerr << "Exception opening/reading file - " << e.what() << std::endl;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "An exception was caught - " << e.what() << std::endl
                  << "Exciting" << std::endl;
        return 3;
    } catch (...) {
        std::cerr << "Exception of unknown type was caught." << std::endl << "ERRNO:" << std::endl
                  << strerror(errno) << std::endl;
        return 4;
    }

    return 0;
}
