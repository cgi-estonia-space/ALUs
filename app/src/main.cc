#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <string>

#include <boost/program_options.hpp>

#include "alg_bond.h"

namespace po = boost::program_options;

int main(int argc, char* argv[]) {

    std::string alg_to_run{};
    std::string input_file{};
    std::string output_path{};
    std::string aux_location{};
    size_t tile_width{};
    size_t tile_height{};

    po::options_description options("Alus options");
    options.add_options()
        ("help,h", "Print help")
        ("alg_name", po::value<std::string>(&alg_to_run)->required(), "Specify algorithm to run")
        ("input,i", po::value<std::string>(&input_file)->required(), "Input dataset path/name GeoTIFF files only.")
        ("output,o", po::value<std::string>(&output_path)->required(), "Output dataset path/name")
        ("tile_width,x", po::value<size_t>(&tile_width)->default_value(500), "Tile width.")
        ("tile_height,y", po::value<size_t>(&tile_height)->default_value(500), "Tile height.")
        ("aux", po::value<std::string>(&aux_location)->required(), "Auxiliary file locations "
                                                                   "(metadata, incident angle, etc).")
        ("conf,c", po::value<std::vector<std::string>>(), "Algorithm specific configuration. Must be "
                                                          "supplied as key=value pairs separated by whitespace.")
        ("list_algs,l", "Print available algorithms")
        ;

    void* alg_lib{nullptr};

    try {
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, options), vm);
        po::notify(vm);

        if (vm.count("help") || vm.empty()) {
            std::cout << options << std::endl;
            return 0;
        }

        auto const alg_name = vm["alg_name"].as<std::string>();
        std::cout << "Algorithm to run: " << alg_name << std::endl;

        auto const lib_file_path = "./lib" + alg_name + ".so";
        alg_lib = dlopen(lib_file_path.c_str(), RTLD_LAZY);
        if (alg_lib == nullptr) {
            std::cout << "Cannot load library " << dlerror() << std::endl;
            return 1;
        }

        auto tc_creator = (AlgorithmBondEntry)dlsym(alg_lib, "CreateAlgorithm");
        if (tc_creator == nullptr) {
            std::cout << "Cannot load symbol - " << dlerror() << std::endl;
            dlclose(alg_lib);
            return 3;
        }

        auto tc_deleter = (AlgorithmBondEnd)dlsym(alg_lib, "DeleteAlgorithm");
        if (tc_deleter == nullptr) {
            std::cout << "Cannot load symbol - " << dlerror() << std::endl;
            dlclose(alg_lib);
            return 4;
        }

        {
            auto alg = std::unique_ptr<alus::AlgBond, AlgorithmBondEnd>(tc_creator(), tc_deleter);
            alg->SetInputs(input_file, aux_location);
            alg->SetTileSize(tile_width, tile_height);
            alg->SetOutputFilename(output_path);
            alg->Execute();
        }

        dlclose(alg_lib);

    } catch (const po::error& e) {
        std::cout << e.what() << std::endl;
        std::cout << options << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cout << "An exception was caught - " << e.what() << std::endl;
        std::cout << "Exciting" << std::endl;
        return 2;
    } catch (...) {
        std::cout << "Exception of unknown type was caught." << std::endl << "ERRNO:" << std::endl;
        std::cout << strerror(errno) << std::endl;
        return 3;
    }

    return 0;
}
