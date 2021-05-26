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

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <boost/program_options.hpp>

#include "alg_load.h"
#include "algorithm_parameters.h"
#include "command_line_options.h"
#include "dem_assistant.h"

#include "../../VERSION"

namespace po = boost::program_options;

namespace {
void PrintHelp(const std::string& options_help) {
    std::cout << "ALUS - EO processing on steroids" << std::endl;
    std::cout << "Version " << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
    std::cout << std::endl;
    std::cout << options_help << std::endl;
    std::cout << std::endl;
    std::cout << "https://bitbucket.org/cgi-ee-space/alus" << std::endl;
}
}  // namespace

int main(int argc, const char* argv[]) {

    std::string help_string{};
    int alg_execute_status{};
    try {
        alus::app::CommandLineOptions options{};
        help_string = options.GetHelp();
        options.ParseArgs(argc, argv);

        if (options.DoRequireHelp()) {
            PrintHelp(help_string);
            return 0;
        }

        const auto alg_to_run = options.GetSelectedAlgorithm();
        std::cout << "Algorithm to run: " << alg_to_run << std::endl;
        const alus::AlgorithmLoadGuard alg_guard{alg_to_run};

        alus::app::AlgorithmParameters::AlgParamTables command_line_parameters;
        const auto alg_params = options.GetAlgorithmParameters();
        if (!alg_params.empty()) {
            command_line_parameters = alus::app::AlgorithmParameters::TryCreateFrom(alg_params);
        }

        alus::app::AlgorithmParameters::AlgParamTables file_conf_parameters;
        const auto alg_params_file = options.GetAlgorithmParametersFile();
        if (!alg_params_file.empty()) {
            file_conf_parameters = alus::app::AlgorithmParameters::TryCreateFromFile(alg_params_file);
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

        if (options.DoRequireAlgorithmHelp()) {
            std::cout << alg_guard.GetInstanceHandle()->GetArgumentsHelp();
        } else {
            std::shared_ptr<alus::app::DemAssistant> dem_assistant{};
            const auto dem_files_param = options.GetDemFiles();
            if (!dem_files_param.empty()) {
                std::cout << "Processing DEM files and creating EGM96." << std::endl;
                dem_assistant = alus::app::DemAssistant::CreateFormattedSrtm3TilesOnGpuFrom(std::move(dem_files_param));
            }

            alg_guard.GetInstanceHandle()->SetInputFilenames(options.GetInputDatasets(), options.GetAux());
            alg_guard.GetInstanceHandle()->SetTileSize(options.GetTileWidth(), options.GetTileHeight());
            alg_guard.GetInstanceHandle()->SetOutputFilename(options.GetOutputStem());

            if (dem_assistant != nullptr) {
                alg_guard.GetInstanceHandle()->SetSrtm3Manager(dem_assistant->GetSrtm3Manager());
                alg_guard.GetInstanceHandle()->SetEgm96Manager(dem_assistant->GetEgm96Manager());
            }

            alg_execute_status = alg_guard.GetInstanceHandle()->Execute();
        }
    } catch (const po::error& e) {
        std::cerr << e.what() << std::endl;
        PrintHelp(help_string);
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
