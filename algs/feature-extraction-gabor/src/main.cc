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

#include <boost/program_options.hpp>
#include <iostream>
#include <string>

#include "algorithm_exception.h"
#include "alus_log.h"
#include "app_error_code.h"
#include "command_line_options.h"
#include "constants.h"
#include "cuda_device_init.h"
#include "execute.h"

int main(int argc, char* argv[]) {
    std::string help_string{};
    try {
        alus::common::log::Initialize();
        auto cuda_init = alus::cuda::CudaInit();

        alus::featurextractiongabor::Arguments args;
        help_string = args.GetHelp();
        args.ParseArgs(std::vector<char*>(argv, argv + argc));

        if (args.IsHelpRequested()) {
            std::cout << alus::app::GenerateHelpMessage(alus::featurextractiongabor::ALG_NAME, help_string);
            return alus::app::errorcode::ALG_SUCCESS;
        }

        args.Check();
        alus::common::log::SetLevel(args.GetLogLevel());

        alus::featurextractiongabor::Execute exe(args.GetOrientationCount(), args.GetFrequencyCount(),
                                                 args.GetPatchSize(), args.GetInput(), cuda_init,
                                                 args.GetGpuMemoryPercentage());

        LOGI << "Generating patches";
        exe.GenerateInputs();
        if (args.IsConvolutionInputsRequested()) {
            exe.SaveGaborInputsTo(args.GetConvolutionInputsStorePath());
        }
        LOGI << "Calculating results";
        exe.CalculateGabor();
        exe.SaveResultsTo(args.GetOutputPath());

    } catch (const boost::program_options::error& e) {
        LOGE << e.what();
        std::cout << alus::app::GenerateHelpMessage(alus::featurextractiongabor::ALG_NAME, help_string);
        return alus::app::errorcode::ARGUMENT_PARSE;
    } catch (const alus::common::AlgorithmException& e) {
        LOGE << e.what();
        return alus::app::errorcode::ALGORITHM_EXCEPTION;
    } catch (const std::exception& e) {
        LOGE << e.what();
        LOGE << "Exiting because of an error." << std::endl;
        return alus::app::errorcode::GENERAL_EXCEPTION;
    } catch (...) {
        LOGE << "Caught an unknown exception." << std::endl;
        return alus::app::errorcode::UNKNOWN_EXCEPTION;
    }

    return alus::app::errorcode::ALG_SUCCESS;
}