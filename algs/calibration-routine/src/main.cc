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

#include <filesystem>
#include <iostream>
#include <string_view>

#include <boost/program_options.hpp>

#include "algorithm_exception.h"
#include "alus_log.h"
#include "app_error_code.h"
#include "app_utils.h"
#include "cli_args.h"
#include "constants.h"
#include "cuda_device_init.h"
#include "cuda_util.h"
#include "execute.h"
#include "gdal_util.h"

namespace {

std::string ConditionAoi(const std::optional<std::string>& aoi_arg) {
    if (!aoi_arg.has_value()) {
        return "";
    }

    const auto arg_val = aoi_arg.value();
    if (!std::filesystem::exists(arg_val)) {
        return arg_val;
    }

    return alus::ConvertToWkt(arg_val);
}

alus::calibrationroutine::Execute::Parameters AssembleParameters(const alus::calibrationroutine::Arguments& args) {
    alus::calibrationroutine::Execute::Parameters params;
    params.input = args.GetInput();
    params.output = args.GetOutput();
    params.wif = args.DoSaveIntermediateResults();
    params.subswath = args.GetSubswath();
    params.polarisation = args.GetPolarisation();
    params.calibration_type = args.GetCalibrationType();
    params.aoi = ConditionAoi(args.GetAoi());
    params.burst_first_index = alus::calibrationroutine::INVALID_BURST_INDEX;
    params.burst_last_index = alus::calibrationroutine::INVALID_BURST_INDEX;
    if (const auto bursts = args.GetBurstIndexes(); bursts.has_value()) {
        params.burst_first_index = std::get<0>(bursts.value());
        params.burst_last_index = std::get<1>(bursts.value());
    }

    return params;
}

template <typename T>
void ExceptionMessagePrint(const T& e) {
    LOGE << "Caught an exception" << std::endl << e.what();
    LOGE << "Exiting.";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string args_help{};
    try {
        alus::common::log::Initialize();
        auto cuda_init = alus::cuda::CudaInit();

        alus::calibrationroutine::Arguments args;
        args_help = args.GetHelp();
        args.Parse(std::vector<char*>(argv, argv + argc));

        if (args.IsHelpRequested()) {
            std::cout << alus::app::GenerateHelpMessage(alus::calibrationroutine::ALG_NAME, args_help);
            return alus::app::errorcode::ALG_SUCCESS;
        }

        args.Check();
        alus::common::log::SetLevel(args.GetLogLevel());

        alus::calibrationroutine::Execute exe(AssembleParameters(args), args.GetDemFiles());
        exe.Run(cuda_init, args.GetGpuMemoryPercentage());

    } catch (const boost::program_options::error& e) {
        std::cout << alus::app::GenerateHelpMessage(alus::calibrationroutine::ALG_NAME, args_help);
        ExceptionMessagePrint(e);
        return alus::app::errorcode::ARGUMENT_PARSE;
    } catch (const alus::CudaErrorException& e) {
        ExceptionMessagePrint(e);
        return alus::app::errorcode::GPU_DEVICE_ERROR;
    } catch (const alus::common::AlgorithmException& e) {
        ExceptionMessagePrint(e);
        return alus::app::errorcode::ALGORITHM_EXCEPTION;
    } catch (const std::exception& e) {
        ExceptionMessagePrint(e);
        return alus::app::errorcode::GENERAL_EXCEPTION;
    } catch (...) {
        LOGE << "Caught an unknown exception.";
        return alus::app::errorcode::UNKNOWN_EXCEPTION;
    }

    return alus::app::errorcode::ALG_SUCCESS;
}