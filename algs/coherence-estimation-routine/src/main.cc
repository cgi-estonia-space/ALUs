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

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

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

namespace {

constexpr bool RUN_TIMELINE = COHERENCE_TIMELINE;

alus::coherenceestimationroutine::Execute::Parameters AssembleParameters(
    const alus::coherenceestimationroutine::Arguments& args) {
    alus::coherenceestimationroutine::Execute::Parameters params;

    params.input_reference = args.GetInputReference();
    params.input_secondary = args.GetInputSecondary();
    params.output = args.GetOutput();
    params.subswath = args.GetSubswath();
    params.polarisation = args.GetPolarisation();
    params.burst_index_start_reference = alus::coherenceestimationroutine::INVALID_BURST_INDEX;
    params.burst_index_last_reference = alus::coherenceestimationroutine::INVALID_BURST_INDEX;
    params.burst_index_start_secondary = alus::coherenceestimationroutine::INVALID_BURST_INDEX;
    params.burst_index_last_secondary = alus::coherenceestimationroutine::INVALID_BURST_INDEX;
    if (const auto burst_ref = args.GetBurstIndexesReference(); burst_ref.has_value()) {
        params.burst_index_start_reference = std::get<0>(burst_ref.value());
        params.burst_index_last_reference = std::get<1>(burst_ref.value());
    }
    if (const auto burst_sec = args.GetBurstIndexesSecondary(); burst_sec.has_value()) {
        params.burst_index_start_secondary = std::get<0>(burst_sec.value());
        params.burst_index_last_secondary = std::get<1>(burst_sec.value());
    }

    params.mask_out_area_without_elevation = args.DoMaskOutAreaWithoutElevation();
    params.aoi = args.GetAoi().value_or("");
    params.orbit_reference = args.GetOrbitFileReference();
    params.orbit_secondary = args.GetOrbitFileSecondary();
    params.orbit_dir = args.GetOrbitDirectory();
    params.subtract_flat_earth = args.DoSubtractFlatEarthPhase();
    params.srp_number_points = args.GetSrpNumberPoints();
    params.srp_polynomial_degree = args.GetSrpPolynomialDegree();
    params.orbit_degree = args.GetOrbitDegree();
    params.rg_window = args.GetRangeWindow();
    params.az_window = args.GetAzimuthWindow();
    params.wif = args.DoSaveIntermediateResults();

    params.timeline_start = args.GetTimelineStart();
    params.timeline_end = args.GetTimelineEnd();
    params.timeline_input = args.GetTimelineInput();
    params.timeline_mission = args.GetTimelineMission();

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

        alus::coherenceestimationroutine::Arguments args(RUN_TIMELINE);
        args_help = args.GetHelp();
        args.Parse(std::vector<char*>(argv, argv + argc));

        if (args.IsHelpRequested()) {
            std::cout << alus::app::GenerateHelpMessage(alus::coherenceestimationroutine::ALG_NAME, args_help);
            return alus::app::errorcode::ALG_SUCCESS;
        }

        args.Check();
        alus::common::log::SetLevel(args.GetLogLevel());
        alus::coherenceestimationroutine::Execute exe(AssembleParameters(args), args.GetDemFiles());
        if (RUN_TIMELINE) {
            exe.RunTimeline(cuda_init, args.GetGpuMemoryPercentage());
        } else {
            exe.RunSinglePair(cuda_init, args.GetGpuMemoryPercentage());
        }

    } catch (const boost::program_options::error& e) {
        std::cout << alus::app::GenerateHelpMessage(alus::coherenceestimationroutine::ALG_NAME, args_help);
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