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

#include <boost/algorithm/string/predicate.hpp>

#include "algorithm_exception.h"
#include "alus_log.h"
#include "constants.h"
#include "gdal_management.h"

namespace alus::sarsegment {

Execute::Execute(Parameters params, const std::vector<std::string>& dem_files)
    : params_{std::move(params)}, dem_files_{dem_files} {
    alus::gdalmanagement::Initialize();
}

void Execute::Run(alus::cuda::CudaInit& cuda_init, size_t gpu_memory_percentage) {
    ParseCalibrationType(params_.calibration_type);
    PrintProcessingParameters();
    (void)gpu_memory_percentage;

    while (!cuda_init.IsFinished())
        ;
    cuda_init.CheckErrors();
    const auto cuda_device = cuda_init.GetDevices().front();
    cuda_device.Set();
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

void Execute::PrintProcessingParameters() const {
    LOGI << "Processing parameters:" << std::endl
         << "Input product - " << params_.input << std::endl
         << "Calibration type - " << params_.calibration_type << std::endl
         << "Write intermediate files - " << (params_.wif ? "YES" : "NO") << std::endl;
}

Execute::~Execute() { alus::gdalmanagement::Deinitialize(); }

}  // namespace alus::sarsegment