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
#include "sentinel1_calibrate_kernel_utils.cuh"
#include "sentinel1_calibrate_kernel_utils.h"

namespace alus {
namespace sentinel1calibrate {
size_t GetCalibrationVectorIndex(int y, int count, const int* line_values) {
    return GetCalibrationVectorIndexImpl(y, count, line_values);
}
void SetupTileLine(int y, CalibrationKernelArgs& args, CalibrationLineParameters& line_parameters) {
    SetupTileLineImpl(y, args, line_parameters);
}
int64_t GetPixelIndex(int x, const s1tbx::CalibrationVectorComputation* calibration_vector) {
    return GetPixelIndexImpl(x, calibration_vector);
}
double CalculateLutVal(CalibrationLineParameters& line_parameters, CalibrationPixelParameters& pixel_parameters) {
    return CalculateLutValImpl(line_parameters, pixel_parameters);
}
void CalculatePixelParams(int x, int y, CalibrationKernelArgs& args, CalibrationLineParameters& line_parameters,
                          CalibrationPixelParameters& pixel_parameters) {
    CalculatePixelParamsImpl(x, y, args, line_parameters, pixel_parameters);
}
void AdjustDn(double dn, double& calibration_value, double calibration_factor) {
    AdjustDnImpl(dn, calibration_value, calibration_factor);
}
void CalculateAmplitude(CalibrationPixelParameters& parameters, double& calibration_value) {
    CalculateAmplitudeImpl(parameters, calibration_value);
}
void CalculateIntensityWithRetro(CalibrationLineParameters& line_parameters,
                                 CalibrationPixelParameters& pixel_parameters, double& calibration_value) {
    CalculateIntensityWithRetroImpl(line_parameters, pixel_parameters, calibration_value);
}
void CalculateIntensityWithoutRetro(CalibrationPixelParameters& pixel_parameters, double& calibration_value) {
    CalculateIntensityWithoutRetroImpl(pixel_parameters, calibration_value);
}
void CalculateReal(CalibrationKernelArgs args, CalibrationPixelParameters& parameters, double& calibration_value) {
    CalculateRealImpl(args, parameters, calibration_value);
}
void CalculateImaginary(CalibrationKernelArgs args, CalibrationPixelParameters& parameters, double& calibration_value) {
    CalculateImaginaryImpl(args, parameters, calibration_value);
}
void CalculateComplexIntensity(CalibrationKernelArgs args, CalibrationPixelParameters& parameters,
                               double& calibration_value) {
    CalculateComplexIntensityImpl(args, parameters, calibration_value);
}
void CalculateIntensityDB(CalibrationPixelParameters& parameters, double& calibration_value) {
    CalculateIntensityDBImpl(parameters, calibration_value);
}
}  // namespace sentinel1calibrate
}  // namespace alus
