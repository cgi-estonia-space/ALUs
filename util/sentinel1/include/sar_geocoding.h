/**
 * This file is a filtered duplicate of a SNAP's SARGeocoding.java ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/s1tbx) repository originally stated to be implemented
 * by "Copyright (C) 2016 by Array Systems Computing Inc. http://www.array.ca"
 *
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
#pragma once

#include <thrust/device_vector.h>

#include "PosVector.hpp"
#include "cuda_util.cuh"

namespace slap {
namespace s1tbx {
namespace sarGeocoding {
/**
 * Compute zero Doppler time for given earth point using bisection method.
 *
 * Duplicate of a SNAP's SARGeocoding.java's getEarthPointZeroDopplerTime().
 * This actually exists as an inline version for CUDA calls as GetEarthPointZeroDopplerTime_impl() in sar_geocoding.cuh
 * This procedure is duplicated by the nvcc for host processing in sar_geocoding.cu.
 *
 * @param firstLineUTC     The zero Doppler time for the first range line.
 * @param lineTimeInterval The line time interval.
 * @param wavelength       The radar wavelength.
 * @param earthPoint       The earth point in xyz coordinate.
 * @param sensorPosition   Array of sensor positions for all range lines.
 * @param sensorVelocity   Array of sensor velocities for all range lines.
 * @return The zero Doppler time in days if it is found, -1 otherwise.
 */
double GetEarthPointZeroDopplerTime(double firstLineUTC,
                                    double lineTimeInterval,
                                    double wavelength,
                                    snapEngine::PosVector earthPoint,
                                    KernelArray<snapEngine::PosVector> sensorPosition,
                                    KernelArray<snapEngine::PosVector> sensorVelocity);

}  // namespace sarGeocoding
}  // namespace s1tbx
}  // namespace slap