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

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>

#include "algorithm_exception.h"
#include "raster_properties.h"
#include "resample_method.h"
#include "type_parameter.h"

namespace alus::resample {

void PrepareDeviceBuffers(const void* input_data, void** device_mem_input, size_t input_size, void** device_mem_output,
                          size_t output_size);
void UnloadDeviceBuffers(void* device_mem_input, void* device_mem_output);
void UnloadDeviceBuffers(void* device_mem_input, void* device_mem_output, void* output_data, size_t output_size);
int DoResampling(const void* input, RasterDimension input_dimension, void* output, RasterDimension output_dimension,
                 TypeParameters type_parameter, Method method);

struct NppiResampleArguments {
    const void* source_buffer;
    size_t source_size;
    RasterDimension source_dimension;
    void* destination_buffer;
    size_t destination_size;
    RasterDimension destination_dimension;
    Method method;
    TypeParameters type_parameter;
};

inline void NppiResample(const NppiResampleArguments& args) {
    void* device_mem_input{};
    void* device_mem_output{};

    PrepareDeviceBuffers(args.source_buffer, &device_mem_input, args.source_size, &device_mem_output,
                         args.destination_size);

    if (const auto res = DoResampling(device_mem_input, args.source_dimension, device_mem_output,
                                      args.destination_dimension, args.type_parameter, args.method);
        res != 0) {
        UnloadDeviceBuffers(device_mem_input, device_mem_output);
        THROW_ALGORITHM_EXCEPTION(APP_NAME, "Unsuccessful resampling, error code - " + std::to_string(res));
    }

    UnloadDeviceBuffers(device_mem_input, device_mem_output, args.destination_buffer, args.destination_size);
}

}  // namespace alus::resample