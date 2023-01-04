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

#include <cufft.h>

#include "algorithm_exception.h"

#define CHECK_CUFFT_ERR(x) alus::palsar::CheckCuFFT(x, __FILE__, __LINE__)
#define CHECK_NULLPTR(x) alus::palsar::CheckNullptr(x, __FILE__, __LINE__)

namespace alus::palsar {

// copy paste from cuda-11.2/samples/common/inc/helper_cuda.h
inline const char* CufftErrorStr(cufftResult error) {
    switch (error) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";

        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";

        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";

        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";

        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";

        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";

        case CUFFT_NOT_SUPPORTED:
            return "CUFFT_NOT_SUPPORTED";
    }

    return "<unknown>";
}

inline void CheckCuFFT(cufftResult err, const char* file, int line) {
    if (err != CUFFT_SUCCESS) {
        std::string error_msg =
            std::string("cuFFT error = ") + CufftErrorStr(err) + " file = " + file + " line = " + std::to_string(line);
        throw std::runtime_error(error_msg);
    }
}

inline void CheckNullptr(void* ptr, const char* file, int line) {
    if (!ptr) {
        std::string error_msg = std::string("nullptr file = ") + file + " line = " + std::to_string(line);
        throw std::invalid_argument(error_msg);
    }
}

inline void CheckCufftSize(size_t workspace_size, cufftHandle plan) {
    size_t fft_workarea = 0;
    auto cufft_err = cufftGetSize(plan, &fft_workarea);
    if (cufft_err != CUFFT_SUCCESS || workspace_size < fft_workarea) {
        throw std::runtime_error("workspace size not enough for FFT plan");
    }
}

}  // namespace alus::palsar
