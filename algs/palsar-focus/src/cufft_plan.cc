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

#include "cufft_plan.h"

#include "checks.h"

namespace {
int IntPow(int base, int power) {
    int r = 1;
    for (int i = 0; i < power; i++) {
        r *= base;
    }
    return r;
}
}  // namespace

namespace alus::palsar {

// 2D image C2C float 1D FFT on each row(aka range direction)
[[nodiscard]] cufftHandle PlanRangeFFT(int range_size, int azimuth_size, bool auto_alloc) {
    cufftHandle plan = 0;
    auto cufft_err = cufftCreate(&plan);
    if (cufft_err == CUFFT_SUCCESS) {
        cufft_err = cufftSetAutoAllocation(plan, static_cast<int>(auto_alloc));
        if (cufft_err == CUFFT_SUCCESS) {
            int rank = 1;
            int n[] = {range_size};
            int inembed[1] = {range_size};
            int istride = 1;
            int idist = range_size;
            int onembed[1] = {range_size};
            int ostride = 1;
            int odist = range_size;
            auto type = CUFFT_C2C;
            int batch = azimuth_size;
            size_t ws = 0;
            cufft_err =
                cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, &ws);
        }
    }

    if (cufft_err != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        CHECK_CUFFT_ERR(cufft_err);
    }
    return plan;
}

cufftHandle PlanAzimuthFFT(int range_size, int azimuth_size, int range_stride, bool auto_alloc) {
    cufftHandle plan = 0;
    auto cufft_err = cufftCreate(&plan);
    if (cufft_err == CUFFT_SUCCESS) {
        cufft_err = cufftSetAutoAllocation(plan, static_cast<int>(auto_alloc));
        if (cufft_err == CUFFT_SUCCESS) {
            int rank = 1;
            int n[] = {azimuth_size};
            int inembed[1] = {azimuth_size};
            int istride = range_stride;
            int idist = 1;
            int onembed[1] = {azimuth_size};
            int ostride = range_stride;
            int odist = 1;
            auto type = CUFFT_C2C;
            int batch = range_size;
            size_t ws = 0;
            CHECK_CUFFT_ERR(
                cufftMakePlanMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, &ws));
        }
    }
    if (cufft_err != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        CHECK_CUFFT_ERR(cufft_err);
    }
    return plan;
}

cufftHandle Plan2DFFT(int range_size, int azimuth_size) {
    cufftHandle plan;
    // NB! weird order, nx for azimuth/columns?
    int nx = azimuth_size;  // strided dimension - weird order
    int ny = range_size;    // continuous dimension
    auto type = CUFFT_C2C;
    CHECK_CUFFT_ERR(cufftPlan2d(&plan, nx, ny, type));
    return plan;
}

size_t EstimateRangeFFT(int range_size, int azimuth_size) {
    int rank = 1;
    int n[] = {range_size};
    int inembed[1] = {range_size};
    int istride = 1;
    int idist = range_size;
    int onembed[1] = {range_size};
    int ostride = 1;
    int odist = range_size;
    auto type = CUFFT_C2C;
    int batch = azimuth_size;
    size_t work_size = 0;
    CHECK_CUFFT_ERR(
        cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, &work_size));
    return work_size;
}

size_t EstimateAzimuthFFT(int range_size, int azimuth_size, int range_stride) {
    int rank = 1;
    int n[] = {azimuth_size};
    int inembed[1] = {azimuth_size};
    int istride = range_stride;
    int idist = 1;
    int onembed[1] = {azimuth_size};
    int ostride = range_stride;
    int odist = 1;
    auto type = CUFFT_C2C;
    int batch = range_size;
    size_t work_size = 0;
    CHECK_CUFFT_ERR(
        cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, &work_size));
    return work_size;
}

size_t Estimate2DFFT(int range_size, int azimuth_size) {
    // NB! weird order, nx for azimuth/columns?
    int nx = azimuth_size;  // strided dimension - weird order
    int ny = range_size;    // continuous dimension
    auto type = CUFFT_C2C;
    size_t work_size = 0;
    CHECK_CUFFT_ERR(cufftEstimate2d(nx, ny, type, &work_size));
    return work_size;
}

std::map<int, std::array<int, 4>> GetOptimalFFTSizes() {
    std::map<int, std::array<int, 4>> result;

    // cuFFT documentation states that the most optimal FFT size is expressed with the following: 2^a×3^b×5^c×7^d
    // find all values up to a reasonable limit that makes sense for PALSAR
    const int limit = 131072;
    for (int i = 0;; i++) {
        int p1 = IntPow(2, i);
        if (p1 > limit) {
            break;
        }
        for (int j = 0;; j++) {
            int p2 = IntPow(3, j);
            if (p1 * p2 > limit) {
                break;
            }
            for (int k = 0;; k++) {
                int p3 = IntPow(5, k);
                if (p1 * p2 * p3 > limit) {
                    break;
                }
                for (int l = 0;; l++) {
                    int p4 = IntPow(7, l);
                    int val = p1 * p2 * p3 * p4;
                    if (val <= limit) {
                        result[val] = {i, j, k, l};
                    } else {
                        break;
                    }
                }
            }
        }
    }
    return result;
}
}  // namespace alus::palsar