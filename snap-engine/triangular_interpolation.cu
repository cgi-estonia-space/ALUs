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
#include "triangular_interpolation_computation.h"



#include "cuda_util.h"
//#include "bcg_ctx.h"


namespace alus {
namespace snapengine {
/**
 * The contents of this namespace come from org.jlinda.core.delaunay.TriangleInterpolator class from s1tbx.
 * A lot of the code has not changed much, so there should be some pretty obvious similarities.
 */
namespace triangularinterpolation {

inline __device__ int test(double x, double y, double* xt, double* yt, PointInTriangle pit) {
    int iRet0 = (pit.xtd0 * (y - yt[0])) > ((x - xt[0]) * pit.ytd0);
    int iRet1 = (pit.xtd1 * (y - yt[1])) > ((x - xt[1]) * pit.ytd1);
    int iRet2 = (pit.xtd2 * (y - yt[2])) > ((x - xt[2]) * pit.ytd2);

    return (iRet0 != 0 && iRet1 != 0 && iRet2 != 0) || (iRet0 == 0 && iRet1 == 0 && iRet2 == 0);
}

inline __device__ Zdataabc GetABC(double* vx, double* vy, double* vz, Zdata data, Zdataabc abc, const double f,
                                  const double xkj, const double ykj, const double xlj, const double ylj) {
    double zj, zk, zl;

    const int i0 = (int)(vz[0] / data.input_height);
    const int j0 = (int)(vz[0] - i0 * data.input_height);
    zj = data.input_arr[i0 * data.input_height + j0];

    const int i1 = (int)(vz[1] / data.input_height);
    const int j1 = (int)(vz[1] - i1 * data.input_height);
    zk = data.input_arr[i1 * data.input_height + j1];

    const int i2 = (int)(vz[2] / data.input_height);
    const int j2 = (int)(vz[2] - i2 * data.input_height);
    zl = data.input_arr[i2 * data.input_height + j2];

    const double zkj = zk - zj;
    const double zlj = zl - zj;

    abc.a = -f * (ykj * zlj - zkj * ylj);
    abc.b = -f * (zkj * xlj - xkj * zlj);
    abc.c = -abc.a * vx[1] - abc.b * vy[1] + zk;

    return abc;
}

inline __device__ long CoordToIndex(const double coord, const double coord0, const double deltaCoord,
                                    const double offset) {
    return (long)floor((((coord - coord0) / (deltaCoord)) - offset) + 0.5);
}

__global__ void FindValidTriangles(delaunay::DelaunayTriangle2D* triangles, TriangleInterpolationParams params,
                                   Zdata* zdata, Zdataabc* abc, TriangleDto* dtos, int* selected_triangles,
                                   unsigned int* amount_of_triangles) {
    const size_t idx = threadIdx.x + (blockDim.x * blockIdx.x);
    size_t accepted_index;
    size_t abc_index;
    TriangleDto dto;

    if (idx >= params.triangle_count) {
        return;
    }

    const double x_min = params.window.linelo;
    const double y_min = params.window.pixlo;

    double xkj, ykj, xlj, ylj;
    double f;  // function
    delaunay::DelaunayTriangle2D triangle;

    double vz[3];

    const int nx = params.window.lines;
    const int ny = params.window.pixels;

    triangle = triangles[idx];

    // store triangle coordinates in local variables
    dto.vx[0] = triangle.ax;
    dto.vy[0] = triangle.ay / params.xy_ratio;

    dto.vx[1] = triangle.bx;
    dto.vy[1] = triangle.by / params.xy_ratio;

    dto.vx[2] = triangle.cx;
    dto.vy[2] = triangle.cy / params.xy_ratio;

    // skip invalid indices
    if (dto.vx[0] == params.invalid_index || dto.vx[1] == params.invalid_index || dto.vx[2] == params.invalid_index ||
        dto.vy[0] == params.invalid_index || dto.vy[1] == params.invalid_index || dto.vy[2] == params.invalid_index) {
        return;
    }

    // Compute grid indices the current triangle may cover
    dto.xp = min(min(dto.vx[0], dto.vx[1]), dto.vx[2]);
    dto.i_min = CoordToIndex(dto.xp, x_min, params.x_scale, params.offset);

    dto.xp = max(max(dto.vx[0], dto.vx[1]), dto.vx[2]);
    dto.i_max = CoordToIndex(dto.xp, x_min, params.x_scale, params.offset);

    dto.yp = min(min(dto.vy[0], dto.vy[1]), dto.vy[2]);
    dto.j_min = CoordToIndex(dto.yp, y_min, params.y_scale, params.offset);

    dto.yp = max(max(dto.vy[0], dto.vy[1]), dto.vy[2]);
    dto.j_max = CoordToIndex(dto.yp, y_min, params.y_scale, params.offset);

    // skip triangle that is above, below, left or right of the region
    if ((dto.i_max < 0) || (dto.i_min >= nx) || (dto.j_max < 0) || (dto.j_min >= ny)) {
        return;
    }

    // triangle covers the upper or lower boundary
    dto.i_min = dto.i_min * (dto.i_min >= 0);
    dto.i_max = (dto.i_max >= nx) * (nx - 1) + (dto.i_max < nx) * dto.i_max;

    // triangle covers left or right boundary
    dto.j_min = dto.j_min * (dto.j_min >= 0);
    dto.j_max = (dto.j_max >= ny) * (ny - 1) + (dto.j_max < ny) * dto.j_max;

    accepted_index = atomicInc(amount_of_triangles, params.triangle_count);
    selected_triangles[accepted_index] = idx;
    abc_index = accepted_index * params.z_data_count;

    // compute plane defined by the three vertices of the triangle: z = ax + by + c
    xkj = dto.vx[1] - dto.vx[0];
    ykj = dto.vy[1] - dto.vy[0];
    xlj = dto.vx[2] - dto.vx[0];
    ylj = dto.vy[2] - dto.vy[0];

    f = 1.0 / (xkj * ylj - ykj * xlj);

    vz[0] = triangle.a_index;
    vz[1] = triangle.b_index;
    vz[2] = triangle.c_index;

    for (size_t i = 0; i < params.z_data_count; i++) {
        abc[abc_index + i] = GetABC(dto.vx, dto.vy, vz, zdata[i], abc[abc_index + i], f, xkj, ykj, xlj, ylj);
    }

    dto.point_in_triangle.xtd0 = dto.vx[2] - dto.vx[0];
    dto.point_in_triangle.xtd1 = dto.vx[0] - dto.vx[1];
    dto.point_in_triangle.xtd2 = dto.vx[1] - dto.vx[2];
    dto.point_in_triangle.ytd0 = dto.vy[2] - dto.vy[0];
    dto.point_in_triangle.ytd1 = dto.vy[0] - dto.vy[1];
    dto.point_in_triangle.ytd2 = dto.vy[1] - dto.vy[2];

    dtos[accepted_index] = dto;
}

__global__ void Interpolate(Zdata* zdata, Zdataabc* abc, TriangleDto* dtos, InterpolationParams params) {
    const unsigned int index = blockIdx.x + (blockIdx.y * gridDim.x);
    const size_t abc_index = index * params.z_data_count;
    TriangleDto dto;
    const double x_min = params.window.linelo;
    const double y_min = params.window.pixlo;

    if (index >= params.accepted_triangles) {
        return;
    }
    dto = dtos[index];

    for (int i = (int)dto.i_min + threadIdx.x; i <= dto.i_max; i += blockDim.x) {
        dto.xp = x_min + i * params.x_scale + params.offset;
        for (int j = (int)dto.j_min + threadIdx.y; j <= dto.j_max; j += blockDim.y) {
            dto.yp = y_min + j * params.y_scale + params.offset;

            if (!test(dto.xp, dto.yp, dto.vx, dto.vy, dto.point_in_triangle)) {
                continue;
            }

            for (int d = 0; d < params.z_data_count; d++) {
                double result = abc[abc_index + d].a * dto.xp + abc[abc_index + d].b * dto.yp + abc[abc_index + d].c;
                zdata[d].output_arr[i * zdata[d].output_height + j] = result;
                int int_result = (int)floor(result);
                atomicMin(&zdata[d].min_int, int_result);
                atomicMax(&zdata[d].max_int, int_result);
            }
        }
    }
}

cudaError_t LaunchInterpolation(delaunay::DelaunayTriangle2D* triangles, Zdata* zdata,
                                TriangleInterpolationParams params, alus::backgeocoding::ComputeCtx* ctx) {
    dim3 block_size(512);
    dim3 grid_size(cuda::GetGridDim(512, params.triangle_count));
    Zdataabc* device_abc;
    TriangleDto* device_dtos;
    int* selected_triangles;
    unsigned int* amount_of_triangles;
    unsigned int accepted_triangles;

    //CHECK_CUDA_ERR(cudaMalloc(&selected_triangles, params.triangle_count * sizeof(int)));
    //CHECK_CUDA_ERR(cudaMalloc(&amount_of_triangles, sizeof(unsigned int)));
    //CHECK_CUDA_ERR(cudaMalloc(&device_abc, params.triangle_count * params.z_data_count * sizeof(Zdataabc)));
    //CHECK_CUDA_ERR(cudaMalloc(&device_dtos, params.triangle_count * sizeof(TriangleDto)));

    selected_triangles = ctx->selected_triangles.EnsureBuffer<int>(params.triangle_count);
    amount_of_triangles = ctx->amount_of_triangles.EnsureBuffer<unsigned int>(1);
    device_abc = ctx->device_abc.EnsureBuffer<Zdataabc>(params.triangle_count * params.z_data_count);
    device_dtos = ctx->device_dtos.EnsureBuffer<TriangleDto>(params.triangle_count);
    CHECK_CUDA_ERR(cudaMemset(amount_of_triangles, 0, sizeof(unsigned int)));

    FindValidTriangles<<<grid_size, block_size, 0, ctx->stream>>>(triangles, params, zdata, device_abc, device_dtos, selected_triangles,
                                                  amount_of_triangles);
    cudaError_t result = cudaGetLastError();

    CHECK_CUDA_ERR(cudaMemcpy(&accepted_triangles, amount_of_triangles, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    InterpolationParams params2;
    params2.z_data_count = params.z_data_count;
    params2.x_scale = params.x_scale;
    params2.y_scale = params.y_scale;
    params2.offset = params.offset;
    params2.accepted_triangles = accepted_triangles;
    params2.window = params.window;

    int grid_dim = (int)sqrt(accepted_triangles);
    grid_dim++;

    dim3 interpolation_grid_size(grid_dim, grid_dim);

    // TODO investigate further
    // block size dimension has major impact the kernel compute time on different GPUs
    dim3 interpolation_block_size(8, 8);
    Interpolate<<<interpolation_grid_size, interpolation_block_size, 0, ctx->stream>>>(zdata, device_abc, device_dtos, params2);

    //CHECK_CUDA_ERR(cudaFree(device_abc));
    //CHECK_CUDA_ERR(cudaFree(selected_triangles));
    //CHECK_CUDA_ERR(cudaFree(amount_of_triangles));
    //CHECK_CUDA_ERR(cudaFree(device_dtos));

    ctx->device_abc.Free();
    ctx->selected_triangles.Free();
    ctx->device_dtos.Free();

    return result;
}

}  // namespace triangularinterpolation
}  // namespace snapengine
}  // namespace alus