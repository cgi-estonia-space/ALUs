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
#include "triangular_interpolation.cuh"

#include "cuda_util.hpp"

namespace alus {
namespace snapengine {
/**
 * The contents of this namespace come from org.jlinda.core.delaunay.TriangleInterpolator class from s1tbx.
 * A lot of the code has not changed much, so there should be some pretty obvious similarities.
 */
namespace triangularinterpolation {

inline __device__ int test(double x, double y, double *xt, double *yt, PointInTriangle pit) {
    int iRet0 = (pit.xtd0 * (y - yt[0])) > ((x - xt[0]) * pit.ytd0);
    int iRet1 = (pit.xtd1 * (y - yt[1])) > ((x - xt[1]) * pit.ytd1);
    int iRet2 = (pit.xtd2 * (y - yt[2])) > ((x - xt[2]) * pit.ytd2);

    return (iRet0 != 0 && iRet1 != 0 && iRet2 != 0) || (iRet0 == 0 && iRet1 == 0 && iRet2 == 0);
}

inline __device__ Zdataabc GetABC(double *vx,
                                  double *vy,
                                  double *vz,
                                  Zdata data,
                                  Zdataabc abc,
                                  const double f,
                                  const double xkj,
                                  const double ykj,
                                  const double xlj,
                                  const double ylj) {
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

inline __device__ long CoordToIndex(const double coord,
                                    const double coord0,
                                    const double deltaCoord,
                                    const double offset) {
    return (long)floor((((coord - coord0) / (deltaCoord)) - offset) + 0.5);
}

__global__ void Interpolate(delaunay::DelaunayTriangle2D *triangles,
                            Zdata *zdata,
                            Zdataabc *abc,
                            InterpolationParams params) {
    const unsigned int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    const size_t abc_index = idx * params.z_data_count;

    if (idx >= params.triangle_count) {
        return;
    }

    const double x_min = params.window.linelo;
    const double y_min = params.window.pixlo;

    long i_min, i_max, j_min, j_max;  // minimas/maximas
    double xp, yp;
    double xkj, ykj, xlj, ylj;
    double f;  // function
    delaunay::DelaunayTriangle2D triangle;

    double vx[3];
    double vy[3];
    double vz[3];

    const int nx = params.window.lines;
    const int ny = params.window.pixels;

    triangle = triangles[idx];

    // store triangle coordinates in local variables
    vx[0] = triangle.ax;
    vy[0] = triangle.ay / params.xy_ratio;

    vx[1] = triangle.bx;
    vy[1] = triangle.by / params.xy_ratio;

    vx[2] = triangle.cx;
    vy[2] = triangle.cy / params.xy_ratio;

    // skip invalid indices
    if (vx[0] == params.invalid_index || vx[1] == params.invalid_index || vx[2] == params.invalid_index ||
        vy[0] == params.invalid_index || vy[1] == params.invalid_index || vy[2] == params.invalid_index) {
        return;
    }

    // Compute grid indices the current triangle may cover
    xp = min(min(vx[0], vx[1]), vx[2]);
    i_min = CoordToIndex(xp, x_min, params.x_scale, params.offset);

    xp = max(max(vx[0], vx[1]), vx[2]);
    i_max = CoordToIndex(xp, x_min, params.x_scale, params.offset);

    yp = min(min(vy[0], vy[1]), vy[2]);
    j_min = CoordToIndex(yp, y_min, params.y_scale, params.offset);

    yp = max(max(vy[0], vy[1]), vy[2]);
    j_max = CoordToIndex(yp, y_min, params.y_scale, params.offset);
    // printf("imax %ld imin %ld nx %d -- jmax %ld jmin %ld ny %d\n", i_max, i_min, nx, j_max, j_min, ny);
    // skip triangle that is above, below, left or right of the region
    if ((i_max < 0) || (i_min >= nx) || (j_max < 0) || (j_min >= ny)) {
        return;
    }

    // triangle covers the upper or lower boundary
    i_min = i_min * (i_min >= 0);
    i_max = (i_max >= nx) * (nx - 1) + (i_max < nx) * i_max;

    // triangle covers left or right boundary
    j_min = j_min * (j_min >= 0);
    j_max = (j_max >= ny) * (ny - 1) + (j_max < ny) * j_max;

    // compute plane defined by the three vertices of the triangle: z = ax + by + c
    xkj = vx[1] - vx[0];
    ykj = vy[1] - vy[0];
    xlj = vx[2] - vx[0];
    ylj = vy[2] - vy[0];

    f = 1.0 / (xkj * ylj - ykj * xlj);

    vz[0] = triangle.a_index;
    vz[1] = triangle.b_index;
    vz[2] = triangle.c_index;

    for (size_t i = 0; i < params.z_data_count; i++) {
        abc[abc_index + i] = GetABC(vx, vy, vz, zdata[i], abc[abc_index + i], f, xkj, ykj, xlj, ylj);
    }

    PointInTriangle point_in_triangle;
    point_in_triangle.xtd0 = vx[2] - vx[0];
    point_in_triangle.xtd1 = vx[0] - vx[1];
    point_in_triangle.xtd2 = vx[1] - vx[2];
    point_in_triangle.ytd0 = vy[2] - vy[0];
    point_in_triangle.ytd1 = vy[0] - vy[1];
    point_in_triangle.ytd2 = vy[1] - vy[2];

    for (int i = (int)i_min; i <= i_max; i++) {
        xp = x_min + i * params.x_scale + params.offset;
        for (int j = (int)j_min; j <= j_max; j++) {
            yp = y_min + j * params.y_scale + params.offset;

            if (!test(xp, yp, vx, vy, point_in_triangle)) {
                continue;
            }
            // printf("printing results at %d\n", idx);
            for (size_t d = 0; d < params.z_data_count; d++) {
                double result = abc[abc_index + d].a * xp + abc[abc_index + d].b * yp + abc[abc_index + d].c;
                zdata[d].output_arr[i * zdata[d].output_height + j] = result;
                int int_result = (int)floor(result);
                atomicMin(&zdata[d].min_int, int_result);
                atomicMax(&zdata[d].max_int, int_result);

            }
        }
    }
}

cudaError_t LaunchInterpolation(delaunay::DelaunayTriangle2D *triangles, Zdata *zdata, InterpolationParams params) {
    dim3 block_size(400);
    dim3 grid_size(cuda::GetGridDim(400, params.triangle_count));
    Zdataabc *device_abc;

    CHECK_CUDA_ERR(cudaMalloc((void **)&device_abc, params.triangle_count * params.z_data_count * sizeof(Zdataabc)));

    Interpolate<<<grid_size, block_size>>>(triangles, zdata, device_abc, params);
    cudaError_t result = cudaGetLastError();

    CHECK_CUDA_ERR(cudaFree(device_abc));
    return result;
}

}  // namespace triangularinterpolation
}  // namespace snapengine
}  // namespace alus