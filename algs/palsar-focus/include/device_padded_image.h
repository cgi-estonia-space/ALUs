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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include "cuda_util.h"
#include "cuda_workplace.h"
#include "math_utils.h"

namespace alus::palsar {
/**
 * Abstraction of a complex float image in device memory with paddings in right and bottom side of the image
 * for effcient SAR math operations
 * ASCII to illustrate layout:
 * ______________
 * |       |    |
 * |   1   |    |
 * |_______|    |
 * |            |
 * |   2        |
 * |____________|
 *
 * Data is in section 1, dimensions are XSize() and YSize()
 * Paddings are in section 2, total dimensions with section 1 are XStride() and YStride()
 */
class DevicePaddedImage {
public:
    void InitPadded(int width, int height, int width_stride, int height_stride) {
        VerifyEmpty();
        if (width_stride < width || height_stride < height) {
            throw std::logic_error("Stride smaller than dimension");
        }
        const size_t bsize = sizeof(d_data_[0]) * width_stride * height_stride;
        CHECK_CUDA_ERR(cudaMalloc(&d_data_, bsize));
        x_size_ = width;
        y_size_ = height;
        x_stride_ = width_stride;
        y_stride_ = height_stride;
    }

    void InitNoPad(int width, int height) { InitPadded(width, height, width, height); }

    void InitPadded(const DevicePaddedImage& oth) {
        InitPadded(oth.x_size_, oth.y_size_, oth.x_stride_, oth.y_stride_);
    }

    void InitExtPtr(int width, int height, int width_stride, int height_stride, CudaWorkspace& d_ext_memory) {
        VerifyEmpty();
        if (width_stride < width || height_stride < height) {
            throw std::logic_error("Stride smaller than dimension");
        }
        size_t mem_needed = static_cast<size_t>(width_stride) * height_stride * sizeof(d_data_[0]);
        VerifyExtMemorySize(mem_needed, d_ext_memory.ByteSize());
        d_data_ = static_cast<cufftComplex*>(d_ext_memory.ReleaseMemory());
        x_size_ = width;
        y_size_ = height;
        x_stride_ = width_stride;
        y_stride_ = height_stride;
    }
    [[nodiscard]] int XSize() const { return x_size_; }
    [[nodiscard]] int YSize() const { return y_size_; }
    [[nodiscard]] int XStride() const { return x_stride_; }
    [[nodiscard]] int YStride() const { return y_stride_; }

    void SetXSize(int x_size) { x_size_ = x_size; }

    [[nodiscard]] cufftComplex* Data() { return d_data_; }
    [[nodiscard]] const cufftComplex* Data() const { return d_data_; }

    [[nodiscard]] size_t TotalByteSize() const {
        return static_cast<size_t>(x_stride_) * y_stride_ * sizeof(d_data_[0]);
    }
    [[nodiscard]] size_t DataByteSize() const { return static_cast<size_t>(x_size_) * y_size_ * sizeof(d_data_[0]); }

    void CopyToHostPaddedSize(cufftComplex* h_dst) const {
        CHECK_CUDA_ERR(cudaMemcpy(h_dst, d_data_, TotalByteSize(), cudaMemcpyDeviceToHost));
    }

    void CopyToHostLogicalSize(cufftComplex* h_dst) const {
        constexpr size_t ELEM_SZ = sizeof(d_data_[0]);
        auto e = cudaMemcpy2D(h_dst, x_size_ * ELEM_SZ, d_data_, ELEM_SZ * x_stride_, x_size_ * ELEM_SZ, y_size_,
                              cudaMemcpyDeviceToHost);

        CHECK_CUDA_ERR(e);
    }

    void FreeMemory() {
        CHECK_CUDA_ERR(cudaFree(d_data_));
        d_data_ = nullptr;
        x_size_ = x_stride_ = 0;
        y_size_ = y_stride_ = 0;
    }

    [[nodiscard]] cufftComplex* ReleasePtr() {
        x_size_ = x_stride_ = 0;
        y_size_ = y_stride_ = 0;
        cufftComplex* ret = d_data_;
        d_data_ = nullptr;
        return ret;
    }

    [[nodiscard]] CudaWorkspace ReleaseMemory() {
        CudaWorkspace ret;
        ret.Reset(d_data_, TotalByteSize());
        x_size_ = x_stride_ = 0;
        y_size_ = y_stride_ = 0;
        d_data_ = nullptr;
        return ret;
    }

    void MultiplyData(float multiplier);
    void ZeroFillPaddings();

    void ZeroMemory() { CHECK_CUDA_ERR(cudaMemset(d_data_, 0, TotalByteSize())); }

    // Transpose via memory allocation, this can be slow with big memory allocations
    void Transpose();

    // Transpose via workspace. Memory used by the object is given to the
    // CudaWorkspace object and the DevicePaddedImage now contains the transposed image
    void Transpose(CudaWorkspace& d_workspace);

    double CalcTotalIntensity(size_t sm_count);

    DevicePaddedImage() = default;

    DevicePaddedImage(const DevicePaddedImage&) = delete;
    DevicePaddedImage& operator=(const DevicePaddedImage&) = delete;

    ~DevicePaddedImage() { cudaFree(d_data_); }

private:
    void VerifyEmpty() const {
        if (d_data_) {
            throw std::logic_error("DevicePaddedImage already initialized");
        }
    }

    static void VerifyExtMemorySize(size_t needed_size, size_t ext_size) {
        if (needed_size > ext_size) {
            throw std::logic_error("External workspace does not have enough memory");
        }
    }

    cufftComplex* d_data_ = nullptr;
    int x_size_ = 0;
    int y_size_ = 0;
    int x_stride_ = 0;
    int y_stride_ = 0;
};
}  // namespace alus::palsar
