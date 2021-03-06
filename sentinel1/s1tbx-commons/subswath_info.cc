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
#include "s1tbx-commons/subswath_info.h"

#include "allocators.h"
#include "cuda_copies.h"

namespace alus::s1tbx {

SubSwathInfo::SubSwathInfo() = default;
SubSwathInfo::~SubSwathInfo() {
    Deallocate2DArray<double>(doppler_rate_);
    Deallocate2DArray<double>(doppler_centroid_);
    Deallocate2DArray<double>(range_depend_doppler_rate_);
    Deallocate2DArray<double>(reference_time_);
    Deallocate2DArray<double>(azimuth_time_);
    Deallocate2DArray<double>(slant_range_time_);
    Deallocate2DArray<double>(latitude_);
    Deallocate2DArray<double>(longitude_);
    Deallocate2DArray<double>(incidence_angle_);

    DeviceFree();
}

void SubSwathInfo::HostToDevice() {
    int doppler_size_x = this->num_of_bursts_;
    int doppler_size_y = this->samples_per_burst_;
    int elems = doppler_size_x * doppler_size_y;
    DeviceSubswathInfo temp_pack;

    // TODO: before copy make sure to check if these are even available. In our demo these are forced available //
    // NOLINT
    // TODO(anton): should the above even be TODO? Seems like a notice than a future task
    CHECK_CUDA_ERR(cudaMalloc((void**)&temp_pack.device_doppler_rate, elems * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&temp_pack.device_doppler_centroid, elems * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&temp_pack.device_reference_time, elems * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&temp_pack.device_range_depend_doppler_rate, elems * sizeof(double)));

    cuda::CopyArrayH2D(temp_pack.device_doppler_rate, doppler_rate_[0], elems);
    cuda::CopyArrayH2D(temp_pack.device_doppler_centroid, doppler_centroid_[0], elems);
    cuda::CopyArrayH2D(temp_pack.device_reference_time, reference_time_[0], elems);
    cuda::CopyArrayH2D(temp_pack.device_range_depend_doppler_rate, range_depend_doppler_rate_[0], elems);

    temp_pack.first_valid_pixel = this->first_valid_pixel_;
    temp_pack.last_valid_pixel = this->last_valid_pixel_;
    temp_pack.first_line_time = this->first_line_time_;
    temp_pack.last_line_time = this->last_line_time_;
    temp_pack.slr_time_to_first_pixel = this->slr_time_to_first_pixel_;
    temp_pack.slr_time_to_last_pixel = this->slr_time_to_last_pixel_;
    temp_pack.range_pixel_spacing = this->range_pixel_spacing_;
    temp_pack.azimuth_time_interval = this->azimuth_time_interval_;
    temp_pack.radar_frequency = this->radar_frequency_;
    temp_pack.azimuth_steering_rate = this->azimuth_steering_rate_;

    temp_pack.doppler_size_x = doppler_size_x;
    temp_pack.doppler_size_y = doppler_size_y;

    temp_pack.lines_per_burst = this->lines_per_burst_;
    temp_pack.num_of_bursts = this->num_of_bursts_;
    temp_pack.samples_per_burst = this->samples_per_burst_;

    temp_pack.num_of_geo_lines = this->num_of_geo_lines_;
    temp_pack.num_of_geo_points_per_line = this->num_of_geo_points_per_line_;

    temp_pack.burst_line_times_count = this->burst_first_line_time_.size();
    CHECK_CUDA_ERR(
        cudaMalloc((void**)&temp_pack.device_burst_first_line_time, temp_pack.burst_line_times_count * sizeof(double)));
    CHECK_CUDA_ERR(
        cudaMalloc((void**)&temp_pack.device_burst_last_line_time, temp_pack.burst_line_times_count * sizeof(double)));

    cuda::CopyArrayH2D(temp_pack.device_burst_first_line_time, burst_first_line_time_.data(),
                       temp_pack.burst_line_times_count);
    cuda::CopyArrayH2D(temp_pack.device_burst_last_line_time, burst_last_line_time_.data(),
                       temp_pack.burst_line_times_count);

    const int subswath_geo_grid_size = num_of_geo_lines_ * num_of_geo_points_per_line_;
    CHECK_CUDA_ERR(cudaMalloc(&temp_pack.device_subswath_azimuth_times, sizeof(double) * subswath_geo_grid_size));
    CHECK_CUDA_ERR(cudaMalloc(&temp_pack.device_subswath_slant_range_times, sizeof(double) * subswath_geo_grid_size));
    CHECK_CUDA_ERR(cudaMalloc(&temp_pack.device_longitudes, sizeof(double) * subswath_geo_grid_size));
    CHECK_CUDA_ERR(cudaMalloc(&temp_pack.device_latidudes, sizeof(double) * subswath_geo_grid_size));

    cuda::CopyArrayH2D(temp_pack.device_subswath_azimuth_times, azimuth_time_[0], subswath_geo_grid_size);
    cuda::CopyArrayH2D(temp_pack.device_subswath_slant_range_times, slant_range_time_[0], subswath_geo_grid_size);
    cuda::CopyArrayH2D(temp_pack.device_longitudes, longitude_[0], subswath_geo_grid_size);
    cuda::CopyArrayH2D(temp_pack.device_latidudes, latitude_[0], subswath_geo_grid_size);

    this->devicePointersHolder = temp_pack;

    CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_subswath_info_, sizeof(DeviceSubswathInfo)));
    cuda::CopyH2D(device_subswath_info_, &temp_pack);
}
void SubSwathInfo::DeviceToHost() { CHECK_CUDA_ERR(cudaErrorNotYetImplemented); }
void SubSwathInfo::DeviceFree() {
    if (this->devicePointersHolder.device_burst_first_line_time != nullptr) {
        cudaFree(this->devicePointersHolder.device_burst_first_line_time);
        this->devicePointersHolder.device_burst_first_line_time = nullptr;
    }
    if (this->devicePointersHolder.device_burst_last_line_time != nullptr) {
        cudaFree(this->devicePointersHolder.device_burst_last_line_time);
        this->devicePointersHolder.device_burst_last_line_time = nullptr;
    }

    if (this->devicePointersHolder.device_doppler_rate != nullptr) {
        cudaFree(this->devicePointersHolder.device_doppler_rate);
        this->devicePointersHolder.device_doppler_rate = nullptr;
    }
    if (this->devicePointersHolder.device_doppler_centroid != nullptr) {
        cudaFree(this->devicePointersHolder.device_doppler_centroid);
        this->devicePointersHolder.device_doppler_centroid = nullptr;
    }
    if (this->devicePointersHolder.device_reference_time != nullptr) {
        cudaFree(this->devicePointersHolder.device_reference_time);
        this->devicePointersHolder.device_reference_time = nullptr;
    }
    if (this->devicePointersHolder.device_range_depend_doppler_rate != nullptr) {
        cudaFree(this->devicePointersHolder.device_range_depend_doppler_rate);
        this->devicePointersHolder.device_range_depend_doppler_rate = nullptr;
    }

    if (devicePointersHolder.device_subswath_azimuth_times) {
        cudaFree(devicePointersHolder.device_subswath_azimuth_times);
        devicePointersHolder.device_subswath_azimuth_times = nullptr;
    }
    if (devicePointersHolder.device_subswath_slant_range_times) {
        cudaFree(devicePointersHolder.device_subswath_slant_range_times);
        devicePointersHolder.device_subswath_slant_range_times = nullptr;
    }
    if (devicePointersHolder.device_longitudes) {
        cudaFree(devicePointersHolder.device_longitudes);
        devicePointersHolder.device_longitudes = nullptr;
    }
    if (devicePointersHolder.device_latidudes) {
        cudaFree(devicePointersHolder.device_latidudes);
        devicePointersHolder.device_latidudes = nullptr;
    }

    if (this->device_subswath_info_ != nullptr) {
        cudaFree(this->device_subswath_info_);
        this->device_subswath_info_ = nullptr;
    }
}

}  // namespace alus::s1tbx
