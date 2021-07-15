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
#include "backgeocoding.h"

#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <string_view>

#include <boost/numeric/conversion/cast.hpp>

#include "interpolation_constants.h"
#include "srtm3_elevation_model.h"
#include "srtm3_elevation_model_constants.h"
#include "triangular_interpolation_computation.h"
#include "alus_log.h"
#include "backgeocoding_constants.h"
#include "bilinear_computation.h"
#include "burst_offset_computation.h"
#include "cuda_util.h"
#include "delaunay_triangulator.h"
#include "deramp_demod_computation.h"
#include "elevation_mask_computation.h"
#include "extended_amount_computation.h"
#include "slave_pixpos_computation.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96_computation.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"

namespace alus::backgeocoding {

Backgeocoding::~Backgeocoding() {
    if (d_master_orbit_vectors_.array != nullptr) {
        cudaFree(d_master_orbit_vectors_.array);
        d_master_orbit_vectors_ = {};
    }

    if (d_slave_orbit_vectors_.array != nullptr) {
        cudaFree(d_slave_orbit_vectors_.array);
        d_slave_orbit_vectors_ = {};
    }
}

void Backgeocoding::PrepareToCompute(std::shared_ptr<snapengine::Product> master_product,
                                     std::shared_ptr<snapengine::Product> slave_product) {
    slave_utils_ = std::make_unique<s1tbx::Sentinel1Utils>(slave_product);
    master_utils_ = std::make_unique<s1tbx::Sentinel1Utils>(master_product);
    PrepareToComputeBody();
}

void Backgeocoding::PrepareToCompute(std::string_view master_metadata_file, std::string_view slave_metadata_file) {
    slave_utils_ = std::make_unique<s1tbx::Sentinel1Utils>(slave_metadata_file);
    master_utils_ = std::make_unique<s1tbx::Sentinel1Utils>(master_metadata_file);
    PrepareToComputeBody();
}

void Backgeocoding::PrepareToComputeBody() {
    // TODO: Exclusively supporting srtm3 atm
    dem_sampling_lat_ = static_cast<double>(snapengine::Srtm3ElevationModel::GetTileWidthInDegrees()) /
                        static_cast<double>(snapengine::Srtm3ElevationModel::GetTileWidth());
    dem_sampling_lon_ = dem_sampling_lat_;

    slave_utils_->ComputeDopplerRate();
    slave_utils_->ComputeReferenceTime();
    slave_utils_->subswath_.at(0)->HostToDevice();
    slave_utils_->HostToDevice();

    master_utils_->ComputeDopplerRate();
    master_utils_->ComputeReferenceTime();
    master_utils_->subswath_.at(0)->HostToDevice();
    master_utils_->HostToDevice();

    const std::vector<snapengine::OrbitStateVectorComputation>& master_orbit_vectors_computation =
        this->master_utils_->GetOrbitStateVectors()->orbit_state_vectors_computation_;
    const size_t master_orbit_vectors_size =
        sizeof(snapengine::OrbitStateVectorComputation) * master_orbit_vectors_computation.size();
    CHECK_CUDA_ERR(cudaMalloc(&d_master_orbit_vectors_.array, master_orbit_vectors_size));
    CHECK_CUDA_ERR(cudaMemcpy(d_master_orbit_vectors_.array, master_orbit_vectors_computation.data(),
                              master_orbit_vectors_size, cudaMemcpyHostToDevice));
    d_master_orbit_vectors_.size = master_orbit_vectors_computation.size();

    const std::vector<snapengine::OrbitStateVectorComputation>& slave_orbit_vectors_computation =
        this->slave_utils_->GetOrbitStateVectors()->orbit_state_vectors_computation_;
    const size_t slave_orbit_vectors_size =
        sizeof(snapengine::OrbitStateVectorComputation) * slave_orbit_vectors_computation.size();
    CHECK_CUDA_ERR(cudaMalloc(&d_slave_orbit_vectors_.array, slave_orbit_vectors_size));
    CHECK_CUDA_ERR(cudaMemcpy(d_slave_orbit_vectors_.array, slave_orbit_vectors_computation.data(),
                              slave_orbit_vectors_size, cudaMemcpyHostToDevice));
    d_slave_orbit_vectors_.size = slave_orbit_vectors_computation.size();

    slave_burst_offset_ = ComputeBurstOffset();
}

Rectangle Backgeocoding::PositionCompute(int m_burst_index, int s_burst_index, Rectangle master_area,
                                         double* device_x_points, double* device_y_points) {
    CoordMinMax coord_min_max;

    if (s_burst_index < 0 || s_burst_index >= slave_utils_->subswath_.at(0)->num_of_bursts_) {
        return {0, 0, 0, 0};
    }

    AzimuthAndRangeBounds az_rg_bounds =
        ComputeExtendedAmount(master_area.x, master_area.y, master_area.width, master_area.height);

    bool result = ComputeSlavePixPos(m_burst_index, s_burst_index, master_area, az_rg_bounds, &coord_min_max,
                                     device_x_points, device_y_points);
    if (!result) {
        return {0, 0, 0, 0};
    }
    const int margin = snapengine::BILINEAR_INTERPOLATION_KERNEL_SIZE;
    Rectangle source_rectangle;

    const int firstLineIndex = s_burst_index * slave_utils_->subswath_.at(0)->lines_per_burst_;
    const int lastLineIndex = firstLineIndex + slave_utils_->subswath_.at(0)->lines_per_burst_ - 1;
    const int firstPixelIndex = 0;
    const int lastPixelIndex = slave_utils_->subswath_.at(0)->samples_per_burst_ - 1;

    coord_min_max.x_min = std::max(coord_min_max.x_min - margin, firstPixelIndex);
    coord_min_max.x_max = std::min(coord_min_max.x_max + margin, lastPixelIndex);
    coord_min_max.y_min = std::max(coord_min_max.y_min - margin, firstLineIndex);
    coord_min_max.y_max = std::min(coord_min_max.y_max + margin, lastLineIndex);

    source_rectangle.x = coord_min_max.x_min;
    source_rectangle.y = coord_min_max.y_min;
    source_rectangle.width = coord_min_max.x_max - coord_min_max.x_min + 1;
    source_rectangle.height = coord_min_max.y_max - coord_min_max.y_min + 1;

    return source_rectangle;
}

void Backgeocoding::CoreCompute(CoreComputeParams params) {
    CHECK_CUDA_ERR(LaunchDerampDemod(params.slave_rectangle, params.device_slave_i, params.device_slave_q,
                                     params.device_demod_phase, params.device_demod_i, params.device_demod_q,
                                     slave_utils_->subswath_.at(0)->device_subswath_info_, params.s_burst_index));

    BilinearParams bilinear_params;

    bilinear_params.point_width = params.target_area.width;
    bilinear_params.point_height = params.target_area.height;
    bilinear_params.demod_width = params.slave_rectangle.width;
    bilinear_params.demod_height = params.slave_rectangle.height;
    bilinear_params.start_x = params.target_area.x;
    bilinear_params.start_y = params.target_area.y;
    bilinear_params.rectangle_x = params.slave_rectangle.x;
    bilinear_params.rectangle_y = params.slave_rectangle.y;
    bilinear_params.disable_reramp = disable_reramp_;
    bilinear_params.subswath_start = slave_utils_->subswath_.at(0)->lines_per_burst_ * params.s_burst_index;
    bilinear_params.subswath_end = slave_utils_->subswath_.at(0)->lines_per_burst_ * (params.s_burst_index + 1);
    bilinear_params.no_data_value = 0.0;  // TODO: placeholder.

    CHECK_CUDA_ERR(LaunchBilinearInterpolation(params.device_x_points, params.device_y_points,
                                               params.device_demod_phase, params.device_demod_i, params.device_demod_q,
                                               bilinear_params, params.device_i_results, params.device_q_results));

    LOGV << "all computations ended.";
}

bool Backgeocoding::ComputeSlavePixPos(int m_burst_index, int s_burst_index, Rectangle master_area,
                                       AzimuthAndRangeBounds az_rg_bounds, CoordMinMax* coord_min_max,
                                       double* device_x_points, double* device_y_points) {
    bool result = true;
    alus::delaunay::DelaunayTriangle2D* device_triangles{nullptr};

    double *device_lat_array{nullptr}, *device_lon_array{nullptr};
    std::vector<double> test_out;

    CHECK_CUDA_ERR(
        LaunchFillXAndY(device_x_points, device_y_points, master_area.width * master_area.height, INVALID_INDEX));

    SlavePixPosData calc_data;
    calc_data.m_burst_index = m_burst_index;
    calc_data.s_burst_index = s_burst_index;
    int xmin = master_area.x - boost::numeric_cast<int>(az_rg_bounds.range_max);
    int ymin = master_area.y - boost::numeric_cast<int>(az_rg_bounds.azimuth_max);
    int ymax = master_area.y + master_area.height + boost::numeric_cast<int>(abs(az_rg_bounds.azimuth_min));
    int xmax = master_area.x + master_area.width + boost::numeric_cast<int>(abs(az_rg_bounds.range_min));

    std::vector<double> lat_lon_min_max =
        this->ComputeImageGeoBoundary(master_utils_->subswath_.at(0).get(), m_burst_index, xmin, xmax, ymin, ymax);

    double delta = fmax(this->dem_sampling_lat_, this->dem_sampling_lon_);
    double extralat = 20 * delta;
    double extralon = 20 * delta;

    double lat_min = lat_lon_min_max.at(0) - extralat;
    double lat_max = lat_lon_min_max.at(1) + extralat;
    double lon_min = lat_lon_min_max.at(2) - extralon;
    double lon_max = lat_lon_min_max.at(3) + extralon;

    double upper_left_x =
        (lon_min + 180.0) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE_INVERTED;
    double upper_left_y =
        (60.0 - lat_max) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE_INVERTED;
    double lower_right_x =
        (lon_max + 180.0) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE_INVERTED;
    double lower_right_y =
        (60.0 - lat_min) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE_INVERTED;

    calc_data.lat_max_idx = boost::numeric_cast<int>(floor(upper_left_y));
    calc_data.lat_min_idx = boost::numeric_cast<int>(ceil(lower_right_y));
    calc_data.lon_min_idx = boost::numeric_cast<int>(floor(upper_left_x));
    calc_data.lon_max_idx = boost::numeric_cast<int>(ceil(lower_right_x));

    calc_data.num_lines = calc_data.lat_min_idx - calc_data.lat_max_idx;
    calc_data.num_pixels = calc_data.lon_max_idx - calc_data.lon_min_idx;
    calc_data.tiles = srtm3_tiles_;
    calc_data.egm = const_cast<float*>(egm96_device_array_);
    calc_data.max_lats = alus::snapengine::earthgravitationalmodel96computation::MAX_LATS;
    calc_data.max_lons = alus::snapengine::earthgravitationalmodel96computation::MAX_LONS;
    // TODO: we may have to rewire this in the future, but no idea to where atm.
    calc_data.dem_no_data_value = alus::snapengine::srtm3elevationmodel::NO_DATA_VALUE;
    calc_data.mask_out_area_without_elevation = 1;  // TODO: placeholder should be coming in with user arguments

    size_t valid_index_count = 0;
    const size_t az_rg_size = calc_data.num_lines * calc_data.num_pixels;

    calc_data.device_master_subswath = this->master_utils_->subswath_.at(0)->device_subswath_info_;
    calc_data.device_slave_subswath = this->slave_utils_->subswath_.at(0)->device_subswath_info_;

    calc_data.device_master_utils = this->master_utils_->device_sentinel_1_utils_;
    calc_data.device_slave_utils = this->slave_utils_->device_sentinel_1_utils_;

    alus::s1tbx::OrbitStateVectors* master_orbit = this->master_utils_->GetOrbitStateVectors();
    alus::s1tbx::OrbitStateVectors* slave_orbit = this->slave_utils_->GetOrbitStateVectors();

    calc_data.device_master_orbit_state_vectors = d_master_orbit_vectors_.array;
    calc_data.device_slave_orbit_state_vectors = d_slave_orbit_vectors_.array;
    calc_data.nr_of_master_vectors = d_master_orbit_vectors_.size;
    calc_data.nr_of_slave_vectors = d_slave_orbit_vectors_.size;
    calc_data.master_dt = master_orbit->GetDt();
    calc_data.slave_dt = slave_orbit->GetDt();

    CHECK_CUDA_ERR(cudaMalloc((void**)&calc_data.device_master_az, az_rg_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&calc_data.device_master_rg, az_rg_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&calc_data.device_slave_az, az_rg_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&calc_data.device_slave_rg, az_rg_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&calc_data.device_lats, az_rg_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&calc_data.device_lons, az_rg_size * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&calc_data.device_valid_index_counter, sizeof(size_t)));
    CHECK_CUDA_ERR(
        cudaMemcpy(calc_data.device_valid_index_counter, &valid_index_count, sizeof(size_t), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(LaunchSlavePixPos(calc_data));

    CHECK_CUDA_ERR(
        cudaMemcpy(&valid_index_count, calc_data.device_valid_index_counter, sizeof(size_t), cudaMemcpyDeviceToHost));

    // If we get any valid indexes then begin triangular interpolation, starting with triangulation.
    if (valid_index_count) {
        snapengine::triangularinterpolation::Window window;
        window.linelo = master_area.y;
        window.linehi = master_area.y + master_area.height - 1;
        window.pixlo = master_area.x;
        window.pixhi = master_area.x + master_area.width - 1;
        window.lines = window.linehi - window.linelo + 1;
        window.pixels = window.pixhi - window.pixlo + 1;

        snapengine::triangularinterpolation::TriangleInterpolationParams params;
        alus::snapengine::triangularinterpolation::Zdata zdata[Z_DATA_SIZE];
        alus::snapengine::triangularinterpolation::Zdata* device_zdata;

        double rg_az_ratio = master_utils_->range_spacing_ / master_utils_->azimuth_spacing_;
        std::vector<double> master_az(az_rg_size);
        std::vector<double> master_rg(az_rg_size);

        CHECK_CUDA_ERR(cudaMemcpy(master_az.data(), calc_data.device_master_az, az_rg_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(master_rg.data(), calc_data.device_master_rg, az_rg_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));

        alus::delaunay::DelaunayTriangulator triangulator;
        triangulator.TriangulateCPU2(master_az.data(), 1.0, master_rg.data(), rg_az_ratio, az_rg_size, INVALID_INDEX);

        CHECK_CUDA_ERR(cudaMalloc((void**)&device_triangles,
                                  triangulator.triangle_count_ * sizeof(alus::delaunay::DelaunayTriangle2D)));
        CHECK_CUDA_ERR(cudaMemcpy(device_triangles, triangulator.host_triangles_.data(),
                                  triangulator.triangle_count_ * sizeof(alus::delaunay::DelaunayTriangle2D),
                                  cudaMemcpyHostToDevice));

        int array_size = window.lines * window.pixels;

        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lat_array, array_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lon_array, array_size * sizeof(double)));

        zdata[0].input_arr = calc_data.device_slave_az;
        zdata[0].input_width = window.lines;
        zdata[0].input_height = window.pixels;
        zdata[0].output_arr = device_y_points;
        zdata[0].output_width = window.lines;
        zdata[0].output_height = window.pixels;
        zdata[0].min_int = std::numeric_limits<int>::max();
        zdata[0].max_int = std::numeric_limits<int>::lowest();

        zdata[1].input_arr = calc_data.device_slave_rg;
        zdata[1].input_width = window.lines;
        zdata[1].input_height = window.pixels;
        zdata[1].output_arr = device_x_points;
        zdata[1].output_width = window.lines;
        zdata[1].output_height = window.pixels;
        zdata[1].min_int = std::numeric_limits<int>::max();
        zdata[1].max_int = std::numeric_limits<int>::lowest();

        zdata[2].input_arr = calc_data.device_lats;
        zdata[2].input_width = window.lines;
        zdata[2].input_height = window.pixels;
        zdata[2].output_arr = device_lat_array;
        zdata[2].output_width = window.lines;
        zdata[2].output_height = window.pixels;
        zdata[2].min_int = std::numeric_limits<int>::max();
        zdata[2].max_int = std::numeric_limits<int>::lowest();

        zdata[3].input_arr = calc_data.device_lons;
        zdata[3].input_width = window.lines;
        zdata[3].input_height = window.pixels;
        zdata[3].output_arr = device_lon_array;
        zdata[3].output_width = window.lines;
        zdata[3].output_height = window.pixels;
        zdata[3].min_int = std::numeric_limits<int>::max();
        zdata[3].max_int = std::numeric_limits<int>::lowest();

        params.triangle_count = triangulator.triangle_count_;
        params.z_data_count = Z_DATA_SIZE;
        params.xy_ratio = rg_az_ratio;
        params.invalid_index = INVALID_INDEX;
        params.x_scale = 1;
        params.y_scale = 1;
        params.offset = 0;
        params.window = window;

        CHECK_CUDA_ERR(
            cudaMalloc((void**)&device_zdata, Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata)));
        CHECK_CUDA_ERR(cudaMemcpy(device_zdata, zdata,
                                  Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata),
                                  cudaMemcpyHostToDevice));

        CHECK_CUDA_ERR(
            snapengine::triangularinterpolation::LaunchInterpolation(device_triangles, device_zdata, params));

        CHECK_CUDA_ERR(cudaMemcpy(zdata, device_zdata,
                                  Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata),
                                  cudaMemcpyDeviceToHost));

        coord_min_max->x_min = zdata[1].min_int;
        coord_min_max->x_max = zdata[1].max_int;
        coord_min_max->y_min = zdata[0].min_int;
        coord_min_max->y_max = zdata[0].max_int;

        ElevationMaskData mask_data;
        mask_data.device_x_points = device_x_points;
        mask_data.device_y_points = device_y_points;
        mask_data.device_lat_array = device_lat_array;
        mask_data.device_lon_array = device_lon_array;

        mask_data.size = array_size;
        mask_data.tiles = srtm3_tiles_;

        int not_invalid_counter = 0;

        cudaMalloc((void**)&mask_data.not_null_counter, sizeof(int));
        cudaMemcpy(mask_data.not_null_counter, &not_invalid_counter, sizeof(int), cudaMemcpyHostToDevice);

        CHECK_CUDA_ERR(LaunchElevationMask(mask_data));

        cudaMemcpy(&not_invalid_counter, mask_data.not_null_counter, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(mask_data.not_null_counter);

        if (!not_invalid_counter) {
            result = false;
        }

        CHECK_CUDA_ERR(cudaFree(device_zdata));
        CHECK_CUDA_ERR(cudaFree(device_triangles));

        CHECK_CUDA_ERR(cudaFree(device_lat_array));
        CHECK_CUDA_ERR(cudaFree(device_lon_array));
    } else {
        result = false;
    }

    CHECK_CUDA_ERR(cudaFree(calc_data.device_master_az));
    CHECK_CUDA_ERR(cudaFree(calc_data.device_master_rg));
    CHECK_CUDA_ERR(cudaFree(calc_data.device_slave_az));
    CHECK_CUDA_ERR(cudaFree(calc_data.device_slave_rg));
    CHECK_CUDA_ERR(cudaFree(calc_data.device_lats));
    CHECK_CUDA_ERR(cudaFree(calc_data.device_lons));
    CHECK_CUDA_ERR(cudaFree(calc_data.device_valid_index_counter));

    return result;
}

// usually we use the subswath from master product.
std::vector<double> Backgeocoding::ComputeImageGeoBoundary(s1tbx::SubSwathInfo* sub_swath, int burst_index, int x_min,
                                                           int x_max, int y_min, int y_max) {
    std::vector<double> results;
    results.resize(4);

    double az_time_min = sub_swath->burst_first_line_time_[burst_index] +
                         (y_min - burst_index * sub_swath->lines_per_burst_) * sub_swath->azimuth_time_interval_;

    double az_time_max = sub_swath->burst_first_line_time_[burst_index] +
                         (y_max - burst_index * sub_swath->lines_per_burst_) * sub_swath->azimuth_time_interval_;

    double rg_time_min = sub_swath->slr_time_to_first_pixel_ +
                         x_min * master_utils_->range_spacing_ / snapengine::eo::constants::LIGHT_SPEED;

    double rg_time_max = sub_swath->slr_time_to_first_pixel_ +
                         x_max * master_utils_->range_spacing_ / snapengine::eo::constants::LIGHT_SPEED;

    double latUL = master_utils_->GetLatitude(az_time_min, rg_time_min, sub_swath);
    double lonUL = master_utils_->GetLongitude(az_time_min, rg_time_min, sub_swath);
    double latUR = master_utils_->GetLatitude(az_time_min, rg_time_max, sub_swath);
    double lonUR = master_utils_->GetLongitude(az_time_min, rg_time_max, sub_swath);
    double latLL = master_utils_->GetLatitude(az_time_max, rg_time_min, sub_swath);
    double lonLL = master_utils_->GetLongitude(az_time_max, rg_time_min, sub_swath);
    double latLR = master_utils_->GetLatitude(az_time_max, rg_time_max, sub_swath);
    double lonLR = master_utils_->GetLongitude(az_time_max, rg_time_max, sub_swath);

    double lat_min = 90.0;
    double lat_max = -90.0;
    double lon_min = 180.0;
    double lon_max = -180.0;

    std::vector<double> lats{lat_min, latUL, latUR, latLL, latLR, lat_max};
    std::vector<double> lons{lon_min, lonUL, lonUR, lonLL, lonLR, lon_max};

    lat_min = *std::min_element(lats.begin(), lats.end() - 1);
    lat_max = *std::max_element(lats.begin() + 1, lats.end());
    lon_min = *std::min_element(lons.begin(), lons.end() - 1);
    lon_max = *std::max_element(lons.begin() + 1, lons.end());

    results.at(0) = lat_min;
    results.at(1) = lat_max;
    results.at(2) = lon_min;
    results.at(3) = lon_max;

    return results;
}

AzimuthAndRangeBounds Backgeocoding::ComputeExtendedAmount(int x_0, int y_0, int w, int h) {
    AzimuthAndRangeBounds extended_amount{};

    CHECK_CUDA_ERR(LaunchComputeExtendedAmount(
        {x_0, y_0, w, h}, extended_amount,
        master_utils_->GetOrbitStateVectors()->orbit_state_vectors_computation_.data(),
        master_utils_->GetOrbitStateVectors()->orbit_state_vectors_computation_.size(),
        master_utils_->GetOrbitStateVectors()->GetDt(), *master_utils_->subswath_.at(0),
        master_utils_->device_sentinel_1_utils_, master_utils_->subswath_.at(0)->device_subswath_info_, srtm3_tiles_,
        const_cast<float*>(egm96_device_array_)));
    return extended_amount;
}

void PrepareBurstOffsetKernelArguments(BurstOffsetKernelArgs& args, PointerArray srtm3_tiles,
                                       s1tbx::Sentinel1Utils* master_utils, s1tbx::Sentinel1Utils* slave_utils) {
    s1tbx::OrbitStateVectors* master_vectors = master_utils->GetOrbitStateVectors();
    s1tbx::OrbitStateVectors* slave_vectors = slave_utils->GetOrbitStateVectors();

    snapengine::OrbitStateVectorComputation* d_master_orbit_state_vector;
    snapengine::OrbitStateVectorComputation* d_slave_orbit_state_vector;

    const size_t master_orbit_byte_size =
        sizeof(snapengine::OrbitStateVectorComputation) * master_vectors->orbit_state_vectors_computation_.size();
    const size_t slave_orbit_byte_size =
        sizeof(snapengine::OrbitStateVectorComputation) * slave_vectors->orbit_state_vectors_computation_.size();
    CHECK_CUDA_ERR(cudaMalloc(&d_master_orbit_state_vector, master_orbit_byte_size));
    CHECK_CUDA_ERR(cudaMalloc(&d_slave_orbit_state_vector, slave_orbit_byte_size));

    CHECK_CUDA_ERR(cudaMemcpy(d_master_orbit_state_vector, master_vectors->orbit_state_vectors_computation_.data(),
                              master_orbit_byte_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_slave_orbit_state_vector, slave_vectors->orbit_state_vectors_computation_.data(),
                              slave_orbit_byte_size, cudaMemcpyHostToDevice));

    args.srtm3_tiles.array = srtm3_tiles.array;
    args.srtm3_tiles.size = srtm3_tiles.size;

    args.master_sentinel_utils = master_utils->device_sentinel_1_utils_;
    args.master_subswath_info = master_utils->subswath_.at(0)->device_subswath_info_;
    args.master_orbit = d_master_orbit_state_vector;
    args.master_num_orbit_vec = master_vectors->orbit_state_vectors_computation_.size();
    args.master_dt = master_vectors->GetDt();

    args.slave_sentinel_utils = slave_utils->device_sentinel_1_utils_;
    args.slave_subswath_info = slave_utils->subswath_.at(0)->device_subswath_info_;
    args.slave_orbit = d_slave_orbit_state_vector;
    args.slave_num_orbit_vec = slave_vectors->orbit_state_vectors_computation_.size();
    args.slave_dt = slave_vectors->GetDt();

    const s1tbx::SubSwathInfo* h_master_subswath = master_utils->subswath_.at(0).get();
    int subswath_geo_grid_size = h_master_subswath->num_of_geo_lines_ * h_master_subswath->num_of_geo_points_per_line_;

    args.width = h_master_subswath->num_of_geo_points_per_line_;
    args.height = h_master_subswath->num_of_geo_lines_;

    int burst_offset = INVALID_BURST_OFFSET;
    CHECK_CUDA_ERR(cudaMalloc(&args.burst_offset, sizeof(int)));
    CHECK_CUDA_ERR(cudaMemcpy(args.burst_offset, &burst_offset, sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMalloc(&args.longitudes, sizeof(double) * subswath_geo_grid_size));
    CHECK_CUDA_ERR(cudaMalloc(&args.latitudes, sizeof(double) * subswath_geo_grid_size));

    CHECK_CUDA_ERR(cudaMemcpy(args.longitudes, h_master_subswath->longitude_[0],
                              sizeof(double) * subswath_geo_grid_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(args.latitudes, h_master_subswath->latitude_[0], sizeof(double) * subswath_geo_grid_size,
                              cudaMemcpyHostToDevice));
}

void FreeBurstOffsetArguments(BurstOffsetKernelArgs& args) {
    CHECK_CUDA_ERR(cudaFree(args.latitudes));
    CHECK_CUDA_ERR(cudaFree(args.longitudes));
    CHECK_CUDA_ERR(cudaFree(args.burst_offset));
    CHECK_CUDA_ERR(cudaFree(args.master_orbit));
    CHECK_CUDA_ERR(cudaFree(args.slave_orbit));
}

int Backgeocoding::ComputeBurstOffset() {
    BurstOffsetKernelArgs args{};

    PrepareBurstOffsetKernelArguments(args, srtm3_tiles_, this->master_utils_.get(), this->slave_utils_.get());

    int burst_offset;
    CHECK_CUDA_ERR(LaunchBurstOffsetKernel(args, &burst_offset));
    FreeBurstOffsetArguments(args);

    return burst_offset;
}

}  // namespace alus::backgeocoding
