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

namespace alus {
namespace backgeocoding {

Backgeocoding::~Backgeocoding() {
    if (device_demod_i_ != nullptr) {
        cudaFree(device_demod_i_);
        device_demod_i_ = nullptr;
    }
    if (device_demod_q_ != nullptr) {
        cudaFree(device_demod_q_);
        device_demod_q_ = nullptr;
    }

    if (device_demod_phase_ != nullptr) {
        cudaFree(device_demod_phase_);
        device_demod_phase_ = nullptr;
    }

    if (device_x_points_ != nullptr) {
        cudaFree(device_x_points_);
        device_x_points_ = nullptr;
    }

    if (device_y_points_ != nullptr) {
        cudaFree(device_y_points_);
        device_y_points_ = nullptr;
    }

    if (device_i_results_ != nullptr) {
        cudaFree(device_i_results_);
        device_i_results_ = nullptr;
    }

    if (device_q_results_ != nullptr) {
        cudaFree(device_q_results_);
        device_q_results_ = nullptr;
    }

    if (device_params_ != nullptr) {
        cudaFree(device_params_);
        device_params_ = nullptr;
    }

    if (device_slave_i_ != nullptr) {
        cudaFree(device_slave_i_);
        device_slave_i_ = nullptr;
    }
    if (device_slave_q_ != nullptr) {
        cudaFree(device_slave_q_);
        device_slave_q_ = nullptr;
    }
}

void Backgeocoding::AllocateGPUData() {
    CHECK_CUDA_ERR(cudaMalloc((void **)&device_demod_i_, this->demod_size_ * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void **)&device_demod_q_, this->demod_size_ * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void **)&device_demod_phase_, this->demod_size_ * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void **)&device_slave_i_, this->demod_size_ * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void **)&device_slave_q_, this->demod_size_ * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void **)&device_x_points_, this->tile_size_ * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void **)&device_y_points_, this->tile_size_ * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void **)&device_i_results_, this->tile_size_ * sizeof(float)));

    CHECK_CUDA_ERR(cudaMalloc((void **)&device_q_results_, this->tile_size_ * sizeof(float)));

    CHECK_CUDA_ERR(cudaMalloc((void **)&device_params_, this->param_size_ * sizeof(int)));
}

void Backgeocoding::CopySlaveTiles(double *slave_tile_i, double *slave_tile_q) {
    CHECK_CUDA_ERR(
        cudaMemcpy(this->device_slave_i_, slave_tile_i, this->demod_size_ * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(
        cudaMemcpy(this->device_slave_q_, slave_tile_q, this->demod_size_ * sizeof(double), cudaMemcpyHostToDevice));
}

void Backgeocoding::CopyGPUData() {

    CHECK_CUDA_ERR(cudaMemcpy(
        this->device_params_, this->params_.data(), this->param_size_ * sizeof(int), cudaMemcpyHostToDevice));
}

void Backgeocoding::FeedPlaceHolders() {

    this->tile_x_ = 100;
    this->tile_y_ = 100;
    this->tile_size_ = this->tile_x_ * this->tile_y_;

    std::ifstream param_stream(params_file_);
    if (!param_stream.is_open()) {
        throw std::ios::failure("Params file not open.");
    }
    this->param_size_ = 15;
    this->params_.resize(this->param_size_);

    for (int i = 0; i < this->param_size_; i++) {
        param_stream >> params_.at(i);
    }

    param_stream.close();

    this->demod_size_ = 108 * 108;

    this->dem_sampling_lat_ = 8.333333333333334E-4;
    this->dem_sampling_lon_ = 8.333333333333334E-4;
}

void Backgeocoding::PrepareToCompute() {
    this->AllocateGPUData();

    std::cout << "making new results with size:" << this->tile_size_ << '\n';
    this->q_result_.resize(this->tile_size_);
    this->i_result_.resize(this->tile_size_);
    this->slave_utils_ = std::make_unique<s1tbx::Sentinel1Utils>(2);
    this->slave_utils_->SetPlaceHolderFiles(this->slave_orbit_state_vectors_file_,
                                            this->dc_estimate_list_file_,
                                            this->azimuth_list_file_,
                                            this->slave_burst_line_time_file_,
                                            this->slave_geo_location_file_);
    this->slave_utils_->ReadPlaceHolderFiles();
    this->slave_utils_->ComputeDopplerRate();
    this->slave_utils_->ComputeReferenceTime();
    this->slave_utils_->subswath_[0].HostToDevice();
    this->slave_utils_->HostToDevice();

    this->master_utils_ = std::make_unique<s1tbx::Sentinel1Utils>(1);
    this->master_utils_->SetPlaceHolderFiles(this->master_orbit_state_vectors_file_,
                                             this->dc_estimate_list_file_,
                                             this->azimuth_list_file_,
                                             this->master_burst_line_time_file_,
                                             this->master_geo_location_file_);
    this->master_utils_->ReadPlaceHolderFiles();
    this->master_utils_->ComputeDopplerRate();
    this->master_utils_->ComputeReferenceTime();
    this->master_utils_->subswath_[0].HostToDevice();
    this->master_utils_->HostToDevice();

    alus::s1tbx::OrbitStateVectors *master_orbit = this->master_utils_->GetOrbitStateVectors();
    alus::s1tbx::OrbitStateVectors *slave_orbit = this->slave_utils_->GetOrbitStateVectors();
    master_orbit->HostToDevice();
    slave_orbit->HostToDevice();

    this->PrepareSrtm3Data();
}

void Backgeocoding::PrepareSrtm3Data() {
    this->egm96_ = std::make_unique<snapengine::EarthGravitationalModel96>(this->grid_file_);
    this->egm96_->HostToDevice();

    // placeholders
    Point srtm_41_01 = {41, 1};
    Point srtm_42_01 = {42, 1};
    std::vector<Point> files;
    files.push_back(srtm_41_01);
    files.push_back(srtm_42_01);
    this->srtm3Dem_ = std::make_unique<snapengine::SRTM3ElevationModel>(files, this->srtms_directory_);
    this->srtm3Dem_->ReadSrtmTiles(this->egm96_.get());
    this->srtm3Dem_->HostToDevice();
}

void Backgeocoding::ComputeTile(Rectangle slave_rect, double *slave_tile_i, double *slave_tile_q) {
    this->CopySlaveTiles(slave_tile_i, slave_tile_q);

    this->CopyGPUData();

    std::vector<double> extended_amount;
    extended_amount.push_back(-0.01773467106249882);
    extended_amount.push_back(0.0);
    extended_amount.push_back(-3.770974349203243);
    extended_amount.push_back(3.8862058607542167);

    // TODO: check out all of those placeholders :)
    this->ComputeSlavePixPos(11, 11, 4000, 17000, 100, 100, extended_amount);

    // TODO: using placeholder as number 11
    CHECK_CUDA_ERR(this->LaunchDerampDemodComp(slave_rect, 11));

    CHECK_CUDA_ERR(this->LaunchBilinearComp());

    this->GetGPUEndResults();
    std::cout << "all computations ended." << '\n';
}

void Backgeocoding::GetGPUEndResults() {
    CHECK_CUDA_ERR(cudaMemcpy(
        this->i_result_.data(), this->device_i_results_, this->tile_size_ * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(cudaMemcpy(
        this->q_result_.data(), this->device_q_results_, this->tile_size_ * sizeof(float), cudaMemcpyDeviceToHost));
}

bool Backgeocoding::ComputeSlavePixPos(int m_burst_index,
                                       int s_burst_index,
                                       int x0,
                                       int y0,
                                       int w,
                                       int h,
                                       std::vector<double> extended_amount) {

    bool result = true;
    alus::delaunay::DelaunayTriangle2D *device_triangles{nullptr};

    double *device_lat_array{nullptr}, *device_lon_array{nullptr};
    std::vector<double> test_out;

    SlavePixPosData calc_data;
    calc_data.m_burst_index = m_burst_index;
    calc_data.s_burst_index = s_burst_index;
    int xmin = x0 - (int)extended_amount.at(3);
    int ymin = y0 - (int)extended_amount.at(1);
    int ymax = y0 + h + (int)abs(extended_amount.at(0));
    int xmax = x0 + w + (int)abs(extended_amount.at(2));

    std::vector<double> lat_lon_min_max =
        this->ComputeImageGeoBoundary(&this->master_utils_->subswath_[0], m_burst_index, xmin, xmax, ymin, ymax);

    double delta = fmax(this->dem_sampling_lat_, this->dem_sampling_lon_);
    double extralat = 20 * delta;
    double extralon = 20 * delta;

    double lat_min = lat_lon_min_max.at(0) - extralat;
    double lat_max = lat_lon_min_max.at(1) + extralat;
    double lon_min = lat_lon_min_max.at(2) - extralon;
    double lon_max = lat_lon_min_max.at(3) + extralon;

    double upper_left_x = (lon_min + 180.0) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    double upper_left_y = (60.0 - lat_max) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    double lower_right_x = (lon_max + 180.0) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    double lower_right_y = (60.0 - lat_min) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;

    calc_data.lat_max_idx = (int)floor(upper_left_y);
    calc_data.lat_min_idx = (int)ceil(lower_right_y);
    calc_data.lon_min_idx = (int)floor(upper_left_x);
    calc_data.lon_max_idx = (int)ceil(lower_right_x);

    calc_data.num_lines = calc_data.lat_min_idx - calc_data.lat_max_idx;
    calc_data.num_pixels = calc_data.lon_max_idx - calc_data.lon_min_idx;
    calc_data.tiles.array = this->srtm3Dem_->device_srtm3_tiles_;
    calc_data.egm = this->egm96_->device_egm_;
    calc_data.max_lats = alus::snapengine::earthgravitationalmodel96::MAX_LATS;
    calc_data.max_lons = alus::snapengine::earthgravitationalmodel96::MAX_LONS;
    // TODO: we may have to rewire this in the future, but no idea to where atm.
    calc_data.dem_no_data_value = alus::snapengine::srtm3elevationmodel::NO_DATA_VALUE;
    calc_data.mask_out_area_without_elevation = 1;  // TODO: placeholder should be coming in with user arguments

    size_t valid_index_count = 0;
    const size_t az_rg_size = calc_data.num_lines * calc_data.num_pixels;

    calc_data.device_master_subswath = this->master_utils_->subswath_[0].device_subswath_info_;
    calc_data.device_slave_subswath = this->slave_utils_->subswath_[0].device_subswath_info_;

    calc_data.device_master_utils = this->master_utils_->device_sentinel_1_utils_;
    calc_data.device_slave_utils = this->slave_utils_->device_sentinel_1_utils_;

    alus::s1tbx::OrbitStateVectors *master_orbit = this->master_utils_->GetOrbitStateVectors();
    alus::s1tbx::OrbitStateVectors *slave_orbit = this->slave_utils_->GetOrbitStateVectors();

    calc_data.device_master_orbit_state_vectors = master_orbit->device_orbit_state_vectors_;
    calc_data.device_slave_orbit_state_vectors = slave_orbit->device_orbit_state_vectors_;
    calc_data.nr_of_master_vectors = master_orbit->orbitStateVectors.size();
    calc_data.nr_of_slave_vectors = slave_orbit->orbitStateVectors.size();
    calc_data.master_dt = master_orbit->GetDT();
    calc_data.slave_dt = slave_orbit->GetDT();

    CHECK_CUDA_ERR(cudaMalloc((void **)&calc_data.device_master_az, az_rg_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&calc_data.device_master_rg, az_rg_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&calc_data.device_slave_az, az_rg_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&calc_data.device_slave_rg, az_rg_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&calc_data.device_lats, az_rg_size * sizeof(double)));
    CHECK_CUDA_ERR(cudaMalloc((void **)&calc_data.device_lons, az_rg_size * sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void **)&calc_data.device_valid_index_counter, sizeof(size_t)));
    CHECK_CUDA_ERR(cudaMemcpy(
        calc_data.device_valid_index_counter, &valid_index_count, sizeof(size_t), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(this->LaunchSlavePixPosComp(calc_data));

    CHECK_CUDA_ERR(cudaMemcpy(&valid_index_count, calc_data.device_valid_index_counter, sizeof(size_t), cudaMemcpyDeviceToHost));

    //If we get any valid indexes then begin triangular interpolation, starting with triangulation.
    if(valid_index_count) {

        snapengine::triangularinterpolation::Window window;
        window.linelo = y0;
        window.linehi = y0 + h - 1;
        window.pixlo = x0;
        window.pixhi = x0 + w - 1;
        window.lines = window.linehi - window.linelo + 1;
        window.pixels = window.pixhi - window.pixlo + 1;

        snapengine::triangularinterpolation::InterpolationParams params;
        alus::snapengine::triangularinterpolation::Zdata zdata[Z_DATA_SIZE];
        alus::snapengine::triangularinterpolation::Zdata *device_zdata;

        double rg_az_ratio = master_utils_->range_spacing_ / master_utils_->azimuth_spacing_;
        std::vector<double> master_az(az_rg_size);
        std::vector<double> master_rg(az_rg_size);

        CHECK_CUDA_ERR(cudaMemcpy(master_az.data(), calc_data.device_master_az, az_rg_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(master_rg.data(), calc_data.device_master_rg, az_rg_size * sizeof(double), cudaMemcpyDeviceToHost));

        alus::delaunay::DelaunayTriangulator triangulator;
        triangulator.TriangulateCPU2(master_az.data(), 1.0, master_rg.data(), rg_az_ratio, az_rg_size);

        CHECK_CUDA_ERR(cudaMalloc((void **)&device_triangles,
                                  triangulator.triangle_count_ * sizeof(alus::delaunay::DelaunayTriangle2D)));
        CHECK_CUDA_ERR(cudaMemcpy(device_triangles,
                                  triangulator.host_triangles_.data(),
                                  triangulator.triangle_count_ * sizeof(alus::delaunay::DelaunayTriangle2D),
                                  cudaMemcpyHostToDevice));

        int array_size = window.lines * window.pixels;

        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lat_array, array_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lon_array, array_size * sizeof(double)));

        zdata[0].input_arr = calc_data.device_slave_az;
        zdata[0].input_width = window.lines;
        zdata[0].input_height = window.pixels;
        zdata[0].output_arr = device_y_points_;
        zdata[0].output_width = window.lines;
        zdata[0].output_height = window.pixels;

        zdata[1].input_arr = calc_data.device_slave_rg;
        zdata[1].input_width = window.lines;
        zdata[1].input_height = window.pixels;
        zdata[1].output_arr = device_x_points_;
        zdata[1].output_width = window.lines;
        zdata[1].output_height = window.pixels;

        zdata[2].input_arr = calc_data.device_lats;
        zdata[2].input_width = window.lines;
        zdata[2].input_height = window.pixels;
        zdata[2].output_arr = device_lat_array;
        zdata[2].output_width = window.lines;
        zdata[2].output_height = window.pixels;

        zdata[3].input_arr = calc_data.device_lons;
        zdata[3].input_width = window.lines;
        zdata[3].input_height = window.pixels;
        zdata[3].output_arr = device_lon_array;
        zdata[3].output_width = window.lines;
        zdata[3].output_height = window.pixels;

        params.triangle_count = triangulator.triangle_count_;
        params.z_data_count = Z_DATA_SIZE;
        params.xy_ratio = rg_az_ratio;
        params.invalid_index = INVALID_INDEX;
        params.x_scale = 1;
        params.y_scale = 1;
        params.offset = 0;
        params.window = window;

        CHECK_CUDA_ERR(cudaMalloc((void**)&device_zdata, Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata)));
        CHECK_CUDA_ERR(cudaMemcpy(device_zdata,
                                  zdata,
                                  Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata),
                                  cudaMemcpyHostToDevice));

        CHECK_CUDA_ERR(snapengine::triangularinterpolation::LaunchInterpolation(device_triangles, device_zdata, params));

        CHECK_CUDA_ERR(cudaFree(device_zdata));
        CHECK_CUDA_ERR(cudaFree(device_triangles));

        CHECK_CUDA_ERR(cudaFree(device_lat_array));
        CHECK_CUDA_ERR(cudaFree(device_lon_array));
    }else{
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
std::vector<double> Backgeocoding::ComputeImageGeoBoundary(s1tbx::SubSwathInfo *sub_swath,
                                                           int burst_index,
                                                           int x_min,
                                                           int x_max,
                                                           int y_min,
                                                           int y_max) {
    std::vector<double> results;
    results.resize(4);

    double az_time_min = sub_swath->burst_first_line_time_[burst_index] +
                         (y_min - burst_index * sub_swath->lines_per_burst_) * sub_swath->azimuth_time_interval_;

    double az_time_max = sub_swath->burst_first_line_time_[burst_index] +
                         (y_max - burst_index * sub_swath->lines_per_burst_) * sub_swath->azimuth_time_interval_;

    double rg_time_min = sub_swath->slr_time_to_first_pixel_ +
                         x_min * master_utils_->range_spacing_ / alus::snapengine::constants::lightSpeed;

    double rg_time_max = sub_swath->slr_time_to_first_pixel_ +
                         x_max * master_utils_->range_spacing_ / alus::snapengine::constants::lightSpeed;

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

void Backgeocoding::SetPlaceHolderFiles(std::string params_file) {
    this->params_file_ = params_file;
}

void Backgeocoding::SetSRTMDirectory(std::string directory) { this->srtms_directory_ = directory; }

void Backgeocoding::SetEGMGridFile(std::string grid_file) { this->grid_file_ = grid_file; }

void Backgeocoding::SetSentinel1Placeholders(std::string dc_estimate_list_file,
                                             std::string azimuth_list_file,
                                             std::string master_burst_line_time_file,
                                             std::string slave_burst_line_time_file,
                                             std::string master_geo_location_file,
                                             std::string slave_geo_location_file) {
    this->dc_estimate_list_file_ = dc_estimate_list_file;
    this->azimuth_list_file_ = azimuth_list_file;
    this->master_burst_line_time_file_ = master_burst_line_time_file;
    this->slave_burst_line_time_file_ = slave_burst_line_time_file;
    this->master_geo_location_file_ = master_geo_location_file;
    this->slave_geo_location_file_ = slave_geo_location_file;
}

cudaError_t Backgeocoding::LaunchBilinearComp() {
    cudaError_t status;
    dim3 grid_size(5, 5);
    dim3 block_size(20, 20);

    LaunchBilinearInterpolation(grid_size,
                                block_size,
                                this->device_x_points_,
                                this->device_y_points_,
                                this->device_demod_phase_,
                                this->device_demod_i_,
                                this->device_demod_q_,
                                this->device_params_,
                                0.0,
                                this->device_i_results_,
                                this->device_q_results_);
    status = cudaGetLastError();

    return status;
}

cudaError_t Backgeocoding::LaunchDerampDemodComp(Rectangle slave_rect, int s_burst_index) {
    cudaError_t status;
    dim3 grid_size(6, 6);
    dim3 block_size(20, 20);

    LaunchDerampDemod(grid_size,
                      block_size,
                      slave_rect,
                      this->device_slave_i_,
                      this->device_slave_q_,
                      this->device_demod_phase_,
                      this->device_demod_i_,
                      this->device_demod_q_,
                      this->slave_utils_->subswath_.at(0).device_subswath_info_,
                      s_burst_index);
    status = cudaGetLastError();

    return status;
}

cudaError_t Backgeocoding::LaunchSlavePixPosComp(SlavePixPosData calc_data) {
    dim3 block_size(20, 20);
    dim3 grid_size(cuda::GetGridDim(20, calc_data.num_lines), cuda::GetGridDim(20, calc_data.num_pixels));

    LaunchSlavePixPos(grid_size, block_size, calc_data);
    return cudaGetLastError();
}
void Backgeocoding::SetOrbitVectorsFiles(std::string master_orbit_state_vectors_file,
                                         std::string slave_orbit_state_vectors_file) {
    this->master_orbit_state_vectors_file_ = master_orbit_state_vectors_file;
    this->slave_orbit_state_vectors_file_ = slave_orbit_state_vectors_file;
}

}  // namespace backgeocoding
}  // namespace alus
