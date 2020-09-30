#include "terrain_correction.hpp"

#include <algorithm>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <numeric>

#include "cuda_util.hpp"
#include "dem.hpp"
#include "local_dem.cuh"

namespace alus {

TerrainCorrection::TerrainCorrection(alus::Dataset coh_ds, alus::Dataset dem)
    : coh_ds_{std::move(coh_ds)}, dem_ds_{std::move(dem)}, coh_ds_elevations_(coh_ds_.GetDataBuffer().size()) {
    FillMetadata();
}

/**
 * Method for launching RangeDoppler Terrain Correction algorithm.
 *
 * @todo WIP: The method is going to be renamed and completed as part of SNAPGPU-119 and SNAPGUP-121 issues
 */
void TerrainCorrection::DoWork() { LocalDemCuda(); }

void TerrainCorrection::LocalDemCpu() {
    auto const result = dem_ds_.GetLocalDemFor(coh_ds_, 0, 0, coh_ds_.GetXSize(), coh_ds_.GetYSize());
}

void TerrainCorrection::LocalDemCuda() {
    auto const h_dem_array = dem_ds_.GetData();

    double* d_dem_array;
    double* d_product_array;

    try {
        CHECK_CUDA_ERR(cudaMalloc(&d_dem_array, sizeof(double) * h_dem_array.size()));
        CHECK_CUDA_ERR(cudaMalloc(&d_product_array, sizeof(double) * coh_ds_elevations_.size()));
        CHECK_CUDA_ERR(
            cudaMemcpy(d_dem_array, h_dem_array.data(), sizeof(double) * h_dem_array.size(), cudaMemcpyHostToDevice));

        struct LocalDemKernelArgs kernel_args {};
        kernel_args.dem_cols = dem_ds_.GetColumnCount();
        kernel_args.dem_rows = dem_ds_.GetRowCount();
        kernel_args.target_cols = coh_ds_.GetXSize();
        kernel_args.target_rows = coh_ds_.GetYSize();
        dem_ds_.FillGeoTransform(kernel_args.dem_origin_lon,
                                 kernel_args.dem_origin_lat,
                                 kernel_args.dem_pixel_size_lon,
                                 kernel_args.dem_pixel_size_lat);
        coh_ds_.FillGeoTransform(kernel_args.target_origin_lon,
                                 kernel_args.target_origin_lat,
                                 kernel_args.target_pixel_size_lon,
                                 kernel_args.target_pixel_size_lat);

        CHECK_CUDA_ERR(cudaGetLastError());

        RunElevationKernel(d_dem_array, d_product_array, kernel_args);

        CHECK_CUDA_ERR(cudaDeviceSynchronize());
        CHECK_CUDA_ERR(cudaGetLastError());

        CHECK_CUDA_ERR(cudaMemcpy(coh_ds_elevations_.data(),
                                  d_product_array,
                                  sizeof(double) * coh_ds_elevations_.size(),
                                  cudaMemcpyDeviceToHost));
    } catch (alus::CudaErrorException const& cudaEx) {
        cudaFree(d_dem_array);
        cudaFree(d_product_array);

        throw;
    }
    cudaFree(d_dem_array);
    cudaFree(d_product_array);
}

TerrainCorrection::~TerrainCorrection() {}

// Only the first three vectors are being copied as later the list of vectors should be populated from .dim file
std::vector<alus::snapengine::OrbitStateVector> GetOrbitStateVectors() {
    std::vector<alus::snapengine::OrbitStateVector> vectors;
    vectors.push_back({alus::snapengine::old::Utc("15-JUL-2019 16:04:33.800577"),
                       3727040.7077331543,
                       1103842.85256958,
                       5902738.6076049805,
                       -5180.233733266592,
                       -3857.165526404977,
                       3982.913521885872});
    vectors.push_back({alus::snapengine::old::Utc("15-JUL-2019 16:04:34.800577"),
                       3721858.106201172,
                       1099985.447479248,
                       5906718.189788818,
                       -5184.967764496803,
                       -3857.643955528736,
                       3976.251023232937});
    vectors.push_back({alus::snapengine::old::Utc("15-JUL-2019 16:04:35.800577"),
                       3716670.7736206055,
                       1096127.5664367676,
                       5910691.107452393,
                       -5189.69604575634,
                       -3858.1173707023263,
                       3969.5840579867363});
    return vectors;
}

void TerrainCorrection::FillMetadata() {
    metadata_.product = "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6";
    metadata_.product_type = metadata::ProductType::SLC;
    metadata_.sph_descriptor = "Sentinel-1 IW Level-1 SLC Product";
    metadata_.mission = "SENTINEL-1A";
    metadata_.acquisition_mode = metadata::AcquisitionMode::IW;
    metadata_.antenna_pointing = metadata::AntennaDirection::RIGHT;
    metadata_.beams = "";
    metadata_.swath = alus::metadata::Swath::IW1;
    metadata_.proc_time = alus::snapengine::old::Utc("15-JUL-2019 18:36:47.607732");
    metadata_.processing_system_identifier = "ESA Sentinel-1 IPF 003.10";
    metadata_.orbit_cycle = 175;
    metadata_.rel_orbit = 58;
    metadata_.abs_orbit = 28130;
    metadata_.state_vector_time = alus::snapengine::old::Utc("15-JUL-2019 16:03:37.964246");
    metadata_.vector_source = "";
    metadata_.incidence_near = 29.786540985107422;
    metadata_.incidence_far = 36.43135039339361;
    metadata_.slice_num = 24;
    metadata_.first_line_time = alus::snapengine::old::Utc("15-JUL-2019 16:04:43.800577");
    metadata_.last_line_time = alus::snapengine::old::Utc("15-JUL-2019 16:04:46.885967");
    metadata_.first_near_lat = 58.21324157714844;
    metadata_.first_near_long = 21.98597526550293;
    metadata_.first_far_lat = 58.392906188964844;
    metadata_.first_far_long = 23.64056968688965;
    metadata_.last_near_lat = 58.3963737487793;
    metadata_.last_near_long = 21.90845489501953;
    metadata_.last_far_lat = 58.57649612426758;
    metadata_.last_far_long = 23.571735382080078;
    metadata_.pass = alus::metadata::Pass::ASCENDING;
    metadata_.sample_type = alus::metadata::SampleType::COMPLEX;
    metadata_.mds1_tx_rx_polar = alus::metadata::Polarisation::VH;
    metadata_.mds2_tx_rx_polar = alus::metadata::Polarisation::VH;
    metadata_.azimuth_looks = 1.0;
    metadata_.range_looks = 1.0;
    metadata_.range_spacing = 2.329562;
    metadata_.azimuth_spacing = 13.91157;
    metadata_.pulse_repetition_frequency = 1717.128973878037;
    metadata_.radar_frequency = 5405.000454334349;
    metadata_.line_time_interval = 0.002055556299999998;
    metadata_.total_size = 158871714;
    metadata_.num_output_lines = 1500;
    metadata_.num_samples_per_line = 23278;
    metadata_.subset_offset_x = 0;
    metadata_.subset_offset_y = 3004;
    metadata_.srgr_flag = false;
    metadata_.avg_scene_height = 23.65084248584435;
    metadata_.map_projection = "";
    metadata_.is_terrain_corrected = false;
    metadata_.dem = "";
    metadata_.geo_ref_system = "";
    metadata_.lat_pixel_res = 99999.0;
    metadata_.long_pixel_res = 99999.0;
    metadata_.slant_range_to_first_pixel = 799303.6132771898;
    metadata_.ant_elev_corr_flag = false;
    metadata_.range_spread_comp_flag = false;
    metadata_.replica_power_corr_flag = false;
    metadata_.abs_calibration_flag = false;
    metadata_.calibration_factor = 99999.0;
    metadata_.chirp_power = 99999.0;
    metadata_.inc_angle_comp_flag = false;
    metadata_.ref_inc_angle = 99999.0;
    metadata_.ref_slant_range = 99999.0;
    metadata_.ref_slant_range_exp = 99999.0;
    metadata_.rescaling_factor = 99999.0;
    metadata_.bistatic_correction_applied = true;
    metadata_.range_sampling_rate = 64.34523812571427;
    metadata_.range_bandwidth = 56.5;
    metadata_.azimuth_bandwidth = 327.0;
    metadata_.multilook_flag = false;
    metadata_.coregistered_stack = false;
    metadata_.external_calibration_file = "";
    metadata_.orbit_state_vector_file =
        "Sentinel Precise S1A_OPER_AUX_POEORB_OPOD_20190804T120708_V20190714T225942_20190716T005942.EOF.zip";
    metadata_.metadata_version = "6.0";
    metadata_.centre_lat = 58.86549898479201;
    metadata_.centre_lon = 24.04291372551365;
    metadata_.centre_heading = 349.45641790421365;
    metadata_.centre_heading_2 = 169.4528423214929;
    metadata_.first_valid_pixel = 1919;
    metadata_.last_valid_pixel = 22265;
    metadata_.slr_time_to_first_valid_pixel = 0.0026811016131539564;
    metadata_.slr_time_to_last_valid_pixel = 0.0028392018828145094;
    metadata_.first_valid_line_time = 6.165218838437437E8;
    metadata_.last_valid_line_time = 6.165218868469114E8;
    metadata_.orbit_state_vectors = GetOrbitStateVectors();
}


alus::Dataset TerrainCorrection::ExecuteTerrainCorrection() { return alus::Dataset("goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb_TC.tif"); }
}  // namespace alus
