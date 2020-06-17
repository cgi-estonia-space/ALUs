#pragma once

#include <memory>
#include <cmath>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_util.hpp"
#include "earth_gravitational_model96.h"
#include "general_constants.h"
#include "pointer_holders.h"
#include "sentinel1_utils.h"
#include "srtm3_elevation_model.h"
#include "srtm3_elevation_model_constants.h"

#include "bilinear.cuh"
#include "deramp_demod.cuh"
#include "slave_pixpos.cuh"



namespace alus {

class Backgeocoding{
private:
    std::vector<float> q_result_;
    std::vector<float> i_result_;
    std::vector<double> x_points_; //slave pixel pos x
    std::vector<double> y_points_; //slave pixel pos y
    std::vector<int> params_;
    double *device_x_points_{nullptr}, *device_y_points_{nullptr};
    double *device_demod_i_{nullptr}, *device_demod_q_{nullptr}, *device_demod_phase_{nullptr};
    float *device_i_results_{nullptr}, *device_q_results_{nullptr}; //I phase and Q pahse
    double *device_slave_i_{nullptr}, *device_slave_q_{nullptr};
    int *device_params_{nullptr};

    int tile_x_, tile_y_, param_size_, tile_size_, demod_size_;

    std::unique_ptr<Sentinel1Utils> master_utils_;
    std::unique_ptr<Sentinel1Utils> slave_utils_;
    double dem_sampling_lat_ = 0.0;
    double dem_sampling_lon_ = 0.0;
    std::unique_ptr<snapengine::EarthGravitationalModel96> egm96_;
    std::unique_ptr<snapengine::SRTM3ElevationModel> srtm3Dem_;



    void AllocateGPUData();
    void CopySlaveTiles(double *slave_tile_i, double *slave_tile_q);
    void CopyGPUData();
    cudaError_t LaunchBilinearComp();
    cudaError_t LaunchDerampDemodComp(Rectangle slave_rect, int s_burst_index);
    cudaError_t LaunchSlavePixPosComp(SlavePixPosData calc_data);
    void GetGPUEndResults();
    void PrepareSrtm3Data();

    std::vector<double> ComputeImageGeoBoundary(
            SubSwathInfo *sub_swath,
            int burst_index,
            int x_min,
            int x_max,
            int y_min,
            int y_max);
    void ComputeSlavePixPos(
            int m_burst_index,
            int s_burst_index,
            int x0,
            int y0,
            int w,
            int h,
            std::vector<double> extended_amount);
//            double **slavePixelPosAz,
//            double **slavePixelPosRg); add those later.



    //placeholder files
    std::string params_file_ = "../test/goods/backgeocoding/params.txt";
    std::string x_points_file_ = "../test/goods/backgeocoding/xPoints.txt";
    std::string y_points_file_ = "../test/goods/backgeocoding/yPoints.txt";

    std::string orbit_state_vectors_file_ = "../test/goods/backgeocoding/orbitStateVectors.txt";
    std::string dc_estimate_list_file_ = "../test/goods/backgeocoding/dcEstimateList.txt";
    std::string azimuth_list_file_ = "../test/goods/backgeocoding/azimuthList.txt";
    std::string burst_line_time_file_ = "../test/goods/backgeocoding/burstLineTimes.txt";
    std::string geo_location_file_ = "../test/goods/backgeocoding/geoLocation.txt";

    //std::string srtm_41_01File = "../test/goods/srtm_41_01.tif";
    //std::string srtm_42_01File = "../test/goods/srtm_42_01.tif";
    std::string srtms_directory_ = "../test/goods/";
    std::string grid_file_ = "../test/goods/ww15mgh_b.grd";

public:

    void FeedPlaceHolders();
    void PrepareToCompute();
    void ComputeTile(Rectangle slave_rect, double *slave_tile_i, double *slave_tile_q);
    Backgeocoding() = default;
    ~Backgeocoding();

    void SetPlaceHolderFiles(std::string params_file,std::string x_points_file, std::string y_points_file);
    void SetSentinel1Placeholders(
        std::string orbit_state_vectors_file,
        std::string dc_estimate_list_file,
        std::string azimuth_list_file,
        std::string burst_line_time_file,
        std::string geo_location_file);

    void SetSRTMDirectory(std::string directory);
    void SetEGMGridFile(std::string grid_file);

    float const *GetIResult(){
        return this->i_result_.data();
    }

    float const *GetQResult(){
        return this->q_result_.data();
    }
};

}//namespace
