#include "deramp_demod.cuh"

namespace alus {
__global__ void DerampDemod(alus::Rectangle rectangle,
                            double *slave_i,
                            double *slave_q,
                            double *demod_phase,
                            double *demod_i,
                            double *demod_q,
                            alus::DeviceSubswathInfo *subSwath, int s_burst_index){


    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
    const int global_index = rectangle.width * idy + idx;
    const int first_line_in_burst = s_burst_index * subSwath->lines_per_burst;
    const int y = rectangle.y + idy;
    const int x = rectangle.x + idx;
    double ta, kt, deramp, demod;
    double value_i, value_q, value_phase, cos_phase, sin_phase;

    if(idx < rectangle.width && idy < rectangle.height){

        ta = (y - first_line_in_burst)* subSwath->azimuth_time_interval;
        kt = subSwath->device_doppler_rate[s_burst_index *subSwath->doppler_size_y + x];
        deramp = -alus::snapengine::constants::PI * kt * pow(ta -
                subSwath->device_reference_time[s_burst_index *subSwath->doppler_size_y + x],2);
        demod = -alus::snapengine::constants::TWO_PI *
                subSwath->device_doppler_centroid[s_burst_index *subSwath->doppler_size_y +
                                                                                 x] * ta;
        value_phase = deramp + demod;

        demod_phase[global_index] = value_phase;

        value_i = slave_i[global_index];
        value_q = slave_q[global_index];

        cos_phase = cos(value_phase);
        sin_phase = sin(value_phase);
        demod_i[global_index] = value_i * cos_phase - value_q * sin_phase;
        demod_q[global_index] = value_i * sin_phase + value_q * cos_phase;

    }
}

cudaError_t LaunchDerampDemod(
        dim3 grid_size,
        dim3 block_size,
        alus::Rectangle rectangle,
        double *slave_i,
        double *slave_q,
        double *demod_phase,
        double *demod_i,
        double *demod_q,
        alus::DeviceSubswathInfo *sub_swath,
        int s_burst_index){
    DerampDemod<<<grid_size, block_size>>>(
        rectangle, slave_i, slave_q, demod_phase, demod_i, demod_q, sub_swath, s_burst_index);
    return cudaGetLastError();
}

}//namespace
