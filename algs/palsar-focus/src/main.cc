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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include "algorithm_exception.h"
#include "alus_log.h"
#include "app_error_code.h"
#include "app_utils.h"
#include "binary_output.h"
#include "cli_args.h"
#include "cuda_cleanup.h"
#include "cuda_device_init.h"
#include "cuda_workplace.h"
#include "cufft_plan.h"
#include "dc_bias.h"
#include "doppler_centroid.h"
#include "gdal_management.h"
#include "img_output.h"
#include "palsar_file.h"
#include "palsar_metadata.h"
#include "processing_velocity_estimate.h"
#include "range_compression.h"
#include "range_doppler_algorithm.h"

namespace {

constexpr std::string_view ALG_NAME = "PALSAR focus";

constexpr double TO_GB = 1024.0 * 1024.0 * 1024.0;

auto TimeStart() { return std::chrono::steady_clock::now(); }

void TimeStop(std::chrono::steady_clock::time_point start, const char* msg) {
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    LOGD << msg << " - completed in " << diff << " ms";
}

void TimeStopFileIO(std::chrono::steady_clock::time_point start, size_t bytes, const char* msg) {
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double gb_size = bytes / TO_GB;
    char str[100] = {};
    snprintf(str, 100, "File IO: size = %.2f GB, time = %ld ms, speed = ~%.2f GB/s", gb_size, diff,
             gb_size / (0.001 * diff));
    LOGD << msg << " - " << str;
}

void ExceptionMessagePrint(const std::exception& e) {
    LOGE << "Caught an exception" << std::endl << e.what();
    LOGE << "Exiting.";
}
}  // namespace

namespace alus::palsar {

void EstimatePaddings(size_t max_device_memory, size_t free_device_memory, int range_size, int azimuth_size,
                      int chirp_size, int& range_pad, int& azimuth_pad) {
    auto fft_sizes = GetOptimalFFTSizes();

    constexpr size_t ELEM_SIZE = 8;  // IQ Float32

    {
        auto range_it = fft_sizes.upper_bound(range_size + chirp_size);
        auto azimuth_it = fft_sizes.upper_bound(azimuth_size);
        if (range_it == fft_sizes.end() || azimuth_it == fft_sizes.end()) {
            return;
        }

        const int range_calc = range_it->first;
        const int azimuth_calc = azimuth_it->first;

        size_t padded_size = range_calc * azimuth_calc * ELEM_SIZE;
        size_t peak_usage = 2 * padded_size;
        peak_usage *= 1.05;
        LOGD << "Estimated mem usage " << peak_usage / TO_GB << " GB free = " << free_device_memory / TO_GB << " GB"
             << " allowed to use " << max_device_memory / TO_GB << " GB";
        if (peak_usage < free_device_memory && peak_usage < max_device_memory) {
            auto r = range_it->second;
            LOGD << "range " << range_size << " -> " << range_calc << " (2^" << r[0] << " * 3^" << r[1] << " * 5^"
                 << r[2] << " * 7^" << r[3] << ")";
            auto a = azimuth_it->second;
            LOGD << "azimuth " << azimuth_size << " -> " << azimuth_calc << " (2^" << a[0] << " * 3^" << a[1] << " * 5^"
                 << a[2] << " * 7^" << a[3] << ")";
            range_pad = range_calc;
            azimuth_pad = azimuth_calc;
        }
    }
}

CudaMallocCleanup BuildIQ8Gpu(FileSystem& files, SARMetadata& metadata) {
    IQ8* d_img_data;
    IQ8* d_img_shifted;
    const size_t byte_size_raw = metadata.img.record_length * metadata.img.azimuth_size;
    const size_t byte_size_corrected = metadata.img.range_size * metadata.img.azimuth_size * 2;
    CHECK_CUDA_ERR(cudaMalloc(&d_img_data, byte_size_raw));
    CudaMallocCleanup clean_data(d_img_data);

    CHECK_CUDA_ERR(cudaMalloc(&d_img_shifted, byte_size_corrected));
    CudaMallocCleanup cleanup_shift(d_img_shifted);

    uint32_t min_slant_range = UINT32_MAX;
    uint32_t prev_slant_range = UINT32_MAX;
    std::vector<uint32_t> slant_ranges;
    std::vector<uint32_t> offsets;
    for (int i = 0; i < metadata.img.azimuth_size; i++) {
        auto r = files.GetSignalDataRecord(i, metadata.img.record_length);
        uint32_t slant_range = r.ReadB4(117);
        if (slant_range < min_slant_range) {
            min_slant_range = slant_range;
        }
        if (slant_range != prev_slant_range) {
            LOGD << "Azimuth idx = " << i << " slant range = " << slant_range;
            prev_slant_range = slant_range;
        }

        slant_ranges.push_back(slant_range);
    }

    offsets.resize(slant_ranges.size());
    for (size_t i = 0; i < slant_ranges.size(); i++) {
        const uint32_t slant_range = slant_ranges[i];
        uint32_t offset = 0;
        if (slant_range != min_slant_range) {
            offset = std::round((slant_range - min_slant_range) / metadata.range_spacing);
        }
        offsets[i] = offset;
    }

    auto& img_file = files.GetImageFile();
    CHECK_CUDA_ERR(cudaMemcpy(d_img_data, img_file.SignalData(), byte_size_raw, cudaMemcpyHostToDevice));

    FormatRawIQ8(d_img_data, d_img_shifted, metadata.img, offsets);

    metadata.left_range_offsets = std::move(offsets);
    metadata.slant_range_times = std::move(slant_ranges);
    metadata.slant_range_first_sample = min_slant_range;

    return cleanup_shift;
}

int Execute(alus::cuda::CudaInit& cuda_init, const Arguments& args) {
    const bool write_temps = args.GetWriteIntermediateFiles();
    const bool output_complex = !args.GetOutputIntensity();

    FileSystem palsar_files;
    palsar_files.InitFromPath(args.GetInputDirectory().c_str(), args.GetPolarisation());

    // TODO(priit) output file naming?
    std::filesystem::path output_path(args.GetOutputPath());
    std::string temp_file_root;
    std::string final_path;
    std::string filename_start = "/BULPP_" + palsar_files.GetSceneId() + "_" + args.GetPolarisation();
    if (std::filesystem::is_directory(output_path)) {
        temp_file_root = output_path.string() + filename_start;
        final_path = output_path.string() + filename_start + (output_complex ? "_slc.tif" : "_intensity.tif");
    } else {
        temp_file_root = output_path.parent_path().string() + filename_start;
        final_path = output_path;
    }
    std::string metadata_path = final_path;
    metadata_path.erase(metadata_path.find(".tif"));
    metadata_path += ".xml";

    auto& img_file = palsar_files.GetImageFile();
    if (args.GetPrintMetadataOnly()) {
        img_file.LoadHeader();
    } else {
        auto load_file_start = TimeStart();
        img_file.LoadFile();
        TimeStopFileIO(load_file_start, img_file.Size(), "Load image file");
    }

    SARMetadata metadata = {};
    metadata.polarisation = args.GetPolarisation();
    palsar::ReadMetadata(palsar_files, metadata);
    palsar::PrintMetadata(metadata);

    metadata.results.Vr = metadata.platform_velocity * 0.94;
    LOGD << "Initial Vr = " << metadata.results.Vr;

    metadata.results.Vr = EstimateProcessingVelocity(metadata);

    LOGI << "Calculated Vr = " << metadata.results.Vr;

    if (args.GetPrintMetadataOnly()) {
        return 0;
    }

    // IMG  file Disk -> CPU/Host
    const size_t signal_size = static_cast<size_t>(metadata.img.record_length) * metadata.img.azimuth_size;

    if (signal_size > img_file.SignalSize()) {
        std::string err_msg = "IMG file size mismatch - expected = " + std::to_string(signal_size) +
                              " real = " + std::to_string(img_file.SignalSize());
        throw std::runtime_error(err_msg);
    }

    while (!cuda_init.IsFinished())
        ;
    cuda_init.CheckErrors();
    const auto& cuda_device = cuda_init.GetDevices().front();
    cuda_device.Set();

    LOGI << "Using '" << cuda_device.GetName() << "' device nr " << cuda_device.GetDeviceNr() << " for calculations";

    int range_padded = 0;
    int azimuth_padded = 0;
    EstimatePaddings(cuda_device.GetTotalGlobalMemory(), cuda_device.GetFreeGlobalMemory() * args.GetGpuMemFraction(),
                     metadata.img.range_size, metadata.img.azimuth_size, metadata.chirp.n_samples, range_padded,
                     azimuth_padded);

    std::string err_msg = "No valid padding found for " + std::to_string(metadata.img.range_size) + " " +
                          std::to_string(metadata.img.azimuth_size);
    if (range_padded == 0 || azimuth_padded == 0) {
        THROW_ALGORITHM_EXCEPTION(ALG_NAME, err_msg);
    }

    // IMG file Host -> Device
    palsar::DevicePaddedImage d_img;
    {
        auto signal_data = BuildIQ8Gpu(palsar_files, metadata);
        const auto* d_signal_data = static_cast<const IQ8*>(signal_data.get());

        // find average I & Q
        auto dc_estimate = TimeStart();
        palsar::CalculateDCBias(d_signal_data, metadata.img, cuda_device.GetSmCount(), metadata.results);
        TimeStop(dc_estimate, "DC estimate");
        LOGI << "DC offset I,Q = " << metadata.results.dc_i << " " << metadata.results.dc_q;
        LOGI << "total samples = " << metadata.results.total_samples;
        // 2x 5 bit ADC -> corrected CFloat32
        auto dc_apply = TimeStart();

        LOGI << "Range padding " << metadata.img.range_size << " -> " << range_padded;
        LOGI << "Azimuth padding " << metadata.img.azimuth_size << " -> " << azimuth_padded;
        d_img.InitPadded(metadata.img.range_size, metadata.img.azimuth_size, range_padded, azimuth_padded);
        palsar::ApplyDCBias(d_signal_data, metadata.results, metadata.img, d_img);
        TimeStop(dc_apply, "IQ8 -> CFloat32");
        if (write_temps) {
            std::string path = temp_file_root + (output_complex ? "_raw.tif" : "_raw_intensity.tif");
            palsar::WriteImage(d_img, path.c_str(), output_complex);
        }
    }

    auto dc_start = TimeStart();
    palsar::CalculateDopplerCentroid(d_img, metadata.pulse_repetition_frequency, metadata.results.doppler_centroid);
    TimeStop(dc_start, "Doppler centroid done");
    LOGI << "Doppler centroid(Hz) = " << metadata.results.doppler_centroid;

    CudaWorkspace d_workspace(d_img.TotalByteSize());
    auto rc_start = TimeStart();
    auto chirp_data = palsar::GenerateChirpData(metadata.chirp, d_img.XStride());
    if (write_temps) {
        WriteRawBinary(chirp_data.data(), chirp_data.size(), (temp_file_root + "_chirp.cf32").c_str());
    }

    palsar::RangeCompression(d_img, chirp_data, metadata.chirp.n_samples, d_workspace);

    TimeStop(rc_start, "Range compression done");

    if (write_temps) {
        const std::string path = temp_file_root + (output_complex ? "_rc.tif" : "_rc_intensity.tif");
        palsar::WriteImage(d_img, path.c_str(), output_complex);
    }

    palsar::DevicePaddedImage d_azi_output;
    auto az_comp_start = TimeStart();
    RangeDopplerAlgorithm(metadata, d_img, d_azi_output, std::move(d_workspace));
    TimeStop(az_comp_start, "Azimuth compression done");

    auto final_write = TimeStart();
    palsar::WriteImage(d_azi_output, final_path.c_str(), output_complex, metadata);
    size_t final_bytes = d_azi_output.DataByteSize();
    if (!output_complex) {
        final_bytes /= 2;
    }
    palsar::ExportMetadata(metadata, metadata_path.c_str());
    TimeStopFileIO(final_write, final_bytes, "Final image storage");

    return 0;
}
}  // namespace alus::palsar

int main(int argc, char* argv[]) {
    alus::palsar::Arguments args;
    std::thread cufft_warmup;
    int return_code = 0;
    try {
        alus::cuda::CudaInit cuda_init;
        alus::common::log::Initialize();
        alus::gdalmanagement::Initialize();
        args.ParseArgs(argc, argv);
        args.Check();
        if (args.GetHelpRequested()) {
            std::cout << alus::app::GenerateHelpMessage(ALG_NAME, args.GetHelp());
            return alus::app::errorcode::ALG_SUCCESS;
        }
        alus::common::log::SetLevel(args.GetLogLevel());
        if (!args.GetPrintMetadataOnly()) {
            cufft_warmup = std::thread([]() {
                size_t ws;
                cufftEstimate1d(1, CUFFT_C2C, 1, &ws);
            });
        }
        int exec = alus::palsar::Execute(cuda_init, args);
        return_code = exec;
    }

    catch (const boost::program_options::error& e) {
        std::cout << alus::app::GenerateHelpMessage(ALG_NAME, args.GetHelp());
        ExceptionMessagePrint(e);
        return_code = alus::app::errorcode::ARGUMENT_PARSE;
    } catch (const alus::CudaErrorException& e) {
        ExceptionMessagePrint(e);
        return_code = alus::app::errorcode::GPU_DEVICE_ERROR;
    } catch (const alus::common::AlgorithmException& e) {
        ExceptionMessagePrint(e);
        return_code = alus::app::errorcode::ALGORITHM_EXCEPTION;
    } catch (const std::exception& e) {
        ExceptionMessagePrint(e);
        return_code = alus::app::errorcode::GENERAL_EXCEPTION;
    } catch (...) {
        LOGE << "Caught an unknown exception.";
        return_code = alus::app::errorcode::UNKNOWN_EXCEPTION;
    }
    if (cufft_warmup.joinable()) {
        cufft_warmup.join();
    }
    return return_code;
}