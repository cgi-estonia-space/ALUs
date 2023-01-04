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

#include "img_output.h"

#include <condition_variable>
#include <queue>
#include <thread>

#include <gdal/gdal_priv.h>
#include <boost/date_time.hpp>

#include "alus_log.h"
#include "checks.h"
#include "gdal_util.h"
#include "math_utils.h"
#include "pugixml.hpp"

// 0 -> Tiff
// 1 -> Envi
// 2 -> BE Envi via hack, no direct gdal support?

#define OUTPUT_FORMAT 0

namespace {

float Bswapf32(float i) {
    uint32_t t;
    memcpy(&t, &i, 4);
    t = __builtin_bswap32(t);
    memcpy(&i, &t, 4);
    return i;
}

struct ImgSlice {
    cufftComplex* slice_data;
    int slice_cnt;
    int slice_start_idx;
    int slice_stride;
};

struct SharedData {
    // gpu thread moves data from device to host and gives exclusive access for that buffer for file IO
    struct {
        std::queue<ImgSlice> queue;
        std::condition_variable cv;
        std::mutex mutex;
        bool stop_file_write;
    } gpu;

    // file thread consumes the memory for file IO and tells the gpu threads the memory can be reused for cudaMemcpy
    struct {
        std::queue<cufftComplex*> queue;
        std::condition_variable cv;
        std::mutex mutex;
        CPLErr gdal_err;
    } file;
};

void RowWriterTask(GDALRasterBand* band, SharedData* shared) {
    int rows_written = 0;
    const int total_rows = band->GetYSize();
    while (rows_written < total_rows) {
        ImgSlice el;
        {
            std::unique_lock l(shared->gpu.mutex);
            shared->gpu.cv.wait(l, [=]() { return shared->gpu.stop_file_write || !shared->gpu.queue.empty(); });
            if (shared->gpu.stop_file_write) {
                return;
            }
            el = shared->gpu.queue.front();
            shared->gpu.queue.pop();
        }

        // write blocks from
        for (int i = 0; i < el.slice_cnt; i++) {
            int y_block = el.slice_start_idx + i;
            auto* ptr = el.slice_data + i * el.slice_stride;
            auto e = band->WriteBlock(0, y_block, ptr);
            if (e == CE_None) {
                rows_written++;
            } else {
                {
                    std::unique_lock l(shared->file.mutex);
                    shared->file.gdal_err = e;
                }
                shared->file.cv.notify_one();
                return;
            }
        }

        {
            // release memory back for reuse
            std::unique_lock l(shared->file.mutex);
            shared->file.queue.push(el.slice_data);
        }
        shared->file.cv.notify_one();
    }
}

// Optimized file write with GDAL datasets, which are stored as strips as opposed to tiles
// 1) Uses WriteBlock API, to bypass GDAL Cache
// 2) Run cudaMemcpy(& complex -> intensity transform) in parallel to file writing
void WriteStripedDataset(const alus::palsar::DevicePaddedImage& img, GDALRasterBand* band, bool complex_output,
                         std::optional<alus::palsar::SARMetadata> metadata) {
    constexpr size_t QUEUE_SIZE = 3;
    SharedData sd = {};
    std::array<std::unique_ptr<cufftComplex[]>, QUEUE_SIZE> ptr_array;

    const int n_rows_per_slice = 10;
    const int x_stride = img.XStride();
    const int y_size = img.YSize();

    int aperature_size = 0;
    if (metadata.has_value()) {
        // TODO(priit) top of image vs bottom of image
        aperature_size = alus::palsar::CalcAperturePixels(metadata.value(), 0);
    }

    for (auto& el : ptr_array) {
        // TODO(priit) benchmark if pinned memory be worth if the file thread writes to ramdisk?
        el = std::unique_ptr<cufftComplex[]>(new cufftComplex[x_stride * n_rows_per_slice]);
        sd.file.queue.push(el.get());
    }

    std::thread gdal_thread(RowWriterTask, band, &sd);

    int rows_transferred = 0;
    cudaError cuda_err = {};

    while (rows_transferred < y_size) {
        cufftComplex* h_ptr = nullptr;
        {
            std::unique_lock l(sd.file.mutex);
            sd.file.cv.wait(l, [&]() { return sd.file.gdal_err != CE_None || !sd.file.queue.empty(); });

            if (sd.file.gdal_err != CE_None) {
                break;
            }
            h_ptr = sd.file.queue.front();
            sd.file.queue.pop();
        }

        const auto* d_src = img.Data() + rows_transferred * x_stride;

        int act_rows = n_rows_per_slice;
        if (rows_transferred + n_rows_per_slice > y_size) {
            act_rows = y_size - rows_transferred;
        }
        ImgSlice slice = {};
        cuda_err = cudaMemcpy(h_ptr, d_src, act_rows * x_stride * 8, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            {
                std::unique_lock l(sd.gpu.mutex);
                sd.gpu.stop_file_write = true;
            }
            sd.gpu.cv.notify_one();
            break;
        }
#if OUTPUT_FORMAT == 2
        for (int i = 0; i < act_rows; i++) {
            auto* ptr = h_ptr + (i * x_stride);
            for (int j = 0; j < x_stride; j++) {
                ptr[j].x = Bswapf32(ptr[j].x);
                ptr[j].y = Bswapf32(ptr[j].y);
            }
        }
#endif
        if (metadata.has_value()) {
            for (int i = 0; i < act_rows; i++) {
                auto* ptr = h_ptr + (i * x_stride);
                const int y = rows_transferred + i;
                if (y < aperature_size || y + aperature_size > y_size) {
                    memset(ptr, 0, sizeof(ptr[0]) * x_stride);
                    continue;
                }
                uint32_t offset = metadata.value().left_range_offsets.at(y);
                if (offset > 0U) {
                    for (uint32_t x = 0; x < offset; x++) {
                        ptr[x] = {};
                    }
                }
            }
        }
        if (!complex_output) {
            const int x_size = band->GetXSize();
            for (int i = 0; i < act_rows; i++) {
                alus::palsar::InplaceComplexToIntensity(h_ptr + (i * x_stride), x_size);
            }
        }

        slice.slice_data = h_ptr;
        slice.slice_stride = x_stride;
        slice.slice_start_idx = rows_transferred;
        slice.slice_cnt = act_rows;
        {
            std::unique_lock l(sd.gpu.mutex);
            sd.gpu.queue.push(slice);
        }
        sd.gpu.cv.notify_one();
        rows_transferred += act_rows;
    }

    gdal_thread.join();

    CHECK_CUDA_ERR(cuda_err);
    CHECK_GDAL_ERROR(sd.file.gdal_err);
}

void WriteTiff(const alus::palsar::DevicePaddedImage& img, bool complex_output, bool padding, const char* path,
               std::optional<alus::palsar::SARMetadata> metadata = {}) {
    const int w = padding ? img.XStride() : img.XSize();
    const int h = padding ? img.YStride() : img.YSize();
#if OUTPUT_FORMAT == 1 || OUTPUT_FORMAT == 2
    std::string wo_ext = path;

    auto idx = wo_ext.find(".tif");
    if (idx < wo_ext.size()) {
        wo_ext.resize(idx);
        LOGD << "Envi name = " << wo_ext;
        path = wo_ext.c_str();
    }
    if (!complex_output) {
        complex_output = true;
        LOGD << "COMPLEX OUTPUT OVERRIDE";
    }
#endif

    LOGI << "Writing " << (complex_output ? "complex" : "intensity") << " file @ " << path;
    const int buf_size = w * h;

    const auto gdal_type = complex_output ? GDT_CFloat32 : GDT_Float32;

    char** output_driver_options = nullptr;
#if OUTPUT_FORMAT == 0
    auto* drv = alus::GetGdalGeoTiffDriver();
#elif OUTPUT_FORMAT == 1 || OUTPUT_FORMAT == 2
    auto* drv = GetGDALDriverManager()->GetDriverByName("ENVI");
#endif
    auto* ds = drv->Create(path, w, h, 1, gdal_type, output_driver_options);
    CHECK_NULLPTR(ds);

    if (metadata.has_value() && !complex_output) {
        // Horrible hack to display intensity GTiff on a map to get a rough estimate where it is,
        // but the image is in slant range and is not displayed correctly

        ds->GetRasterBand(1)->SetNoDataValue(0.0);
        OGRSpatialReference target_crs;
        // EPSG:4326 is a WGS84 code
        target_crs.importFromEPSG(4326);
        char* projection_wkt = nullptr;
        auto cpl_free = [](char* csl_data) { CPLFree(csl_data); };
        target_crs.exportToWkt(&projection_wkt);
        std::unique_ptr<char, decltype(cpl_free)> projection_guard(projection_wkt, cpl_free);
        ds->SetProjection(projection_wkt);

        double gt[6] = {};
        double spacing_rg = (metadata->range_spacing / 6378137.0) * (180 / M_PI);
        double spacing_az = (metadata->azimuth_spacing / 6378137.0) * (180 / M_PI);

        gt[0] = metadata->center_lon - metadata->img.range_size / 2 * spacing_rg;
        gt[1] = spacing_rg;
        gt[3] = metadata->center_lat - metadata->img.azimuth_size / 2 * spacing_az;
        gt[5] = spacing_az;

        ds->SetGeoTransform(gt);
    }

    std::unique_ptr<GDALDataset, decltype(&GDALClose)> gdal_close(ds, GDALClose);

    auto* band = ds->GetRasterBand(1);
    int x_block_size;
    int y_block_size;
    band->GetBlockSize(&x_block_size, &y_block_size);
    if (x_block_size == w && y_block_size == 1) {
        LOGD << "Write fast path";
        WriteStripedDataset(img, band, complex_output, metadata);
        return;
    }

    LOGD << "Write slow path";
    std::unique_ptr<cufftComplex[]> data(new cufftComplex[buf_size]);
    if (padding) {
        img.CopyToHostPaddedSize(data.get());
    } else {
        img.CopyToHostLogicalSize(data.get());
    }

    auto* b = ds->GetRasterBand(1);
    if (complex_output) {
#if OUTPUT_FORMAT == 2
        for (int i = 0; i < w * h; i++) {
            data[i].x = Bswapf32(data[i].x);
            data[i].y = Bswapf32(data[i].y);
        }
#endif
        CHECK_GDAL_ERROR(b->RasterIO(GF_Write, 0, 0, w, h, data.get(), w, h, GDT_CFloat32, 0, 0));
    } else {
        // convert to intensity, each pixel being I^2 + Q^2
        std::unique_ptr<float[]> intens_buf(new float[buf_size]);
        for (int j = 0; j < buf_size; j++) {
            float i = data[j].x;
            float q = data[j].y;
            intens_buf[j] = i * i + q * q;
        }
        CHECK_GDAL_ERROR(b->RasterIO(GF_Write, 0, 0, w, h, intens_buf.get(), w, h, GDT_Float32, 0, 0));
    }
}

}  // namespace

namespace alus::palsar {

void WriteImage(const DevicePaddedImage& img, const char* path, bool complex, std::optional<SARMetadata> metadata) {
    WriteTiff(img, complex, false, path, std::move(metadata));
}

void WriteComplexImg(const DevicePaddedImage& img, const char* path) { WriteTiff(img, true, false, path); }

void WriteIntensityImg(const DevicePaddedImage& img, const char* path) { WriteTiff(img, false, false, path); }

void WriteComplexPaddedImg(const DevicePaddedImage& img, const char* path) { WriteTiff(img, true, true, path); }

void WriteIntensityPaddedImg(const DevicePaddedImage& img, const char* path) { WriteTiff(img, false, true, path); }

void ExportMetadata(const SARMetadata& metadata, const char* path) {
    pugi::xml_document doc;
    auto root = doc.append_child("metadata");

    root.append_child("processing_time")
        .text()
        .set(boost::posix_time::to_iso_extended_string(boost::posix_time::second_clock::local_time()).c_str());

    root.append_child("range_size").text().set(metadata.img.range_size);
    root.append_child("azimuth_size").text().set(metadata.img.azimuth_size);
    root.append_child("range_spacing").text().set(metadata.range_spacing);
    root.append_child("polarisation").text().set(metadata.polarisation.c_str());
    root.append_child("prf").text().set(metadata.pulse_repetition_frequency);
    root.append_child("frequency").text().set(metadata.carrier_frequency);
    root.append_child("wavelength").text().set(metadata.wavelength);
    root.append_child("slant_range").text().set(metadata.slant_range_first_sample);
    root.append_child("Vr").text().set(metadata.results.Vr);
    root.append_child("DC_I").text().set(metadata.results.dc_i);
    root.append_child("DC_Q").text().set(metadata.results.dc_q);
    root.append_child("RangeSamplingRate").text().set(metadata.chirp.range_sampling_rate);
    root.append_child("DopplerCentroid").text().set(metadata.results.doppler_centroid);
    root.append_child("AzimuthBandwidthFraction").text().set(metadata.azimuth_bandwidth_fraction);

    root.append_child("center_point_latitude").text().set(metadata.center_lat);
    root.append_child("center_point_longitude").text().set(metadata.center_lon);
    root.append_child("center_time")
        .text()
        .set(boost::posix_time::to_iso_extended_string(metadata.center_time).c_str());

    auto orbit = root.append_child("orbit_state_vectors");
    orbit.append_child("first_vector_time")
        .text()
        .set(boost::posix_time::to_iso_extended_string(metadata.first_orbit_time).c_str());
    orbit.append_child("interval").text().set(metadata.orbit_interval);

    for (const auto& osv : metadata.orbit_state_vectors) {
        pugi::xml_node orbit_node = orbit.append_child("orbit_state_vector");
        orbit_node.append_child("x_pos").text().set(osv.x_pos);
        orbit_node.append_child("y_pos").text().set(osv.y_pos);
        orbit_node.append_child("z_pos").text().set(osv.z_pos);
        orbit_node.append_child("x_vel").text().set(osv.x_vel);
        orbit_node.append_child("y_vel").text().set(osv.y_vel);
        orbit_node.append_child("z_vel").text().set(osv.z_vel);
    }

    LOGI << "Metadata file @ " << path;
    doc.save_file(path);
    // doc.print(std::cout);
}
}  // namespace alus::palsar