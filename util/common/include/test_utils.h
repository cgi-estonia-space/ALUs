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

#include <cmath>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gdal_priv.h>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <boost/functional/hash.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include "dataset.h"
#include "gdal_util.h"

namespace alus::utils::test {
inline std::string HashFromBand(std::string_view file_path) {
    auto* const dataset = static_cast<GDALDataset*>(GDALOpen(file_path.data(), GA_ReadOnly));
    const auto x_size = dataset->GetRasterXSize();
    const auto y_size = dataset->GetRasterYSize();

    std::vector<float> raster_data(x_size * y_size);
    auto error = dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, x_size, y_size, raster_data.data(), x_size, y_size,
                                                     GDALDataType::GDT_Float32, 0, 0);

    GDALClose(dataset);
    CHECK_GDAL_ERROR(error);

    std::ostringstream s_out;
    s_out << std::hex << std::setfill('0') << boost::hash<std::vector<float>>{}(raster_data);

    return s_out.str();
}

inline std::string SHA256FromFile(const std::string& path) {
    unsigned char result[SHA256_DIGEST_LENGTH];
    boost::iostreams::mapped_file_source src(path);
    SHA256(reinterpret_cast<const unsigned char*>(src.data()), src.size(), result);
    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result) {
        sout << std::setw(2) << static_cast<int>(c);
    }
    return sout.str();
}

inline bool AreDatasetsEqual(const std::shared_ptr<GDALDataset>& comparand_dataset,
                             const std::shared_ptr<GDALDataset>& reference_dataset, double error_margin) {
    const auto width = reference_dataset->GetRasterXSize();
    const auto height = reference_dataset->GetRasterYSize();

    if (width != comparand_dataset->GetRasterXSize() || height != comparand_dataset->GetRasterYSize()) {
        return false;
    }

    std::vector<float> reference_data(width * height);
    std::vector<double> comparand_data(width * height);

    auto read_error = reference_dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, reference_data.data(),
                                                                    width, height, GDT_Float32, 0, 0);
    CHECK_GDAL_ERROR(read_error);

    read_error = comparand_dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, comparand_data.data(),
                                                               width, height, GDT_Float64, 0, 0);
    CHECK_GDAL_ERROR(read_error);

    for (size_t i = 0; i < reference_data.size(); ++i) {
        const auto ref_value = reference_data.at(i);
        const auto comparand_value = comparand_data.at(i);

        if (std::fabs(ref_value - comparand_value) > error_margin) {
            const auto y = i / width;
            const auto x = i - width * y;
            std::cout << std::setprecision(static_cast<int>(std::log10(1 / error_margin)) + 1);
            std::cout << "Value mismatch at " << x << "x" << y << ". Expected " << ref_value << ", received "
                      << comparand_value << "." << std::endl;
            return false;
        }
    }

    return true;
}

inline bool AreDatasetsEqual(const std::string_view comparand, const std::string_view reference, double error_margin) {
    auto dataset_closer = [](GDALDataset* dataset) { GDALClose(dataset); };

    auto* const reference_dataset = static_cast<GDALDataset*>(GDALOpen(reference.data(), GA_ReadOnly));
    auto* const comparand_dataset = static_cast<GDALDataset*>(GDALOpen(comparand.data(), GA_ReadOnly));

    std::shared_ptr<GDALDataset> comparand_dataset_pointer(comparand_dataset, dataset_closer);
    std::shared_ptr<GDALDataset> reference_dataset_pointer(reference_dataset, dataset_closer);

    return AreDatasetsEqual(comparand_dataset_pointer, reference_dataset_pointer, error_margin);
}
}  // namespace alus::utils::test