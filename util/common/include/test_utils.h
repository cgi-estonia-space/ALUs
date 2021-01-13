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

#include <ios>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <gdal_priv.h>
#include <openssl/md5.h>
#include <boost/functional/hash.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include "dataset.h"
#include "gdal_util.h"

namespace alus {
namespace utils {
namespace test {
std::string HashFromBand(std::string_view file_path) {
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

std::string Md5FromFile(const std::string& path) {
    unsigned char result[MD5_DIGEST_LENGTH];
    boost::iostreams::mapped_file_source src(path);
    MD5(reinterpret_cast<const unsigned char*>(src.data()), src.size(), result);
    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result) sout << std::setw(2) << static_cast<int>(c);
    return sout.str();
}
}  // namespace test
}  // namespace utils
}  // namespace alus