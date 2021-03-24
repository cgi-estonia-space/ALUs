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
#include <algorithm>
#include <vector>

#include "data_bands_buffer.h"

namespace alus {
void DataBandsBuffer::DataToTensors(int tile_size_x, int tile_size_y, const std::vector<float>& gdal_data) {
    int band_nr;
    int tile_size = tile_size_x * tile_size_y;

    // FLOAT vs. SHORT !!!
    band_master_real_ = tensorflow::Tensor{tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{tile_size}};
    band_master_imag_ = tensorflow::Tensor{tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{tile_size}};
    band_slave_real_ = tensorflow::Tensor{tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{tile_size}};
    band_slave_imag_ = tensorflow::Tensor{tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{tile_size}};

    band_nr = 1;
    std::copy_n(gdal_data.begin() + ((band_nr - 1) * tile_size), tile_size, band_master_real_.flat<float>().data());
    band_nr = 2;
    std::copy_n(gdal_data.begin() + ((band_nr - 1) * tile_size), tile_size, band_master_imag_.flat<float>().data());
    //!!FLOAT vs. SHORT!! (also tensorflow needs float for complex number, but maybe we can use bands sepparately)
    band_nr = 3;
    std::copy_n(gdal_data.begin() + ((band_nr - 1) * tile_size), tile_size, band_slave_real_.flat<float>().data());
    band_nr = 4;
    std::copy_n(gdal_data.begin() + ((band_nr - 1) * tile_size), tile_size, band_slave_imag_.flat<float>().data());
}
}  // namespace alus