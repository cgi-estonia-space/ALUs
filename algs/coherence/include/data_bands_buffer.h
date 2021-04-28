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

#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace alus {

class DataBandsBuffer {
private:
    tensorflow::Tensor band_master_real_, band_master_imag_, band_slave_real_, band_slave_imag_;

public:
    void DataToTensors(int tile_size_x, int tile_size_y, const std::vector<float>& gdal_data);
    // constructor needs to know size of tile, to create Tensors
    // copy actual data which gdal read into tensors which tensorflow will use
    [[nodiscard]] tensorflow::Tensor& GetBandMasterReal() { return band_master_real_; }
    [[nodiscard]] tensorflow::Tensor& GetBandMasterImag() { return band_master_imag_; }
    [[nodiscard]] tensorflow::Tensor& GetBandSlaveReal() { return band_slave_real_; }
    [[nodiscard]] tensorflow::Tensor& GetBandSlaveImag() { return band_slave_imag_; }
};
}  // namespace alus