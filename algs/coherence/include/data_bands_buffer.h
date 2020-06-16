#pragma once

#include "tensorflow/core/framework/tensor.h"

namespace alus {

class DataBandsBuffer {
   private:
    tensorflow::Tensor band_master_real_, band_master_imag_, band_slave_real_, band_slave_imag_;

   public:
    void DataToTensors(int tile_size_x, int tile_size_y, float *gdal_data);
    // constructor needs to know size of tile, to create Tensors
    // copy actual data which gdal read into tensors which tensorflow will use
    // void copyDataToTFTensor(std::vector<int> & band, int bandNr, int & gdalDataBuffer,  Tensor & tensorToPopulate);
    [[nodiscard]] tensorflow::Tensor &GetBandMasterReal() { return band_master_real_; }
    [[nodiscard]] tensorflow::Tensor &GetBandMasterImag() { return band_master_imag_; }
    [[nodiscard]] tensorflow::Tensor &GetBandSlaveReal() { return band_slave_real_; }
    [[nodiscard]] tensorflow::Tensor &GetBandSlaveImag() { return band_slave_imag_; }
};
}  // namespace alus