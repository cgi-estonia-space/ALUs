#include "data_bands_buffer.h"

namespace alus {
void DataBandsBuffer::DataToTensors(int tile_size_x, int tile_size_y, float *gdal_data) {
    int band_nr;
    int tile_size = tile_size_x * tile_size_y;

    // FLOAT vs. SHORT !!!
    band_master_real_ = tensorflow::Tensor{tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{tile_size}};
    band_master_imag_ = tensorflow::Tensor{tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{tile_size}};
    band_slave_real_ = tensorflow::Tensor{tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{tile_size}};
    band_slave_imag_ = tensorflow::Tensor{tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{tile_size}};

    // band_nr 1,2,5,6 (reworked source image to contain only 4 needed bands)
    band_nr = 1;
    std::vector<float> v_band_master_real(&gdal_data[(band_nr - 1) * tile_size], &gdal_data[band_nr * tile_size]);
    band_nr = 2;
    std::vector<float> v_band_master_imag(&gdal_data[(band_nr - 1) * tile_size], &gdal_data[band_nr * tile_size]);
    // todo:at start only use images with correct bands in correct order (later make it better)
    // band_nr = 5;
    band_nr = 3;
    std::vector<float> v_band_slave_real(&gdal_data[(band_nr - 1) * tile_size], &gdal_data[band_nr * tile_size]);
    // band_nr = 6;
    band_nr = 4;
    std::vector<float> v_band_slave_imag(&gdal_data[(band_nr - 1) * tile_size], &gdal_data[band_nr * tile_size]);
    std::copy_n(v_band_master_real.begin(), v_band_master_real.size(), band_master_real_.flat<float>().data());
    std::copy_n(v_band_master_imag.begin(), v_band_master_imag.size(), band_master_imag_.flat<float>().data());
    //!!FLOAT vs. SHORT!! (also tensorflow needs float for complex number, but maybe we can use bands sepparately)
    std::copy_n(v_band_slave_real.begin(), v_band_slave_real.size(), band_slave_real_.flat<float>().data());
    std::copy_n(v_band_slave_imag.begin(), v_band_slave_imag.size(), band_slave_imag_.flat<float>().data());

    // is this needed?
    v_band_master_real.clear();
    v_band_master_imag.clear();
    v_band_slave_real.clear();
    v_band_slave_imag.clear();
}
}  // namespace alus