#pragma once

#include <tuple>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "coh_tile.h"
#include "data_bands_buffer.h"
#include "i_algo.h"
#include "i_data_tile_reader.h"
#include "meta_data.h"

namespace alus {
class MetaData;
class Coh : virtual public IAlgo {
   private:
    const int srp_number_points_;
    const int srp_polynomial_degree_;
    const bool subtract_flat_earth_;
    const int coh_win_rg_;
    const int coh_win_az_;
    const int tile_x_;
    const int tile_y_;
    const int orbit_degree_;
    MetaData &meta_master_;
    MetaData &meta_slave_;

    std::vector<tensorflow::Input> tile_calc_inputs_;
    std::vector<tensorflow::ops::Placeholder> tile_calc_placeholder_inputs_;
    // todo: Try to map correct bands from input master&slave (probaly should be user input)
    DataBandsBuffer bands_;

    static std::tuple<std::vector<int>, std::vector<int>> DistributePoints(
        int num_of_points, int band_x_size, int band_x_min, int band_y_size, int band_y_min);
    auto Norm(tensorflow::Scope &root, tensorflow::Input tensor);
    auto ComplexAbs(tensorflow::Scope &scope, tensorflow::Input complex_nr);
    auto CoherenceProduct(tensorflow::Scope &root, tensorflow::Input sum_t, tensorflow::Input power_t);
    std::vector<int> GetXPows(int srp_polynomial_degree);
    std::vector<int> GetYPows(int srp_polynomial_degree);
    auto NormalizeDoubleMatrix3(tensorflow::Scope &root,
                                tensorflow::Input t,
                                tensorflow::Input min,
                                tensorflow::Input max);
    auto GetA(tensorflow::Scope &root,
              tensorflow::Input lines,
              tensorflow::Input pixels,
              int x_min_pixel,
              int x_max_pixel,
              int y_min_line,
              int y_max_line,
              tensorflow::Input x_pows,
              tensorflow::Input y_pows);
    std::vector<double> GenerateY(std::tuple<std::vector<int>, std::vector<int>> lines_pixels,
                                  MetaData &meta_master,
                                  MetaData &meta_slave);
    auto GetCoefs(tensorflow::Scope &root, tensorflow::Input a, tensorflow::Input generate_y);
    auto Polyval2DDim(tensorflow::Scope &root,
                      tensorflow::Input x,
                      tensorflow::Input y,
                      tensorflow::Input coefs,
                      tensorflow::Input x_pows,
                      tensorflow::Input y_pows,
                      int x_size,
                      int y_size);
    auto ComputeFlatEarthPhase(tensorflow::Scope &root,
                               CohTile &tile,
                               tensorflow::Input x_pows,
                               tensorflow::Input y_pows,
                               tensorflow::Input coefs);
    auto CoherenceTileCalc(tensorflow::Scope &root,
                           CohTile &coh_tile,
                           tensorflow::Input mst_real,
                           tensorflow::Input mst_imag,
                           tensorflow::Input slv_real,
                           tensorflow::Input slv_imag,
                           tensorflow::Input x_pows,
                           tensorflow::Input y_pows,
                           tensorflow::Input coefs);

   public:
    // todo: replace with external inputs struct
    Coh(int srp_number_points,
        int srp_polynomial_degree,
        bool subtract_flat_earth,
        int coh_win_rg,
        int coh_win_az,
        int tile_x,
        int tile_y,
        int orbit_degree,
        MetaData &meta_master,
        MetaData &meta_slave);
    void CoherencePreTileCalc(tensorflow::Scope &root);
    // todo:might want to make it even more general
    template <typename T>
    [[nodiscard]] tensorflow::Tensor GetTensor(const std::vector<T> &data);
    [[nodiscard]] tensorflow::ClientSession::FeedType &GetInputs() override;
    void PreTileCalc(tensorflow::Scope &scope) override;
    [[nodiscard]] tensorflow::Output TileCalc(tensorflow::Scope &scope, CohTile &tile) override;
    void DataToTensors(const Tile &tile, const IDataTileReader &reader) override;
};

}  // namespace alus
