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

#include <tuple>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"

#include "coh_tile.h"
#include "coh_window.h"
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
    const int orbit_degree_;
    MetaData& meta_master_;
    MetaData& meta_slave_;

    std::vector<tensorflow::Input> tile_calc_inputs_;
    std::vector<tensorflow::ops::Placeholder> tile_calc_placeholder_inputs_;
    // todo: Try to map correct bands from input master&slave (probaly should be user input)
    DataBandsBuffer bands_;

    static std::tuple<std::vector<int>, std::vector<int>> DistributePoints(int num_of_points, int band_x_size,
                                                                           int band_x_min, int band_y_size,
                                                                           int band_y_min);
    static auto Norm(const tensorflow::Scope& root, const tensorflow::Input& tensor);
    static auto ComplexAbs(const tensorflow::Scope& root, const tensorflow::Input& complex_nr);
    static auto CoherenceProduct(const tensorflow::Scope& root, const tensorflow::Input& sum_t,
                                 const tensorflow::Input& power_t);
    static std::vector<int> GetXPows(int srp_polynomial_degree);
    static std::vector<int> GetYPows(int srp_polynomial_degree);
    static auto NormalizeDoubleMatrix3(const tensorflow::Scope& root, const tensorflow::Input& t,
                                       const tensorflow::Input& min, const tensorflow::Input& max);
    static auto GetA(const tensorflow::Scope& root, const tensorflow::Input& lines, const tensorflow::Input& pixels,
                     int x_min_pixel, int x_max_pixel, int y_min_line, int y_max_line, const tensorflow::Input& x_pows,
                     const tensorflow::Input& y_pows);
    std::vector<double> GenerateY(std::tuple<std::vector<int>, std::vector<int>> lines_pixels, MetaData& meta_master,
                                  MetaData& meta_slave) const;
    static auto GetCoefs(const tensorflow::Scope& root, const tensorflow::Input& a,
                         const tensorflow::Input& generate_y);
    static auto Polyval2DDim(const tensorflow::Scope& root, const tensorflow::Input& x, const tensorflow::Input& y,
                             const tensorflow::Input& coefs, const tensorflow::Input& x_pows,
                             const tensorflow::Input& y_pows, int x_size, int y_size);
    auto ComputeFlatEarthPhase(const tensorflow::Scope& root, const CohTile& tile, const tensorflow::Input& x_pows,
                               const tensorflow::Input& y_pows, const tensorflow::Input& coefs) const;
    auto CoherenceTileCalc(const tensorflow::Scope& root, const CohTile& coh_tile, const tensorflow::Input& mst_real,
                           const tensorflow::Input& mst_imag, const tensorflow::Input& slv_real,
                           const tensorflow::Input& slv_imag, const tensorflow::Input& x_pows,
                           const tensorflow::Input& y_pows, const tensorflow::Input& coefs) const;

public:
    // todo: replace with external inputs struct
    Coh(int srp_number_points, int srp_polynomial_degree, bool subtract_flat_earth, const CohWindow& coh_window,
        int orbit_degree, MetaData& meta_master, MetaData& meta_slave);
    void CoherencePreTileCalc(const tensorflow::Scope& root);
    // todo:might want to make it even more general
    template <typename T>
    [[nodiscard]] tensorflow::Tensor GetTensor(const std::vector<T>& data);
    void PreTileCalc(const tensorflow::Scope& root) override;
    [[nodiscard]] tensorflow::Output TileCalc(const tensorflow::Scope& root, CohTile& tile) override;
    void DataToTensors(const Tile& tile, const IDataTileReader& reader) override;
};

}  // namespace alus
