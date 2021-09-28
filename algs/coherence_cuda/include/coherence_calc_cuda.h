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

#include <array>
#include <tuple>
#include <vector>

#include "band_params.h"
#include "coh_tile.h"
#include "coh_window.h"
#include "coherence_computation.h"
#include "i_algo_cuda.h"
#include "i_data_tile_reader.h"
#include "meta_data.h"

namespace alus {
namespace coherence_cuda {
class MetaData;
class CohCuda : public IAlgoCuda {
private:
    const int srp_number_points_;
    const int srp_polynomial_degree_;
    const bool subtract_flat_earth_;
    CohWindow coh_win_;
    const int orbit_degree_;
    MetaData& meta_master_;
    MetaData& meta_slave_;
    BandParams band_params_;
    coherence_cuda::CoherenceComputation coherence_computation_;

    static std::tuple<std::vector<int>, std::vector<int>> DistributePoints(int num_of_points, int band_x_size,
                                                                           int band_x_min, int band_y_size,
                                                                           int band_y_min);
    static std::vector<int> GetXPows(int srp_polynomial_degree);
    static std::vector<int> GetYPows(int srp_polynomial_degree);
    std::vector<double> GenerateY(std::tuple<std::vector<int>, std::vector<int>> lines_pixels, MetaData& meta_master,
                                  MetaData& meta_slave) const;

public:
    CohCuda(int srp_number_points, int srp_polynomial_degree, bool subtract_flat_earth, const CohWindow& coh_window,
            int orbit_degree, MetaData& meta_master, MetaData& meta_slave);
    void CoherencePreTileCalc();
    void PreTileCalc() override;
    void TileCalc(CohTile& tile, const std::array<std::vector<float>, 4>& data, std::vector<float>& data_out) override;
    void Cleanup() override;
};
}  // namespace coherence_cuda
}  // namespace alus
