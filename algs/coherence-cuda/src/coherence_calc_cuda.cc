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
#include "coherence_calc_cuda.h"

#include <cmath>

#include <Eigen/Dense>
#include "alus_log.h"

#include "coherence_computation.h"
#include "jlinda/jlinda-core/constants.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"

#include "cuda_copies.h"

namespace alus {
namespace coherence_cuda {
CohCuda::CohCuda(const int srp_number_points, const int srp_polynomial_degree, const bool subtract_flat_earth,
                 const CohWindow& coh_window, const int orbit_degree, MetaData& meta_master, MetaData& meta_slave)
    : srp_number_points_{srp_number_points},
      srp_polynomial_degree_{srp_polynomial_degree},
      subtract_flat_earth_{subtract_flat_earth},
      coh_win_{coh_window},
      orbit_degree_{orbit_degree},
      meta_master_{meta_master},
      meta_slave_{meta_slave},
      //      todo: take band map and band count from product and refactor local methods to use band_params!!!
      band_params_{std::vector<int>{}, 4, meta_master_.GetBandXSize(), meta_master_.GetBandYSize(), 0, 0} {}

std::tuple<std::vector<int>, std::vector<int>> CohCuda::DistributePoints(int num_of_points, int band_x_size,
                                                                         int band_x_min, int band_y_size,
                                                                         int band_y_min) {
    std::vector<int> result_lines;
    std::vector<int> result_pixels;
    result_lines.reserve(static_cast<unsigned long>(num_of_points));
    result_pixels.reserve(static_cast<unsigned long>(num_of_points));
    double win_p = sqrt(num_of_points / (static_cast<double>(band_y_size) / band_x_size));
    double win_l = num_of_points / win_p;
    if (win_l < win_p) {
        win_l = win_p;
    }
    int win_l_int = static_cast<int>(std::floor(win_l));
    double delta_lin = (band_y_size - 1) / static_cast<double>((win_l_int - 1));
    int total_pix = static_cast<int>(std::floor(band_x_size * win_l_int));
    double delta_pix = static_cast<double>((total_pix - 1)) / static_cast<double>((num_of_points - 1));
    double pix = -delta_pix;
    int l_counter = 0;
    double lin;
    for (int i = 0; i < num_of_points; i++) {
        pix += delta_pix;
        while (std::floor(pix) >= band_x_size) {
            pix -= band_x_size;
            l_counter += 1;
        }
        lin = l_counter * delta_lin;
        result_lines.push_back(static_cast<int>(std::floor(lin)) + band_y_min);
        result_pixels.push_back(static_cast<int>(std::floor(pix)) + band_x_min);
    }
    return std::make_tuple(result_lines, result_pixels);
}

std::vector<int> CohCuda::GetXPows(int srp_polynomial_degree) {
    std::vector<int> x_pows;
    for (int i = 0; i < srp_polynomial_degree + 1; i++) {
        for (int j = 0; j < i + 1; j++) {
            x_pows.push_back(i - j);
        }
    }
    return x_pows;
}

std::vector<int> CohCuda::GetYPows(int srp_polynomial_degree) {
    std::vector<int> y_pows;
    for (int i = 0; i < srp_polynomial_degree + 1; i++) {
        for (int j = 0; j < i + 1; j++) {
            y_pows.push_back(j);
        }
    }
    return y_pows;
}

void CohCuda::CoherencePreTileCalc() {
    int nr_of_bursts = meta_master_.burst_meta.size();
    std::vector<cuda::DeviceBuffer<double>> burst_coeffs(nr_of_bursts);

    auto b = std::chrono::steady_clock::now();

    for (int burst_i = 0; burst_i < nr_of_bursts; burst_i++) {
        std::tuple<std::vector<int>, std::vector<int>> position_lines_pixels =
            DistributePoints(srp_number_points_, meta_master_.GetBandXSize(), 0, meta_master_.lines_per_burst - 1, 0);

        double master_min_pi_4_div_lam =
            static_cast<double>(-4.0L * jlinda::SNAP_PI * snapengine::eo::constants::LIGHT_SPEED) /
            meta_master_.GetRadarWaveLength();
        double slave_min_pi_4_div_lam =
            static_cast<double>(-4.0L * jlinda::SNAP_PI * snapengine::eo::constants::LIGHT_SPEED) /
            meta_slave_.GetRadarWaveLength();

        std::vector<int> lines = std::get<0>(position_lines_pixels);
        std::vector<int> pixels = std::get<1>(position_lines_pixels);

        int nr_coeffs = (int)(0.5 * (pow(srp_polynomial_degree_ + 1, 2) + srp_polynomial_degree_ + 1));

        Eigen::MatrixXd y(srp_number_points_, 1);
        Eigen::MatrixXd A(srp_number_points_, nr_coeffs);

        int minLine = 0;
        int maxLine = meta_master_.lines_per_burst - 1;
        int minPixel = 0;
        int maxPixel = meta_master_.GetBandXSize() - 1;

        for (int i = 0; i < srp_number_points_; i++) {
            double master_time_range = meta_master_.PixelToTimeRange(pixels.at(static_cast<unsigned long>(i)) + 1);

            const auto rows = lines.at(static_cast<unsigned long>(i)) + 1;
            const auto columns = pixels.at(static_cast<unsigned long>(i)) + 1;
            const auto mst_az_time = meta_master_.Line2Ta(burst_i, rows);
            const auto rg_time = meta_master_.PixelToTimeRange(columns);
            auto ellipsoid_position = meta_master_.burst_meta.at(burst_i).approx_xyz_centre;
            s1tbx::Point xyz_master =
                meta_master_.GetOrbit()->RowsColumns2Xyz(rows, columns, mst_az_time, rg_time, ellipsoid_position);
            const auto slave_az_time = meta_slave_.central_avg_az_time;
            s1tbx::Point slave_time_vector = meta_slave_.GetOrbit()->Xyz2T(xyz_master, slave_az_time);
            double slave_time_range = slave_time_vector.GetX();
            y(i) = (master_min_pi_4_div_lam * master_time_range) - (slave_min_pi_4_div_lam * slave_time_range);

            double line = rows;
            line -= (0.5 * (minLine + maxLine));
            line /= (0.25 * (maxLine - minLine));

            double pixel = columns;
            pixel -= (0.5 * (minPixel + maxPixel));
            pixel /= (0.25 * (maxPixel - minPixel));
            double posL = line;
            double posP = pixel;

            int index = 0;

            for (int j = 0; j <= srp_polynomial_degree_; j++) {
                for (int k = 0; k <= j; k++) {
                    A(i, index) = std::pow(posL, j - k) * std::pow(posP, k);
                    index++;
                }
            }
        }

        Eigen::MatrixXd At = A.transpose();

        Eigen::MatrixXd N = At * A;

        Eigen::MatrixXd rhs = At * y;

        Eigen::MatrixXd r = N.completeOrthogonalDecomposition().solve(rhs);

        auto& d_buf = burst_coeffs.at(burst_i);

        d_buf.Resize(r.size());

        cuda::CopyArrayH2D(d_buf.data(), r.data(), r.size());
    }

    auto x_pows = GetXPows(srp_polynomial_degree_);
    auto y_pows = GetYPows(srp_polynomial_degree_);
    coherence_computation_.LaunchCoherencePreTileCalc(x_pows, y_pows, std::move(burst_coeffs));

    auto e = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(e-b).count();
    LOGI << nr_of_bursts << " burst flat earth phase polynomial estimation = " << diff << " ms";
}

void CohCuda::TileCalc(const CohTile& tile, ThreadContext& buffers) {
    coherence_computation_.LaunchCoherence(tile, buffers, coh_win_, band_params_);
}

void CohCuda::PreTileCalc() {
    if (subtract_flat_earth_) {
        CoherencePreTileCalc();
    }
}

}  // namespace coherence_cuda
}  // namespace alus
