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
#include "coherence_calc.h"

#include <algorithm>
#include <cmath>

#include "tensorflow/cc/ops/math_ops_internal.h"

#include "jlinda-core/constants.h"

namespace alus {

Coh::Coh(const int srp_number_points, const int srp_polynomial_degree, const bool subtract_flat_earth,
         const CohWindow& coh_window, const int orbit_degree, MetaData& meta_master, MetaData& meta_slave)
    : srp_number_points_{srp_number_points},
      srp_polynomial_degree_{srp_polynomial_degree},
      subtract_flat_earth_{subtract_flat_earth},
      coh_win_rg_{coh_window.rg},
      coh_win_az_{coh_window.az},
      orbit_degree_{orbit_degree},
      meta_master_{meta_master},
      meta_slave_{meta_slave} {}

std::tuple<std::vector<int>, std::vector<int>> Coh::DistributePoints(int num_of_points, int band_x_size, int band_x_min,
                                                                     int band_y_size, int band_y_min) {
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

auto Coh::Norm(const tensorflow::Scope& root, const tensorflow::Input& tensor) {
    return tensorflow::ops::Add(root,
                                tensorflow::ops::Multiply(root.WithOpName("1"), tensorflow::ops::Real(root, tensor),
                                                          tensorflow::ops::Real(root, tensor)),
                                tensorflow::ops::Multiply(root.WithOpName("2"), tensorflow::ops::Imag(root, tensor),
                                                          tensorflow::ops::Imag(root, tensor)));
}

auto Coh::ComplexAbs(const tensorflow::Scope& root, const tensorflow::Input& complex_nr) {
    return tensorflow::ops::Sqrt(
        root, tensorflow::ops::Add(root, tensorflow::ops::Square(root, tensorflow::ops::Real(root, complex_nr)),
                                   tensorflow::ops::Square(root, tensorflow::ops::Imag(root, complex_nr))));
}

auto Coh::CoherenceProduct(const tensorflow::Scope& root, const tensorflow::Input& sum_t,
                           const tensorflow::Input& power_t) {
    auto product_t = tensorflow::ops::Multiply(root.WithOpName("3"), tensorflow::ops::Real(root, power_t),
                                               tensorflow::ops::Imag(root, power_t));
    auto zero_t = tensorflow::ops::ZerosLike(root, product_t);
    auto condition = tensorflow::ops::Greater(root, product_t, zero_t);

    return tensorflow::ops::Where3(
        root, condition, tensorflow::ops::Div(root, ComplexAbs(root, sum_t), tensorflow::ops::Sqrt(root, product_t)),
        zero_t);
}

// tensorflow c++ api did not support while loop and ragged tensors, so I gave up and give this as input (which might be
// faster anyway)
std::vector<int> Coh::GetXPows(int srp_polynomial_degree) {
    // todo: copy this to get correct number of x_pows
    // PolyUtils.numberOfCoefficients(srp_polynomial_degree)  to reserve space
    std::vector<int> x_pows;
    for (int i = 0; i < srp_polynomial_degree + 1; i++) {
        for (int j = 0; j < i + 1; j++) {
            x_pows.push_back(i - j);
        }
    }
    return x_pows;
}

std::vector<int> Coh::GetYPows(int srp_polynomial_degree) {
    // todo: copy this to get correct number of xPows
    // PolyUtils.numberOfCoefficients(srp_polynomial_degree)  to reserve space
    std::vector<int> y_pows;
    for (int i = 0; i < srp_polynomial_degree + 1; i++) {
        for (int j = 0; j < i + 1; j++) {
            y_pows.push_back(j);
        }
    }
    return y_pows;
}

auto Coh::NormalizeDoubleMatrix3(const tensorflow::Scope& root, const tensorflow::Input& t,
                                 const tensorflow::Input& min, const tensorflow::Input& max) {
    auto hf = tensorflow::ops::Const(root, 0.5);
    auto qt = tensorflow::ops::Const(root, 0.25);
    // todo: cast before here
    return tensorflow::ops::Div(
        root,
        tensorflow::ops::Subtract(root, t, tensorflow::ops::Multiply(root, hf, tensorflow::ops::Add(root, min, max))),
        tensorflow::ops::Multiply(root, qt, tensorflow::ops::Subtract(root, max, min)));
}

auto Coh::GetA(const tensorflow::Scope& root, const tensorflow::Input& lines, const tensorflow::Input& pixels,
               int x_min_pixel, int x_max_pixel, int y_min_line, int y_max_line, const tensorflow::Input& x_pows,
               const tensorflow::Input& y_pows) {
    auto min_line = tensorflow::ops::Const(root, static_cast<double>(y_min_line));
    auto max_line = tensorflow::ops::Const(root, static_cast<double>(y_max_line));

    auto min_pixel = tensorflow::ops::Const(root, static_cast<double>(x_min_pixel));
    auto max_pixel = tensorflow::ops::Const(root, static_cast<double>(x_max_pixel));

    auto x =
        NormalizeDoubleMatrix3(root, tensorflow::ops::Cast(root, lines, tensorflow::DT_DOUBLE), min_line, max_line);
    auto y =
        NormalizeDoubleMatrix3(root, tensorflow::ops::Cast(root, pixels, tensorflow::DT_DOUBLE), min_pixel, max_pixel);

    return tensorflow::ops::Multiply(
        root.WithOpName("7"),
        tensorflow::ops::Pow(root.WithOpName("pow/1"), tensorflow::ops::Cast(root, x, tensorflow::DT_DOUBLE),
                             tensorflow::ops::Cast(root, x_pows, tensorflow::DT_DOUBLE)),
        tensorflow::ops::Pow(root.WithOpName("pow/2"), tensorflow::ops::Cast(root, y, tensorflow::DT_DOUBLE),
                             tensorflow::ops::Cast(root, y_pows, tensorflow::DT_DOUBLE)));
}

// void GenerateY(int lines, int pixels)

// todo:MetaData should come from this->operator or something like that

std::vector<double> Coh::GenerateY(std::tuple<std::vector<int>, std::vector<int>> lines_pixels, MetaData& meta_master,
                                   MetaData& meta_slave) const {
    double master_min_pi_4_div_lam =
        static_cast<double>(-4.0L * jlinda::SNAP_PI * jlinda::C) / meta_master.GetRadarWaveLength();
    double slave_min_pi_4_div_lam =
        static_cast<double>(-4.0L * jlinda::SNAP_PI * jlinda::C) / meta_slave.GetRadarWaveLength();

    std::vector<int> lines = std::get<0>(lines_pixels);
    std::vector<int> pixels = std::get<1>(lines_pixels);
    std::vector<double> y;
    y.reserve(srp_number_points_);

    for (int i = 0; i < srp_number_points_; i++) {
        double master_time_range = meta_master.PixelToTimeRange(pixels.at(static_cast<unsigned long>(i)) + 1);

        const auto rows = lines.at(static_cast<unsigned long>(i)) + 1;
        const auto columns = pixels.at(static_cast<unsigned long>(i)) + 1;
        const auto az_time = meta_master.Line2Ta(rows);
        const auto rg_time = meta_master.PixelToTimeRange(columns);
        auto ellipsoid_position = meta_master.GetApproxXyzCentreOriginal();
        s1tbx::Point xyz_master =
            meta_master.GetOrbit()->RowsColumns2Xyz(rows, columns, az_time, rg_time, ellipsoid_position);
        const auto line_2_a =
            meta_slave.Line2Ta(static_cast<int>(0.5 * meta_slave.GetApproxRadarCentreOriginal().GetY()));
        s1tbx::Point slave_time_vector = meta_slave.GetOrbit()->Xyz2T(xyz_master, line_2_a);
        double slave_time_range = slave_time_vector.GetX();
        y.push_back(static_cast<double&&>((master_min_pi_4_div_lam * master_time_range) -
                                          (slave_min_pi_4_div_lam * slave_time_range)));
    }

    return y;
}

auto Coh::GetCoefs(const tensorflow::Scope& root, const tensorflow::Input& a, const tensorflow::Input& generate_y) {
    auto rs_y = tensorflow::ops::Reshape(root.WithOpName("rs/16"), generate_y, {-1, 1});
    auto y = tensorflow::ops::Cast(root, rs_y, tensorflow::DT_DOUBLE);
    auto n = tensorflow::ops::MatMul(root, a, a, tensorflow::ops::MatMul::TransposeB(true));
    auto rhs = tensorflow::ops::MatMul(root, a, y);
    auto solve = tensorflow::ops::MatrixSolve(root, n, rhs);

    /*return Const(root, {{-4503.2426667999025       },
         {   -3.410390210210569     },
         { -175.45287510908182      },
         {    0.0022445807962841696 },
         {   -0.137375319953189     },
         {    7.457431849660493     },
         {    0.00004944766023813719},
         {    0.00003437512128983022},
         {    0.00627940379677397   },
         {   -0.3741010944421739    },
         {   -0.00000218024199519107},
         {    0.00001764008057906517},
         {   -0.00000317914482244708},
         {   -0.0003315944430138992 },
         {    0.023134823537937305  },
         {   -0.00001025374749496674},
         {    0.00000972891748855349},
         {   -0.00000955564797113269},
         {   -0.00000537438207954155},
         {    0.00002308499919378773},
         {   -0.0016041266212625392 }});*/
    return solve;
}

auto Coh::Polyval2DDim(const tensorflow::Scope& root, const tensorflow::Input& x, const tensorflow::Input& y,
                       const tensorflow::Input& coefs, const tensorflow::Input& x_pows, const tensorflow::Input& y_pows,
                       int x_size, int y_size) {
    auto tile_y = tensorflow::ops::Tile(root, tensorflow::ops::ExpandDims(root, y, 1), {1, y_size});
    auto tile_y_expand = tensorflow::ops::ExpandDims(root, tile_y, 2);
    auto tile_x = tensorflow::ops::Tile(root, tensorflow::ops::ExpandDims(root, x, 0), {x_size, 1});
    auto tile_x_expand = tensorflow::ops::ExpandDims(root, tile_x, 2);
    auto yc = tensorflow::ops::Reshape(root.WithOpName("rs/1"), tile_y_expand, {-1});
    auto xc = tensorflow::ops::Reshape(root.WithOpName("rs/1"), tile_x_expand, {-1});
    auto xxyy = tensorflow::ops::Multiply(
        root.WithOpName("8"),
        tensorflow::ops::Pow(root.WithOpName("pow/3"), tensorflow::ops::Cast(root, xc, tensorflow::DT_DOUBLE),
                             tensorflow::ops::Cast(root, x_pows, tensorflow::DT_DOUBLE)),
        tensorflow::ops::Pow(root.WithOpName("pow/4"), tensorflow::ops::Cast(root, yc, tensorflow::DT_DOUBLE),
                             tensorflow::ops::Cast(root, y_pows, tensorflow::DT_DOUBLE)));
    auto coef_xxyy = tensorflow::ops::Multiply(root.WithOpName("9"), coefs, xxyy);
    // todo:check if axis=0 is 3rd arg
    auto sum = tensorflow::ops::ReduceSum(root, coef_xxyy, 0);
    //#return tf.reduce_sum(coefXXYYdot)
    return sum;
}

auto Coh::ComputeFlatEarthPhase(const tensorflow::Scope& root, const CohTile& tile, const tensorflow::Input& x_pows,
                                const tensorflow::Input& y_pows, const tensorflow::Input& coefs) const {
    auto min_line = tensorflow::ops::Const(root, static_cast<double>(0));
    auto max_line = tensorflow::ops::Const(root, static_cast<double>(meta_master_.GetBandYSize() - 1));
    auto min_pixel = tensorflow::ops::Const(root, static_cast<double>(0));
    auto max_pixel = tensorflow::ops::Const(root, static_cast<double>(meta_master_.GetBandXSize() - 1));

    auto xx = NormalizeDoubleMatrix3(
        root,
        tensorflow::ops::Cast(root,
                              tensorflow::ops::internal::LinSpace(
                                  root, tensorflow::ops::Cast(root, tile.GetCalcXMin(), tensorflow::DT_DOUBLE),
                                  tensorflow::ops::Cast(root, tile.GetCalcXMax(), tensorflow::DT_DOUBLE), tile.GetWw()),
                              tensorflow::DT_DOUBLE),
        min_pixel, max_pixel);

    auto yy = NormalizeDoubleMatrix3(
        root,
        tensorflow::ops::Cast(root,
                              tensorflow::ops::internal::LinSpace(
                                  root, tensorflow::ops::Cast(root, tile.GetCalcYMin(), tensorflow::DT_DOUBLE),
                                  tensorflow::ops::Cast(root, tile.GetCalcYMax(), tensorflow::DT_DOUBLE), tile.GetHh()),
                              tensorflow::DT_DOUBLE),
        min_line, max_line);
    return Coh::Polyval2DDim(root, yy, xx, coefs, x_pows, y_pows, tile.GetWw(), tile.GetHh());
}

auto Coh::CoherenceTileCalc(const tensorflow::Scope& root, const CohTile& coh_tile, const tensorflow::Input& mst_real,
                            const tensorflow::Input& mst_imag, const tensorflow::Input& slv_real,
                            const tensorflow::Input& slv_imag, const tensorflow::Input& x_pows,
                            const tensorflow::Input& y_pows, const tensorflow::Input& coefs) const {
    // todo: not sure if we need this step now (will try without at start and if it has issues I will reconsider)
    auto mst_real_rs = tensorflow::ops::Reshape(root.WithOpName("mst_real_rs"), mst_real,
                                                {coh_tile.GetTileIn().GetYSize(), coh_tile.GetTileIn().GetXSize()});
    auto mst_imag_rs = tensorflow::ops::Reshape(root.WithOpName("mst_imag_rs"), mst_imag,
                                                {coh_tile.GetTileIn().GetYSize(), coh_tile.GetTileIn().GetXSize()});
    auto slv_real_rs = tensorflow::ops::Reshape(root.WithOpName("slv_real_rs"), slv_real,
                                                {coh_tile.GetTileIn().GetYSize(), coh_tile.GetTileIn().GetXSize()});
    auto slv_imag_rs = tensorflow::ops::Reshape(root.WithOpName("slv_imag_rs"), slv_imag,
                                                {coh_tile.GetTileIn().GetYSize(), coh_tile.GetTileIn().GetXSize()});

    // todo:move this also to CohTile?
    auto padding = tensorflow::ops::Const(
        root, {{coh_tile.GetYMinPad(), coh_tile.GetYMaxPad()}, {coh_tile.GetXMinPad(), coh_tile.GetXMaxPad()}});
    auto mst_real_p = tensorflow::ops::Pad(root, mst_real_rs, padding);
    auto mst_imag_p = tensorflow::ops::Pad(root, mst_imag_rs, padding);
    auto slv_real_p = tensorflow::ops::Pad(root, slv_real_rs, padding);
    auto slv_imag_p = tensorflow::ops::Pad(root, slv_imag_rs, padding);

    // todo:can we get rid of complex numbers!?  for some reason it only allows float32 and double as input, but we have
    // int16!!!!! real: A Tensor. Must be one of the following types: float32, float64. imag: A Tensor. Must have the
    // same type as real. (FROM PYTHON API AND PRINING OUT GRAPH)
    auto data_master_c = tensorflow::ops::Complex(root, mst_real_p, mst_imag_p);
    auto data_slave_c = tensorflow::ops::Complex(root, slv_real_p, slv_imag_p);
    auto data_master = tensorflow::ops::Reshape(root, data_master_c, {coh_tile.GetHh(), coh_tile.GetWw()});

    tensorflow::Output data_slave{};
    if (subtract_flat_earth_) {
        auto flat_earth_phase = ComputeFlatEarthPhase(root, coh_tile, x_pows, y_pows, coefs);
        auto complex_reference_phase_1 = tensorflow::ops::Complex(
            root, tensorflow::ops::Cast(root, tensorflow::ops::Cos(root, flat_earth_phase), tensorflow::DT_FLOAT),
            tensorflow::ops::Cast(root, tensorflow::ops::Sin(root, flat_earth_phase), tensorflow::DT_FLOAT));
        auto complex_reference_phase =
            tensorflow::ops::Transpose(root,
                                       tensorflow::ops::Reshape(root.WithOpName("rs/4"), complex_reference_phase_1,
                                                                {coh_tile.GetWw(), coh_tile.GetHh()}),
                                       {1, 0});
        data_slave = tensorflow::ops::Multiply(
            root.WithOpName("10"),
            tensorflow::ops::Reshape(root.WithOpName("rs/3"), data_slave_c, {coh_tile.GetHh(), coh_tile.GetWw()}),
            complex_reference_phase);
    } else {
        data_slave =
            tensorflow::ops::Reshape(root.WithOpName("rs/9"), data_slave_c, {coh_tile.GetHh(), coh_tile.GetWw()});
    }
    auto sumconv = tensorflow::ops::Const(root, 1.f, {coh_win_rg_, coh_win_az_});
    auto sumconv_reshape =
        tensorflow::ops::Reshape(root.WithOpName("rs/10"), sumconv, {coh_win_az_, coh_win_rg_, 1, 1});
    auto tmp = Norm(root, data_master);
    auto data_master_norm =
        tensorflow::ops::Multiply(root.WithOpName("12"), data_master, tensorflow::ops::Conj(root, data_slave));
    auto data_slave_norm = tensorflow::ops::Complex(root, Norm(root, data_slave), tmp);
    auto data_master_norm_reshape = tensorflow::ops::Reshape(root.WithOpName("rs/11"), data_master_norm,
                                                             {-1, coh_tile.GetHh(), coh_tile.GetWw(), 1});
    auto data_slave_norm_reshape = tensorflow::ops::Reshape(root.WithOpName("rs/12"), data_slave_norm,
                                                            {-1, coh_tile.GetHh(), coh_tile.GetWw(), 1});
    // todo:check if we can do it In a single operation now?

    auto data_master_sum_real = tensorflow::ops::Conv2D(root, tensorflow::ops::Real(root, data_master_norm_reshape),
                                                        sumconv_reshape, {1, 1, 1, 1}, "VALID");
    auto data_master_sum_imag = tensorflow::ops::Conv2D(root, tensorflow::ops::Imag(root, data_master_norm_reshape),
                                                        sumconv_reshape, {1, 1, 1, 1}, "VALID");
    auto data_slave_sum_real = tensorflow::ops::Conv2D(root, tensorflow::ops::Real(root, data_slave_norm_reshape),
                                                       sumconv_reshape, {1, 1, 1, 1}, "VALID");
    auto data_slave_sum_imag = tensorflow::ops::Conv2D(root, tensorflow::ops::Imag(root, data_slave_norm_reshape),
                                                       sumconv_reshape, {1, 1, 1, 1}, "VALID");

    auto data_master_sum = tensorflow::ops::Complex(root, data_master_sum_real, data_master_sum_imag);
    auto data_slave_sum = tensorflow::ops::Complex(root, data_slave_sum_real, data_slave_sum_imag);
    float slave_real_no_data = 0.0f;

    auto coherence_result = tensorflow::ops::Squeeze(root, CoherenceProduct(root, data_master_sum, data_slave_sum));
    auto slv_real_reshape =
        tensorflow::ops::Reshape(root.WithOpName("rs/13"), slv_real_p, {coh_tile.GetHh(), coh_tile.GetWw()});
    auto azw = (coh_win_az_ - 1) / 2;
    auto rgw = (coh_win_rg_ - 1) / 2;
    //    todo: look throught slicing logic when we have more time, on first glance I think this should contain tile
    //    specific coh windows
    auto slv_real_cut = tensorflow::ops::StridedSlice(
        root, slv_real_reshape, {azw, rgw},
        {-(azw + static_cast<int>(coh_win_az_ % 2 == 0)), -(rgw + static_cast<int>(coh_win_rg_ % 2 == 0))}, {1, 1});
    return tensorflow::ops::Where3(root, tensorflow::ops::Equal(root, slv_real_cut, slave_real_no_data), slv_real_cut,
                                   coherence_result);
}

void Coh::CoherencePreTileCalc(const tensorflow::Scope& root) {
    auto b_1 = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
    auto b_2 = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
    // todo: for some inputs we can use INT instead of FLOAT (needs changes down the road e.g Complex is limited atm.)
    auto b_3 = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);
    auto b_4 = tensorflow::ops::Placeholder(root, tensorflow::DT_FLOAT);

    //  this->tile_calc_placeholder_inputs_.emplace_back(root, DT_FLOAT);
    // todo: check where can we use emplace_back instead!
    tile_calc_placeholder_inputs_.push_back(b_1);
    tile_calc_placeholder_inputs_.push_back(b_2);
    tile_calc_placeholder_inputs_.push_back(b_3);
    tile_calc_placeholder_inputs_.push_back(b_4);

    //  todo:think this over later
    // these are used for each tile calculation as inputs
    tile_calc_inputs_.push_back(b_1);
    tile_calc_inputs_.push_back(b_2);
    tile_calc_inputs_.push_back(b_3);
    tile_calc_inputs_.push_back(b_4);

    auto x_pows = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
    auto y_pows = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);

    auto position_lines = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
    auto position_pixels = tensorflow::ops::Placeholder(root, tensorflow::DT_INT32);
    auto get_y = tensorflow::ops::Placeholder(root, tensorflow::DT_DOUBLE);

    auto x_pows_rs = tensorflow::ops::Reshape(root, x_pows, {-1, 1});
    auto y_pows_rs = tensorflow::ops::Reshape(root, y_pows, {-1, 1});

    tile_calc_inputs_.push_back(x_pows_rs);
    tile_calc_inputs_.push_back(y_pows_rs);

    std::vector<int> x_pows_vector = GetXPows(srp_polynomial_degree_);
    std::vector<int> y_pows_vector = GetYPows(srp_polynomial_degree_);

    tensorflow::Tensor x_pows_tensor = GetTensor(x_pows_vector);
    tensorflow::Tensor y_pows_tensor = GetTensor(y_pows_vector);
    std::tuple<std::vector<int>, std::vector<int>> position_lines_pixels =
        DistributePoints(srp_number_points_, meta_master_.GetBandXSize(), 0, meta_master_.GetBandYSize(), 0);
    tensorflow::Tensor position_lines_tensor = GetTensor(std::get<0>(position_lines_pixels));
    tensorflow::Tensor position_pixels_tensor = GetTensor(std::get<1>(position_lines_pixels));
    inputs_.try_emplace(position_lines, position_lines_tensor);
    inputs_.try_emplace(position_pixels, position_pixels_tensor);

    // GenerateY
    std::vector<double> generate_y_vector = GenerateY(position_lines_pixels, meta_master_, meta_slave_);
    tensorflow::Tensor generate_y_tensor = GetTensor(generate_y_vector);

    inputs_.try_emplace(get_y, generate_y_tensor);
    inputs_.try_emplace(x_pows, x_pows_tensor);
    inputs_.try_emplace(y_pows, y_pows_tensor);

    auto a = GetA(root, position_lines, position_pixels, 0, meta_master_.GetBandXSize() - 1, 0,
                  meta_master_.GetBandYSize() - 1, x_pows_rs, y_pows_rs);
    auto coefs = GetCoefs(root, a, get_y);
    tile_calc_inputs_.push_back(coefs);
    // logic to control filling external vector which will hold data we take using gdal.
    // only important part is to understand where data ends for gdal
    // paddings which are only needed on edges
    // we overlap tiles using half coherence window, last and first contain no data so we cut it there and use padding
    // with zeros instead
}

template <typename T>
tensorflow::Tensor Coh::GetTensor(const std::vector<T>& data) {
    tensorflow::Tensor tensor =
        tensorflow::Tensor{tensorflow::DataTypeToEnum<T>::v(), tensorflow::TensorShape{data.size()}};
    std::copy_n(data.begin(), data.size(), tensor.flat<T>().data());
    return tensor;
}

tensorflow::Output Coh::TileCalc(const tensorflow::Scope& root, CohTile& tile) {
    inputs_.insert_or_assign(tile_calc_placeholder_inputs_.at(0), bands_.GetBandMasterReal());
    inputs_.insert_or_assign(tile_calc_placeholder_inputs_.at(1), bands_.GetBandMasterImag());
    inputs_.insert_or_assign(tile_calc_placeholder_inputs_.at(2), bands_.GetBandSlaveReal());
    inputs_.insert_or_assign(tile_calc_placeholder_inputs_.at(3), bands_.GetBandSlaveImag());

    return CoherenceTileCalc(root, tile, tile_calc_inputs_.at(0), tile_calc_inputs_.at(1), tile_calc_inputs_.at(2),
                             tile_calc_inputs_.at(3), tile_calc_inputs_.at(4), tile_calc_inputs_.at(5),
                             tile_calc_inputs_.at(6));
}

void Coh::PreTileCalc(const tensorflow::Scope& root) { CoherencePreTileCalc(root); }

void Coh::DataToTensors(const Tile& tile, const IDataTileReader& reader) {
    bands_.DataToTensors(tile.GetXSize(), tile.GetYSize(), reader.GetData());
}

}  // namespace alus
