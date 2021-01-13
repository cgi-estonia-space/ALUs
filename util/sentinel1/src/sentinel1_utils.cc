#include "sentinel1_utils.h"

#include <map>
#include <memory>
#include <string>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "general_constants.h"
#include "meta_data_node_names.h"
#include "product_data_utc.h"

namespace alus {
namespace s1tbx {

Sentinel1Utils::Sentinel1Utils() { WritePlaceolderInfo(2); }

Sentinel1Utils::Sentinel1Utils(int placeholderType) { WritePlaceolderInfo(placeholderType); }

Sentinel1Utils::~Sentinel1Utils() { DeviceFree(); }

// TODO: using placeholder data
void Sentinel1Utils::WritePlaceolderInfo(int placeholder_type) {
    num_of_sub_swath_ = 1;

    SubSwathInfo temp;
    subswath_.push_back(temp);

    // master
    switch (placeholder_type) {
        case 1:

            first_line_utc_ = 6362.663629015139;
            last_line_utc_ = 6362.664239377373;
            line_time_interval_ = 2.3791160879629606E-8;
            near_edge_slant_range_ = 803365.0384269019;
            wavelength_ = 0.05546576;
            range_spacing_ = 2.329562;
            azimuth_spacing_ = 13.91421;

            source_image_width_ = 21401;
            source_image_height_ = 28557;
            near_range_on_left_ = 1;
            srgr_flag_ = 0;
            // srgrConvParams = null;

            subswath_.at(0).azimuth_time_interval_ = 0.002055556299999998;
            subswath_.at(0).num_of_bursts_ = 19;
            subswath_.at(0).lines_per_burst_ = 1503;
            subswath_.at(0).samples_per_burst_ = 21401;
            subswath_.at(0).first_valid_pixel_ = 267;
            subswath_.at(0).last_valid_pixel_ = 20431;
            subswath_.at(0).range_pixel_spacing_ = 2.329562;
            subswath_.at(0).slr_time_to_first_pixel_ = 0.002679737321566982;
            subswath_.at(0).slr_time_to_last_pixel_ = 0.0028460277850849134;
            subswath_.at(0).subswath_name_ = "IW1";
            subswath_.at(0).first_line_time_ = 5.49734137546908E8;
            subswath_.at(0).last_line_time = 5.49734190282205E8;
            subswath_.at(0).radar_frequency_ = 5.40500045433435E9;
            subswath_.at(0).azimuth_steering_rate_ = 1.590368784;
            subswath_.at(0).num_of_geo_lines_ = 21;
            subswath_.at(0).num_of_geo_points_per_line_ = 21;
            break;
        // slave
        case 2:

            first_line_utc_ = 6374.66363659448;
            last_line_utc_ = 6374.6642469804865;
            line_time_interval_ = 2.3791160879629606E-8;
            near_edge_slant_range_ = 803365.0384269019;
            wavelength_ = 0.05546576;
            range_spacing_ = 2.329562;
            azimuth_spacing_ = 13.91417;

            source_image_width_ = 21401;
            source_image_height_ = 28557;
            near_range_on_left_ = 1;
            srgr_flag_ = 0;
            // srgrConvParams = null;

            subswath_.at(0).azimuth_time_interval_ = 0.002055556299999998;
            subswath_.at(0).num_of_bursts_ = 19;
            subswath_.at(0).lines_per_burst_ = 1503;
            subswath_.at(0).samples_per_burst_ = 21401;
            subswath_.at(0).first_valid_pixel_ = 267;
            subswath_.at(0).last_valid_pixel_ = 20431;
            subswath_.at(0).range_pixel_spacing_ = 2.329562;
            subswath_.at(0).slr_time_to_first_pixel_ = 0.002679737321566982;
            subswath_.at(0).slr_time_to_last_pixel_ = 0.0028460277850849134;
            subswath_.at(0).subswath_name_ = "IW1";
            subswath_.at(0).first_line_time_ = 5.50770938201763E8;
            subswath_.at(0).last_line_time = 5.50770990939114E8;
            subswath_.at(0).radar_frequency_ = 5.40500045433435E9;
            subswath_.at(0).azimuth_steering_rate_ = 1.590368784;
            subswath_.at(0).num_of_geo_lines_ = 21;
            subswath_.at(0).num_of_geo_points_per_line_ = 21;
            break;
    }
}

void Sentinel1Utils::ReadPlaceHolderFiles() {
    int size;
    std::ifstream burst_line_time_reader(burst_line_time_file_);
    if (!burst_line_time_reader.is_open()) {
        throw std::ios::failure("Burst Line times file not open.");
    }
    burst_line_time_reader >> size;

    subswath_.at(0).burst_first_line_time_.resize(size);
    subswath_.at(0).burst_last_line_time_.resize(size);

    for (int i = 0; i < size; i++) {
        burst_line_time_reader >> subswath_.at(0).burst_first_line_time_[i];
    }
    for (int i = 0; i < size; i++) {
        burst_line_time_reader >> subswath_.at(0).burst_last_line_time_[i];
    }

    burst_line_time_reader.close();

    std::ifstream geo_location_reader(geo_location_file_);
    if (!geo_location_reader.is_open()) {
        throw std::ios::failure("Geo Location file not open.");
    }
    int num_of_geo_lines2, num_of_geo_points_per_line2;

    geo_location_reader >> num_of_geo_lines2 >> num_of_geo_points_per_line2;
    if ((num_of_geo_lines2 != subswath_.at(0).num_of_geo_lines_) ||
        (num_of_geo_points_per_line2 != subswath_.at(0).num_of_geo_points_per_line_)) {
        throw std::runtime_error("Geo lines and Geo points per lines are not equal to ones in the file.");
    }
    subswath_.at(0).azimuth_time_ = Allocate2DArray<double>(num_of_geo_lines2, num_of_geo_points_per_line2);
    subswath_.at(0).slant_range_time_ = Allocate2DArray<double>(num_of_geo_lines2, num_of_geo_points_per_line2);
    subswath_.at(0).latitude_ = Allocate2DArray<double>(num_of_geo_lines2, num_of_geo_points_per_line2);
    subswath_.at(0).longitude_ = Allocate2DArray<double>(num_of_geo_lines2, num_of_geo_points_per_line2);
    subswath_.at(0).incidence_angle_ = Allocate2DArray<double>(num_of_geo_lines2, num_of_geo_points_per_line2);

    for (int i = 0; i < num_of_geo_lines2; i++) {
        for (int j = 0; j < num_of_geo_points_per_line2; j++) {
            geo_location_reader >> subswath_.at(0).azimuth_time_[i][j];
        }
    }
    for (int i = 0; i < num_of_geo_lines2; i++) {
        for (int j = 0; j < num_of_geo_points_per_line2; j++) {
            geo_location_reader >> subswath_.at(0).slant_range_time_[i][j];
        }
    }
    for (int i = 0; i < num_of_geo_lines2; i++) {
        for (int j = 0; j < num_of_geo_points_per_line2; j++) {
            geo_location_reader >> subswath_.at(0).latitude_[i][j];
        }
    }
    for (int i = 0; i < num_of_geo_lines2; i++) {
        for (int j = 0; j < num_of_geo_points_per_line2; j++) {
            geo_location_reader >> subswath_.at(0).longitude_[i][j];
        }
    }
    for (int i = 0; i < num_of_geo_lines2; i++) {
        for (int j = 0; j < num_of_geo_points_per_line2; j++) {
            geo_location_reader >> subswath_.at(0).incidence_angle_[i][j];
        }
    }

    geo_location_reader.close();
}

void Sentinel1Utils::SetPlaceHolderFiles(std::string_view orbit_state_vectors_file, std::string_view dc_estimate_list_file,
                                         std::string_view azimuth_list_file, std::string_view burst_line_time_file,
                                         std::string_view geo_location_file) {
    orbit_state_vectors_file_ = orbit_state_vectors_file;
    dc_estimate_list_file_ = dc_estimate_list_file;
    azimuth_list_file_ = azimuth_list_file;
    burst_line_time_file_ = burst_line_time_file;
    geo_location_file_ = geo_location_file;
}

double* Sentinel1Utils::ComputeDerampDemodPhase(int subswath_index, int s_burst_index, Rectangle rectangle) {
    const int x0 = rectangle.x;
    const int y0 = rectangle.y;
    const int w = rectangle.width;
    const int h = rectangle.height;

    const int x_max = x0 + w;
    const int y_max = y0 + h;
    const int s = subswath_index - 1;
    const int first_line_in_burst = s_burst_index * subswath_.at(s).lines_per_burst_;

    double* result = new double[h * w * sizeof(double)];
    int yy, xx, x, y;
    double ta, kt, deramp, demod;

    for (y = y0; y < y_max; y++) {
        yy = y - y0;
        ta = (y - first_line_in_burst) * subswath_.at(s).azimuth_time_interval_;
        for (x = x0; x < x_max; x++) {
            xx = x - x0;
            kt = subswath_.at(s).doppler_rate_[s_burst_index][x];
            deramp = -alus::snapengine::constants::PI * kt *
                     pow(ta - subswath_.at(s).reference_time_[s_burst_index][x], 2);
            demod =
                -alus::snapengine::constants::TWO_PI * subswath_.at(s).doppler_centroid_[s_burst_index][x] * ta;
            result[yy * w + xx] = deramp + demod;
        }
    }

    return result;
}

// TODO: is using placeholder info
void Sentinel1Utils::GetProductOrbit() {
    std::vector<snapengine::OrbitStateVector> original_vectors;
    int i;
    int count;
    snapengine::OrbitStateVector temp_vector;

    std::ifstream vector_reader(orbit_state_vectors_file_);
    if (!vector_reader.is_open()) {
        throw std::ios::failure("Vector reader is not open.");
    }
    vector_reader >> count;
    std::cout << "writing original vectors: " << count << '\n';
    for (i = 0; i < count; i++) {
        int days{}, seconds{}, microseconds{};
        vector_reader >> days >> seconds >> microseconds;
        temp_vector.time_ = std::make_shared<snapengine::Utc>(days, seconds, microseconds);
        vector_reader >> temp_vector.time_mjd_;
        vector_reader >> temp_vector.x_pos_ >> temp_vector.y_pos_ >> temp_vector.z_pos_;
        vector_reader >> temp_vector.x_vel_ >> temp_vector.y_vel_ >> temp_vector.z_vel_;
        original_vectors.push_back(temp_vector);
    }
    vector_reader >> count;
    orbit_ = std::make_unique<alus::s1tbx::OrbitStateVectors>(original_vectors);

    vector_reader.close();
    is_orbit_available_ = true;
}

double Sentinel1Utils::GetVelocity(double time) {
    snapengine::PosVector velocity = *orbit_->GetVelocity(time);
    return sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
}

void Sentinel1Utils::ComputeDopplerRate() {
    double wave_length, az_time, v, steering_rate, krot;

    if (!is_orbit_available_) {
        GetProductOrbit();
    }

    if (!is_range_depend_doppler_rate_available_) {
        ComputeRangeDependentDopplerRate();
    }

    wave_length = alus::snapengine::constants::lightSpeed / subswath_.at(0).radar_frequency_;
    for (int s = 0; s < num_of_sub_swath_; s++) {
        az_time = (subswath_.at(s).first_line_time_ + subswath_.at(s).last_line_time) / 2.0;
        subswath_.at(s).doppler_rate_ =
            Allocate2DArray<double>(subswath_.at(s).num_of_bursts_, subswath_.at(s).samples_per_burst_);
        v = GetVelocity(az_time / alus::snapengine::constants::secondsInDay);  // DLR: 7594.0232
        steering_rate = subswath_.at(s).azimuth_steering_rate_ * alus::snapengine::constants::DTOR;
        krot = 2 * v * steering_rate / wave_length;  // doppler rate by antenna steering

        for (int b = 0; b < subswath_.at(s).num_of_bursts_; b++) {
            for (int x = 0; x < subswath_.at(s).samples_per_burst_; x++) {
                subswath_.at(s).doppler_rate_[b][x] =
                    subswath_.at(s).range_depend_doppler_rate_[b][x] * krot /
                    (subswath_.at(s).range_depend_doppler_rate_[b][x] - krot);
            }
        }
    }
}

double Sentinel1Utils::GetSlantRangeTime(int x, int subswath_index) {
    return subswath_.at(subswath_index - 1).slr_time_to_first_pixel_ +
           x * subswath_.at(subswath_index - 1).range_pixel_spacing_ / alus::snapengine::constants::lightSpeed;
}

// TODO: using mock data
std::vector<DCPolynomial> Sentinel1Utils::GetDCEstimateList(std::string subswath_name) {
    std::vector<DCPolynomial> result;
    int count;
    int i;
    int j;
    int temp_count;
    double temp;
    std::cout << "Mocking for subswath: " << subswath_name << '\n';
    std::ifstream dc_lister(dc_estimate_list_file_);
    if (!dc_lister.is_open()) {
        throw std::ios::failure("Azimuth list reader is not open.");
    }

    dc_lister >> count;
    result.reserve(count);
    for (i = 0; i < count; i++) {
        DCPolynomial temp_poly;
        dc_lister >> temp_poly.time >> temp_poly.t0 >> temp_count;
        temp_poly.data_dc_polynomial.reserve(temp_count);
        for (j = 0; j < temp_count; j++) {
            dc_lister >> temp;
            temp_poly.data_dc_polynomial.push_back(temp);
        }
        result.push_back(temp_poly);
    }

    dc_lister.close();

    return result;
}

DCPolynomial Sentinel1Utils::ComputeDC(double center_time, std::vector<DCPolynomial> dc_estimate_list) {
    DCPolynomial dc_polynomial;
    double mu;
    int i0 = 0, i1 = 0;
    if (center_time < dc_estimate_list.at(0).time) {
        i0 = 0;
        i1 = 1;
    } else if (center_time > dc_estimate_list.at(dc_estimate_list.size() - 1).time) {
        i0 = dc_estimate_list.size() - 2;
        i1 = dc_estimate_list.size() - 1;
    } else {
        for (unsigned int i = 0; i < dc_estimate_list.size() - 1; i++) {
            if (center_time >= dc_estimate_list.at(i).time && center_time < dc_estimate_list.at(i + 1).time) {
                i0 = i;
                i1 = i + 1;
                break;
            }
        }
    }

    dc_polynomial.time = center_time;
    dc_polynomial.t0 = dc_estimate_list.at(i0).t0;
    dc_polynomial.data_dc_polynomial.reserve(dc_estimate_list.at(i0).data_dc_polynomial.size());
    mu = (center_time - dc_estimate_list.at(i0).time) / (dc_estimate_list.at(i1).time - dc_estimate_list.at(i0).time);
    for (unsigned int j = 0; j < dc_estimate_list.at(i0).data_dc_polynomial.size(); j++) {
        dc_polynomial.data_dc_polynomial[j] = (1 - mu) * dc_estimate_list.at(i0).data_dc_polynomial[j] +
                                              mu * dc_estimate_list.at(i1).data_dc_polynomial[j];
    }

    return dc_polynomial;
}

// TODO: Half of this function will not work due to missing data. We just got lucky atm.
std::vector<DCPolynomial> Sentinel1Utils::ComputeDCForBurstCenters(std::vector<DCPolynomial> dc_estimate_list,
                                                                   int subswath_index) {
    double center_time;
    if ((int)dc_estimate_list.size() >= subswath_[subswath_index - 1].num_of_bursts_) {
        std::cout << "used the fast lane" << '\n';
        return dc_estimate_list;
    }

    std::vector<DCPolynomial> dcBurstList(subswath_[subswath_index - 1].num_of_bursts_);
    for (int b = 0; b < subswath_[subswath_index - 1].num_of_bursts_; b++) {
        if (b < (int)dc_estimate_list.size()) {
            dcBurstList[b] = dc_estimate_list[b];
            std::cout << "using less list" << '\n';
        } else {
            std::cout << "using more list" << '\n';
            center_time = 0.5 * (subswath_[subswath_index - 1].burst_first_line_time_[b] +
                                 subswath_[subswath_index - 1].burst_last_line_time_[b]);

            dcBurstList[b] = ComputeDC(center_time, dc_estimate_list);
        }
    }

    return dcBurstList;
}

void Sentinel1Utils::ComputeDopplerCentroid() {
    double slrt, dt, dc_value;
    for (int s = 0; s < num_of_sub_swath_; s++) {
        std::vector<DCPolynomial> dc_estimate_list = GetDCEstimateList(subswath_.at(s).subswath_name_);
        std::vector<DCPolynomial> dc_burst_list = ComputeDCForBurstCenters(dc_estimate_list, s + 1);
        subswath_.at(s).doppler_centroid_ =
            Allocate2DArray<double>(subswath_.at(s).num_of_bursts_, subswath_.at(s).samples_per_burst_);
        for (int b = 0; b < subswath_.at(s).num_of_bursts_; b++) {
            for (int x = 0; x < subswath_.at(s).samples_per_burst_; x++) {
                slrt = GetSlantRangeTime(x, s + 1) * 2;
                dt = slrt - dc_burst_list[b].t0;

                dc_value = 0.0;
                for (unsigned int i = 0; i < dc_burst_list[b].data_dc_polynomial.size(); i++) {
                    dc_value += dc_burst_list[b].data_dc_polynomial[i] * pow(dt, i);
                }
                subswath_.at(s).doppler_centroid_[b][x] = dc_value;
            }
        }
    }

    is_doppler_centroid_available_ = true;
}

// TODO: useing mock data
std::vector<AzimuthFmRate> Sentinel1Utils::GetAzimuthFmRateList(std::string subswath_name) {
    std::vector<AzimuthFmRate> result;
    int count, i;
    std::cout << "Getting azimuth FM list for subswath: " << subswath_name << '\n';
    std::ifstream azimuth_list_reader(azimuth_list_file_);
    if (!azimuth_list_reader.is_open()) {
        throw std::ios::failure("Azimuth list reader is not open.");
    }

    azimuth_list_reader >> count;
    result.reserve(count);
    for (i = 0; i < count; i++) {
        AzimuthFmRate temp;
        azimuth_list_reader >> temp.time >> temp.t0 >> temp.c0 >> temp.c1 >> temp.c2;
        result.push_back(temp);
    }

    azimuth_list_reader.close();

    return result;
}

void Sentinel1Utils::ComputeRangeDependentDopplerRate() {
    double slrt, dt;

    for (int s = 0; s < num_of_sub_swath_; s++) {
        std::vector<AzimuthFmRate> az_fm_rate_list = GetAzimuthFmRateList(subswath_.at(s).subswath_name_);
        subswath_.at(s).range_depend_doppler_rate_ =
            Allocate2DArray<double>(subswath_.at(s).num_of_bursts_, subswath_.at(s).samples_per_burst_);
        for (int b = 0; b < subswath_.at(s).num_of_bursts_; b++) {
            for (int x = 0; x < subswath_.at(s).samples_per_burst_; x++) {
                slrt = GetSlantRangeTime(x, s + 1) * 2;  // 1-way to 2-way
                dt = slrt - az_fm_rate_list[b].t0;
                subswath_.at(s).range_depend_doppler_rate_[b][x] =
                    az_fm_rate_list[b].c0 + az_fm_rate_list[b].c1 * dt + az_fm_rate_list[b].c2 * dt * dt;
            }
        }
    }
    is_range_depend_doppler_rate_available_ = true;
}

void Sentinel1Utils::ComputeReferenceTime() {
    double tmp1, tmp2;
    if (!is_doppler_centroid_available_) {
        ComputeDopplerCentroid();
    }

    if (!is_range_depend_doppler_rate_available_) {
        ComputeRangeDependentDopplerRate();
    }

    for (int s = 0; s < num_of_sub_swath_; s++) {
        subswath_.at(s).reference_time_ =
            Allocate2DArray<double>(subswath_.at(s).num_of_bursts_, subswath_.at(s).samples_per_burst_);
        tmp1 = subswath_.at(s).lines_per_burst_ * subswath_.at(s).azimuth_time_interval_ / 2.0;

        for (int b = 0; b < subswath_.at(s).num_of_bursts_; b++) {
            tmp2 = tmp1 +
                   subswath_.at(s).doppler_centroid_[b][subswath_.at(s).first_valid_pixel_] /
                       subswath_.at(s).range_depend_doppler_rate_[b][subswath_.at(s).first_valid_pixel_];

            for (int x = 0; x < subswath_.at(s).samples_per_burst_; x++) {
                subswath_.at(s).reference_time_[b][x] =
                    tmp2 - subswath_.at(s).doppler_centroid_[b][x] /
                               subswath_.at(s).range_depend_doppler_rate_[b][x];
            }
        }
    }
}

double Sentinel1Utils::GetLatitude(double azimuth_time, double slant_range_time, SubSwathInfo* subswath) {
    return GetLatitudeValue(ComputeIndex(azimuth_time, slant_range_time, subswath), subswath);
}
double Sentinel1Utils::GetLongitude(double azimuth_time, double slant_range_time, SubSwathInfo* subswath) {
    return GetLongitudeValue(ComputeIndex(azimuth_time, slant_range_time, subswath), subswath);
}

Sentinel1Index Sentinel1Utils::ComputeIndex(double azimuth_time, double slant_range_time, SubSwathInfo* subswath) {
    Sentinel1Index result;
    int j0 = -1, j1 = -1;
    double mu_x = 0;
    if (slant_range_time < subswath->slant_range_time_[0][0]) {
        j0 = 0;
        j1 = 1;
    } else if (slant_range_time > subswath->slant_range_time_[0][subswath->num_of_geo_points_per_line_ - 1]) {
        j0 = subswath->num_of_geo_points_per_line_ - 2;
        j1 = subswath->num_of_geo_points_per_line_ - 1;
    } else {
        for (int j = 0; j < subswath->num_of_geo_points_per_line_ - 1; j++) {
            if (subswath->slant_range_time_[0][j] <= slant_range_time &&
                subswath->slant_range_time_[0][j + 1] > slant_range_time) {
                j0 = j;
                j1 = j + 1;
                break;
            }
        }
    }

    mu_x = (slant_range_time - subswath->slant_range_time_[0][j0]) /
           (subswath->slant_range_time_[0][j1] - subswath->slant_range_time_[0][j0]);

    int i0 = -1, i1 = -1;
    double mu_y = 0;
    for (int i = 0; i < subswath->num_of_geo_lines_ - 1; i++) {
        double i0_az_time = (1 - mu_x) * subswath->azimuth_time_[i][j0] + mu_x * subswath->azimuth_time_[i][j1];

        double i1_az_time = (1 - mu_x) * subswath->azimuth_time_[i + 1][j0] + mu_x * subswath->azimuth_time_[i + 1][j1];

        if ((i == 0 && azimuth_time < i0_az_time) ||
            (i == subswath->num_of_geo_lines_ - 2 && azimuth_time >= i1_az_time) ||
            (i0_az_time <= azimuth_time && i1_az_time > azimuth_time)) {
            i0 = i;
            i1 = i + 1;
            mu_y = (azimuth_time - i0_az_time) / (i1_az_time - i0_az_time);
            break;
        }
    }

    result.i0 = i0;
    result.i1 = i1;
    result.j0 = j0;
    result.j1 = j1;
    result.mu_x = mu_x;
    result.mu_y = mu_y;

    return result;
}

double Sentinel1Utils::GetLatitudeValue(Sentinel1Index index, SubSwathInfo* subswath) {
    double lat00 = subswath->latitude_[index.i0][index.j0];
    double lat01 = subswath->latitude_[index.i0][index.j1];
    double lat10 = subswath->latitude_[index.i1][index.j0];
    double lat11 = subswath->latitude_[index.i1][index.j1];

    return (1 - index.mu_y) * ((1 - index.mu_x) * lat00 + index.mu_x * lat01) +
           index.mu_y * ((1 - index.mu_x) * lat10 + index.mu_x * lat11);
}

double Sentinel1Utils::GetLongitudeValue(Sentinel1Index index, SubSwathInfo* subswath) {
    double lon00 = subswath->longitude_[index.i0][index.j0];
    double lon01 = subswath->longitude_[index.i0][index.j1];
    double lon10 = subswath->longitude_[index.i1][index.j0];
    double lon11 = subswath->longitude_[index.i1][index.j1];

    return (1 - index.mu_y) * ((1 - index.mu_x) * lon00 + index.mu_x * lon01) +
           index.mu_y * ((1 - index.mu_x) * lon10 + index.mu_x * lon11);
}

void Sentinel1Utils::HostToDevice() {
    DeviceSentinel1Utils temp_pack;

    temp_pack.first_line_utc = first_line_utc_;
    temp_pack.last_line_utc = last_line_utc_;
    temp_pack.line_time_interval = line_time_interval_;
    temp_pack.near_edge_slant_range = near_edge_slant_range_;
    temp_pack.wavelength = wavelength_;
    temp_pack.range_spacing = range_spacing_;
    temp_pack.azimuth_spacing = azimuth_spacing_;

    temp_pack.source_image_width = source_image_width_;
    temp_pack.source_image_height = source_image_height_;
    temp_pack.near_range_on_left = near_range_on_left_;
    temp_pack.srgr_flag = srgr_flag_;

    CHECK_CUDA_ERR(cudaMalloc((void**)&device_sentinel_1_utils_, sizeof(DeviceSentinel1Utils)));
    CHECK_CUDA_ERR(
        cudaMemcpy(device_sentinel_1_utils_, &temp_pack, sizeof(DeviceSentinel1Utils), cudaMemcpyHostToDevice));
}

void Sentinel1Utils::DeviceToHost() { CHECK_CUDA_ERR(cudaErrorNotYetImplemented); }

void Sentinel1Utils::DeviceFree() {
    if (device_sentinel_1_utils_ != nullptr) {
        cudaFree(device_sentinel_1_utils_);
        device_sentinel_1_utils_ = nullptr;
    }
}

std::shared_ptr<snapengine::Utc> Sentinel1Utils::GetTime(std::shared_ptr<snapengine::MetadataElement> element,
                                                         std::string_view tag) {
    auto start = element->GetAttributeString(tag, snapengine::MetaDataNodeNames::NO_METADATA_STRING);

    return snapengine::Utc::Parse(start, "%Y-%m-%dT%H:%M:%S");
}

template <typename T>
void AddToVector(std::vector<T>& vector, std::string_view csv_string, std::string_view delim) {
    std::vector<std::string> tokens;
    boost::split(tokens, csv_string, boost::is_any_of(delim));
    for (auto&& token : tokens) {
        vector.push_back(boost::lexical_cast<T>(token));
    }
}

std::vector<CalibrationVector> Sentinel1Utils::GetCalibrationVectors(
    const std::shared_ptr<snapengine::MetadataElement>& calibration_vector_list_element, bool output_sigma_band,
    bool output_beta_band, bool output_gamma_band, bool output_dn_band) {
    auto check_that_metadata_exists = [](std::shared_ptr<snapengine::MetadataElement> element,
                                         std::string_view parent_element, std::string_view element_name) {
        if (!element) {
            throw std::runtime_error(std::string(parent_element) + " is missing " + std::string(element_name) +
                                     " metadata element");
        }
    };

    auto get_element = [&](std::shared_ptr<snapengine::MetadataElement> parent_element, std::string_view element_name) {
        auto element = parent_element->GetElement(element_name);
        check_that_metadata_exists(element, parent_element->GetName(), element_name);

        return element;
    };

    std::map<std::string_view, bool> selected_bands{{snapengine::MetaDataNodeNames::SIGMA_NOUGHT, output_sigma_band},
                                                    {snapengine::MetaDataNodeNames::BETA_NOUGHT, output_beta_band},
                                                    {snapengine::MetaDataNodeNames::GAMMA, output_gamma_band},
                                                    {snapengine::MetaDataNodeNames::DN, output_dn_band}};

    const auto calibration_vector_elements = calibration_vector_list_element->GetElements();

    std::vector<CalibrationVector> calibration_vectors;
    calibration_vectors.reserve(calibration_vector_elements.size());

    for (auto&& calibration_vector_element : calibration_vector_elements) {
        std::map<std::string_view, std::vector<float>> unit_data{{snapengine::MetaDataNodeNames::SIGMA_NOUGHT, {}},
                                                                 {snapengine::MetaDataNodeNames::BETA_NOUGHT, {}},
                                                                 {snapengine::MetaDataNodeNames::GAMMA, {}},
                                                                 {snapengine::MetaDataNodeNames::DN, {}}};

        const auto time = GetTime(calibration_vector_element, snapengine::MetaDataNodeNames::AZIMUTH_TIME);
        const auto line = calibration_vector_element->GetAttributeInt(snapengine::MetaDataNodeNames::LINE);

        const auto pixel_element = get_element(calibration_vector_element, snapengine::MetaDataNodeNames::PIXEL);
        const auto pixel_string = pixel_element->GetAttributeString(snapengine::MetaDataNodeNames::PIXEL);
        const auto count = pixel_element->GetAttributeInt(snapengine::MetaDataNodeNames::COUNT);

        std::vector<int> pixels;
        pixels.reserve(count);
        AddToVector(pixels, pixel_string, " ");

        auto fetch_unit_data = [&](std::string_view selected_type) {
            auto& unit_vector = unit_data.at(selected_type);
            unit_vector.reserve(count);
            const auto unit_element = get_element(calibration_vector_element, selected_type);
            const auto unit_string = unit_element->GetAttributeString(selected_type);
            AddToVector(unit_vector, unit_string, " ");
        };

        for (auto&& unit : selected_bands) {
            if (unit.second) {
                fetch_unit_data(unit.first);
            }
        }

        calibration_vectors.push_back({time->GetMjd(), line, pixels, unit_data.at(snapengine::MetaDataNodeNames::SIGMA_NOUGHT),
                                       unit_data.at(snapengine::MetaDataNodeNames::BETA_NOUGHT),
                                       unit_data.at(snapengine::MetaDataNodeNames::GAMMA), unit_data.at(snapengine::MetaDataNodeNames::DN),
                                       static_cast<size_t>(count)});
    }

    return calibration_vectors;
}
}  // namespace s1tbx
}  // namespace alus