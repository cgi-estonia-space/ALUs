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

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "allocators.h"
#include "s1tbx-commons/subswath_info.h"

namespace testing {

class Sentinel1UtilsTester {
public:
    double** doppler_rate_2_{nullptr};
    double** doppler_centroid_2_{nullptr};
    double** reference_time_2_{nullptr};
    double** range_depend_doppler_rate_2_{nullptr};

    std::vector<std::unique_ptr<alus::s1tbx::SubSwathInfo>> subswath_;

    void Read4Arrays(const std::string& doppler_rate_file, const std::string& doppler_centroid_file,
                     const std::string& range_depend_file, const std::string& reference_time_file) {
        std::ifstream doppler_rate_reader(doppler_rate_file);
        std::ifstream doppler_centroid_reader(doppler_centroid_file);
        std::ifstream range_depend_doppler_rate_reader(range_depend_file);
        std::ifstream reference_time_reader(reference_time_file);

        int x;
        int y;
        int i;
        int j;

        doppler_rate_reader >> x >> y;
        doppler_rate_2_ = alus::Allocate2DArray<double>(x, y);

        for (i = 0; i < x; i++) {
            for (j = 0; j < y; j++) {
                doppler_rate_reader >> doppler_rate_2_[i][j];
            }
        }

        doppler_centroid_reader >> x >> y;
        doppler_centroid_2_ = alus::Allocate2DArray<double>(x, y);
        for (i = 0; i < x; i++) {
            for (j = 0; j < y; j++) {
                doppler_centroid_reader >> doppler_centroid_2_[i][j];
            }
        }

        range_depend_doppler_rate_reader >> x >> y;
        range_depend_doppler_rate_2_ = alus::Allocate2DArray<double>(x, y);
        for (i = 0; i < x; i++) {
            for (j = 0; j < y; j++) {
                range_depend_doppler_rate_reader >> range_depend_doppler_rate_2_[i][j];
            }
        }

        reference_time_reader >> x >> y;
        reference_time_2_ = alus::Allocate2DArray<double>(x, y);
        for (i = 0; i < x; i++) {
            for (j = 0; j < y; j++) {
                reference_time_reader >> reference_time_2_[i][j];
            }
        }

        doppler_rate_reader.close();
        doppler_centroid_reader.close();
        range_depend_doppler_rate_reader.close();
        reference_time_reader.close();
    }

    void ReadOriginalPlaceHolderFiles(const std::string& burst_line_file, const std::string& geo_location_file,
                                      int geo_lines, int points_per_line) {
        int size;
        std::ifstream burst_line_time_reader(burst_line_file);
        if (!burst_line_time_reader.is_open()) {
            throw std::ios::failure("Burst Line times file not open.");
        }
        burst_line_time_reader >> size;
        subswath_.push_back(std::make_unique<alus::s1tbx::SubSwathInfo>());
        subswath_.at(0)->num_of_geo_lines_ = geo_lines;
        subswath_.at(0)->num_of_geo_points_per_line_ = points_per_line;

        subswath_.at(0)->burst_first_line_time_.resize(size);
        subswath_.at(0)->burst_last_line_time_.resize(size);

        for (int i = 0; i < size; i++) {
            burst_line_time_reader >> subswath_.at(0)->burst_first_line_time_.at(i);
        }
        for (int i = 0; i < size; i++) {
            burst_line_time_reader >> subswath_.at(0)->burst_last_line_time_.at(i);
        }

        burst_line_time_reader.close();

        std::ifstream geo_location_reader(geo_location_file);
        if (!geo_location_reader.is_open()) {
            throw std::ios::failure("Geo Location file not open.");
        }
        int num_of_geo_lines2;
        int num_of_geo_points_per_line2;

        geo_location_reader >> num_of_geo_lines2 >> num_of_geo_points_per_line2;
        if ((num_of_geo_lines2 != subswath_.at(0)->num_of_geo_lines_) ||
            (num_of_geo_points_per_line2 != subswath_.at(0)->num_of_geo_points_per_line_)) {
            geo_location_reader.close();

            std::stringstream ss;
            ss << "Geo lines and Geo points per lines are not equal to ones in the file. The numbers: "
               << num_of_geo_lines2 << " vs " << subswath_.at(0)->num_of_geo_lines_ << " and "
               << num_of_geo_points_per_line2 << " vs " << subswath_.at(0)->num_of_geo_points_per_line_ << std::endl;
            throw std::runtime_error(ss.str());
        }
        subswath_.at(0)->azimuth_time_ = alus::Allocate2DArray<double>(num_of_geo_lines2, num_of_geo_points_per_line2);
        subswath_.at(0)->slant_range_time_ =
            alus::Allocate2DArray<double>(num_of_geo_lines2, num_of_geo_points_per_line2);
        subswath_.at(0)->latitude_ = alus::Allocate2DArray<double>(num_of_geo_lines2, num_of_geo_points_per_line2);
        subswath_.at(0)->longitude_ = alus::Allocate2DArray<double>(num_of_geo_lines2, num_of_geo_points_per_line2);
        subswath_.at(0)->incidence_angle_ =
            alus::Allocate2DArray<double>(num_of_geo_lines2, num_of_geo_points_per_line2);

        for (int i = 0; i < num_of_geo_lines2; i++) {
            for (int j = 0; j < num_of_geo_points_per_line2; j++) {
                geo_location_reader >> subswath_.at(0)->azimuth_time_[i][j];
            }
        }
        for (int i = 0; i < num_of_geo_lines2; i++) {
            for (int j = 0; j < num_of_geo_points_per_line2; j++) {
                geo_location_reader >> subswath_.at(0)->slant_range_time_[i][j];
            }
        }
        for (int i = 0; i < num_of_geo_lines2; i++) {
            for (int j = 0; j < num_of_geo_points_per_line2; j++) {
                geo_location_reader >> subswath_.at(0)->latitude_[i][j];
            }
        }
        for (int i = 0; i < num_of_geo_lines2; i++) {
            for (int j = 0; j < num_of_geo_points_per_line2; j++) {
                geo_location_reader >> subswath_.at(0)->longitude_[i][j];
            }
        }
        for (int i = 0; i < num_of_geo_lines2; i++) {
            for (int j = 0; j < num_of_geo_points_per_line2; j++) {
                geo_location_reader >> subswath_.at(0)->incidence_angle_[i][j];
            }
        }

        geo_location_reader.close();
    }

    Sentinel1UtilsTester() = default;
    ~Sentinel1UtilsTester() {
        alus::Deallocate2DArray(doppler_rate_2_);
        alus::Deallocate2DArray(doppler_centroid_2_);
        alus::Deallocate2DArray(range_depend_doppler_rate_2_);
        alus::Deallocate2DArray(reference_time_2_);
    }
    Sentinel1UtilsTester(const Sentinel1UtilsTester&) = delete;  // class does not support copying(and moving)
    Sentinel1UtilsTester& operator=(const Sentinel1UtilsTester&) = delete;
};

}  // namespace testing