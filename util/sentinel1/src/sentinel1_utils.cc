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
#include "sentinel1_utils.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "abstract_metadata.h"
#include "general_constants.h"
#include "product_data_utc.h"
#include "pugixml_meta_data_reader.h"
#include "sar_utils.h"

namespace alus {
namespace s1tbx {

Sentinel1Utils::Sentinel1Utils(std::string_view metadata_file_name) : num_of_sub_swath_(1) {
    metadata_reader_ = std::make_unique<snapengine::PugixmlMetaDataReader>(metadata_file_name);

    subswath_.push_back(std::make_unique<SubSwathInfo>());

    FillSubswathMetaData(subswath_.at(0).get());
    FillUtilsMetadata();
}

Sentinel1Utils::~Sentinel1Utils() { DeviceFree(); }

void Sentinel1Utils::FillUtilsMetadata() {
    std::shared_ptr<snapengine::MetadataElement> abstract_metadata =
        metadata_reader_->Read(alus::snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);

    srgr_flag_ =
        snapengine::AbstractMetadata::GetAttributeBoolean(abstract_metadata, snapengine::AbstractMetadata::SRGR_FLAG);
    wavelength_ = s1tbx::SarUtils::GetRadarFrequency(abstract_metadata);
    range_spacing_ = snapengine::AbstractMetadata::GetAttributeDouble(abstract_metadata,
                                                                      snapengine::AbstractMetadata::RANGE_SPACING);
    azimuth_spacing_ = snapengine::AbstractMetadata::GetAttributeDouble(abstract_metadata,
                                                                        snapengine::AbstractMetadata::AZIMUTH_SPACING);
    first_line_utc_ = abstract_metadata->GetAttributeUtc(snapengine::AbstractMetadata::FIRST_LINE_TIME)->GetMjd();
    last_line_utc_ = abstract_metadata->GetAttributeUtc(snapengine::AbstractMetadata::LAST_LINE_TIME)->GetMjd();
    line_time_interval_ = abstract_metadata->GetAttributeDouble(snapengine::AbstractMetadata::LINE_TIME_INTERVAL) /
                          snapengine::constants::secondsInDay;

    std::shared_ptr<snapengine::MetadataElement> size_info =
        metadata_reader_->Read("product")
            ->GetElement(alus::snapengine::AbstractMetadata::IMAGE_ANNOTATION)
            ->GetElement(alus::snapengine::AbstractMetadata::IMAGE_INFORMATION);

    source_image_width_ = size_info->GetAttributeInt(snapengine::AbstractMetadata::NUMBER_OF_SAMPLES);
    source_image_height_ = size_info->GetAttributeInt(snapengine::AbstractMetadata::NUMBER_OF_LINES);

    if (srgr_flag_) {
        throw std::runtime_error("Use of srgr flag currently not supported in Sentinel 1 utils.");
    } else {
        near_edge_slant_range_ = snapengine::AbstractMetadata::GetAttributeDouble(
            abstract_metadata, snapengine::AbstractMetadata::SLANT_RANGE_TO_FIRST_PIXEL);
    }

    if (subswath_.empty()) {
        throw std::runtime_error("Subswath size 0 currently not supported in Sentinel 1 Utils");
    }
    near_range_on_left_ = (subswath_.at(0)->incidence_angle_[0][0] < subswath_.at(0)->incidence_angle_[0][1]);

}

void Sentinel1Utils::FillSubswathMetaData(SubSwathInfo *subswath) {

    std::shared_ptr<snapengine::MetadataElement> product =
        metadata_reader_->Read("product");
    std::shared_ptr<snapengine::MetadataElement> image_annotation =
        product->GetElement(alus::snapengine::AbstractMetadata::IMAGE_ANNOTATION);
    std::shared_ptr<snapengine::MetadataElement> image_information =
        image_annotation->GetElement(alus::snapengine::AbstractMetadata::IMAGE_INFORMATION);

    std::shared_ptr<snapengine::MetadataElement> swath_timing =
        product->GetElement(snapengine::AbstractMetadata::SWATH_TIMING);
    std::shared_ptr<snapengine::MetadataElement> burst_list =
        swath_timing->GetElement(snapengine::AbstractMetadata::BURST_LIST);
    std::shared_ptr<snapengine::MetadataElement> general_annotation =
        product->GetElement(snapengine::AbstractMetadata::GENERAL_ANNOTATION);
    std::shared_ptr<snapengine::MetadataElement> product_information =
        general_annotation->GetElement(snapengine::AbstractMetadata::PRODUCT_INFORMATION);

    std::shared_ptr<snapengine::MetadataElement> antenna_pattern =
        product->GetElement(snapengine::AbstractMetadata::ANTENNA_PATTERN);
    std::shared_ptr<snapengine::MetadataElement> antenna_pattern_list =
        antenna_pattern->GetElement(snapengine::AbstractMetadata::ANTENNA_PATTERN_LIST);

    subswath->azimuth_time_interval_ =
        std::stod(image_information->GetAttributeString(snapengine::AbstractMetadata::AZIMUTH_TIME_INTERVAL));
    subswath->num_of_samples_ =
        std::stoi(image_information->GetAttributeString(snapengine::AbstractMetadata::NUMBER_OF_SAMPLES));
    subswath->num_of_lines_ =
        std::stoi(image_information->GetAttributeString(snapengine::AbstractMetadata::NUMBER_OF_LINES));
    subswath->num_of_bursts_ = std::stoi(burst_list->GetAttributeString(snapengine::AbstractMetadata::COUNT));
    subswath->lines_per_burst_ =
        std::stoi(swath_timing->GetAttributeString(snapengine::AbstractMetadata::LINES_PER_BURST));
    subswath->samples_per_burst_ =
        std::stoi(swath_timing->GetAttributeString(snapengine::AbstractMetadata::SAMPLES_PER_BURST));
    subswath->range_pixel_spacing_ =
        std::stod(image_information->GetAttributeString(snapengine::AbstractMetadata::RANGE_PIXEL_SPACING));
    subswath->slr_time_to_first_pixel_ = std::stod(image_information->GetAttributeString("slantRangeTime")) / 2.0;
    subswath->slr_time_to_last_pixel_ = subswath->slr_time_to_first_pixel_ + (double)(subswath->num_of_samples_ - 1) *
                                                                                 subswath->range_pixel_spacing_ /
                                                                                 snapengine::constants::lightSpeed;

    subswath->first_line_time_ =
        GetTime(image_information, snapengine::AbstractMetadata::PRODUCT_FIRST_LINE_UTC_TIME)->GetMjd() *
        snapengine::constants::secondsInDay;

    subswath->last_line_time_ =
        GetTime(image_information, snapengine::AbstractMetadata::PRODUCT_LAST_LINE_UTC_TIME)->GetMjd() *
        snapengine::constants::secondsInDay;

    subswath->radar_frequency_ = std::stod(product_information->GetAttributeString("radarFrequency"));
    subswath->azimuth_steering_rate_ =
        std::stod(product_information->GetAttributeString(snapengine::AbstractMetadata::AZIMUTH_STEERING_RATE));

    subswath->first_valid_pixel_ = 0;
    subswath->last_valid_pixel_ = subswath->num_of_samples_;

    int k = 0;
    if (subswath->num_of_bursts_ > 0) {
        int first_valid_pixel = 0;
        int last_valid_pixel = subswath->num_of_samples_;
        std::vector<std::shared_ptr<snapengine::MetadataElement>> burst_list_elem = burst_list->GetElements();
        for (auto list_elem : burst_list_elem) {

            subswath->burst_first_line_time_.push_back(
                GetTime(list_elem, snapengine::AbstractMetadata::AZIMUTH_TIME)->GetMjd() *
                snapengine::constants::secondsInDay);

            subswath->burst_last_line_time_.push_back(subswath->burst_first_line_time_.at(k) +
                                                      (subswath->lines_per_burst_ - 1) *
                                                          subswath->azimuth_time_interval_);

            std::shared_ptr<snapengine::MetadataElement> first_valid_sample_elem =
                list_elem->GetElement(snapengine::AbstractMetadata::FIRST_VALID_SAMPLE);
            std::shared_ptr<snapengine::MetadataElement> last_valid_sample_elem =
                list_elem->GetElement(snapengine::AbstractMetadata::LAST_VALID_SAMPLE);

            subswath->first_valid_sample_.push_back(GetIntVector(
                first_valid_sample_elem->GetAttribute(snapengine::AbstractMetadata::FIRST_VALID_SAMPLE), " "));
            subswath->last_valid_sample_.push_back(GetIntVector(
                last_valid_sample_elem->GetAttribute(snapengine::AbstractMetadata::LAST_VALID_SAMPLE), " "));

            int first_valid_line_idx = -1;
            int last_valid_line_idx = -1;
            for (size_t line_idx = 0; line_idx < subswath->first_valid_sample_.at(k).size(); line_idx++) {
                if (subswath->first_valid_sample_.at(k).at(line_idx) != -1) {
                    if (subswath->first_valid_sample_.at(k).at(line_idx) > first_valid_pixel) {
                        first_valid_pixel = subswath->first_valid_sample_.at(k).at(line_idx);
                    }

                    if (first_valid_line_idx == -1) {
                        first_valid_line_idx = line_idx;
                        last_valid_line_idx = line_idx;
                    } else {
                        last_valid_line_idx++;
                    }
                }
            }

            for (auto last_pixel : subswath->last_valid_sample_.at(k)) {
                if (last_pixel != -1 && last_pixel < last_valid_pixel) {
                    last_valid_pixel = last_pixel;
                }
            }

            subswath->burst_first_valid_line_time_.push_back(subswath->burst_first_line_time_.at(k) +
                                                             first_valid_line_idx * subswath->azimuth_time_interval_);

            subswath->burst_last_valid_line_time_.push_back(subswath->burst_first_line_time_.at(k) +
                                                            last_valid_line_idx * subswath->azimuth_time_interval_);

            subswath->first_valid_line_.push_back(first_valid_line_idx);
            subswath->last_valid_line_.push_back(last_valid_line_idx);

            k++;
        }
        subswath->first_valid_pixel_ = first_valid_pixel;
        subswath->last_valid_pixel_ = last_valid_pixel;
        subswath->first_valid_line_time_ = subswath->burst_first_valid_line_time_.at(0);
        subswath->last_valid_line_time_ = subswath->burst_last_valid_line_time_.at(subswath->num_of_bursts_ - 1);
    }

    subswath->slr_time_to_first_valid_pixel_ =
        subswath->slr_time_to_first_pixel_ +
        subswath->first_valid_pixel_ * subswath->range_pixel_spacing_ / snapengine::constants::lightSpeed;

    subswath->slr_time_to_last_valid_pixel_ =
        subswath->slr_time_to_first_pixel_ +
        subswath->last_valid_pixel_ * subswath->range_pixel_spacing_ / snapengine::constants::lightSpeed;

    // get geolocation grid points
    std::shared_ptr<snapengine::MetadataElement> geolocation_grid =
        product->GetElement(snapengine::AbstractMetadata::GEOLOCATION_GRID);
    std::shared_ptr<snapengine::MetadataElement> geolocation_grid_point_list =
        geolocation_grid->GetElement(snapengine::AbstractMetadata::GEOLOCATION_GRID_POINT_LIST);

    const int num_of_geo_location_grid_points =
        std::stoi(geolocation_grid_point_list->GetAttributeString(snapengine::AbstractMetadata::COUNT));
    std::vector<std::shared_ptr<snapengine::MetadataElement>> geolocation_grid_point_list_elem =
        geolocation_grid_point_list->GetElements();
    int num_of_geo_points_per_line = 0;

    int line = 0;
    for (auto list_elem : geolocation_grid_point_list_elem) {
        if (num_of_geo_points_per_line == 0) {
            line = std::stoi(list_elem->GetAttributeString(snapengine::AbstractMetadata::LINE));
            num_of_geo_points_per_line++;
        } else if (line == std::stoi(list_elem->GetAttributeString(snapengine::AbstractMetadata::LINE))) {
            num_of_geo_points_per_line++;
        } else {
            break;
        }
    }

    if (num_of_geo_points_per_line == 0) {
        throw std::runtime_error("Number of geopoints in sentinel 1 utils is not allowed to be 0.");
    }

    int num_of_geo_lines = num_of_geo_location_grid_points / num_of_geo_points_per_line;
    bool missing_tie_points = false;
    int first_missing_line_idx = -1;
    if (num_of_geo_lines <= subswath->num_of_bursts_) {
        missing_tie_points = true;
        first_missing_line_idx = num_of_geo_lines;
        num_of_geo_lines = subswath->num_of_bursts_ + 1;
    }

    subswath->num_of_geo_lines_ = num_of_geo_lines;
    subswath->num_of_geo_points_per_line_ = num_of_geo_points_per_line;
    subswath->azimuth_time_ = Allocate2DArray<double>(num_of_geo_lines, num_of_geo_points_per_line);
    subswath->slant_range_time_ = Allocate2DArray<double>(num_of_geo_lines, num_of_geo_points_per_line);
    subswath->latitude_ = Allocate2DArray<double>(num_of_geo_lines, num_of_geo_points_per_line);
    subswath->longitude_ = Allocate2DArray<double>(num_of_geo_lines, num_of_geo_points_per_line);
    subswath->incidence_angle_ = Allocate2DArray<double>(num_of_geo_lines, num_of_geo_points_per_line);

    k = 0;
    for (auto list_elem : geolocation_grid_point_list_elem) {
        int i = k / num_of_geo_points_per_line;
        int j = k - i * num_of_geo_points_per_line;
        subswath->azimuth_time_[i][j] = GetTime(list_elem, snapengine::AbstractMetadata::AZIMUTH_TIME)->GetMjd() *
                                        snapengine::constants::secondsInDay;
        subswath->slant_range_time_[i][j] = std::stod(list_elem->GetAttributeString("slantRangeTime")) / 2.0;
        subswath->latitude_[i][j] = std::stod(list_elem->GetAttributeString(snapengine::AbstractMetadata::LATITUDE));
        subswath->longitude_[i][j] = std::stod(list_elem->GetAttributeString(snapengine::AbstractMetadata::LONGITUDE));
        subswath->incidence_angle_[i][j] =
            std::stod(list_elem->GetAttributeString(snapengine::AbstractMetadata::INCIDENCE_ANGLE));
        k++;
    }

    // compute the missing tie points by extrapolation assuming the missing lines are at the bottom
    if (missing_tie_points && first_missing_line_idx >= 2) {
        for (int line_idx = first_missing_line_idx; line_idx < num_of_geo_lines; line_idx++) {
            double mu = line_idx - first_missing_line_idx + 2.0;
            for (int pixel_idx = 0; pixel_idx < num_of_geo_points_per_line; pixel_idx++) {
                subswath->azimuth_time_[line_idx][pixel_idx] =
                    mu * subswath->azimuth_time_[first_missing_line_idx - 1][pixel_idx] +
                    (1 - mu) * subswath->azimuth_time_[first_missing_line_idx - 2][pixel_idx];

                subswath->slant_range_time_[line_idx][pixel_idx] =
                    mu * subswath->slant_range_time_[first_missing_line_idx - 1][pixel_idx] +
                    (1 - mu) * subswath->slant_range_time_[first_missing_line_idx - 2][pixel_idx];

                subswath->latitude_[line_idx][pixel_idx] =
                    mu * subswath->latitude_[first_missing_line_idx - 1][pixel_idx] +
                    (1 - mu) * subswath->latitude_[first_missing_line_idx - 2][pixel_idx];

                subswath->longitude_[line_idx][pixel_idx] =
                    mu * subswath->longitude_[first_missing_line_idx - 1][pixel_idx] +
                    (1 - mu) * subswath->longitude_[first_missing_line_idx - 2][pixel_idx];

                subswath->incidence_angle_[line_idx][pixel_idx] =
                    mu * subswath->incidence_angle_[first_missing_line_idx - 1][pixel_idx] +
                    (1 - mu) * subswath->incidence_angle_[first_missing_line_idx - 2][pixel_idx];
            }
        }
    }

    int num_ap_records = std::stoi(antenna_pattern_list->GetAttributeString(snapengine::AbstractMetadata::COUNT));

    if (num_ap_records > 0) {
        std::vector<std::shared_ptr<snapengine::MetadataElement>> antenna_pattern_list_elem =
            antenna_pattern_list->GetElements();
        for (auto list_elem : antenna_pattern_list_elem) {
            std::shared_ptr<snapengine::MetadataElement> slant_range_time_elem =
                list_elem->GetElement("slantRangeTime");
            std::shared_ptr<snapengine::MetadataElement> elevation_angle_elem =
                list_elem->GetElement(snapengine::AbstractMetadata::ELEVATION_ANGLE);

            subswath->ap_slant_range_time_.push_back(
                GetDoubleVector(slant_range_time_elem->GetAttribute("slantRangeTime"), " "));
            subswath->ap_elevation_angle_.push_back(GetDoubleVector(
                elevation_angle_elem->GetAttribute(snapengine::AbstractMetadata::ELEVATION_ANGLE), " "));
        }
    }
}

[[nodiscard]] std::vector<double> Sentinel1Utils::GetDoubleVector(std::shared_ptr<snapengine::MetadataAttribute> attribute,
                                                    std::string_view delimiter) const {
    std::vector<double> result;

    if (attribute->GetDataType() == snapengine::ProductData::TYPE_ASCII) {
        std::string data_str = attribute->GetData()->GetElemString();
        size_t pos = 0;
        std::string token;
        while ((pos = data_str.find(delimiter)) != std::string::npos) {
            token = data_str.substr(0, pos);
            result.push_back(std::stod(token));
            data_str.erase(0, pos + delimiter.length());
        }
        result.push_back(std::stod(data_str));
    }

    return result;
}

[[nodiscard]] std::vector<int> Sentinel1Utils::GetIntVector(std::shared_ptr<snapengine::MetadataAttribute> attribute,
                                              std::string_view delimiter) const {
    std::vector<int> result;

    if (attribute->GetDataType() == snapengine::ProductData::TYPE_ASCII) {
        std::string data_str = attribute->GetData()->GetElemString();
        size_t pos = 0;
        std::string token;
        while ((pos = data_str.find(delimiter)) != std::string::npos) {
            token = data_str.substr(0, pos);
            result.push_back(std::stoi(token));
            data_str.erase(0, pos + delimiter.length());
        }
        result.push_back(std::stoi(data_str));
    }

    return result;
}

double* Sentinel1Utils::ComputeDerampDemodPhase(int subswath_index, int s_burst_index, Rectangle rectangle) {
    const int x0 = rectangle.x;
    const int y0 = rectangle.y;
    const int w = rectangle.width;
    const int h = rectangle.height;

    const int x_max = x0 + w;
    const int y_max = y0 + h;
    const int s = subswath_index - 1;
    const int first_line_in_burst = s_burst_index * subswath_.at(s)->lines_per_burst_;

    double* result = new double[h * w * sizeof(double)];
    int yy, xx, x, y;
    double ta, kt, deramp, demod;

    for (y = y0; y < y_max; y++) {
        yy = y - y0;
        ta = (y - first_line_in_burst) * subswath_.at(s)->azimuth_time_interval_;
        for (x = x0; x < x_max; x++) {
            xx = x - x0;
            kt = subswath_.at(s)->doppler_rate_[s_burst_index][x];
            deramp =
                -alus::snapengine::constants::PI * kt * pow(ta - subswath_.at(s)->reference_time_[s_burst_index][x], 2);
            demod = -alus::snapengine::constants::TWO_PI * subswath_.at(s)->doppler_centroid_[s_burst_index][x] * ta;
            result[yy * w + xx] = deramp + demod;
        }
    }

    return result;
}

void Sentinel1Utils::GetProductOrbit() {
    std::shared_ptr<snapengine::MetadataElement> abstract_metadata =
        metadata_reader_->Read(alus::snapengine::AbstractMetadata::ABSTRACT_METADATA_ROOT);
    std::vector<snapengine::OrbitStateVector> original_vectors =
        snapengine::AbstractMetadata::GetOrbitStateVectors(abstract_metadata);

    orbit_ = std::make_unique<alus::s1tbx::OrbitStateVectors>(original_vectors);

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

    wave_length = alus::snapengine::constants::lightSpeed / subswath_.at(0)->radar_frequency_;
    for (int s = 0; s < num_of_sub_swath_; s++) {
        az_time = (subswath_.at(s)->first_line_time_ + subswath_.at(s)->last_line_time_) / 2.0;
        subswath_.at(s)->doppler_rate_ =
            Allocate2DArray<double>(subswath_.at(s)->num_of_bursts_, subswath_.at(s)->samples_per_burst_);
        v = GetVelocity(az_time / alus::snapengine::constants::secondsInDay);  // DLR: 7594.0232
        steering_rate = subswath_.at(s)->azimuth_steering_rate_ * alus::snapengine::constants::DTOR;
        krot = 2 * v * steering_rate / wave_length;  // doppler rate by antenna steering

        for (int b = 0; b < subswath_.at(s)->num_of_bursts_; b++) {
            for (int x = 0; x < subswath_.at(s)->samples_per_burst_; x++) {
                subswath_.at(s)->doppler_rate_[b][x] = subswath_.at(s)->range_depend_doppler_rate_[b][x] * krot /
                                                      (subswath_.at(s)->range_depend_doppler_rate_[b][x] - krot);
            }
        }
    }
}

double Sentinel1Utils::GetSlantRangeTime(int x, int subswath_index) {
    return subswath_.at(subswath_index - 1)->slr_time_to_first_pixel_ +
           x * subswath_.at(subswath_index - 1)->range_pixel_spacing_ / alus::snapengine::constants::lightSpeed;
}

std::vector<DCPolynomial> Sentinel1Utils::GetDCEstimateList(std::string subswath_name) {
    std::cout << "We are supposed to select subswath " << subswath_name << " for dc list, but we will use default."
              << std::endl;

    std::shared_ptr<snapengine::MetadataElement> product =
        metadata_reader_->Read("product");
    std::shared_ptr<snapengine::MetadataElement> image_annotation = product->GetElement("imageAnnotation");
    std::shared_ptr<snapengine::MetadataElement> processing_information =
        image_annotation->GetElement("processingInformation");
    std::string dc_method = processing_information->GetAttributeString("dcMethod");
    std::shared_ptr<snapengine::MetadataElement> doppler_centroid = product->GetElement("dopplerCentroid");
    std::shared_ptr<snapengine::MetadataElement> dc_estimate_list = doppler_centroid->GetElement("dcEstimateList");
    int count = std::stoi(dc_estimate_list->GetAttributeString("count"));

    std::vector<DCPolynomial> results;

    if (count > 0) {
        std::vector<std::shared_ptr<snapengine::MetadataElement>> dc_estimate_list_elem =
            dc_estimate_list->GetElements();
        for (size_t i = 0; i < dc_estimate_list_elem.size(); i++) {
            std::shared_ptr<snapengine::MetadataElement> list_elem = dc_estimate_list_elem.at(i);
            results.emplace_back();

            results.at(i).time = GetTime(list_elem, "azimuthTime")->GetMjd() * snapengine::constants::secondsInDay;
            results.at(i).t0 = list_elem->GetAttributeDouble("t0");

            if (dc_method.find("Data Analysis") != std::string::npos) {
                std::shared_ptr<snapengine::MetadataElement> data_dc_polynomial_elem =
                    list_elem->GetElement("dataDcPolynomial");
                results.at(i).data_dc_polynomial =
                    GetDoubleVector(data_dc_polynomial_elem->GetAttribute("dataDcPolynomial"), " ");
            } else {
                std::shared_ptr<snapengine::MetadataElement> geometry_dc_polynomial_elem =
                    list_elem->GetElement("geometryDcPolynomial");
                results.at(i).data_dc_polynomial =
                    GetDoubleVector(geometry_dc_polynomial_elem->GetAttribute("geometryDcPolynomial"), " ");
            }
        }
    }

    return results;
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
    if ((int)dc_estimate_list.size() >= subswath_.at(subswath_index - 1)->num_of_bursts_) {
        std::cout << "used the fast lane" << '\n';
        return dc_estimate_list;
    }

    std::vector<DCPolynomial> dcBurstList(subswath_.at(subswath_index - 1)->num_of_bursts_);
    for (int b = 0; b < subswath_.at(subswath_index - 1)->num_of_bursts_; b++) {
        if (b < (int)dc_estimate_list.size()) {
            dcBurstList[b] = dc_estimate_list[b];
            std::cout << "using less list" << '\n';
        } else {
            std::cout << "using more list" << '\n';
            center_time = 0.5 * (subswath_.at(subswath_index - 1)->burst_first_line_time_[b] +
                                 subswath_.at(subswath_index - 1)->burst_last_line_time_[b]);

            dcBurstList[b] = ComputeDC(center_time, dc_estimate_list);
        }
    }

    return dcBurstList;
}

void Sentinel1Utils::ComputeDopplerCentroid() {
    double slrt, dt, dc_value;
    for (int s = 0; s < num_of_sub_swath_; s++) {
        std::vector<DCPolynomial> dc_estimate_list = GetDCEstimateList(subswath_.at(s)->subswath_name_);
        std::vector<DCPolynomial> dc_burst_list = ComputeDCForBurstCenters(dc_estimate_list, s + 1);
        subswath_.at(s)->doppler_centroid_ =
            Allocate2DArray<double>(subswath_.at(s)->num_of_bursts_, subswath_.at(s)->samples_per_burst_);
        for (int b = 0; b < subswath_.at(s)->num_of_bursts_; b++) {
            for (int x = 0; x < subswath_.at(s)->samples_per_burst_; x++) {
                slrt = GetSlantRangeTime(x, s + 1) * 2;
                dt = slrt - dc_burst_list[b].t0;

                dc_value = 0.0;
                for (unsigned int i = 0; i < dc_burst_list[b].data_dc_polynomial.size(); i++) {
                    dc_value += dc_burst_list[b].data_dc_polynomial[i] * pow(dt, i);
                }
                subswath_.at(s)->doppler_centroid_[b][x] = dc_value;
            }
        }
    }

    is_doppler_centroid_available_ = true;
}

std::vector<AzimuthFmRate> Sentinel1Utils::GetAzimuthFmRateList(std::string subswath_name) {
    std::cout << "We are supposed to read subswath " << subswath_name
              << " Azimuth fm rate list, but we will be using default subswath." << std::endl;

    std::shared_ptr<snapengine::MetadataElement> product =
        metadata_reader_->Read("product");
    std::shared_ptr<snapengine::MetadataElement> general_annotation = product->GetElement("generalAnnotation");
    std::shared_ptr<snapengine::MetadataElement> azimuth_fm_rate_list =
        general_annotation->GetElement("azimuthFmRateList");
    int count = std::stoi(azimuth_fm_rate_list->GetAttributeString("count"));
    std::vector<AzimuthFmRate> az_fm_rate_list;

    if (count > 0) {
        std::vector<std::shared_ptr<snapengine::MetadataElement>> az_fm_rate_list_elem =
            azimuth_fm_rate_list->GetElements();
        for (size_t i = 0; i < az_fm_rate_list_elem.size(); i++) {
            std::shared_ptr<snapengine::MetadataElement> list_elem = az_fm_rate_list_elem.at(i);
            az_fm_rate_list.emplace_back();

            az_fm_rate_list.at(i).time = GetTime(list_elem, "azimuthTime")->GetMjd() * snapengine::constants::secondsInDay;
            az_fm_rate_list.at(i).t0 = std::stod(list_elem->GetAttributeString("t0"));

            std::shared_ptr<snapengine::MetadataElement> azimuth_fm_rate_polynomial_elem =
                list_elem->GetElement("azimuthFmRatePolynomial");
            if (azimuth_fm_rate_polynomial_elem) {
                std::vector<double> coeffs =
                    GetDoubleVector(azimuth_fm_rate_polynomial_elem->GetAttribute("azimuthFmRatePolynomial"), " ");
                az_fm_rate_list.at(i).c0 = coeffs.at(0);
                az_fm_rate_list.at(i).c1 = coeffs.at(1);
                az_fm_rate_list.at(i).c2 = coeffs.at(2);
            } else {
                az_fm_rate_list.at(i).c0 = std::stod(list_elem->GetAttributeString("c0"));
                az_fm_rate_list.at(i).c1 = std::stod(list_elem->GetAttributeString("c1"));
                az_fm_rate_list.at(i).c2 = std::stod(list_elem->GetAttributeString("c2"));
            }
        }
    }
    return az_fm_rate_list;
}

void Sentinel1Utils::ComputeRangeDependentDopplerRate() {
    double slrt, dt;

    for (int s = 0; s < num_of_sub_swath_; s++) {
        std::vector<AzimuthFmRate> az_fm_rate_list = GetAzimuthFmRateList(subswath_.at(s)->subswath_name_);
        subswath_.at(s)->range_depend_doppler_rate_ =
            Allocate2DArray<double>(subswath_.at(s)->num_of_bursts_, subswath_.at(s)->samples_per_burst_);
        for (int b = 0; b < subswath_.at(s)->num_of_bursts_; b++) {
            for (int x = 0; x < subswath_.at(s)->samples_per_burst_; x++) {
                slrt = GetSlantRangeTime(x, s + 1) * 2;  // 1-way to 2-way
                dt = slrt - az_fm_rate_list[b].t0;
                subswath_.at(s)->range_depend_doppler_rate_[b][x] =
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
        subswath_.at(s)->reference_time_ =
            Allocate2DArray<double>(subswath_.at(s)->num_of_bursts_, subswath_.at(s)->samples_per_burst_);
        tmp1 = subswath_.at(s)->lines_per_burst_ * subswath_.at(s)->azimuth_time_interval_ / 2.0;

        for (int b = 0; b < subswath_.at(s)->num_of_bursts_; b++) {
            tmp2 = tmp1 + subswath_.at(s)->doppler_centroid_[b][subswath_.at(s)->first_valid_pixel_] /
                              subswath_.at(s)->range_depend_doppler_rate_[b][subswath_.at(s)->first_valid_pixel_];

            for (int x = 0; x < subswath_.at(s)->samples_per_burst_; x++) {
                subswath_.at(s)->reference_time_[b][x] =
                    tmp2 - subswath_.at(s)->doppler_centroid_[b][x] / subswath_.at(s)->range_depend_doppler_rate_[b][x];
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
    auto start = element->GetAttributeString(tag, snapengine::AbstractMetadata::NO_METADATA_STRING);

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

    std::map<std::string_view, bool> selected_bands{{snapengine::AbstractMetadata::SIGMA_NOUGHT, output_sigma_band},
                                                    {snapengine::AbstractMetadata::BETA_NOUGHT, output_beta_band},
                                                    {snapengine::AbstractMetadata::GAMMA, output_gamma_band},
                                                    {snapengine::AbstractMetadata::DN, output_dn_band}};

    const auto calibration_vector_elements = calibration_vector_list_element->GetElements();

    std::vector<CalibrationVector> calibration_vectors;
    calibration_vectors.reserve(calibration_vector_elements.size());

    for (auto&& calibration_vector_element : calibration_vector_elements) {
        std::map<std::string_view, std::vector<float>> unit_data{{snapengine::AbstractMetadata::SIGMA_NOUGHT, {}},
                                                                 {snapengine::AbstractMetadata::BETA_NOUGHT, {}},
                                                                 {snapengine::AbstractMetadata::GAMMA, {}},
                                                                 {snapengine::AbstractMetadata::DN, {}}};

        const auto time = GetTime(calibration_vector_element, snapengine::AbstractMetadata::AZIMUTH_TIME);
        const auto line = calibration_vector_element->GetAttributeInt(snapengine::AbstractMetadata::LINE);

        const auto pixel_element = get_element(calibration_vector_element, snapengine::AbstractMetadata::PIXEL);
        const auto pixel_string = pixel_element->GetAttributeString(snapengine::AbstractMetadata::PIXEL);
        const auto count = pixel_element->GetAttributeInt(snapengine::AbstractMetadata::COUNT);

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

        calibration_vectors.push_back(
            {time->GetMjd(), line, pixels, unit_data.at(snapengine::AbstractMetadata::SIGMA_NOUGHT),
             unit_data.at(snapengine::AbstractMetadata::BETA_NOUGHT), unit_data.at(snapengine::AbstractMetadata::GAMMA),
             unit_data.at(snapengine::AbstractMetadata::DN), static_cast<size_t>(count)});
    }

    return calibration_vectors;
}
}  // namespace s1tbx
}  // namespace alus