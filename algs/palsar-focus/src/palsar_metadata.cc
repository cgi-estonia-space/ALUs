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

#include "palsar_metadata.h"

#include <cmath>

#include <boost/date_time.hpp>

#include "alus_log.h"
#include "math_utils.h"

namespace {
constexpr int SIGNAL_OFFSET = 412;
}

// NB! All offsets taken from Palsar file format documentation
// Document name - ALOS Product Format Description (PALSAR Level 1.0)
// All offsets in the documentation use 1 based indexing
// The file record reader implementation subtracts 1 to accommodate C's 0 based indexing

namespace alus::palsar {
void ReadMetadata(const FileSystem& files, SARMetadata& metadata) {
    {
        Record data_set_summary = files.GetDataSetSummaryRecord();

        std::string processing_level = data_set_summary.ReadAn(1095, 16);
        processing_level.erase(std::remove(processing_level.begin(), processing_level.end(), ' '),
                               processing_level.end());
        if (processing_level != "1.0") {
            std::string err_msg = "Expected file processing level to be 1.0, actually is = " + processing_level;
            throw std::runtime_error(err_msg);
        }

        metadata.wavelength = data_set_summary.ReadF16(501);
        metadata.carrier_frequency = data_set_summary.ReadF16(493) * 1e9;

        auto& chirp = metadata.chirp;
        chirp.coefficient[0] = data_set_summary.ReadF16(535);  // Hz
        chirp.coefficient[1] = data_set_summary.ReadF16(551);  // Hz/s
        chirp.coefficient[2] = data_set_summary.ReadF16(567);  // quad term
        chirp.coefficient[3] = data_set_summary.ReadF16(583);  // quad term
        chirp.coefficient[4] = data_set_summary.ReadF16(599);  // quad term

        chirp.range_sampling_rate = data_set_summary.ReadF16(711) * 1e6;  // MHz -> Hz;
        chirp.pulse_duration = data_set_summary.ReadF16(743) * 1e-6;      // useconds -> s
        chirp.n_samples = static_cast<int>(std::round(chirp.pulse_duration * chirp.range_sampling_rate));

        chirp.pulse_bandwidth = chirp.pulse_duration * fabs(chirp.coefficient[1]);

        metadata.range_spacing = SOL / (2 * chirp.range_sampling_rate);

        metadata.pulse_repetition_frequency = data_set_summary.ReadF16(935) * 1e-3;  // mHz -> Hz;
        metadata.azimuth_bandwidth_fraction = 0.8;

        if (data_set_summary.ReadI4(389) == 4) {
            // in polarimetry mode PRF needs to be divided by 2
            metadata.pulse_repetition_frequency /= 2.0;
        }
        metadata.center_lat = data_set_summary.ReadF16(117);
        metadata.center_lon = data_set_summary.ReadF16(133);

        auto center_time_str = data_set_summary.ReadAn(69, 32);

        // format YYYYMMDDhhmmssttt
        int year = std::stoi(center_time_str.substr(0, 4));
        int month = std::stoi(center_time_str.substr(4, 2));
        int day = std::stoi(center_time_str.substr(6, 2));
        int hour = std::stoi(center_time_str.substr(8, 2));
        int minute = std::stoi(center_time_str.substr(10, 2));
        int second = std::stoi(center_time_str.substr(12, 2));
        int millisecond = std::stoi(center_time_str.substr(14, center_time_str.size() - 14));

        boost::posix_time::ptime center_time(boost::gregorian::date(year, month, day));

        center_time += boost::posix_time::hours(hour);
        center_time += boost::posix_time::minutes(minute);
        center_time += boost::posix_time::seconds(second);
        center_time += boost::posix_time::milliseconds(millisecond);

        metadata.center_time = center_time;
    }
    {
        Record file_descriptor_record = files.GetFileDescriptorRecord();
        metadata.img.azimuth_size = file_descriptor_record.ReadI6(181);
        metadata.img.data_line_offset = SIGNAL_OFFSET;
        metadata.img.record_length = file_descriptor_record.ReadI6(187);
        metadata.img.range_size = file_descriptor_record.ReadI8(281) / 2;

        if (metadata.img.record_length != metadata.img.range_size * 2 + SIGNAL_OFFSET) {
            // TODO(priit)
            throw std::runtime_error("IMG files with differing signal lengths not yet supported");
        }
    }

    {
        auto& first_pos = metadata.first_position;
        Record platform_position_data = files.GetPlatformPositionDataRecord();
        first_pos.x_pos = platform_position_data.ReadF16(45);
        first_pos.y_pos = platform_position_data.ReadF16(45 + 1 * 16);
        first_pos.z_pos = platform_position_data.ReadF16(45 + 2 * 16);
        first_pos.x_vel = platform_position_data.ReadF16(45 + 3 * 16);
        first_pos.y_vel = platform_position_data.ReadF16(45 + 4 * 16);
        first_pos.z_vel = platform_position_data.ReadF16(45 + 5 * 16);

        const int n_points = platform_position_data.ReadI4(141);

        const int year = platform_position_data.ReadI4(145);
        const int month = platform_position_data.ReadI4(149);
        const int day = platform_position_data.ReadI4(153);
        const double second_of_day = platform_position_data.ReadD22(161);
        metadata.orbit_interval = platform_position_data.ReadD22(183);

        metadata.first_orbit_time = boost::posix_time::ptime(
            boost::gregorian::date(year, month, day), boost::posix_time::seconds(static_cast<int>(second_of_day)));

        for (int i = 0; i < n_points; i++) {
            OrbitInfo osv = {};
            osv.time_point = i * metadata.orbit_interval;
            osv.x_pos = platform_position_data.ReadD22(387 + i * 132);
            osv.y_pos = platform_position_data.ReadD22(387 + 22 + i * 132);
            osv.z_pos = platform_position_data.ReadD22(387 + 44 + i * 132);
            osv.x_vel = platform_position_data.ReadD22(453 + i * 132);
            osv.y_vel = platform_position_data.ReadD22(453 + 22 + i * 132);
            osv.z_vel = platform_position_data.ReadD22(453 + 44 + i * 132);

            metadata.orbit_state_vectors.push_back(osv);
        }

        double acc_vel = 0.0;
        for (const auto& osv : metadata.orbit_state_vectors) {
            acc_vel += CalcVelocity(osv);
        }
        metadata.platform_velocity = acc_vel / metadata.orbit_state_vectors.size();

        //TODO(priit) - better azimuth spacing estimate?
        metadata.azimuth_spacing = (0.88 * metadata.platform_velocity) / metadata.pulse_repetition_frequency;
    }
    {
        Record signal_data_record = files.GetSignalDataRecord(0, metadata.img.record_length);
        metadata.slant_range_first_sample = signal_data_record.ReadB4(117);
    }
}

void PrintMetadata(const SARMetadata& metadata) {
    LOGI << "PRF(Hz) = " << metadata.pulse_repetition_frequency;
    LOGI << "Frequency(GHz) = " << metadata.carrier_frequency / 1e9;
    LOGI << "Wavelength(m) = " << metadata.wavelength;
    LOGI << "Range sampling rate(MHz) = " << metadata.chirp.range_sampling_rate / 1e6;
    LOGI << "Chirp samples = " << metadata.chirp.n_samples;
    LOGI << "Chirp duration(us) = " << metadata.chirp.pulse_duration * 1e6;
    LOGI << "Chirp bandwidth(MHz) = " << metadata.chirp.pulse_bandwidth / 1e6;
    LOGI << "Range x Azimuth = " << metadata.img.range_size << " " << metadata.img.azimuth_size;
    if (metadata.img.record_length != SIGNAL_OFFSET + metadata.img.range_size * 2) {
        // Appendix A-1-6 No19 special case, where some datasets have right padding on signal data lines
        LOGI << "Signal line extra padding: "
             << metadata.img.record_length - SIGNAL_OFFSET - metadata.img.range_size * 2;
    }
    LOGI << "Range spacing(m) = " << metadata.range_spacing;
    LOGI << "Platform velocity(m/s) = " << metadata.platform_velocity;
    LOGI << "First orbit time = " << metadata.first_orbit_time;

    char latlon[60] = {};
    snprintf(latlon, 60, "Center Lat/Lon = (%.10f %.10f)", metadata.center_lat, metadata.center_lon);
    LOGI << latlon;

    LOGI << "Center time = " << metadata.center_time;
    LOGI << "Slant range to first sample(m) = " << metadata.slant_range_first_sample;
}
}  // namespace alus::palsar