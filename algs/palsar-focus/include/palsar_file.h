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

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "palsar_record.h"

namespace alus::palsar {

// abstraction for LED, VOL and TRL files, these do not contain raw signal data
class MetadataFile {
public:
    MetadataFile() = default;

    void Open(const std::string& path);

    [[nodiscard]] bool IsOpen() const { return fd_ > 0; }

    [[nodiscard]] const char* Data() const { return data_; }

    [[nodiscard]] size_t Size() const { return size_; }

    ~MetadataFile();

    MetadataFile(const MetadataFile&) = delete;

    MetadataFile& operator=(const MetadataFile&) = delete;

private:
    char* data_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;
    std::string path_;
};

// IMG file abstraction, contains the raw 5 bit complex samples
class ImgFile {
public:
    ImgFile() = default;

    void Open(const std::string& path);

    [[nodiscard]] const char* Data() const { return buffer_; }

    [[nodiscard]] bool IsOpen() const { return fd_ > 0; }

    [[nodiscard]] size_t Size() const { return file_size_; }

    // 720 is SAR Data File Descriptor size before Signal Data Records
    [[nodiscard]] const char* SignalData() { return buffer_ + 720; }
    [[nodiscard]] size_t SignalSize() const { return file_size_ - 720; }
    [[nodiscard]] const char* SignalDataRow(int row_idx, int record_size) { return buffer_ + 720 + (row_idx * record_size); }

    void LoadHeader();
    void LoadFile();

    ~ImgFile();

    ImgFile(const ImgFile&) = delete;
    ImgFile& operator=(const ImgFile&) = delete;

private:
    char* buffer_ = nullptr;
    size_t file_size_ = 0;
    size_t buf_size_ = 0;
    int fd_ = -1;
    std::string path_;
};

/**
 *  PALSAR LVL0 folder contains 4 relevant files:
 *  VOL - VOLUME DIRECTORY FILE
 *  LED - SAR LEADER FILE
 *  IMG - SAR DATA FILE (1x, 2x or 4x)
 *  TRL - SAR TRAILER FILE
 *
 *  Each file is subdivided into records, from which relevant SAR metadata can be extracted
 */
class FileSystem {
public:
    FileSystem() = default;

    void InitFromPath(const char* path, const char* polarization);

    // Record size and offsets taken from: ALOS/PALSAR Level 1 Product Format Description Vol.1: Level 1.0
    [[nodiscard]] Record GetVolumeDescriptorRecord() const {
        const size_t record_offset = 0;
        const size_t record_size = 360;
        const char* name = "Volume Descriptor";
        VerifyRecordSize(volume_directory_file_.Size(), record_offset, record_size, name);
        return {volume_directory_file_.Data(), record_offset, record_size, name};
    }
    [[nodiscard]] Record GetDataSetSummaryRecord() const {
        const size_t record_offset = 720;
        const size_t record_size = 4096;
        const char* name = "DataSet Summary";
        VerifyRecordSize(leader_file_.Size(), record_offset, record_size, name);
        return {leader_file_.Data(), record_offset, record_size, name};
    }
    [[nodiscard]] Record GetFileDescriptorRecord() const {
        const size_t record_offset = 0;
        const size_t record_size = 720;
        const char* name = "File descriptor";
        VerifyRecordSize(image_file_.Size(), record_offset, record_size, name);
        return {image_file_.Data(), record_offset, record_size, name};
    }
    [[nodiscard]] Record GetPlatformPositionDataRecord() const {
        const size_t record_offset = 720 + 4096;
        const size_t record_size = 4680;
        const char* name = "Platform Position Data";
        VerifyRecordSize(leader_file_.Size(), record_offset, record_size, name);
        return {leader_file_.Data(), record_offset, record_size, name};
    }
    [[nodiscard]] Record GetSignalDataRecord(size_t index, size_t record_size) const {
        const size_t record_offset = 720 + index * record_size;
        const size_t header_record_size = 412;
        const char* name = "Signal Data Record";
        VerifyRecordSize(image_file_.Size(), record_offset, header_record_size, name);
        return {image_file_.Data(), record_offset, header_record_size, name};
    }

    ImgFile& GetImageFile() { return image_file_; }

    const std::string& GetSceneId() { return scene_id_; }

private:
    static void VerifyRecordSize(size_t file_size, size_t record_offset, size_t record_size, const char* record_name) {
        if (record_offset + record_size >= file_size) {
            std::string err_msg = std::string(record_name) + " record offset(" + std::to_string(record_offset) +
                                  ") + size(" + std::to_string(record_size) +
                                  ") not enough file size = " + std::to_string(file_size);
            throw std::runtime_error(err_msg);
        }
    }

    MetadataFile leader_file_;
    MetadataFile volume_directory_file_;
    MetadataFile trailer_file_;
    ImgFile image_file_;
    std::string scene_id_;
    std::string product_id_;
};
}  // namespace alus::palsar
