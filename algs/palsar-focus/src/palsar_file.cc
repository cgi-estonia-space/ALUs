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

#include "palsar_file.h"

#include <algorithm>
#include <filesystem>
#include <iostream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <boost/algorithm/string.hpp>

#include "alus_log.h"

namespace alus::palsar {
void MetadataFile::Open(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ > 0) {
        struct stat st = {};
        fstat(fd_, &st);
        size_ = st.st_size;
        data_ = static_cast<char*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        path_ = path;
    } else {
        throw std::runtime_error("Failed to open file " + path);
    }
}

MetadataFile::~MetadataFile() {
    munmap(data_, size_);
    close(fd_);
}

void ImgFile::Open(const std::string& path) {
    int oflag = O_RDONLY;
    // oflag |= O_DIRECT;  // TODO(priit) disable this or not?
    fd_ = open(path.c_str(), oflag);
    if (fd_ > 0) {
        struct stat st = {};
        fstat(fd_, &st);
        file_size_ = st.st_size;
        path_ = path;
    } else {
        throw std::runtime_error("Failed to open file " + path);
    }
}

void ImgFile::LoadHeader() {
    constexpr ssize_t ALIGN = 512;  // O_DIRECT alignment
    buf_size_ = ALIGN * 2;
    buffer_ = static_cast<char*>(aligned_alloc(ALIGN, ALIGN * 2));
    ssize_t read_cnt = read(fd_, buffer_, ALIGN * 2);
    if (read_cnt != ALIGN * 2) {
        throw std::runtime_error("Failed to file header from " + path_);
    }
}

void ImgFile::LoadFile() {
    constexpr size_t ALIGN = 512;  // O_DIRECT alignment
    buf_size_ = file_size_ + ALIGN - (file_size_ % ALIGN);
    buffer_ = static_cast<char*>(aligned_alloc(ALIGN, buf_size_));
    ssize_t read_cnt = read(fd_, buffer_, buf_size_);
    if (read_cnt != static_cast<ssize_t>(file_size_)) {
        throw std::runtime_error("Failed to read full file from " + path_);
    }
}

ImgFile::~ImgFile() {
    free(buffer_);
    close(fd_);
}

void FileSystem::InitFromPath(const char* path, const char* polarization) {
    std::filesystem::path p(path);
    for (const auto& entry : std::filesystem::directory_iterator{p}) {
        auto path_str = entry.path().string();
        auto filename = entry.path().filename().string();
        std::vector<std::string> tokens;
        boost::algorithm::split(tokens, filename, boost::is_any_of("-"));
        if (tokens.size() < 3U) {
            continue;
        }

        const auto& file_type = tokens.at(0);

        if (!volume_directory_file_.IsOpen() && file_type == "VOL") {
            volume_directory_file_.Open(path_str);
        } else if (!trailer_file_.IsOpen() && file_type == "TRL") {
            trailer_file_.Open(path_str);
        } else if (!leader_file_.IsOpen() && file_type == "LED") {
            leader_file_.Open(path_str);
            scene_id_ = tokens.at(1);
            product_id_ = tokens.at(2);
            LOGI << "Scene ID = " << scene_id_;
        } else if (!image_file_.IsOpen() && file_type == "IMG" && tokens.size() == 4U) {
            const auto& pol = tokens.at(1);
            if (pol == polarization) {
                image_file_.Open(path_str);
            }
        }
    }

    std::string err_msg;
    if (!volume_directory_file_.IsOpen()) {
        err_msg += " VOL ";
    }
    if (!trailer_file_.IsOpen()) {
        err_msg += " TRL ";
    }
    if (!leader_file_.IsOpen()) {
        err_msg += " LED ";
    }
    if (!image_file_.IsOpen()) {
        err_msg += " IMG ";
    }

    if (!err_msg.empty()) {
        throw std::invalid_argument("Missing the following files: " + err_msg);
    }
}
}  // namespace alus::palsar