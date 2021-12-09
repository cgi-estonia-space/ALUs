/**
 * This file is a filtered duplicate of a SNAP's
 * com.bc.ceres.core.VirtualDir.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
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
#include "ceres-core/core/zip.h"

#include <algorithm>
#include <stdexcept>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/copy.hpp>

namespace alus::ceres {

Zip::Zip(const boost::filesystem::path& file)
    : zip_file_(file),
      temp_zip_file_dir_{IVirtualDir::CreateUniqueTempDir()},
      unzipper_(zip_file_.string(), ""),
      entries_{unzipper_.entries()} {
}

std::vector<std::string> Zip::List(std::string_view path) {
    std::string path_str(path);
    if ("." == path_str || path_str.empty()) {
        path_str = "";
    } else if (!boost::algorithm::ends_with(path_str, "/")) {
        path_str += "/";
    }
    bool dir_seen = false;
    //    TreeSet<String> nameSet = new TreeSet<>();
    std::vector<std::string> name_set;
    //    Enumeration < ? extends ZipEntry > enumeration = zip_file_.filename().string().entries();
    for (auto const& entry : entries_) {
        std::string name = entry.name;
        if (boost::algorithm::starts_with(name, path_str)) {
            int i1 = path_str.length();
            int i2 = -1;
            if (name.find('/', i1) != std::string::npos) {
                i2 = name.find('/', i1);
            }

            std::string entry_name;
            if (i2 == -1) {
                entry_name = name.substr(i1);
            } else {
                entry_name = name.substr(i1, (i2 - i1));
            }
            if (!entry_name.empty() && !(std::find(name_set.begin(), name_set.end(), entry_name) != name_set.end())) {
                name_set.emplace_back(entry_name);
            }
            dir_seen = true;
        }
    }
    if (!dir_seen) {
        //        throw new FileNotFoundException(getBasePath() + "!" + path);
        throw std::runtime_error("File not found: " + std::string(path_str));
    }
    return name_set;
}

boost::filesystem::path Zip::GetFile(std::string_view path) {
    zipper::ZipEntry zip_entry = GetEntry(path);

    //    boost::filesystem::path temp_file(boost::filesystem::canonical(temp_zip_file_dir_).string()); //+
    /* boost::filesystem::path::preferred_separator + zip_entry.name);*/
    boost::filesystem::path unzipped_file_target_path = temp_zip_file_dir_ / zip_entry.name;
    if (boost::filesystem::exists(unzipped_file_target_path)) {
        return unzipped_file_target_path;
    }
    //    ok this needs to check if zip reader sees this as directory or file
    //    if (boost::algorithm::ends_with(zip_entry.name, "/")) {
    //        boost::filesystem::create_directories(temp_file);
    //    } else {
    Unzip(zip_entry, temp_zip_file_dir_);
    //    }

    return unzipped_file_target_path;
}

bool Zip::Exists(std::string_view path) {
    try {
        zipper::ZipEntry zip_entry = GetEntry(path);
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

void Zip::GetInputStream(std::string_view path, std::fstream& stream) { return GetInputStream(GetEntry(path), stream); }

zipper::ZipEntry Zip::GetEntry(std::string_view path) {
    //    todo: remove comment if this has been verified
    //        todo: make sure it gets result passed even if this gets closed
    auto itr = std::find_if(entries_.begin(), entries_.end(), [=](const zipper::ZipEntry& e) { return e.name == path; });
    if (itr != entries_.end()) {
        return *itr;
    }
    throw std::runtime_error("File not found from provided archive: " + zip_file_.filename().string() +
                             std::string("!") + std::string(path));
}

void Zip::Unzip(const zipper::ZipEntry& zip_entry, const boost::filesystem::path& temp_file) {
    unzipper_.extractEntry(zip_entry.name, temp_file.string());
}

void Zip::Close() { Cleanup(); }

void Zip::Cleanup() {
    unzipper_.close();
    zip_file_.clear();
    if (boost::filesystem::is_directory(temp_zip_file_dir_)) {
        DeleteFileTree(temp_zip_file_dir_);
    }
}

void Zip::GetInputStream(const zipper::ZipEntry& zip_entry, std::fstream& stream) {
    std::vector<unsigned char> unzipped_entry;
    //    todo: this GetEntry can be shortcutted if we avoid converting into zip_entry but use name directly and throw
    //    exception right here
    //    todo: ignoring possible memory limits atm..
    //    unzipper.extractEntryToMemory(zip_entry.name, unzipped_entry);
    //    unzipper.extractEntryToMemory(zip_entry.name, unzipped_entry);
    // todo: try to make it work without extracting to temporary location
    // todo: check if files are removed from temporary
    auto temp = boost::filesystem::temp_directory_path();
    unzipper_.extractEntry(zip_entry.name, temp.string());
    //    std::ofstream ostream;
    //    unzipper.extractEntryToStream(zip_entry.name, ostream);
    stream.open(boost::filesystem::canonical(zip_entry.name, temp).string(), std::ifstream::in);
    //    boost::iostreams::copy(ostream,stream);
    //        stream.read(reinterpret_cast<char*>(unzipped_entry.data()), unzipped_entry.size());
    //    stream.rdbuf()->pubsetbuf(reinterpret_cast<char*>(unzipped_entry.data()), unzipped_entry.size());

    //    zipper::Unzipper entry_unzipper(zip_file_.filename().string());
    //    entry_unzipper.extractEntryToStream(zip_entry.name, stream);
}

}  // namespace ceres
