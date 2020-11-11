#include "ceres-core/zip.h"

#include <algorithm>
#include <stdexcept>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/copy.hpp>

namespace alus {
namespace ceres {

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
    zipper::Unzipper unzipper(zip_file_.string());
    std::vector<zipper::ZipEntry> entries = unzipper.entries();
    unzipper.close();
    for (auto const& entry : entries) {
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

    if (temp_zip_file_dir_.empty()) {
        temp_zip_file_dir_ = IVirtualDir::CreateUniqueTempDir();
    }

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
    zipper::Unzipper unzipper(zip_file_.string());
    //    todo: remove comment if this has been verified
    std::vector<zipper::ZipEntry> entries = unzipper.entries();
    //        todo: make sure it gets result passed even if this gets closed
    unzipper.close();
    auto itr = std::find_if(entries.begin(), entries.end(), [=](const zipper::ZipEntry& e) { return e.name == path; });
    if (itr != entries.end()) {
        return *itr;
    }
    throw std::runtime_error("File not found from provided archive: " + zip_file_.filename().string() +
                             std::string("!") + std::string(path));
}

void Zip::Unzip(const zipper::ZipEntry& zip_entry, const boost::filesystem::path& temp_file) {
    zipper::Unzipper unzipper(zip_file_.string());
    unzipper.extractEntry(zip_entry.name, temp_file.string());
    unzipper.close();
}

void Zip::Close() { Cleanup(); }

void Zip::Cleanup() {
    zip_file_.clear();
    if (boost::filesystem::is_directory(temp_zip_file_dir_)) {
        DeleteFileTree(temp_zip_file_dir_);
    }
}

void Zip::GetInputStream(const zipper::ZipEntry& zip_entry, std::fstream& stream) {
    zipper::Unzipper unzipper(zip_file_.string());
    std::vector<unsigned char> unzipped_entry;
    //    todo: this GetEntry can be shortcutted if we avoid converting into zip_entry but use name directly and throw
    //    exception right here
    //    todo: ignoring possible memory limits atm..
    //    unzipper.extractEntryToMemory(zip_entry.name, unzipped_entry);
    //    unzipper.extractEntryToMemory(zip_entry.name, unzipped_entry);
    // todo: try to make it work without extracting to temporary location
    // todo: check if files are removed from temporary
    auto temp = boost::filesystem::temp_directory_path();
    unzipper.extractEntry(zip_entry.name, temp.string());
    //    std::ofstream ostream;
    //    unzipper.extractEntryToStream(zip_entry.name, ostream);
    unzipper.close();
    stream.open(boost::filesystem::canonical(zip_entry.name, temp).string(), std::ifstream::in);
    //    boost::iostreams::copy(ostream,stream);
    //        stream.read(reinterpret_cast<char*>(unzipped_entry.data()), unzipped_entry.size());
    //    stream.rdbuf()->pubsetbuf(reinterpret_cast<char*>(unzipped_entry.data()), unzipped_entry.size());
    if (boost::algorithm::ends_with(zip_entry.name, ".gz")) {
        throw std::runtime_error(".gz stream is not yet supported");
    }

    //    zipper::Unzipper entry_unzipper(zip_file_.filename().string());
    //    entry_unzipper.extractEntryToStream(zip_entry.name, stream);
}

}  // namespace ceres
}  // namespace alus
