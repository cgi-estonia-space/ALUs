#include "ceres-core/dir.h"

#include <iostream>
#include <string>

namespace alus {
namespace ceres {

bool ceres::Dir::IsCompressed() { return false; }

std::vector<std::string> Dir::List(std::string_view path) {
//    todo: this must be checked over later
    auto child = GetFile(path);
    std::vector<std::string> name_set;
    if (boost::filesystem::exists(child) && boost::filesystem::is_directory(child)){
        for (auto& entry : boost::filesystem::directory_iterator(child)){
            name_set.emplace_back(entry.path().filename().string());
        }
    }
    return name_set;
}

boost::filesystem::path Dir::GetFile(std::string_view path) {
    //    todo: this must be checked over later
    boost::filesystem::path child{std::string(path)};
    return boost::filesystem::canonical(child, dir_);
}

bool Dir::Exists(std::string_view path) {
    //    todo: this must be checked over later
    boost::filesystem::path child{std::string(path)};
    try{
        auto check_path = boost::filesystem::canonical(child, dir_);
        return true;
    } catch (const boost::filesystem::filesystem_error& ex) {
        return false;
    }
}

void Dir::GetInputStream([[maybe_unused]]std::string_view path, [[maybe_unused]] std::fstream& stream) {
    //todo: add gzip support if used (boost has some good gzip options for this)
    stream.open(GetFile(path).generic_path().string(), std::ifstream::in);
}

void Dir::Close() {
    //this does nothing
}

}  // namespace ceres
}  // namespace alus
