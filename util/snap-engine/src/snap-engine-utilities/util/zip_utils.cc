#include "snap-engine-utilities/util/zip_utils.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "zipper/unzipper.h"
#include "zipper/zipper.h"

namespace alus {
namespace snapengine {

bool ZipUtils::IsZipped(const boost::filesystem::path& file) {
    std::string name = file.filename().string();
    boost::to_lower(name);
    for (auto const& ext : EXT_LIST) {
        if (boost::algorithm::ends_with(name, ext)) {
            return true;
        }
    }
    return false;
}
bool ZipUtils::IsZip(const boost::filesystem::path& input_path) {
    //    auto path = input_path.filename().string();
    auto path = boost::filesystem::canonical(input_path).string();
    boost::to_lower(path);
    return boost::algorithm::ends_with(path, ".zip");
}

// todo: this needs further testing
std::string ZipUtils::GetRootFolder(const boost::filesystem::path& file, std::string_view header_file_name) {
    zipper::Unzipper unzipper(boost::filesystem::canonical(file).string());
    std::vector<zipper::ZipEntry> entries = unzipper.entries();
    unzipper.close();
    auto found_entry = std::find_if(std::begin(entries), std::end(entries), [header_file_name](const auto& entry) {
        if (!boost::ends_with(entry.name, "/")) {
            std::string entry_name_lower = entry.name;
            boost::to_lower(entry_name_lower);
            return boost::ends_with(entry_name_lower, header_file_name);
        }
        return false;
    });

    if (found_entry != entries.end()) {
        std::string found_entry_name = found_entry->name;
        auto sep_index = found_entry_name.rfind('/');
        if (sep_index > 0) {
            return found_entry_name.substr(0, sep_index) + "/";
        }
        return "";
    }
    return "";
}

bool ZipUtils::FindInZip(const boost::filesystem::path& file, std::string_view prefix, std::string_view suffix) {
    try {
        std::string lowercase_suffix(suffix);
        boost::algorithm::to_lower(lowercase_suffix);

        std::string lowercase_prefix(prefix);
        boost::algorithm::to_lower(lowercase_prefix);
        zipper::Unzipper product_zip(boost::filesystem::canonical(file).string());
        auto entries = product_zip.entries();
        auto filtered_entries =
            std::find_if(entries.begin(), entries.end(), [lowercase_suffix, lowercase_prefix](auto const& entry) {
                std::string lowercase_name = entry.name;
                boost::algorithm::to_lower(lowercase_name);
                return !boost::ends_with(entry.name, "/") &&
                       boost::algorithm::ends_with(lowercase_name, lowercase_suffix) &&
                       boost::algorithm::starts_with(lowercase_name, lowercase_prefix);
            });
        if (filtered_entries != entries.end()) {
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "unable to read zip file " << file.filename().string() << ": " << e.what() << std::endl;
    }
    return false;
}

}  // namespace snapengine
}  // namespace alus
