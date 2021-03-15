#include "snap-core/util/io/file_utils.h"

#include <boost/algorithm/string/predicate.hpp>

#include "guardian.h"

namespace alus {
namespace snapengine {

std::string FileUtils::ExchangeExtension(std::string_view path, std::string_view extension) {
    Guardian::AssertNotNullOrEmpty("path", path);
    Guardian::AssertNotNull("extension", extension);
    if (extension.length() > 0 && boost::algorithm::ends_with(path, extension)) {
        return std::string(path);
    }
    auto extension_dot_pos = GetExtensionDotPos(path);
    if (extension_dot_pos > 0) {
        // replace existing extension
        return std::string(path.substr(0, extension_dot_pos)) + std::string(extension);
    } else {
        // append extension
        return std::string(path) + std::string(extension);
    }
}

int FileUtils::GetExtensionDotPos(std::string_view path) {
    Guardian::AssertNotNullOrEmpty("path", path);
//    todo: might need to rethink this logic
    std::size_t extension_dot_pos = 0;
    if( path.find_last_of('.') != std::string_view::npos){
        extension_dot_pos = path.find_last_of('.');
    }
    if (extension_dot_pos > 0) {
        std::size_t last_separator_pos = 0;
        if(path.find_last_of('/') != std::string_view::npos){
            last_separator_pos = path.find_last_of('/');
        }
        if(path.find_last_of('\\') != std::string_view::npos){
            last_separator_pos = std::max(last_separator_pos, path.find_last_of('\\'));
        }
        if(path.find_last_of(':') != std::string_view::npos) {
            last_separator_pos = std::max(last_separator_pos, path.find_last_of(':'));
        }
        if (last_separator_pos < extension_dot_pos - 1) {
            return extension_dot_pos;
        }
    }
    return -1;
}
}  // namespace snapengine
}  // namespace alus
