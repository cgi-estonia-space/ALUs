#include "snap-core/util/string_utils.h"

#include <iterator>
#include <sstream>

namespace alus {
namespace snapengine {

std::vector<std::string> StringUtils::StringToVectorByDelimiter(std::string_view strv, std::string_view delimiter) {
    std::string str(strv);
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string_view token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        tokens.emplace_back(str.substr(0, pos));
        str.erase(0, pos + delimiter.size());
    }
    tokens.emplace_back(str);
    return tokens;
}

std::string StringUtils::PadNum(int num, int max, char c) {
    std::stringstream str;
    size_t len = std::to_string(num).length();
    while (len < static_cast<size_t>(max)) {
        str << c;
        len++;
    }
    str << num;
    return str.str();
}

std::string StringUtils::ArrayToString(std::vector<std::string> vec, std::string_view delimiter) {
    std::ostringstream oss;
    if (!vec.empty()) {
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<std::string>(oss, std::string(delimiter).c_str()));
        // last element with no delimiter
        oss << vec.back();
    }
    return oss.str();
}

}  // namespace snapengine
}  // namespace alus