#include "string_utils.h"

#include <sstream>

namespace alus{
namespace snapengine{

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
    // add last also ??
    return tokens;
}
std::string StringUtils::PadNum(int num, int max, char c) {
        std::stringstream str;
        size_t len = std::to_string(num).length();
        while (len < (size_t) max) {
            str << c;
            len++;
        }
        str << num;
        return str.str();
}
} //namespace snapengine
} //namespace alus