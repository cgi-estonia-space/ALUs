#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace alus {
namespace snapengine {

/**
 * consider using boost instead!
 */
class StringUtils {
   public:
    static std::vector<std::string> StringToVectorByDelimiter(std::string_view str, std::string_view delimiter);
};

}  // namespace snapengine
}  // namespace alus