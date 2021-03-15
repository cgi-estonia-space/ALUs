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

    /**
     * Adds padding to an integer
     * 1 becomes 001 or __1
     * @param num the integer value
     * @param max the desired string length
     * @param c the inserted character
     * @return padded number as string
     */
    static std::string PadNum(int num, int max, char c);

    /**
     * Converts an std::vector into a string.
     *
     * @param array the array object
     * @param s     the separator string, e.g. ","
     * @return a string represenation of the array
     * @throws std::invalid_argument if the given Object is not an <code>array</code> or <code>null</code>.
     */
    static std::string ArrayToString(std::vector<std::string>, std::string_view delimiter);
};

}  // namespace snapengine
}  // namespace alus