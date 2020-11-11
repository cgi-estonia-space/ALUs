#include "guardian.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace alus {
namespace snapengine {

void Guardian::AssertNotNullOrEmpty(std::string_view expr_text, std::string_view expr_value) {
    if (expr_value.empty()) {
        throw std::invalid_argument(std::string(expr_text) + " argument is empty");
    }
}
void Guardian::AssertEquals(std::string_view expr_text, long expr_value, long expected_value) {
    if (expected_value != expr_value) {
        std::stringstream sb;
        sb << "[" << expr_text << "]"
           << " is [" << expr_value << "] but should be equal to [" << expected_value << "]";
        throw std::invalid_argument(sb.str());
    }
}
void Guardian::AssertWithinRange(std::string_view expr_text, long expr_value, long range_min, long range_max) {
    if (expr_value < range_min || expr_value > range_max) {
        std::stringstream sb;
        sb << "[" << expr_text << "]"
           << " is [" << expr_value << "]  but should be in the range [" << range_min << "] to [" << range_max << "]";
        throw std::invalid_argument(sb.str());
    }
}

// void Guardian::AssertSame(std::string_view expr_text, std::any expr_value, std::any expected_value) {
//    if (expected_value != expr_value) {
//        std::stringstream sb;
//        sb << "[" << expr_text << "]"
//           << " is [" << expr_value << "] but should be same as [" << expected_value << "]";
//        throw std::invalid_argument(sb.str());
//    }
//}

}  // namespace snapengine
}  // namespace alus
