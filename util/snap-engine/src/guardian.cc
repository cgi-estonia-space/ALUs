#include "guardian.h"

#include <stdexcept>
#include <string>

namespace alus {
namespace snapengine {

void Guardian::AssertNotNullOrEmpty(std::string_view expr_text, std::string_view expr_value) {
    if (expr_value.empty()) {
        throw std::invalid_argument(std::string(expr_text) + " argument is empty");
    }
}

}  // namespace snapengine
}  // namespace alus
