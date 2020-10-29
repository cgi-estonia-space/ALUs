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
void Guardian::AssertNotNull(std::string_view expr_text, boost::posix_time::time_input_facet *expr_value) {
    if (expr_value == nullptr) {
        throw std::invalid_argument(std::string(expr_text) + " argument is nullptr");
    }
}

void Guardian::AssertNotNull(std::string_view expr_text, std::shared_ptr<ProductData> &expr_value) {
    if (expr_value == nullptr) {
        throw std::invalid_argument(std::string(expr_text) + " argument is nullptr");
    }
}

}  // namespace snapengine
}  // namespace alus
