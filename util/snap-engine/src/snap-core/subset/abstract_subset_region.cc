#include "snap-core/subset/abstract_subset_region.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace alus {
namespace snapengine {

AbstractSubsetRegion::AbstractSubsetRegion(int border_pixels) {
    if (border_pixels < 0) {
        throw std::invalid_argument("The border pixels " + std::to_string(border_pixels) + " is negative.");
    }
    border_pixels_ = border_pixels;
}

void AbstractSubsetRegion::ValidateDefaultSize(int default_product_width, int default_product_height,
                                               std::string_view exception_message_prefix) {
    if (default_product_width < 1) {
        throw std::invalid_argument(std::string(exception_message_prefix) + " width " +
                                    std::to_string(default_product_width) + " must be greater or equal than 1.");
    }
    if (default_product_height < 1) {
        throw std::invalid_argument(std::string(exception_message_prefix) + " height " +
                                    std::to_string(default_product_height) + " must be greater or equal than 1.");
    }
}

}  // namespace snapengine
}  // namespace alus
