#pragma once

#include <string_view>

namespace alus {
namespace snapengine {
class DateUtils {
   public:
    static double DateTimeToSecOfDay(std::string_view date_time);
};
}  // namespace snapengine
}  // namespace alus