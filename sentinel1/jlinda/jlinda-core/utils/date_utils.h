#pragma once

#include <string_view>

namespace alus {
namespace s1tbx {
class DateUtils {
public:
    static double DateTimeToSecOfDay(std::string_view date_time);
};
}  // namespace s1tbx
}  // namespace alus