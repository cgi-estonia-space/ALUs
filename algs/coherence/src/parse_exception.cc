#include "../include/parse_exception.h"

namespace alus {

ParseException::ParseException(const char *string) : runtime_error(string) {}
ParseException::ParseException(const std::string &arg) : runtime_error(arg) {}

}  // namespace alus