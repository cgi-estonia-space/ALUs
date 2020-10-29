#pragma once

#include <stdexcept>

namespace alus{

class ParseException : public std::runtime_error{
   public:
    explicit ParseException(const char *string);
    explicit ParseException(const std::string &arg);
};

}