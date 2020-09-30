#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../../../external/pugixml/include/pugixml.hpp"

namespace alus {
class PugixmlErrorException : public std::runtime_error {
   public:
    PugixmlErrorException(const std::string_view &source_file,
                          const pugi::xml_document &doc,
                          const pugi::xml_parse_result &err,
                          const std::string_view src,
                          const int src_line)
        : std::runtime_error(
              "XML source file [" + std::string{source_file} +
              "]\n"
              //                                                                   "parsed with errors, attr value: [" +
              //                             std::string{doc.child("node").attribute("attr").value()} + "]\n"
              + "Error description: " + err.description() + "\nError offset: " + std::to_string(err.offset) +
              //                             " (error at [..." + std::string{(source.data() + err.offset)} + "]\n\n" +
              "' at file: " + std::string{src} + ":" + std::to_string(src_line)) {}
};

}  // namespace alus

inline void CheckPugixmlError(const std::string_view source, const char *file, const int line) {
    //    todo: check if this is usable, currently not getting good location for error, but we don't have time for that
    pugi::xml_document doc;
    //    pugi::xml_parse_result result = doc.load_string(source.data());
    pugi::xml_parse_result result = doc.load_file(source.data(), pugi::parse_default | pugi::parse_declaration);
    if (!result) {
        throw alus::PugixmlErrorException(source, doc, result, file, line);
    }
}

#define CHECK_PUGIXML_ERROR(err) CheckPugixmlError(err, __FILE__, __LINE__);